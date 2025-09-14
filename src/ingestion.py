import os
import argparse
import hashlib
import netCDF4 as nc
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

load_dotenv()


def compute_checksum(data):
    # deterministic string representation for checksum
    return hashlib.sha256(str(data).encode()).hexdigest()


def _to_py_str(val):
    """
    Convert netCDF string-like objects to Python str.
    Handles fixed-width char arrays and numpy bytes.
    """
    if val is None:
        return None
    if isinstance(val, (bytes, bytearray)):
        return val.decode(errors='ignore').strip()
    try:
        if hasattr(val, 'tobytes'):
            b = val.tobytes()
            return b.decode(errors='ignore').strip()
    except Exception:
        pass
    return str(val).strip()


def get_variable_safe(ds, var_name, prof_idx, default=None, is_string=False):
    """
    Safely extract from ds.variables[var_name] the value for profile index prof_idx.
    - If variable has a 'N_PROF' dimension, index that axis at prof_idx (keeping other axes full slices).
    - If no 'N_PROF', try to return scalar or 1-D variable item (guarded).
    - Return default on any bounds error.
    """
    if var_name not in ds.variables:
        return default
    var = ds.variables[var_name]
    try:
        # dims contain 'N_PROF' -> build index tuple
        if hasattr(var, 'dimensions') and 'N_PROF' in var.dimensions:
            axis = var.dimensions.index('N_PROF')
            axis_size = var.shape[axis]
            if prof_idx < 0 or prof_idx >= axis_size:
                return default
            idx = [slice(None)] * getattr(var, 'ndim', len(var.shape))
            idx[axis] = prof_idx
            idx = tuple(idx)
            value = var[idx]
        else:
            # no N_PROF dim: if scalar
            if getattr(var, 'ndim', 0) == 0:
                value = var[...]
            elif getattr(var, 'ndim', 1) == 1 and prof_idx < var.shape[0]:
                value = var[prof_idx]
            else:
                value = var[...]
    except IndexError:
        return default
    except Exception:
        return default

    # handle masked arrays
    if isinstance(value, np.ma.MaskedArray):
        if value.mask is True or (hasattr(value.mask, 'all') and value.mask.all()):
            return default
        value = value.filled(np.nan)

    if is_string:
        return _to_py_str(value)

    try:
        return np.asarray(value)
    except Exception:
        return value


def ingest_netcdf(nc_dir, pg_uri, batch_size):
    conn = psycopg2.connect(pg_uri)
    cur = conn.cursor()

    # variable name -> variable_id map
    cur.execute("SELECT variable_name, variable_id FROM measurement_variables")
    var_map = {row[0]: row[1] for row in cur.fetchall()}

    float_inserts = []
    profile_inserts = []      # rows: (wmo_id, cycle_number, juld, lat, lon, direction, data_mode, checksum)
    measurement_inserts = []  # rows: (checksum, variable_id, depth, value, qc_flag)

    for nc_file in os.listdir(nc_dir):
        if not nc_file.endswith('.nc'):
            continue
        path = os.path.join(nc_dir, nc_file)
        ds = nc.Dataset(path)
        n_prof = ds.dimensions['N_PROF'].size if 'N_PROF' in ds.dimensions else 1

        for prof_idx in range(n_prof):
            wmo_id = get_variable_safe(ds, 'PLATFORM_NUMBER', prof_idx, default='', is_string=True)

            cycle_val = get_variable_safe(ds, 'CYCLE_NUMBER', prof_idx, default=None)
            try:
                if cycle_val is None:
                    cycle_number = 0
                else:
                    cycle_number = int(np.asscalar(cycle_val)) if (np.size(cycle_val) == 1) else int(cycle_val)
            except Exception:
                cycle_number = 0

            juld_val = get_variable_safe(ds, 'JULD', prof_idx, default=None)
            if juld_val is not None:
                try:
                    juld = pd.to_datetime(np.asscalar(juld_val), unit='D', origin='1950-01-01')
                except Exception:
                    try:
                        juld = pd.to_datetime(juld_val, unit='D', origin='1950-01-01')
                    except Exception:
                        juld = None
            else:
                juld = None

            lat_val = get_variable_safe(ds, 'LATITUDE', prof_idx, default=None)
            try:
                lat = float(np.asscalar(lat_val)) if (lat_val is not None and np.size(lat_val) == 1) else (float(lat_val) if lat_val is not None else None)
            except Exception:
                lat = None

            lon_val = get_variable_safe(ds, 'LONGITUDE', prof_idx, default=None)
            try:
                lon = float(np.asscalar(lon_val)) if (lon_val is not None and np.size(lon_val) == 1) else (float(lon_val) if lon_val is not None else None)
            except Exception:
                lon = None

            direction = get_variable_safe(ds, 'DIRECTION', prof_idx, default='', is_string=True) or ''
            data_mode = get_variable_safe(ds, 'DATA_MODE', prof_idx, default='', is_string=True) or ''

            # build profile_data for checksum -> only variables that actually have prof dimension and include this prof
            profile_data = {}
            for k, var in ds.variables.items():
                if hasattr(var, 'dimensions') and 'N_PROF' in var.dimensions:
                    axis = var.dimensions.index('N_PROF')
                    if prof_idx < var.shape[axis]:
                        v = get_variable_safe(ds, k, prof_idx, default=None, is_string=(getattr(var, 'dtype', None) is not None and (str(var.dtype).startswith('S') or var.dtype.type is np.bytes_)))
                        profile_data[k] = None if v is None else (v.tolist() if hasattr(v, 'tolist') else v)

            checksum = compute_checksum(profile_data)
            cur.execute("SELECT profile_id FROM profiles WHERE checksum = %s", (checksum,))
            if cur.fetchone():
                continue

            float_inserts.append((wmo_id, getattr(ds, 'platform_type', ''), getattr(ds, 'pi_name', ''), getattr(ds, 'country', '')))
            profile_inserts.append((wmo_id, cycle_number, juld, lat, lon, direction, data_mode, checksum))

            # measurements: store checksum as placeholder for profile linkage later
            for varname in ['TEMP', 'PSAL', 'PRES', 'DOXY']:
                if varname not in ds.variables or varname not in var_map:
                    continue
                var_data = ds.variables[varname]
                if not hasattr(var_data, 'dimensions') or 'N_PROF' not in var_data.dimensions:
                    continue
                prof_axis = var_data.dimensions.index('N_PROF')
                if prof_idx >= var_data.shape[prof_axis]:
                    continue

                idx = [slice(None)] * var_data.ndim
                idx[prof_axis] = prof_idx
                idx = tuple(idx)
                vals = var_data[idx]
                if isinstance(vals, np.ma.MaskedArray):
                    vals = vals.compressed()
                else:
                    vals = np.asarray(vals).ravel()

                qc_var = f"{varname}_QC"
                if qc_var in ds.variables and 'N_PROF' in ds.variables[qc_var].dimensions:
                    qc_axis = ds.variables[qc_var].dimensions.index('N_PROF')
                    if prof_idx < ds.variables[qc_var].shape[qc_axis]:
                        qidx = [slice(None)] * ds.variables[qc_var].ndim
                        qidx[qc_axis] = prof_idx
                        qidx = tuple(qidx)
                        qvals = ds.variables[qc_var][qidx]
                        if isinstance(qvals, np.ma.MaskedArray):
                            qvals = qvals.compressed()
                        else:
                            qvals = np.asarray(qvals).ravel()
                    else:
                        qvals = np.array([''] * len(vals))
                else:
                    qvals = np.array([''] * len(vals))

                if varname != 'PRES' and 'PRES' in ds.variables and 'N_PROF' in ds.variables['PRES'].dimensions:
                    pres_var = ds.variables['PRES']
                    pres_axis = pres_var.dimensions.index('N_PROF')
                    if prof_idx < pres_var.shape[pres_axis]:
                        pidx = [slice(None)] * pres_var.ndim
                        pidx[pres_axis] = prof_idx
                        pidx = tuple(pidx)
                        depths = pres_var[pidx]
                        if isinstance(depths, np.ma.MaskedArray):
                            depths = depths.compressed()
                        else:
                            depths = np.asarray(depths).ravel()
                    else:
                        depths = vals
                else:
                    depths = vals

                for depth, val, qc in zip(depths, vals, np.resize(qvals, vals.shape)):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    measurement_inserts.append((checksum, var_map[varname], float(depth), float(val), _to_py_str(qc)))

        ds.close()

    # INSERT floats (wmo-level)
    if float_inserts:
        execute_batch(cur, """
            INSERT INTO floats (wmo_id, platform_type, pi_name, country)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (wmo_id) DO NOTHING
        """, float_inserts, page_size=batch_size)

    # INSERT profiles (no RETURNING). We will fetch profile_ids by checksum afterwards.
    if profile_inserts:
        execute_batch(cur, """
            INSERT INTO profiles (wmo_id, cycle_number, juld, lat, lon, direction, data_mode, checksum)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (checksum) DO NOTHING
        """, profile_inserts, page_size=batch_size)

    conn.commit()

    unique_checksums = list({row[7] for row in profile_inserts}) if profile_inserts else []
    checksum_to_profile_id = {}
    if unique_checksums:
        for i in range(0, len(unique_checksums), batch_size):
            chunk = unique_checksums[i:i+batch_size]
            sql = "SELECT profile_id, checksum FROM profiles WHERE checksum = ANY(%s)"
            cur.execute(sql, (chunk,))
            for pid, ch in cur.fetchall():
                checksum_to_profile_id[ch] = pid

    final_measurements = []
    for chksum, var_id, depth, val, qc in measurement_inserts:
        pid = checksum_to_profile_id.get(chksum)
        if pid is None:
            continue
        final_measurements.append((pid, var_id, depth, val, qc))

    if final_measurements:
        execute_batch(cur, """
            INSERT INTO measurements (profile_id, variable_id, depth, value, qc_flag)
            VALUES (%s, %s, %s, %s, %s)
        """, final_measurements, page_size=batch_size)

    conn.commit()
    print(f"Inserted: profiles={len(profile_inserts)} measurements={len(final_measurements)}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_dir", required=True)
    parser.add_argument("--pg", required=True)
    parser.add_argument("--batch", type=int, default=1000)
    args = parser.parse_args()
    ingest_netcdf(args.nc_dir, args.pg, args.batch)
