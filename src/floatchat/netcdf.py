"""Reading Argo profile NetCDF files with heuristic (schema-agnostic) variable detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np

# Priority-ordered aliases: providers and file generations name the same quantity differently.
ALIASES: dict[str, list[str]] = {
    "lat": ["LATITUDE", "LAT", "latitude"],
    "lon": ["LONGITUDE", "LON", "longitude"],
    "time": ["JULD", "TIME", "time", "JULD_LOCATION"],
    "wmo": ["PLATFORM_NUMBER", "WMO", "WMO_ID", "PLATFORM"],
    "cycle": ["CYCLE_NUMBER", "CYCLE"],
    "pres": ["PRES", "PRESSURE", "DEPTH"],
    "temp": ["TEMP", "TEMPERATURE", "TEMP_ADJUSTED"],
    "psal": ["PSAL", "SALINITY", "PSAL_ADJUSTED"],
    "mode": ["DATA_MODE"],
    "platform_type": ["PLATFORM_TYPE"],
    "pi": ["PI_NAME"],
}
GOOD_QC = {"1", "2"}  # good / probably good (Argo QC flag scale)


def guess_variable_names(ds: nc.Dataset) -> dict[str, str | None]:
    """Map logical field -> actual variable name in this file (first alias that exists)."""
    present = set(ds.variables)
    return {logical: next((a for a in names if a in present), None) for logical, names in ALIASES.items()}


@dataclass
class Profile:
    wmo_id: str
    cycle: int | None
    time: datetime
    lat: float
    lon: float
    data_mode: str | None
    platform_type: str | None
    pi_name: str | None
    pres: np.ndarray
    temp: np.ndarray  # NaN where missing / bad QC
    psal: np.ndarray
    qc_temp: np.ndarray = field(default_factory=lambda: np.array([]))
    qc_psal: np.ndarray = field(default_factory=lambda: np.array([]))


def _text(arr) -> str:
    """Fixed-width char array / bytes -> clean str."""
    try:
        if isinstance(arr, (bytes, np.bytes_)):
            return arr.decode(errors="ignore").strip()
        a = np.ma.filled(arr, b"").ravel()
        return b"".join(x if isinstance(x, bytes) else bytes(str(x), "ascii") for x in a).decode(errors="ignore").strip()
    except Exception:
        return str(arr).strip()


def _qc_chars(var, i: int, n: int) -> np.ndarray:
    try:
        raw = np.ma.filled(var[i], b"9")
        return np.array([(c.decode() if isinstance(c, bytes) else str(c)) for c in raw[:n]])
    except Exception:
        return np.array(["9"] * n)


def _masked(values, qc: np.ndarray) -> np.ndarray:
    out = np.ma.filled(np.ma.asarray(values, dtype="f8"), np.nan)
    ok = np.isin(qc, list(GOOD_QC)) if qc.size == out.size else np.ones(out.size, bool)
    out[~ok] = np.nan
    return out


def _juld_to_dt(value, units: str) -> datetime:
    # Argo JULD: days since 1950-01-01 00:00:00 UTC
    base = datetime(1950, 1, 1, tzinfo=timezone.utc)
    if "since" in units:
        try:
            base = datetime.fromisoformat(units.split("since")[1].strip().replace("Z", "+00:00"))
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return base + timedelta(days=float(value))


def read_profiles(path: str | Path):
    """Yield Profile objects from one file. Skips profiles with no position/time or no valid data."""
    ds = nc.Dataset(str(path))
    try:
        m = guess_variable_names(ds)
        for need in ("lat", "lon", "time", "pres"):
            if m[need] is None:
                raise ValueError(f"{path}: cannot find a '{need}' variable (aliases: {ALIASES[need]})")
        n_prof = ds.dimensions["N_PROF"].size if "N_PROF" in ds.dimensions else 1
        lat, lon, tim = ds[m["lat"]][:], ds[m["lon"]][:], ds[m["time"]]
        units = getattr(tim, "units", "days since 1950-01-01 00:00:00")
        tv = tim[:]
        pres_v = ds[m["pres"]]
        for i in range(n_prof):
            la, lo, t = (np.ma.atleast_1d(x)[i] if np.ma.atleast_1d(x).size > i else np.ma.masked for x in (lat, lon, tv))
            if np.ma.is_masked(la) or np.ma.is_masked(lo) or np.ma.is_masked(t):
                continue
            pres_raw = pres_v[i] if pres_v.ndim > 1 else pres_v[:]
            pres = np.ma.filled(np.ma.asarray(pres_raw, dtype="f8"), np.nan)
            n = pres.size
            series = {}
            for key, qcname in (("temp", "TEMP"), ("psal", "PSAL")):
                name = m[key]
                if name is None:
                    series[key] = (np.full(n, np.nan), np.array(["9"] * n))
                    continue
                var = ds[name]
                mode = _text(ds[m["mode"]][i]) if m["mode"] else ""
                # prefer delayed/adjusted values when the file provides them for this profile
                adj = name + "_ADJUSTED"
                if mode in ("A", "D") and adj in ds.variables and np.ma.count(ds[adj][i]) > 0:
                    var, qcname = ds[adj], name + "_ADJUSTED"
                else:
                    qcname = name
                qc = _qc_chars(ds[qcname + "_QC"], i, n) if (qcname + "_QC") in ds.variables else np.array(["1"] * n)
                series[key] = (_masked(var[i], qc), qc)
            temp, qt = series["temp"]
            psal, qs = series["psal"]
            ok = ~np.isnan(pres) & (~np.isnan(temp) | ~np.isnan(psal))
            if ok.sum() < 3:
                continue
            yield Profile(
                wmo_id=_text(ds[m["wmo"]][i]) if m["wmo"] else "unknown",
                cycle=int(ds[m["cycle"]][i]) if m["cycle"] and not np.ma.is_masked(ds[m["cycle"]][i]) else None,
                time=_juld_to_dt(t, units),
                lat=float(la),
                lon=float(lo),
                data_mode=(_text(ds[m["mode"]][i]) or None) if m["mode"] else None,
                platform_type=(_text(ds[m["platform_type"]][i]) or None) if m["platform_type"] else None,
                pi_name=(_text(ds[m["pi"]][i]) or None) if m["pi"] else None,
                pres=pres[ok], temp=temp[ok], psal=psal[ok], qc_temp=qt[ok], qc_psal=qs[ok],
            )
    finally:
        ds.close()
