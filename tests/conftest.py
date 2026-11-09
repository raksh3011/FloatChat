"""Shared fixtures. DB tests need a PostgreSQL server: set FLOATCHAT_TEST_DSN (a scratch database) to enable them."""
import os
from datetime import datetime

import netCDF4 as nc
import numpy as np
import pytest

TEST_DSN = os.getenv("FLOATCHAT_TEST_DSN")
needs_db = pytest.mark.skipif(not TEST_DSN, reason="set FLOATCHAT_TEST_DSN to run database tests")


def make_argo_file(path, n_prof=3, n_lev=60, base_day=25000.0, wmo=("1900001", "1900002", "1900003"), alias=False):
    """Write a small synthetic Argo-style profile file. alias=True uses alternative variable names."""
    ds = nc.Dataset(path, "w")
    ds.createDimension("N_PROF", n_prof)
    ds.createDimension("N_LEVELS", n_lev)
    ds.createDimension("STR8", 8)
    names = dict(lat="LAT", lon="LON", time="TIME", pres="PRESSURE", temp="TEMPERATURE", psal="SALINITY") if alias \
        else dict(lat="LATITUDE", lon="LONGITUDE", time="JULD", pres="PRES", temp="TEMP", psal="PSAL")
    ds.createVariable(names["lat"], "f8", ("N_PROF",))[:] = [10.0, 15.5, -20.0][:n_prof]
    ds.createVariable(names["lon"], "f8", ("N_PROF",))[:] = [65.0, 88.0, 75.0][:n_prof]
    t = ds.createVariable(names["time"], "f8", ("N_PROF",))
    t.units = "days since 1950-01-01 00:00:00 UTC"
    t[:] = [base_day + i for i in range(n_prof)]
    pn = ds.createVariable("PLATFORM_NUMBER", "S1", ("N_PROF", "STR8"))
    for i in range(n_prof):
        pn[i] = nc.stringtochar(np.array(wmo[i].ljust(8), dtype="S8"))
    ds.createVariable("CYCLE_NUMBER", "i4", ("N_PROF",))[:] = list(range(1, n_prof + 1))
    pres = np.tile(np.linspace(2, 1000, n_lev), (n_prof, 1))
    ds.createVariable(names["pres"], "f8", ("N_PROF", "N_LEVELS"))[:] = pres
    # warm mixed layer to ~50 dbar, then a sharp thermocline
    temp = np.where(pres < 50, 28.0, 28.0 - (pres - 50) * 0.02)
    psal = 35.0 + 0.001 * pres
    tv = ds.createVariable(names["temp"], "f8", ("N_PROF", "N_LEVELS"), fill_value=99999.0)
    sv = ds.createVariable(names["psal"], "f8", ("N_PROF", "N_LEVELS"), fill_value=99999.0)
    tv[:], sv[:] = temp, psal
    qc = ds.createVariable("TEMP_QC", "S1", ("N_PROF", "N_LEVELS"))
    q = np.full((n_prof, n_lev), b"1", dtype="S1")
    q[0, 5] = b"4"  # one bad reading that must be dropped
    qc[:] = q
    ds.close()
    return path


@pytest.fixture
def argo_dir(tmp_path):
    make_argo_file(tmp_path / "a.nc")
    make_argo_file(tmp_path / "b.nc", base_day=25010.0, wmo=("1900011", "1900012", "1900013"), alias=True)
    return tmp_path


@pytest.fixture
def conn():
    import psycopg
    from floatchat import db
    c = psycopg.connect(TEST_DSN)
    c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    c.commit()
    db.apply_schema(c)
    yield c
    c.close()
