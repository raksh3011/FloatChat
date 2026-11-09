"""Idempotent, transactional NetCDF -> PostgreSQL ingestion.

* Each file is fingerprinted with SHA-256. A known checksum is skipped; a known path with a new checksum is reported.
* Profiles and measurements are staged in TEMP tables inside a single transaction, then merged into the real
  tables with set-based SQL. A failure rolls the whole file back.
* Rows are loaded in batches (default 5,000).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import psycopg

from .features import features, summarize
from .netcdf import Profile, read_profiles

log = logging.getLogger("floatchat.ingest")
BATCH = 5000


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class Result:
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_changed: int = 0
    profiles: int = 0
    measurements: int = 0


def _batched(rows: list, n: int) -> Iterable[list]:
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def _measurement_rows(idx: int, p: Profile):
    for var_id, vals, qc in ((1, p.temp, p.qc_temp), (2, p.psal, p.qc_psal)):
        for pres, v, q in zip(p.pres, vals, qc):
            if not (np.isnan(v) or np.isnan(pres)):
                yield (idx, var_id, float(pres), float(v), str(q))


def ingest_file(conn: psycopg.Connection, path: Path, batch: int = BATCH) -> tuple[str, int, int]:
    """Ingest one file. Returns (status, profiles, measurements); status in ingested|skipped|changed."""
    checksum = sha256_file(path)
    if conn.execute("SELECT 1 FROM raw_files WHERE checksum=%s", (checksum,)).fetchone():
        return "skipped", 0, 0
    prior = conn.execute("SELECT checksum FROM raw_files WHERE path=%s", (str(path),)).fetchone()
    if prior:
        log.warning("%s changed since last ingestion (%s -> %s); inspect before re-ingesting", path, prior[0][:8], checksum[:8])
        return "changed", 0, 0

    conn.commit()  # end the read-only implicit transaction so the block below is a real transaction
    profiles = list(read_profiles(path))
    with conn.transaction():
        conn.execute("INSERT INTO raw_files (checksum, path, n_profiles) VALUES (%s,%s,%s)", (checksum, str(path), len(profiles)))
        conn.execute("""CREATE TEMP TABLE tmp_profiles (idx INT, wmo_id TEXT, cycle INT, juld TIMESTAMPTZ, lat FLOAT8,
                        lon FLOAT8, mode CHAR(1), sst REAL, sss REAL, mld REAL, thermo REAL, maxp REAL, summary TEXT,
                        platform_type TEXT, pi TEXT) ON COMMIT DROP""")
        conn.execute("""CREATE TEMP TABLE tmp_meas (idx INT, var SMALLINT, pres REAL, value REAL, qc CHAR(1))
                        ON COMMIT DROP""")
        prof_rows, n_meas = [], 0
        meas_buf: list = []
        with conn.cursor() as cur:
            for i, p in enumerate(profiles):
                f = features(p)
                prof_rows.append((i, p.wmo_id, p.cycle, p.time, p.lat, p.lon, p.data_mode, f["sst"], f["sss"],
                                  f["mld_dbar"], f["thermocline_dbar"], f["max_pres_dbar"], summarize(p, f),
                                  p.platform_type, p.pi_name))
                meas_buf.extend(_measurement_rows(i, p))
                while len(meas_buf) >= batch:
                    cur.executemany("INSERT INTO tmp_meas VALUES (%s,%s,%s,%s,%s)", meas_buf[:batch])
                    n_meas += batch
                    meas_buf = meas_buf[batch:]
            if meas_buf:
                cur.executemany("INSERT INTO tmp_meas VALUES (%s,%s,%s,%s,%s)", meas_buf)
                n_meas += len(meas_buf)
            for chunk in _batched(prof_rows, batch):
                cur.executemany("INSERT INTO tmp_profiles VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", chunk)

        conn.execute("""INSERT INTO floats (wmo_id, platform_type, pi_name)
                        SELECT DISTINCT ON (wmo_id) wmo_id, platform_type, pi FROM tmp_profiles
                        ON CONFLICT (wmo_id) DO NOTHING""")
        conn.execute("""INSERT INTO profiles (wmo_id, cycle_number, juld, lat, lon, data_mode, source_sha256, sst, sss,
                        mld_dbar, thermocline_dbar, max_pres_dbar, summary)
                        SELECT wmo_id, cycle, juld, lat, lon, mode, %s, sst, sss, mld, thermo, maxp, summary
                        FROM tmp_profiles ON CONFLICT (wmo_id, cycle_number, juld) DO NOTHING""", (checksum,))
        conn.execute("""INSERT INTO measurements (profile_id, variable_id, pres_dbar, value, qc_flag)
                        SELECT pr.profile_id, m.var, m.pres, m.value, m.qc
                        FROM tmp_meas m
                        JOIN tmp_profiles t ON t.idx = m.idx
                        JOIN profiles pr ON pr.wmo_id = t.wmo_id AND pr.cycle_number IS NOT DISTINCT FROM t.cycle AND pr.juld = t.juld
                        WHERE pr.source_sha256 = %s
                        ON CONFLICT DO NOTHING""", (checksum,))
    return "ingested", len(profiles), n_meas


def ingest_dir(conn: psycopg.Connection, directory: str | Path, batch: int = BATCH) -> Result:
    res = Result()
    for path in sorted(Path(directory).rglob("*.nc")):
        res.files_seen += 1
        status, n_p, n_m = ingest_file(conn, path, batch)
        if status == "ingested":
            res.files_ingested += 1
            res.profiles += n_p
            res.measurements += n_m
        elif status == "skipped":
            res.files_skipped += 1
        else:
            res.files_changed += 1
        log.info("%s: %s (%d profiles)", path.name, status, n_p)
    return res
