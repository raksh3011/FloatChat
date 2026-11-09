"""Export the PostgreSQL database to the static browser demo: profiles.json + emb.bin (int8, 384-d, L2-normalised).

usage: python tools/export_static.py <out_dir>        (uses FLOATCHAT_DSN)
"""
import json, sys
from pathlib import Path

import numpy as np

from floatchat import db

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
LEVELS = list(range(0, 1001, 20))

with db.connect() as c:
    prof = c.execute("""SELECT profile_id, wmo_id, juld AT TIME ZONE 'UTC', lat, lon, sst, sss, mld_dbar, thermocline_dbar,
                        summary, embedding FROM profiles WHERE embedding IS NOT NULL ORDER BY profile_id""").fetchall()
    series = {}
    for pid, vid, pres, val in c.execute("""SELECT profile_id, variable_id, array_agg(pres_dbar ORDER BY pres_dbar),
                        array_agg(value ORDER BY pres_dbar) FROM measurements GROUP BY 1, 2"""):
        series[(pid, vid)] = (np.array(pres), np.array(val))


def interp(key):
    if key not in series:
        return None
    p, v = series[key]
    res = [round(float(np.interp(L, p, v)), 2) if p[0] - 5 <= L <= p[-1] + 5 else None for L in LEVELS]
    return res if sum(x is not None for x in res) >= 5 else None


profiles, emb = [], []
for i, (pid, wmo, t, lat, lon, sst, sss, mld, th, summ, e) in enumerate(prof):
    T, S = interp((pid, 1)), interp((pid, 2))
    if T is None and S is None:
        continue
    profiles.append({"id": len(profiles), "wmo": wmo, "cy": None, "t": t.strftime("%Y-%m-%dT%H:%MZ"),
                     "lat": round(lat, 3), "lon": round(lon, 3), "T": T, "S": S,
                     "f": [None if x is None else round(float(x), 2) for x in (sst, sss, mld, th)], "s": summ})
    emb.append(np.array(e, dtype="f4"))
M = np.vstack(emb)
q = np.clip(np.round(M * 127), -127, 127).astype("i1")
(out / "emb.bin").write_bytes(q.tobytes())
json.dump({"levels": LEVELS, "dim": M.shape[1], "profiles": profiles}, open(out / "profiles.json", "w"), separators=(",", ":"))
print(len(profiles), "profiles;", M.shape, "emb;", round((out / 'profiles.json').stat().st_size / 1e6, 2), "MB json")
