"""Download the most recent daily Argo profile files for a basin from the public GDAC (Ifremer mirror).

usage: python tools/fetch_argo.py [--days 31] [--basin indian_ocean] [--out data/recent]

The GDAC publishes one NetCDF per day per ocean basin at
https://data-argo.ifremer.fr/geo/<basin>/<YYYY>/<MM>/<YYYYMMDD>_prof.nc, a day or two behind real time.
Walks back from today and keeps the newest `--days` files that exist; files already on disk are skipped.
Then:  floatchat ingest data/recent && floatchat embed
"""
from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = "https://data-argo.ifremer.fr/geo"


def fetch(url: str, dest: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest.with_suffix(".part"), "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        dest.with_suffix(".part").unlink(missing_ok=True)
        if e.code == 404:
            return False
        raise
    dest.with_suffix(".part").replace(dest)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=31)
    ap.add_argument("--basin", default="indian_ocean", choices=["indian_ocean", "atlantic_ocean", "pacific_ocean"])
    ap.add_argument("--out", default="data/recent")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    got, d, misses = 0, date.today(), 0
    while got < a.days and misses < 20:
        name = f"{d:%Y%m%d}_prof.nc"
        dest = out / name
        if dest.exists():
            got += 1
        elif fetch(f"{BASE}/{a.basin}/{d:%Y/%m}/{name}", dest):
            got += 1
            print("downloaded", name)
        else:
            misses += 1  # not published yet
        d -= timedelta(days=1)
    print(f"{got} daily files in {out}")


if __name__ == "__main__":
    main()
