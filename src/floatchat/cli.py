"""Command line: floatchat init | ingest <dir> | embed | ask "<question>" | serve"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from . import db


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="floatchat")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create tables (+ pgvector column and HNSW index if available)")
    p = sub.add_parser("ingest", help="ingest a directory of Argo NetCDF files (idempotent)")
    p.add_argument("directory")
    p.add_argument("--batch", type=int, default=5000)
    sub.add_parser("embed", help="embed profile summaries that have no embedding yet")
    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    sub.add_parser("serve", help="start the Streamlit app")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "serve":
        app = Path(__file__).with_name("app.py")
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))

    with db.connect() as conn:
        if args.cmd == "init":
            print("vector backend:", db.apply_schema(conn))
        elif args.cmd == "ingest":
            from .ingest import ingest_dir
            db.apply_schema(conn)
            r = ingest_dir(conn, args.directory, args.batch)
            print(r)
        elif args.cmd == "embed":
            from .embed import embed_missing
            print("embedded", embed_missing(conn), "profiles")
        elif args.cmd == "ask":
            from .agent import ask
            a = ask(conn, args.question)
            print(a.text, f"\n[{a.mode} · tools: {', '.join(a.tools_used)} · {a.elapsed_ms} ms]")
