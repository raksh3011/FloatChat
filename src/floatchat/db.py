"""Connection handling and schema management."""
from __future__ import annotations

import os
from pathlib import Path

import psycopg

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/floatchat"


def dsn() -> str:
    return os.getenv("FLOATCHAT_DSN", os.getenv("DB_URI", DEFAULT_DSN))


def connect(conninfo: str | None = None) -> psycopg.Connection:
    return psycopg.connect(conninfo or dsn())


def has_pgvector(conn: psycopg.Connection) -> bool:
    return conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone() is not None


def apply_schema(conn: psycopg.Connection) -> str:
    """Create tables; add pgvector column + HNSW index if possible, else a REAL[] fallback column.

    Returns the vector backend in use: 'pgvector' or 'array'.
    """
    conn.execute((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
    conn.commit()
    backend = "array"
    try:
        conn.execute((SQL_DIR / "vector.sql").read_text(encoding="utf-8"))
        conn.commit()
        backend = "pgvector"
    except psycopg.errors.Error:
        conn.rollback()
        conn.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS embedding REAL[]")
        conn.commit()
    return backend


def vector_backend(conn: psycopg.Connection) -> str:
    row = conn.execute(
        "SELECT udt_name FROM information_schema.columns WHERE table_name='profiles' AND column_name='embedding'"
    ).fetchone()
    return "pgvector" if row and row[0] == "vector" else "array"
