"""Semantic search over profile embeddings.

Two backends with the same interface:
  * pgvector — HNSW index, cosine distance, done inside PostgreSQL;
  * array    — embeddings in a REAL[] column, exact cosine search in NumPy (used when pgvector is not installed).
"""
from __future__ import annotations

import numpy as np
import psycopg

from .db import vector_backend


_OPS = {"<", "<=", ">", ">="}
_COLS = {"sst", "sss", "mld_dbar", "thermocline_dbar"}


def _filters(lat=None, lon=None, start=None, end=None, bbox=None, facets=()):
    where, params = ["embedding IS NOT NULL"], []
    for col, op, val in facets:
        assert col in _COLS and op in _OPS  # never interpolate anything else into SQL
        where.append(f"{col} {op} %s")
        params.append(val)
    if bbox:
        where.append("lat BETWEEN %s AND %s AND lon BETWEEN %s AND %s")
        params += [bbox[0], bbox[1], bbox[2], bbox[3]]
    if start:
        where.append("juld >= %s")
        params.append(start)
    if end:
        where.append("juld < %s")
        params.append(end)
    return " AND ".join(where), params


def semantic_search(conn: psycopg.Connection, query_vec: np.ndarray, k: int = 5, bbox=None, start=None, end=None, facets=()) -> list[dict]:
    """Hybrid retrieval: exact numeric `facets` + bbox/date filters, then cosine ranking of the remaining summaries."""
    where, params = _filters(bbox=bbox, start=start, end=end, facets=facets)
    cols = "profile_id, wmo_id, cycle_number, juld, lat, lon, sst, sss, mld_dbar, thermocline_dbar, summary"
    if vector_backend(conn) == "pgvector":
        vec = "[" + ",".join(f"{x:.6f}" for x in query_vec) + "]"
        rows = conn.execute(f"SELECT {cols}, 1 - (embedding <=> %s::vector) AS score FROM profiles WHERE {where} "
                            f"ORDER BY embedding <=> %s::vector LIMIT %s", [vec, *params, vec, k]).fetchall()
    else:
        rows = conn.execute(f"SELECT {cols}, embedding FROM profiles WHERE {where}", params).fetchall()
        if not rows:
            return []
        mat = np.array([r[-1] for r in rows], dtype="f4")
        scores = mat @ query_vec.astype("f4")
        top = np.argsort(-scores)[:k]
        rows = [(*rows[i][:-1], float(scores[i])) for i in top]
    keys = ["profile_id", "wmo_id", "cycle", "time", "lat", "lon", "sst", "sss", "mld_dbar", "thermocline_dbar", "summary", "score"]
    return [dict(zip(keys, r)) for r in rows]
