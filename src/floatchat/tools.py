"""Tools the agent can call. Every tool returns {"text": str, "rows": list[dict], "kind": str}."""
from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import psycopg

from .embed import Embedder
from .features import facets
from .store import semantic_search as _semantic

MAX_ROWS = 200
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|vacuum|set|pg_sleep)\b", re.I)


class ToolError(ValueError):
    pass


def run_sql(conn: psycopg.Connection, sql: str) -> dict:
    """Run one read-only SELECT. Guarded three ways: statement allow-list, READ ONLY transaction, row/time caps."""
    stmt = sql.strip().rstrip(";")
    if ";" in stmt:
        raise ToolError("only a single statement is allowed")
    if not re.match(r"^(select|with)\b", stmt, re.I) or _FORBIDDEN.search(stmt):
        raise ToolError("only read-only SELECT queries are allowed")
    with conn.transaction():
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SET LOCAL statement_timeout = '8s'")
        cur = conn.execute(f"SELECT * FROM ({stmt}) AS q LIMIT {MAX_ROWS}")
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"kind": "table", "rows": rows, "text": f"{len(rows)} row(s)"}


def _where(bbox=None, start=None, end=None, near=None, a=""):
    """WHERE fragment over profiles columns; `a` is an optional table alias prefix such as "pr."."""
    w, p = ["TRUE"], []
    if bbox:
        w.append(f"{a}lat BETWEEN %s AND %s AND {a}lon BETWEEN %s AND %s")
        p += list(bbox)
    if start:
        w.append(f"{a}juld >= %s")
        p.append(start)
    if end:
        w.append(f"{a}juld < %s")
        p.append(end)
    if near:
        lat, lon, km = near
        w.append("12742 * asin(sqrt(power(sin(radians(lat - %s) / 2), 2) + cos(radians(%s)) * cos(radians(lat)) * "
                 "power(sin(radians(lon - %s) / 2), 2))) <= %s")
        p += [lat, lat, lon, km]
    return " AND ".join(w), p


def count_profiles(conn, bbox=None, start=None, end=None, near=None) -> dict:
    w, p = _where(bbox, start, end, near)
    n, floats = conn.execute(f"SELECT count(*), count(DISTINCT wmo_id) FROM profiles WHERE {w}", p).fetchone()
    return {"kind": "count", "rows": [{"profiles": n, "floats": floats}], "text": f"{n} profiles from {floats} floats match."}


def profiles_near(conn, lat: float, lon: float, radius_km: float = 300, start=None, end=None, limit: int = 20) -> dict:
    w, p = _where(start=start, end=end)
    rows = conn.execute(
        f"""SELECT profile_id, wmo_id, juld, lat, lon, sst, sss,
                   12742 * asin(sqrt(power(sin(radians(lat - %s) / 2), 2) + cos(radians(%s)) * cos(radians(lat)) *
                   power(sin(radians(lon - %s) / 2), 2))) AS km
            FROM profiles WHERE {w} ORDER BY km LIMIT %s""", [lat, lat, lon, *p, limit]).fetchall()
    out = [dict(zip(["profile_id", "wmo_id", "time", "lat", "lon", "sst", "sss", "km"], r)) for r in rows if r[-1] <= radius_km]
    return {"kind": "table", "rows": out, "text": f"{len(out)} profile(s) within {radius_km:.0f} km of ({lat:.1f}, {lon:.1f})."}


_VARS = {"temp": ("TEMP", "temperature", "°C"), "psal": ("PSAL", "salinity", "PSU")}


def depth_stats(conn, variable: str = "temp", depth_dbar: float = 0, bbox=None, start=None, end=None, near=None) -> dict:
    """Mean/min/max/p10/p90 of a variable at the level closest to `depth_dbar` (within 15 dbar) per profile."""
    name, label, unit = _VARS[variable]
    w, p = _where(bbox, start, end, near)
    rows = conn.execute(
        f"""WITH sel AS (SELECT profile_id FROM profiles WHERE {w}),
             pick AS (SELECT DISTINCT ON (m.profile_id) m.profile_id, m.value
                      FROM measurements m JOIN sel USING (profile_id)
                      JOIN measurement_variables v USING (variable_id)
                      WHERE v.variable_name = %s AND abs(m.pres_dbar - %s) <= 15
                      ORDER BY m.profile_id, abs(m.pres_dbar - %s))
            SELECT count(*), avg(value), min(value), max(value),
                   percentile_cont(0.1) WITHIN GROUP (ORDER BY value),
                   percentile_cont(0.9) WITHIN GROUP (ORDER BY value) FROM pick""",
        [*p, name, depth_dbar, depth_dbar]).fetchone()
    n, mean, lo, hi, p10, p90 = rows
    if not n:
        return {"kind": "stats", "rows": [], "text": f"No valid {label} readings near {depth_dbar:.0f} dbar for that selection."}
    return {"kind": "stats", "rows": [{"n": n, "mean": mean, "min": lo, "max": hi, "p10": p10, "p90": p90}],
            "text": f"{label.capitalize()} near {depth_dbar:.0f} dbar over {n} profiles: mean {mean:.2f} {unit} "
                    f"(range {lo:.2f}–{hi:.2f}, p10–p90 {p10:.2f}–{p90:.2f})."}


def extreme(conn, variable: str = "temp", depth_dbar: float = 0, highest: bool = True, bbox=None, start=None, end=None) -> dict:
    name, label, unit = _VARS[variable]
    w, p = _where(bbox, start, end, a="pr.")
    order = "DESC" if highest else "ASC"
    row = conn.execute(
        f"""SELECT pr.profile_id, pr.wmo_id, pr.juld, pr.lat, pr.lon, m.pres_dbar, m.value
            FROM profiles pr JOIN measurements m USING (profile_id) JOIN measurement_variables v USING (variable_id)
            WHERE {w}
              AND v.variable_name = %s AND abs(m.pres_dbar - %s) <= 15
            ORDER BY m.value {order} LIMIT 1""", [*p, name, depth_dbar]).fetchone()
    if not row:
        return {"kind": "table", "rows": [], "text": "No matching readings."}
    d = dict(zip(["profile_id", "wmo_id", "time", "lat", "lon", "pres_dbar", "value"], row))
    return {"kind": "table", "rows": [d],
            "text": f"{'Highest' if highest else 'Lowest'} {label} near {depth_dbar:.0f} dbar: {d['value']:.2f} {unit} "
                    f"(float {d['wmo_id']}, {d['lat']:.1f}°, {d['lon']:.1f}°, {d['time']:%Y-%m-%d})."}


def profile_data(conn, profile_id: int) -> dict:
    head = conn.execute("SELECT wmo_id, juld, lat, lon, summary FROM profiles WHERE profile_id=%s", (profile_id,)).fetchone()
    if not head:
        raise ToolError(f"no profile {profile_id}")
    rows = conn.execute("""SELECT v.variable_name, m.pres_dbar, m.value FROM measurements m
                           JOIN measurement_variables v USING (variable_id) WHERE m.profile_id=%s ORDER BY m.pres_dbar""",
                        (profile_id,)).fetchall()
    return {"kind": "profile", "rows": [{"var": r[0], "pres": r[1], "value": r[2]} for r in rows],
            "head": dict(zip(["wmo_id", "time", "lat", "lon", "summary"], head)), "text": head[4] or ""}


def semantic(conn, embedder: Embedder, text: str, k: int = 5, bbox=None, start=None, end=None) -> dict:
    """Hybrid search: numeric constraints implied by the wording (e.g. 'low-salinity' -> sss < 33.8) are applied
    exactly, then the embedding ranks the survivors by meaning."""
    fc = facets(text)
    hits = _semantic(conn, embedder.embed([text])[0], k, bbox, start, end, fc)
    if not hits:
        return {"kind": "semantic", "rows": [], "text": "No matching profiles (run `floatchat embed` if you have not)."}
    shown = " · ".join(f"{h['wmo_id']} {h['time']:%d %b} ({h['score']:.2f})" for h in hits[:3])
    cons = f" Constraints applied: {', '.join(f'{c} {o} {v}' for c, o, v in fc)}." if fc else ""
    return {"kind": "semantic", "rows": hits, "text": f"Top {len(hits)} by meaning for “{text}”: {shown}.{cons}"}


def parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s) if "T" in s else datetime.combine(date.fromisoformat(s), datetime.min.time())
