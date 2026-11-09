"""Question answering: a deterministic planner (no API key needed) and an optional LLM tool-calling loop.

Both call the same tools in tools.py, so answers are always computed from the database, never invented.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import psycopg

from . import tools
from .embed import Embedder
from .features import REGIONS

MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


@dataclass
class Answer:
    text: str
    tools_used: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    mode: str = "planner"
    elapsed_ms: int = 0


# ----------------------------------------------------------------------------- planner
def _region(t: str):
    for name, (lat, lon) in REGIONS.items():
        if name.lower() in t:
            return name, (lat[0], lat[1], lon[0], lon[1])
    if re.search(r"\bequator(ial)?\b", t):
        return "Equatorial band", (-5, 5, -180, 360)
    return None, None


def _near(t: str):
    la = re.search(r"(\d+(?:\.\d+)?)\s*°?\s*([ns])\b", t)
    lo = re.search(r"(\d+(?:\.\d+)?)\s*°?\s*([ew])\b", t)
    if not (la and lo):
        return None
    km = re.search(r"within\s*(\d+)\s*km", t)
    return ((-1 if la.group(2) == "s" else 1) * float(la.group(1)), (-1 if lo.group(2) == "w" else 1) * float(lo.group(1)),
            float(km.group(1)) if km else 500.0)


def _dates(t: str, year: int):
    """Return (start, end_exclusive) or (None, None). Understands ISO dates, 'jan 12', 'jan 5-9', 'first week'."""
    utc = timezone.utc
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if iso:
        d = datetime(int(iso[1]), int(iso[2]), int(iso[3]), tzinfo=utc)
        return d, d + timedelta(days=1)
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{1,2})(?:\s*(?:-|to)\s*(\d{1,2}))?\b", t)
    if m:
        mo, d1 = MONTHS[m[1]], int(m[2])
        d2 = int(m[3]) if m[3] else d1
        return datetime(year, mo, d1, tzinfo=utc), datetime(year, mo, d2, tzinfo=utc) + timedelta(days=1)
    for word, (a, b) in {"first week": (1, 8), "second week": (8, 15), "third week": (15, 22), "last week": (22, 32)}.items():
        if word in t:
            mo = MONTHS.get(next((k for k in MONTHS if k in t), ""), None)
            return ("MONTH", a, b, mo)  # resolved by caller against the data's month
    return None, None


def plan_and_run(conn: psycopg.Connection, question: str, embedder: Embedder | None = None) -> Answer:
    t0 = time.time()
    t = question.lower()
    yr, mo = conn.execute("SELECT EXTRACT(year FROM min(juld AT TIME ZONE 'UTC'))::int, EXTRACT(month FROM min(juld AT TIME ZONE 'UTC'))::int FROM profiles").fetchone() or (2020, 1)
    rname, bbox = _region(t)
    near = _near(t)
    d = _dates(t, yr)
    start = end = None
    if d and d[0] == "MONTH":
        _, a, b, m2 = d
        m2 = m2 or mo
        start = datetime(yr, m2, a, tzinfo=timezone.utc)
        end = datetime(yr, m2, 1, tzinfo=timezone.utc) + timedelta(days=b - 1)
    elif d:
        start, end = d
    var = "psal" if re.search(r"salin|psal|salt|fresh", t) else "temp"
    dm = re.search(r"\b(?:at|around|near)\s*(\d{1,4})\s*(?:m|dbar|metres?|meters?)\b", t)
    depth = float(dm[1]) if dm else 0.0
    used, results, text = [], [], ""

    def call(name, **kw):
        used.append(name)
        r = getattr(tools, name)(conn, **kw)
        results.append({"tool": name, **r})
        return r

    filt = dict(bbox=bbox, start=start, end=end)
    if re.search(r"how many|count|number of", t):
        r = call("count_profiles", **filt, near=near)
    elif re.search(r"warmest|hottest|highest|saltiest|coldest|coolest|lowest|freshest", t):
        hi = bool(re.search(r"warmest|hottest|highest|saltiest", t))
        r = call("extreme", variable=var, depth_dbar=depth, highest=hi, **filt)
    elif near and re.search(r"nearest|closest|near|within", t):
        r = call("profiles_near", lat=near[0], lon=near[1], radius_km=near[2], start=start, end=end)
    elif re.search(r"thermocline|mixed layer", t):
        w, p = tools._where(bbox, start, end)
        row = conn.execute(f"SELECT count(*), percentile_cont(0.5) WITHIN GROUP (ORDER BY thermocline_dbar), "
                           f"percentile_cont(0.5) WITHIN GROUP (ORDER BY mld_dbar) FROM profiles WHERE {w}", p).fetchone()
        used.append("sql:thermocline")
        r = {"kind": "stats", "rows": [dict(zip(["n", "thermocline_median", "mld_median"], row))],
             "text": f"Across {row[0]} profiles the median thermocline is near "
                     f"{row[1] if row[1] is None else round(row[1])} dbar and the median mixed-layer depth "
                     f"{row[2] if row[2] is None else round(row[2])} dbar."}
        results.append({"tool": "sql:thermocline", **r})
    elif re.search(r"similar|like|resembl|influenc|river|upwelling|eddy|anomal|water mass|fresh|saline|monsoon|warm pool|stratif", t):
        r = call("semantic", embedder=embedder or Embedder(), text=question, k=5, bbox=bbox, start=start, end=end)
    elif bbox or near or start or re.search(r"temperature|salinity|average|mean|profile", t):
        r = call("depth_stats", variable=var, depth_dbar=depth, **filt, near=near)
    else:  # no structured filters: treat the question as a semantic query over profile summaries
        r = call("semantic", embedder=embedder or Embedder(), text=question, k=5)
    scope = ", ".join(x for x in (rname, f"near {near[0]:.1f},{near[1]:.1f}" if near else None,
                                  f"{start:%Y-%m-%d}–{(end - timedelta(days=1)):%Y-%m-%d}" if start else None) if x)
    text = r["text"] + (f" (scope: {scope})" if scope else "")
    ans = Answer(text=text, tools_used=used, results=results, mode="planner", elapsed_ms=int((time.time() - t0) * 1000))
    _log(conn, question, ans)
    return ans


# ----------------------------------------------------------------------------- LLM tool-calling loop
TOOL_SPECS = [
    {"type": "function", "function": {"name": "count_profiles", "description": "Count profiles/floats in a region and time window.",
     "parameters": {"type": "object", "properties": {"bbox": {"type": "array", "items": {"type": "number"}, "description": "[lat_min, lat_max, lon_min, lon_max]"},
      "start": {"type": "string", "description": "ISO date"}, "end": {"type": "string", "description": "ISO date, exclusive"}}}}},
    {"type": "function", "function": {"name": "depth_stats", "description": "Mean/min/max/percentiles of temp or psal near a depth.",
     "parameters": {"type": "object", "properties": {"variable": {"enum": ["temp", "psal"]}, "depth_dbar": {"type": "number"},
      "bbox": {"type": "array", "items": {"type": "number"}}, "start": {"type": "string"}, "end": {"type": "string"}}, "required": ["variable"]}}},
    {"type": "function", "function": {"name": "profiles_near", "description": "Profiles closest to a point.",
     "parameters": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}, "radius_km": {"type": "number"}}, "required": ["lat", "lon"]}}},
    {"type": "function", "function": {"name": "semantic", "description": "Semantic search over profile descriptions (e.g. 'low-salinity river-influenced water').",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "k": {"type": "integer"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "run_sql", "description": "Read-only SQL over profiles(profile_id,wmo_id,juld,lat,lon,sst,sss,mld_dbar,thermocline_dbar,summary) "
     "and measurements(profile_id,variable_id,pres_dbar,value).", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
]
SYSTEM = ("You answer questions about Argo float profiles using ONLY the provided tools. Never invent numbers. "
          "If the tools return nothing, say so. Cite float IDs and dates from tool results.")


def _json_safe(o):
    return json.loads(json.dumps(o, default=str))


def llm_answer(conn: psycopg.Connection, question: str, embedder: Embedder | None = None, chat=None, max_steps: int = 6) -> Answer:
    """Tool-calling loop against any OpenAI-compatible chat API (OpenAI GPT-4o, Groq, Ollama…).

    Configure with LLM_API_KEY, LLM_BASE_URL (default https://api.openai.com/v1) and LLM_MODEL (default gpt-4o).
    `chat` can be injected (messages, tools) -> response-dict for tests.
    """
    import requests

    t0 = time.time()
    base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o")

    def default_chat(messages, tools_):
        r = requests.post(f"{base}/chat/completions", timeout=60,
                          headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
                          json={"model": model, "messages": messages, "tools": tools_, "temperature": 0})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

    chat = chat or default_chat
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    used, results = [], []
    for _ in range(max_steps):
        msg = chat(messages, TOOL_SPECS)
        calls = msg.get("tool_calls") or []
        if not calls:
            ans = Answer(msg.get("content") or "", used, results, "llm", int((time.time() - t0) * 1000))
            _log(conn, question, ans)
            return ans
        messages.append(msg)
        for c in calls:
            name, args = c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")
            try:
                fn = getattr(tools, name)
                if name == "semantic":
                    args["embedder"] = embedder or Embedder()
                for k in ("start", "end"):
                    if args.get(k):
                        args[k] = tools.parse_date(args[k])
                out = fn(conn, **args)
            except Exception as e:  # report tool errors back to the model so it can correct itself
                conn.rollback()
                out = {"kind": "error", "rows": [], "text": f"{type(e).__name__}: {e}"}
            used.append(name)
            results.append({"tool": name, **out})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(_json_safe({"text": out["text"], "rows": out["rows"][:20]}))})
    ans = Answer("I couldn't complete that within the step limit.", used, results, "llm", int((time.time() - t0) * 1000))
    _log(conn, question, ans)
    return ans


def ask(conn: psycopg.Connection, question: str, embedder: Embedder | None = None) -> Answer:
    """Use the LLM agent when LLM_API_KEY is set, otherwise the deterministic planner."""
    if os.getenv("LLM_API_KEY"):
        return llm_answer(conn, question, embedder)
    return plan_and_run(conn, question, embedder)


def _log(conn: psycopg.Connection, q: str, a: Answer) -> None:
    try:
        conn.execute("INSERT INTO query_log (question, mode, tools_used, elapsed_ms) VALUES (%s,%s,%s,%s)",
                     (q, a.mode, a.tools_used, a.elapsed_ms))
        conn.commit()
    except psycopg.Error:
        conn.rollback()
