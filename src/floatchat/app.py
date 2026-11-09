"""Streamlit chat UI: ask a question, see the answer, the evidence (rows), a map and depth profiles."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from floatchat import agent, db, tools
from floatchat.embed import Embedder

st.set_page_config(page_title="FloatChat", page_icon="🌊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1300px}
.fc-chip{display:inline-block;padding:2px 10px;border-radius:999px;border:1px solid #2a3b46;color:#7fd6cc;font-size:.75rem;margin-right:6px}
</style>""", unsafe_allow_html=True)

SUGGESTIONS = [
    "How many profiles in the Arabian Sea in the first week?",
    "Mean salinity at 100 m in the Bay of Bengal",
    "Where is the thermocline in the Bay of Bengal?",
    "Warmest profile at 100 m",
    "Floats within 300 km of 10N 70E",
    "Low-salinity river-influenced surface water",
]


@st.cache_resource
def get_conn():
    return db.connect()


@st.cache_resource
def get_embedder():
    return Embedder()


conn = get_conn()
try:
    stats = conn.execute("SELECT count(*), count(DISTINCT wmo_id), min(juld), max(juld), count(embedding) FROM profiles").fetchone()
except Exception as e:  # schema missing / DB down
    conn.rollback()
    st.error(f"Database not ready: {e}. Run `floatchat init`, `floatchat ingest <dir>` and `floatchat embed`.")
    st.stop()

with st.sidebar:
    st.title("🌊 FloatChat")
    st.caption("Ask questions of real Argo float profiles.")
    st.metric("Profiles", f"{stats[0]:,}")
    st.metric("Floats", f"{stats[1]:,}")
    st.caption(f"{stats[2]:%Y-%m-%d} → {stats[3]:%Y-%m-%d} · {stats[4]:,} embedded · vector backend: **{db.vector_backend(conn)}**")
    st.caption("Mode: **LLM tool-calling**" if __import__("os").getenv("LLM_API_KEY") else "Mode: **deterministic planner** (set LLM_API_KEY for GPT-4o / Groq)")
    st.divider()
    for s in SUGGESTIONS:
        if st.button(s, use_container_width=True):
            st.session_state["pending"] = s

if "history" not in st.session_state:
    st.session_state.history = []


def render(res: dict):
    kind, rows = res["kind"], res["rows"]
    if not rows:
        return
    if kind == "semantic":
        df = pd.DataFrame(rows)
        st.dataframe(df[["score", "wmo_id", "time", "lat", "lon", "sst", "sss", "summary"]], hide_index=True, use_container_width=True)
        m = go.Figure(go.Scattergeo(lat=df.lat, lon=df.lon, text=df.wmo_id, mode="markers",
                                    marker=dict(size=10, color=df.score, colorscale="Teal", showscale=True)))
        m.update_geos(lataxis_range=[-45, 30], lonaxis_range=[20, 120], showland=True, landcolor="#1d2a33", bgcolor="rgba(0,0,0,0)")
        m.update_layout(height=340, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(m, use_container_width=True)
        top = rows[0]["profile_id"]
        prof = tools.profile_data(conn, top)
        fig = go.Figure()
        for var, name in (("TEMP", "Temperature °C"), ("PSAL", "Salinity PSU")):
            d = [r for r in prof["rows"] if r["var"] == var]
            fig.add_trace(go.Scatter(x=[r["value"] for r in d], y=[r["pres"] for r in d], name=name, xaxis="x2" if var == "PSAL" else "x"))
        fig.update_layout(yaxis=dict(autorange="reversed", title="Pressure (dbar)"), xaxis=dict(title="Temperature °C"),
                          xaxis2=dict(title="Salinity PSU", overlaying="x", side="top"), height=380, title=f"Best match: float {prof['head']['wmo_id']}")
        st.plotly_chart(fig, use_container_width=True)
    elif kind in ("table", "count", "stats"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a.text)
        st.markdown(" ".join(f"<span class='fc-chip'>{t}</span>" for t in a.tools_used) +
                    f"<span class='fc-chip'>{a.mode} · {a.elapsed_ms} ms</span>", unsafe_allow_html=True)
        for r in a.results:
            render(r)

q = st.chat_input("Ask about temperature, salinity, regions, dates, depths…") or st.session_state.pop("pending", None)
if q:
    with st.spinner("Querying…"):
        try:
            a = agent.ask(conn, q, get_embedder())
        except Exception as e:
            conn.rollback()
            a = agent.Answer(f"Sorry — that failed: {e}", mode="error")
    st.session_state.history.append((q, a))
    st.rerun()
