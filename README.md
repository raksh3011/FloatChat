# FloatChat — ask questions of real Argo ocean data

**Live demo (runs in your browser):** https://raksh3011.github.io/floatchat/

Argo floats are autonomous profiling instruments that measure temperature and salinity from the surface to 2,000 m. The raw
data is thousands of hierarchical NetCDF files with inconsistent variable names. FloatChat turns it into a database a
scientist or student can query in plain English — *"mean salinity at 100 m in the Bay of Bengal"*, *"where is the
thermocline?"*, *"low-salinity river-influenced surface water"* — and get an answer **computed from the data**, with the
evidence (rows, map, depth profile) next to it.

## Why it exists

Ocean data is public but hard to use: you need to know NetCDF, Argo's variable conventions, QC flags and SQL before you can ask
a simple question. FloatChat lowers that barrier while staying honest: every number comes from a tool that queries the
database; the language model (optional) only chooses which tool to call and phrases the result.

## Architecture

```
 Argo NetCDF files                                     question
        │                                                  │
        ▼                                                  ▼
 netcdf.py   heuristic variable detection,        agent.py ── planner (no API key)  ┐ same
             QC-flag filtering, adjusted-data     │           LLM tool loop (opt.)  ┘ tools
             preference                           ▼
        │                                  tools.py  count · depth_stats · extreme · profiles_near ·
        ▼                                            semantic · profile_data · run_sql (read-only, guarded)
 features.py  SST/SSS, mixed-layer depth,                │
              thermocline, plain-language summary        ▼
        │                                   PostgreSQL ── relational (profiles, measurements)
        ▼                                              └─ vector: pgvector HNSW  (or REAL[] + NumPy fallback)
 ingest.py    SHA-256 file registry → idempotent;
              TEMP-table staging in one transaction;
              5,000-row batches
        │
        ▼
 embed.py     all-MiniLM-L6-v2 (384-d, ONNX) over each profile's summary
```

### Design decisions

| Decision | Why |
|---|---|
| **Schema-agnostic reader** (`netcdf.py`) | Providers name variables differently (`LATITUDE`/`LAT`, `JULD`/`TIME`, `PRES`/`PRESSURE`). Priority-ordered aliases mean new sources need no code change. |
| **Checksum-registered, idempotent ingestion** | Re-running is safe: same SHA-256 → skipped; same path but different content → reported, never silently overwritten. |
| **Temp-table staging in one transaction** | A malformed file rolls back completely; foreign keys are resolved set-based in SQL, not row by row. |
| **QC-aware** | Only Argo QC flags 1 (good) and 2 (probably good) are kept; delayed-mode adjusted values are preferred when present. |
| **Embed a *summary*, not numbers** | A profile is described in plain words ("very warm, low-salinity water in the Bay of Bengal, mixed layer 18 dbar") and that text is embedded, so questions phrased in everyday language match. Derived features (mixed-layer depth, thermocline) are also stored as columns for exact filtering. |
| **pgvector HNSW with graceful fallback** | With pgvector: `vector(384)` + HNSW cosine index. Without it: `REAL[]` + exact NumPy search (fine to ~10⁵ profiles). The backend is auto-detected and shown in the UI. |
| **Tools, not free-form SQL, as the default** | The agent calls typed tools. `run_sql` exists for flexibility but is guarded: single `SELECT`/`WITH`, `READ ONLY` transaction, 8 s timeout, 200-row cap. |
| **Works without an API key** | A deterministic planner answers the common question types so the system is usable and testable offline. Set `LLM_API_KEY` (OpenAI GPT-4o, Groq, Ollama — any OpenAI-compatible API) to use the tool-calling LLM loop. |

## Quick start

```bash
docker compose up --build        # Postgres+pgvector, ingest ./data, embed, Streamlit on :8501
```

or locally:

```bash
pip install -e ".[app,dev]"
export FLOATCHAT_DSN=postgresql://postgres:postgres@localhost:5432/floatchat
floatchat init
floatchat ingest ./data/indian_ocean_aug_2025     # idempotent; re-run any time
floatchat embed
floatchat ask "mean salinity at 100 m in the Bay of Bengal"
floatchat serve
```

Optional LLM agent: `export LLM_API_KEY=... LLM_MODEL=gpt-4o` (or `LLM_BASE_URL=https://api.groq.com/openai/v1`).

## Tests

```bash
export FLOATCHAT_TEST_DSN=postgresql://postgres:postgres@localhost:5432/floatchat_test   # a scratch database!
pytest -q
```

Covers variable-alias detection, QC masking, mixed-layer/thermocline maths, idempotent and rollback-safe ingestion, the SQL
guard, the planner and the LLM tool loop (with a scripted fake model). CI runs them against a `pgvector/pgvector` service.

## Data

The bundled sample (`data/indian_ocean_aug_2025/`, 31 daily files, **January 2020**, Indian Ocean) is from the international Argo
programme (https://argo.ucsd.edu), which is freely available. The folder name is historical.

For current data, `python tools/fetch_argo.py --days 31` downloads the newest daily files from the public GDAC into
`data/recent/` (the GDAC runs a day or two behind real time); then `floatchat ingest data/recent && floatchat embed`.
The hosted demo uses 10 Oct - 9 Nov 2026 (2,603 profiles from 861 floats).

## Browser demo

The portfolio site ([raksh3011.github.io/floatchat](https://raksh3011.github.io/floatchat/)) ships a static version: the same profiles, summaries and 384-d embeddings exported from this
database, with in-browser semantic search (Transformers.js, same MiniLM model; keyword fallback when offline), follow-up
questions, drag-a-box area selection, CSV export, and the equivalent query shown for every answer. See `tools/export_static.py`.

## Limitations

* Salinity/temperature only (no BGC variables yet); sample covers one month.
* The planner understands a fixed set of question shapes; the LLM loop is more flexible but needs an API key.
* pgvector HNSW path is exercised in CI (pgvector image); the NumPy fallback is covered locally.
