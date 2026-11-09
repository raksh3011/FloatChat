from datetime import datetime, timezone

import numpy as np
import pytest

from conftest import make_argo_file, needs_db
from floatchat import agent, db, features, ingest, tools
from floatchat.netcdf import guess_variable_names, read_profiles
import netCDF4 as nc


def test_variable_detection_handles_alias_names(tmp_path):
    make_argo_file(tmp_path / "x.nc", alias=True)
    m = guess_variable_names(nc.Dataset(tmp_path / "x.nc"))
    assert (m["lat"], m["pres"], m["temp"], m["psal"], m["time"]) == ("LAT", "PRESSURE", "TEMPERATURE", "SALINITY", "TIME")


def test_reader_drops_bad_qc_and_decodes_metadata(tmp_path):
    make_argo_file(tmp_path / "x.nc")
    profs = list(read_profiles(tmp_path / "x.nc"))
    assert len(profs) == 3
    p = profs[0]
    assert p.wmo_id == "1900001" and p.cycle == 1
    assert p.time.year == 2018 and p.time.tzinfo is not None
    assert not np.isnan(p.psal).any()
    assert len(p.pres) == 60 and np.isnan(p.temp).sum() == 1  # the QC=4 temperature level is masked out


def test_mixed_layer_and_thermocline():
    pres = np.linspace(2, 1000, 200)
    temp = np.where(pres < 50, 28.0, 28.0 - (pres - 50) * 0.02)
    mld = features.mixed_layer_depth(pres, temp)
    assert 50 < mld < 70           # 0.2 degC drop happens ~10 dbar below the 50 dbar knee
    assert 40 < features.thermocline_depth(pres, temp) < 1000


def test_summary_uses_plain_words():
    p = next(read_profiles_from_synthetic())
    s = features.summarize(p, features.features(p))
    assert "very warm" in s and "Arabian Sea" in s and "cycle 1" in s


def read_profiles_from_synthetic():
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    make_argo_file(d / "s.nc")
    return read_profiles(d / "s.nc")


def test_haversine():
    assert features.haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.5)


def test_sql_guard_rejects_writes():
    for bad in ["DROP TABLE profiles", "select 1; select 2", "SELECT 1; DELETE FROM profiles", "update profiles set lat=0"]:
        with pytest.raises(tools.ToolError):
            tools.run_sql(None, bad)


# ---------------------------------------------------------------- database tests
@needs_db
def test_ingest_is_idempotent_and_queryable(conn, argo_dir):
    r1 = ingest.ingest_dir(conn, argo_dir, batch=50)
    assert (r1.files_ingested, r1.profiles) == (2, 6)
    n = conn.execute("SELECT count(*) FROM profiles").fetchone()[0]
    m = conn.execute("SELECT count(*) FROM measurements").fetchone()[0]
    assert n == 6 and m > 6 * 100
    r2 = ingest.ingest_dir(conn, argo_dir)
    assert (r2.files_ingested, r2.files_skipped) == (0, 2)
    assert conn.execute("SELECT count(*) FROM profiles").fetchone()[0] == 6


@needs_db
def test_changed_file_is_reported_not_overwritten(conn, tmp_path):
    f = make_argo_file(tmp_path / "a.nc")
    ingest.ingest_dir(conn, tmp_path)
    make_argo_file(f, base_day=26000.0)  # same path, different content
    r = ingest.ingest_dir(conn, tmp_path)
    assert r.files_changed == 1 and conn.execute("SELECT count(*) FROM profiles").fetchone()[0] == 3


@needs_db
def test_bad_file_rolls_back(conn, tmp_path):
    (tmp_path / "broken.nc").write_bytes(b"not a netcdf file")
    with pytest.raises(Exception):
        ingest.ingest_dir(conn, tmp_path)
    conn.rollback()
    assert conn.execute("SELECT count(*) FROM raw_files").fetchone()[0] == 0


@needs_db
def test_tools_and_planner(conn, argo_dir):
    ingest.ingest_dir(conn, argo_dir)
    assert tools.count_profiles(conn)["rows"][0]["profiles"] == 6
    near = tools.profiles_near(conn, 10.0, 65.0, radius_km=50)
    assert near["rows"][0]["wmo_id"].startswith("19000") and near["rows"][0]["km"] < 1
    st = tools.depth_stats(conn, "temp", 10)
    assert st["rows"][0]["mean"] == pytest.approx(28.0, abs=0.01)
    a = agent.plan_and_run(conn, "how many profiles in the Bay of Bengal")
    assert a.tools_used == ["count_profiles"] and "2 profiles" in a.text
    with pytest.raises(tools.ToolError):
        tools.run_sql(conn, "DELETE FROM profiles")
    assert tools.run_sql(conn, "SELECT count(*) AS n FROM profiles")["rows"][0]["n"] == 6


@needs_db
def test_llm_loop_with_fake_model(conn, argo_dir):
    """The LLM path executes the tools the model asks for and returns its final text."""
    ingest.ingest_dir(conn, argo_dir)
    script = iter([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "count_profiles", "arguments": "{}"}}]},
        {"role": "assistant", "content": "There are 6 profiles."},
    ])
    seen = []
    a = agent.llm_answer(conn, "how many?", chat=lambda m, t: (seen.append(m[-1]), next(script))[1])
    assert a.mode == "llm" and a.tools_used == ["count_profiles"] and a.text == "There are 6 profiles."
    assert '"profiles": 6' in seen[-1]["content"]


def test_facets_turn_words_into_constraints():
    assert ("sss", "<", 33.8) in features.facets("low-salinity river water")
    assert ("sss", ">=", 36.0) in features.facets("salty evaporation-dominated water")
    assert features.facets("freshest profile") == []        # 'freshest' is an extreme query, not a facet
    assert ("sst", ">=", 28.0) in features.facets("warm pool")


@needs_db
def test_hybrid_semantic_respects_facets(conn, argo_dir):
    """Antonym guard: a 'low-salinity' question must never return a salty profile even if the embedding is close."""
    ingest.ingest_dir(conn, argo_dir)
    conn.execute("UPDATE profiles SET sss = CASE WHEN wmo_id LIKE '19000%' THEN 32.0 ELSE 36.5 END")
    class Fake:
        def embed(self, texts):
            v = np.ones((len(texts), 384), dtype="f4"); return v / np.linalg.norm(v, axis=1, keepdims=True)
    unit = [float(1 / 384 ** 0.5)] * 384
    if db.vector_backend(conn) == "pgvector":
        conn.execute("UPDATE profiles SET embedding = %s::vector", ("[" + ",".join(map(str, unit)) + "]",))
    else:
        conn.execute("UPDATE profiles SET embedding = %s", (unit,))
    r = tools.semantic(conn, Fake(), "low-salinity water", k=10)
    assert r["rows"] and all(h["sss"] < 33.8 for h in r["rows"])
