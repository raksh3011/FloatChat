-- FloatChat schema. The embedding column and HNSW index are added by db.apply_schema()
-- (see sql/vector.sql) when the pgvector extension is available.

CREATE TABLE IF NOT EXISTS floats (
    wmo_id        TEXT PRIMARY KEY,
    platform_type TEXT,
    pi_name       TEXT
);

-- One row per ingested NetCDF file. The SHA-256 of the file bytes makes ingestion idempotent:
-- same checksum -> skipped; same path with a different checksum -> reported, never silently overwritten.
CREATE TABLE IF NOT EXISTS raw_files (
    checksum    TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    n_profiles  INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS profiles (
    profile_id    BIGSERIAL PRIMARY KEY,
    wmo_id        TEXT NOT NULL REFERENCES floats(wmo_id),
    cycle_number  INTEGER,
    juld          TIMESTAMPTZ NOT NULL,
    lat           DOUBLE PRECISION NOT NULL CHECK (lat BETWEEN -90 AND 90),
    lon           DOUBLE PRECISION NOT NULL CHECK (lon BETWEEN -180 AND 360),
    data_mode     CHAR(1),
    source_sha256 TEXT REFERENCES raw_files(checksum),
    -- derived features, computed at ingestion
    sst           REAL,      -- temperature at the shallowest valid level (<= 25 dbar)
    sss           REAL,      -- salinity at the shallowest valid level (<= 25 dbar)
    mld_dbar      REAL,      -- mixed-layer depth, 0.2 degC criterion from 10 dbar
    thermocline_dbar REAL,   -- depth of maximum |dT/dz|
    max_pres_dbar REAL,
    summary       TEXT,      -- natural-language description that is embedded
    UNIQUE (wmo_id, cycle_number, juld)
);

CREATE TABLE IF NOT EXISTS measurement_variables (
    variable_id   SMALLINT PRIMARY KEY,
    variable_name TEXT UNIQUE NOT NULL,
    unit          TEXT,
    description   TEXT
);
INSERT INTO measurement_variables VALUES
    (1, 'TEMP', 'degC', 'In-situ sea water temperature (ITS-90)'),
    (2, 'PSAL', 'PSU',  'Practical salinity')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS measurements (
    profile_id  BIGINT   NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
    variable_id SMALLINT NOT NULL REFERENCES measurement_variables(variable_id),
    pres_dbar   REAL     NOT NULL,
    value       REAL     NOT NULL,
    qc_flag     CHAR(1),
    PRIMARY KEY (profile_id, variable_id, pres_dbar)
);

CREATE INDEX IF NOT EXISTS idx_profiles_juld ON profiles (juld);
CREATE INDEX IF NOT EXISTS idx_profiles_lat_lon ON profiles (lat, lon);
CREATE INDEX IF NOT EXISTS idx_profiles_wmo ON profiles (wmo_id);

CREATE TABLE IF NOT EXISTS query_log (
    id          BIGSERIAL PRIMARY KEY,
    asked_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    question    TEXT NOT NULL,
    mode        TEXT NOT NULL,        -- 'planner' | 'llm'
    tools_used  TEXT[],
    elapsed_ms  INTEGER
);
