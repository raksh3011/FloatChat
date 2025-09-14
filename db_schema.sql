-- Tables
CREATE TABLE floats (
    wmo_id VARCHAR(10) PRIMARY KEY,
    platform_type VARCHAR(50),
    pi_name VARCHAR(100),
    country VARCHAR(50)
);

CREATE TABLE profiles (
    profile_id SERIAL PRIMARY KEY,
    wmo_id VARCHAR(10) REFERENCES floats(wmo_id),
    cycle_number INTEGER,
    juld TIMESTAMP,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    direction CHAR(1),
    data_mode CHAR(1),
    checksum VARCHAR(64) UNIQUE
);

CREATE TABLE measurement_variables (
    variable_id SERIAL PRIMARY KEY,
    variable_name VARCHAR(50) UNIQUE,
    description TEXT
);

CREATE TABLE measurements (
    measurement_id SERIAL PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(profile_id),
    variable_id INTEGER REFERENCES measurement_variables(variable_id),
    depth DOUBLE PRECISION,
    value DOUBLE PRECISION,
    qc_flag CHAR(1)
);

-- Indexes
CREATE INDEX idx_profiles_wmo_id ON profiles(wmo_id);
CREATE INDEX idx_profiles_juld ON profiles(juld);
CREATE INDEX idx_measurements_profile_id ON measurements(profile_id);
CREATE INDEX idx_measurements_variable_id ON measurements(variable_id);

-- Pre-populate measurement variables
INSERT INTO measurement_variables (variable_name, description) VALUES
    ('TEMP', 'Sea water temperature (degrees Celsius)'),
    ('PSAL', 'Practical salinity (PSU)'),
    ('PRES', 'Sea water pressure (dbar)'),
    ('DOXY', 'Dissolved oxygen (micromole/kg)');

-- View for Indian Ocean profiles (August 2023 for PoC)
CREATE VIEW indian_ocean_profiles AS
SELECT p.*
FROM profiles p
WHERE p.lat BETWEEN -30 AND 30
  AND p.lon BETWEEN 30 AND 120
  AND p.juld >= '2023-08-01' AND p.juld < '2023-09-01';