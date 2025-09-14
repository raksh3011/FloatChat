import pandas as pd
import plotly.express as px
import folium
from folium.plugins import MarkerCluster

def get_salinity_data(db):
    query = """
    SELECT p.profile_id, p.lat, p.lon, p.juld, m.depth, m.value
    FROM indian_ocean_profiles p
    JOIN measurements m ON p.profile_id = m.profile_id
    JOIN measurement_variables mv ON m.variable_id = mv.variable_id
    WHERE mv.variable_name = 'PSAL'
    """
    return pd.DataFrame(db.run(f"SELECT * FROM ({query}) AS subquery"))

def plot_salinity_profiles(df):
    return px.scatter(df, x="value", y="depth", color="profile_id",
                     title="Salinity Profiles", labels={"value": "Salinity (PSU)", "depth": "Depth (m)"},
                     hover_data=["lat", "lon", "juld"])

def plot_float_locations(db):
    query = """
    SELECT DISTINCT lat, lon, wmo_id
    FROM indian_ocean_profiles
    """
    df = pd.DataFrame(db.run(f"SELECT * FROM ({query}) AS subquery"))
    m = folium.Map(location=[0, 75], zoom_start=4)
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in df.iterrows():
        folium.Marker([row['lat'], row['lon']], popup=row['wmo_id']).add_to(marker_cluster)
    return m