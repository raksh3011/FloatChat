import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports (post v0.2.0)
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from streamlit_folium import st_folium
import visualizations

# Load environment variables
load_dotenv()

st.set_page_config(page_title="ArgoFloatChat", layout="wide")
st.title("ArgoFloatChat")

DB_URI = os.getenv("DB_URI")
if not DB_URI:
    st.error("Missing DB_URI in .env file")
    st.stop()

@st.cache_resource
def create_resources():
    db = SQLDatabase.from_uri(DB_URI)
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
    )
    return db, agent

db, agent = create_resources()

user_input = st.text_input("Ask about Argo float data:")
if user_input:
    try:
        response = agent.run(user_input)
        st.subheader("Agent Response")
        st.write(response)

        query = user_input.lower()
        if "salinity" in query or "psal" in query:
            df = visualizations.get_salinity_data(db)
            if df is not None and not df.empty:
                fig = visualizations.plot_salinity_profiles(df)
                st.plotly_chart(fig, use_container_width=True)
        if "map" in query or "location" in query:
            m = visualizations.plot_float_locations(db)
            st.subheader("Float Locations")
            st_folium(m, width=900, height=600)

    except Exception as e:
        st.error(f"Error: {e}")
