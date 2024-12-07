import os
import streamlit as st

def load_credentials():
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    neo4j_password = st.secrets["NEO4J_PASSWORD"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]

    if not all([openai_api_key, neo4j_password, pinecone_api_key]):
        st.error("Missing credentials. Please ensure all secrets are set.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["NEO4J_PASSWORD"] = neo4j_password
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["NEO4J_URI"] = 'neo4j+s://cbeb0505.databases.neo4j.io'
    os.environ["NEO4J_USERNAME"] = 'neo4j'

