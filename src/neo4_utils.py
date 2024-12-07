import logging
from langchain_community.graphs import Neo4jGraph
import streamlit as st

logger = logging.getLogger(__name__)

@st.cache_resource
def get_neo4j_graph():
    logger.info("Connecting to Neo4j...")
    graph = Neo4jGraph()
    graph.refresh_schema()
    logger.info("Neo4j connected and schema loaded.")
    return graph
