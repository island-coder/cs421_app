# app.py

from env_setup import load_credentials
from help_section import display_help
load_credentials()

import streamlit as st
from setup import llm, retriever, graph_chain

from content import articles, display_about_kg
from query_runners import run_vector_retrieval_only, run_with_neo4j,display_response_with_images

# Load credentials first

load_credentials()

st.title("Demo: Evaluation of Multi-Modal Knowledge Graph Based Retrieval")
with st.sidebar:
    display_help()
# Query Input
query = st.text_input("Enter your query", placeholder="E.g., Images depicting Elon Musk")

if st.button("Run Query"):
    if not query.strip():
        st.error("Please enter a valid query.")
    else:
        st.write(f"Processing query: {query}")

        # Vector Retrieval Only
        st.subheader("RAG Without KG (Vector Retrieval Only)")
        vector_response = run_vector_retrieval_only(query, retriever, llm)
        display_response_with_images(vector_response)

        # RAG with Neo4j
        st.subheader("MMKG Enhanced RAG (Vector Retrieval + MMKG)")
        kg_response = run_with_neo4j(query, retriever, graph_chain, llm)
        display_response_with_images(kg_response)

# Additional Sections
if st.checkbox("About the Knowledge Graph"):
    display_about_kg()

if st.checkbox("Show Source Articles"):
    st.subheader("Source Articles")
    for article in articles:
        st.markdown(f"- [{article}]({article})")
