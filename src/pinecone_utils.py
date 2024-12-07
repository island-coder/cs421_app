import os
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import streamlit as st

logger = logging.getLogger(__name__)

@st.cache_resource
def get_pinecone_index():
    logger.info("Initializing Pinecone...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = "mmkg-doc-index"
    dimension = 1536

    pc = Pinecone(api_key=pinecone_api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    logger.info("Pinecone initialized.")
    return pc.Index(index_name)

@st.cache_resource
def get_vectorstore():
    index = get_pinecone_index()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(index=index, embedding=embeddings)
