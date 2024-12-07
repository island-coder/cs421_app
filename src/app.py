import os
import streamlit as st
import logging
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import GraphCypherQAChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts.prompt import PromptTemplate

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials
st.title("Demo App: Evaluation of Multi-Modal Knowledge Graph based retrieval")

st.write("Loading credentials...")
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

# Initialize Neo4j
@st.cache_resource
def get_neo4j_graph():
    logger.info("Connecting to Neo4j...")
    graph = Neo4jGraph()
    graph.refresh_schema()
    logger.info("Neo4j connected and schema loaded.")
    return graph

graph = get_neo4j_graph()
st.success("Neo4j connected and schema loaded.")

# Initialize Pinecone
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
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    logger.info("Pinecone initialized.")
    return pc.Index(index_name)

index = get_pinecone_index()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# Metadata field info for the SelfQueryRetriever
metadata_field_info = [
    AttributeInfo(
        name="id",
        description="The source URL of the article",
        type="string"
    ),
    AttributeInfo(
        name="title",
        description="The title of the article",
        type="string"
    ),
    AttributeInfo(
        name="description",
        description="A short description of the article",
        type="string"
    ),
]

document_content_description = "The main content of the article"
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_content_description=document_content_description,
    metadata_field_info=metadata_field_info,
    document_contents="page_content",
    verbose=False
)

# Cypher Generation Template
from cypher_template import CYPHER_GENERATION_TEMPLATE 


cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# Graph QA Chain
graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    qa_llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    validate_cypher=True,
    allow_dangerous_requests=True,
    verbose=True
)

# Caching for RAG Without KG
@st.cache_data
def run_vector_retrieval_only(query):
    logger.info("Running Vector Retrieval only...")
    retrieved_docs = retriever.get_relevant_documents(query)
    combined_text = "\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = f"""
    You are an AI assistant. Use the following information to answer the query:

    Pinecone Results:
    {combined_text}

    Query: {query}

    Provide the most accurate and comprehensive response.
    """
    response = llm.predict(final_prompt)
    logger.info("Vector Retrieval complete.")
    return response

# Caching for RAG With KG
@st.cache_data
def run_with_neo4j(query):
    logger.info("Running with Neo4j MMKG integration...")
    retrieved_docs = retriever.get_relevant_documents(query)
    combined_text = "\n".join([doc.page_content for doc in retrieved_docs])
    graph_response = graph_chain.invoke({"query": query})

    final_prompt = f"""
    You are an AI assistant. Use the following information to answer the query:

    Pinecone Results:
    {combined_text}

    Graph Database Results:
    {graph_response}

    Query: {query}

    Provide the most accurate and comprehensive response.
    """
    response = llm.predict(final_prompt)
    logger.info("Neo4j Query complete.")
    return response

# Streamlit Interface
articles = [
    'https://www.bbc.com/news/articles/cy9j8r8gg0do',
    'https://www.bbc.com/news/articles/c04ld19vlg6o',
    'https://www.bbc.com/news/articles/cx24gze60yzo',
    'https://www.bbc.com/news/articles/c0mzl7zygpmo',
    'https://www.bbc.com/news/articles/c2dl0e4l7lzo',
    'https://www.bbc.com/news/business-61234231',
    'https://www.bbc.com/news/articles/c36pxnj01xgo',
    'https://www.bbc.com/news/articles/c3e8z53qyd5o',
    'https://www.bbc.com/news/articles/cx2lknel1xpo',
    'https://www.bbc.com/news/articles/c2k0zd2z53xo',
    'https://www.bbc.com/news/articles/crmzvdn9e18o',
    'https://www.bbc.com/news/articles/c04lvv6ee3lo',
    'https://www.bbc.com/news/articles/cj0jen70m88o',
    'https://www.bbc.com/news/articles/ced961egp65o',
]

if st.checkbox("About the Knowledge Graph"):
    st.subheader("Knowledge Graph Information")
    st.write(
        """
        The Multi-Modal Knowledge Graph used for this application was constructed using data extracted from a selection of articles sourced from the offical website of BBC. 
        The articles primarily focus on topics related to **US politics** and **climate change**. By integrating textual 
        and visual data, this Knowledge Graph aims to provide a comprehensive and interconnected view 
        of the relationships, entities, and multimedia associated with these domains.
        """
    )


query = st.text_input("Enter your query", placeholder="E.g., Images depicting Elon Musk")
if st.button("Run Query"):
    if not query.strip():
        st.error("Please enter a valid query.")
    else:
        logger.info(f"Processing query: {query}")
        st.write(f"Running query: {query}")

        st.subheader("RAG Without KG (Vector Retrieval Only)")
        rag_only_response = run_vector_retrieval_only(query)
        st.text_area("Response", rag_only_response, height=200)

        st.subheader("MMKG enhanced RAG (Vector Retrieval + MMKG)")
        rag_with_kg_response = run_with_neo4j(query)
        st.text_area("Response", rag_with_kg_response, height=200)

if st.checkbox("Show Source Articles"):
    st.subheader("Source Articles")
    for article in articles:
        st.markdown(f"- [{article}]({article})")
