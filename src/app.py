import os
import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import GraphCypherQAChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts.prompt import PromptTemplate

# Load credentials
st.title("Knowledge Graph & Vector Retrieval Demo")

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
st.write("Connecting to Neo4j...")
graph = Neo4jGraph()
graph.refresh_schema()
st.success("Neo4j connected and schema loaded.")

# Initialize Pinecone
def init_pinecone_index():
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
    return pc.Index(index_name)

st.write("Initializing Pinecone...")
index = init_pinecone_index()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
st.success("Pinecone initialized.")

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
    verbose=True
)

# Cypher Generation Template
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Use graph query results as base and formulate a high-quality response. Double-check for query correctness.
Schema:
{schema}
THE FOLLOWING ARE RULES YOU MUST OBEY:
All entities will be coming from one or more referring articles.
When asked about the relationship or association between people, try to use all possible person-to-person relationships from the schema to find relationships, do not limit to a few.
When you need to find the source URL/link for articles, please use the has_source_url relationship.(MATCH (i:article)-[s:has_source_url]->(link)).
When doing any query related to images use the following relationship to get the image URL.(MATCH (i:image)-[s:has_source_url]->(link))
'depiction' nodes contain 'has_bounding_box' property, this gives the bounding box for the person who is depicted in the image.
'image' nodes contain 'has_caption' and 'has_generated_caption' properties which give image descriptions.
If you fail to find an entity the first time, try to search with similar names. Entities will be connected to images using depictions.
Include multimedia details like image URLs, article sources in your results as much as possible.
The question is:
{question}"""

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

# Define RAG without KG
def run_vector_retrieval_only(query):
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
    return response

# Define RAG with KG
def run_with_neo4j(query):
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

query = st.text_input("Enter your query", placeholder="E.g., Images depicting Elon Musk")
if st.button("Run Query"):
    if not query.strip():
        st.error("Please enter a valid query.")
    else:
        st.write(f"Running query: {query}")

        st.subheader("RAG Without KG (Vector Retrieval Only)")
        rag_only_response = run_vector_retrieval_only(query)
        st.text_area("Response", rag_only_response, height=200)

        st.subheader("RAG With KG (Vector Retrieval + Neo4j)")
        rag_with_kg_response = run_with_neo4j(query)
        st.text_area("Response", rag_with_kg_response, height=200)

if st.checkbox("Show Source Articles"):
    st.subheader("Source Articles")
    for article in articles:
        st.markdown(f"- [{article}]({article})")
