import os
import logging
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from prompt_templates import CYPHER_GENERATION_TEMPLATE
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain
from neo4_utils import get_neo4j_graph  # Ensure this is correctly imported

def initialize_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

# Initialize Logging
logger = initialize_logging()

# Load Pinecone Vectorstore
from pinecone_utils import get_vectorstore  # Ensure pinecone_utils.py exists
vectorstore = get_vectorstore()

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Metadata field info for retriever
metadata_field_info = [
    AttributeInfo(name="id", description="The source URL of the article", type="string"),
    AttributeInfo(name="title", description="The title of the article", type="string"),
    AttributeInfo(name="description", description="A short description of the article", type="string"),
]

document_content_description = "The main content of the article"

# Initialize SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_content_description=document_content_description,
    metadata_field_info=metadata_field_info,
    document_contents="page_content",
    verbose=False,
)

# Initialize Cypher Prompt
cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

vqa_prompt = PromptTemplate(template="""
You are an AI assistant answering questions based on data retrieved from a graph database. Ensure that your response includes any source links or URIs for explainability.

### Expected Format:
1. **Answer**: Provide a concise and accurate answer to the query.
2. **Source URIs**: Always include any source URIs found in the graph response. These are usually in the format:
   - Article Source: 'article_source_url': {{'uri': 'https://.....'}}
   - Image Source: 'image_url': {{'uri': 'https://.....'}}
3. **Image Captions**: Include captions and any additional image details if available.

### Handling Missing Data:
- If URIs or captions are missing, focus on providing a complete response based on the available data without highlighting the absence of missing fields.
""")

# Initialize Neo4j Graph
graph = get_neo4j_graph()  # Properly initialize the Neo4j graph here

# Initialize Graph QA Chain
graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    qa_llm=llm,
    graph=graph,  # Pass the initialized graph here
    cypher_prompt=cypher_prompt,
    #qa_prompt=vqa_prompt,
    return_direct=True,
    validate_cypher=True,
    allow_dangerous_requests=True,
    verbose=True,
)
