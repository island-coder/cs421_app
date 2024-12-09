import logging
import re  # For extracting URLs
import streamlit as st
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Helper function to extract image URLs from a response
def extract_unique_image_urls(response_text):
    # Basic regex for URL extraction
    url_pattern = r'(https?://\S+\.(?:jpg|jpeg|png|gif|bmp))'
    urls = re.findall(url_pattern, response_text)
    return list(set(urls))  # Remove duplicates by converting to a set and back to a list


@st.cache_data
def run_vector_retrieval_only(query, _retriever: SelfQueryRetriever, _llm: ChatOpenAI):
    logger.info("Running Vector Retrieval only...")
    retrieved_docs = _retriever.get_relevant_documents(query)
    combined_text = "\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = f"""
    You are an AI assistant. Use the following information to answer the query:

    Pinecone Results:
    {combined_text}

    Query: {query}

    Provide the most accurate and comprehensive response.
    """
    response = _llm.predict(final_prompt)
    logger.info("Vector Retrieval complete.")
    return response

@st.cache_data
def run_with_neo4j(query, _retriever: SelfQueryRetriever, _graph_chain: GraphCypherQAChain, _llm: ChatOpenAI):
    logger.info("Running with Neo4j MMKG integration...")
    retrieved_docs = _retriever.get_relevant_documents(query)
    combined_text = "\n".join([doc.page_content for doc in retrieved_docs])
    graph_response = _graph_chain.invoke({"query": query})
    
    final_prompt = f"""
    You are an AI assistant designed to synthesize data from multiple sources, focusing on key attributes such as Article Source URLs, Image URLs, and their captions. Use the information provided to generate a clear, structured, and relevant response.

    ### Query
    {query}

    ### Instructions
    1. **Prioritize Key Attributes**:
    - Always include Article Source URLs to provide credibility and traceability.
    - If Image URLs are available, include them along with their captions (both image captions and generated captions, if provided).

    2. **Handle Missing Data Gracefully**:
    - If any key attribute (e.g., Image URL, Caption) is `None` or missing, omit it from the response.
    - Focus on synthesizing the available data without highlighting missing fields.

    3. **Emphasize Relationships**:
    - Use relationships (e.g., conflict, collaboration) from the Graph Database to provide context for the query.
    - Connect associates and their relationships to relevant articles or events.

    4. **Integrate Data Sources**:
    - Combine textual data from Pinecone Results with structured data from the Graph Database.
    - Ensure the response is cohesive, integrating insights from all available sources.
    
    5. **Avoid repeating duplicate images or image uris , unless it's required**:

    ### Data Sources
    **Pinecone Results**:
    {combined_text}

    **Graph Database Results**:
    Structured data retrieved from the Graph Database is provided below. Focus on the critical fields:

    - **Article Source URL**: Direct links to relevant articles.
    - **Image URL**: Links to multimedia content, if available.
    - **Image Caption**: Descriptions of the image, if provided.
    - **Generated Caption**: AI-generated descriptions, if available.

    {graph_response}

    ### Task
    Based on the provided data:
    1. Answer the query comprehensively and concisely.
    2. Include key attributes (Article Source URLs, Image URLs, and Captions) where available.
    3. Provide a unified response that combines textual and relational insights from both Pinecone and the Graph Database.
    4. Avoid mentioning missing or `None` values.
    """


    response = _llm.predict(final_prompt)
    logger.info("Neo4j Query complete.")
    return response

# Function to display response with image handling
def display_response_with_images(response_text):
    # Display the response text
    st.text_area("Response", response_text, height=200)

    # Extract image URLs and display them
    image_urls = extract_unique_image_urls(response_text)
    if image_urls:
        st.subheader("Images")
        for url in image_urls:
            st.image(url, caption=f"Image from {url}", use_container_width=True)  # Updated parameter
    else:
        st.write("No images found in the response.")
