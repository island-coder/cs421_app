import streamlit as st

def display_help():
    """
    Displays the help section in a modal or sidebar.
    """
    st.info("""
    # How to Use the App
 

    ### What Does the App Do?
    1. Retrieves relevant articles and images from a set of source articles.
    2. Provides insights into articles, relationships, and visual content.
    3. Synthesizes data from both a knowledge graph and vector database.

    ### Example Prompts
    - **Find articles mentioning 'Harris':**
      Retrieves articles where 'Harris' is mentioned and provides associated insights.

    - **Mentions of 'Ilham Aliyev':**
      Locates references to 'Ilham Aliyev' in articles, along with associated data.

    - **What can you infer from visual content relating to 'Christiana Figueres'?**
      Analyzes visual content linked to 'Christiana Figueres' for deeper context.

    - **What can you infer about 'Kamala Harris' or 'Harris' from photos linked to her or her associates?**
      Synthesizes visual and textual information to provide comprehensive insights.

    ### Tips for Using the App
    - Use clear and specific queries.
    - Combine names with context, e.g., "Harris in conflict-related articles."
    - If you need source articles or images, mention it in the prompt "Images where Elon Musk is present...".
    - Explore relationships, e.g., "Connections between Harris and policy advisors."

    ### Key Features
    - **Image Extraction**: Automatically identifies and displays image URLs and captions.
    - **Graph Integration**: Leverages graph databases for relational insights.
    - **Streamlined Results**: Focuses on essential details while omitting irrelevant information.
    """)
