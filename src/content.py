import streamlit as st
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

def display_about_kg():
    st.subheader("Knowledge Graph Information")
    st.write(
        """
        The Multi-Modal Knowledge Graph used for this application was constructed using data extracted 
        from a selection of articles sourced from the official website of BBC. The articles primarily focus 
        on topics related to **US politics** and **climate change**. By integrating textual and visual data, 
        this Knowledge Graph aims to provide a comprehensive and interconnected view of the relationships, 
        entities, and multimedia associated with these domains.
        """
    )
