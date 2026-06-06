import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader, UnstructuredURLLoader
# to load the environment variables from a .env file
from dotenv import load_dotenv
import os
load_dotenv()

# streamlit app
st.set_page_config(page_title="Summerize YouTube Video or Website", page_icon=":books:")
st.title("🌟 Summerize YouTube Video or Website 🌟")
st.subheader("Summerize URL Content using LangChain and Groq LLM")

# get the groq api key 
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")

generic_url = st.text_input("Enter YouTube Video or Website URL",label_visibility="collapsed")

llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

# Define prompt template for summarization
prompt_template = """
Write short summary of the following content in concise points,
content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summerize Content"):
    if not groq_api_key:
        st.warning("Please enter your Groq API Key.")
    elif not generic_url or not validators.url(generic_url):# Validate URL
        st.warning("Please enter a valid YouTube Video or Website URL.")
    else:
        try:
            with st.spinner("Fetching and summerizing content..."):
                # Load content based on URL type
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False)

                documents = loader.load()

                # Chain for summarization using RefineChain
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt,
                    verbose=True
                )
                summary = chain.run(documents)
                st.subheader("🌟 Summary 🌟")
                st.success(summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")



