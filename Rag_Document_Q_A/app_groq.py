import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the qustion based on the provided context only.
    Please provide the most accurate response based on the context.
    <context>
    {context}
    <context>
    Qustion:{input}
    """
)

def create_vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")#data ingestion
        st.session_state.docs = st.session_state.loader.load()# document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vector = FAISS.from_documents(st.session_state.final_document,st.session_state.embeddings)
    
user_prompt = st.text_input("Enter your query for research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("vectore database is ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vector.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({"input":user_prompt})
    st.write(time.process_time()-start)

    st.write(response['answer'])
    