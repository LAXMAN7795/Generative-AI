# pip install fastapi uvicorn lnagserve
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Define a custom prompt template
generic_template = "Translate the following text into {language}:"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",generic_template),
        ("human","{text}")
    ]
)

# parser
parser = StrOutputParser()

# chaining 
chain = prompt | model | parser

# app definition

app = FastAPI(
    title="LangServe with Groq LLM",
    description="An API to demonstrate LangServe with Groq LLM",
    version="0.1",
)

# adding chain as route
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)