import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

from ingest import build_vectorstore

load_dotenv()

PERSIST_DIR = "vectorstore"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "llama-3.1-8b-instant"


def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


@st.cache_resource
def load_retriever():
    if os.path.isdir(PERSIST_DIR):
        embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        store, _, _ = build_vectorstore()
    return store.as_retriever(search_kwargs={"k": 4})


def format_docs(docs):
    return "\n\n".join(f"[p.{d.metadata.get('page', '?')}] {d.page_content}" for d in docs)


st.title("AI PM Bible — Agentic Product Design")
st.caption("Retrieval-augmented Q&A over Part VI of the AI PM Bible, running on free hosted inference.")

groq_api_key = get_groq_api_key()
if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Add it to a .env file locally, or to Streamlit secrets when deployed.")
    st.stop()

with st.spinner("Loading index..." if os.path.isdir(PERSIST_DIR) else "Building index (first run)..."):
    retriever = load_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using only the context below, drawn from the AI PM Bible. "
            "If the context doesn't contain the answer, say you don't know instead of guessing.\n\n"
            "Context:\n{context}",
        ),
        ("user", "{question}"),
    ]
)

llm = ChatGroq(model=GROQ_MODEL, api_key=groq_api_key, temperature=0)
output_parser = StrOutputParser()

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

question = st.text_input("Ask a question about agentic product design")

if question:
    with st.spinner("Retrieving and generating..."):
        answer = chain.invoke(question)
        sources = retriever.invoke(question)

    st.write(answer)

    with st.expander("Sources"):
        for doc in sources:
            st.caption(f"Page {doc.metadata.get('page', '?')}")
            st.text(doc.page_content[:300] + "...")
