import os

import streamlit as st
from dotenv import load_dotenv

from rag import PERSIST_DIR, build_chain, load_retriever

load_dotenv()


def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


st.title("AI PM Bible — Agentic Product Design")
st.caption("Retrieval-augmented Q&A over Part VI of the AI PM Bible, running on free hosted inference.")

groq_api_key = get_groq_api_key()
if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Add it to a .env file locally, or to Streamlit secrets when deployed.")
    st.stop()

with st.spinner("Loading index..." if os.path.isdir(PERSIST_DIR) else "Building index (first run)..."):
    retriever = st.cache_resource(load_retriever)()

chain = build_chain(retriever, groq_api_key)

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
