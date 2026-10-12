import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

from ingest import build_vectorstore

PERSIST_DIR = "vectorstore"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You answer questions using only the context below, drawn from the AI PM Bible. "
    "If the context doesn't contain the answer, say you don't know instead of guessing.\n\n"
    "Context:\n{context}"
)


def load_retriever(k=6):
    if os.path.isdir(PERSIST_DIR):
        embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        store, _, _ = build_vectorstore()
    return store.as_retriever(search_kwargs={"k": k})


def format_docs(docs):
    return "\n\n".join(f"[p.{d.metadata.get('page', '?')}] {d.page_content}" for d in docs)


def build_chain(retriever, groq_api_key, model=GROQ_MODEL, temperature=0):
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", "{question}")]
    )
    llm = ChatGroq(model=model, api_key=groq_api_key, temperature=temperature)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
