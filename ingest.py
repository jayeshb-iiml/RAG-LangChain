from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "Data/AI_PM_Bible_Part_VI_Agentic_Product_Design.pdf"
PERSIST_DIR = "vectorstore"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def build_vectorstore():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    # Page 1 is the table of contents — pure navigation text with no answering
    # value, which was crowding out real content in retrieval.
    documents = [d for d in documents if d.metadata.get("page") != 1]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return store, len(chunks), len(documents)


if __name__ == "__main__":
    _, num_chunks, num_pages = build_vectorstore()
    print(f"Indexed {num_chunks} chunks from {num_pages} pages into {PERSIST_DIR}/")
