from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "Data/AI_PM_Bible_Part_VI_Agentic_Product_Design.pdf"
PERSIST_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return store, len(chunks), len(documents)


if __name__ == "__main__":
    _, num_chunks, num_pages = build_vectorstore()
    print(f"Indexed {num_chunks} chunks from {num_pages} pages into {PERSIST_DIR}/")
