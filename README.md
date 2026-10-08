# RAG-LangChain

A small retrieval-augmented Q&A app over Part VI of the AI PM Bible ("Agentic Product Design"). Ask a question, it retrieves the relevant chunks and answers from them, and shows its sources.

Everything free and hosted, nothing to run locally:
- **Embeddings:** local, free (`sentence-transformers/all-MiniLM-L6-v2`), built once at startup if no index exists yet
- **Vector store:** [Chroma](https://www.trychroma.com/), persisted to disk
- **LLM:** [Groq](https://groq.com/) (free tier), via `langchain-groq`
- **App hosting:** [Streamlit Community Cloud](https://streamlit.io/cloud) (free)

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your free Groq API key (console.groq.com)
streamlit run app.py
```

The vector index builds automatically on first run (a few seconds for this document) and is cached after that. To rebuild it manually: `python ingest.py`.

## Deploy for free

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and create a new app from this repo, with `app.py` as the entry point.
3. Under the app's **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
4. Deploy. The vector index builds automatically on the app's first run.

## Swapping in a different document

Replace the PDF in `Data/` and update `DATA_PATH` in `ingest.py`, then delete the `vectorstore/` folder (or just let the app rebuild it on next run if it's missing).
