# AI Research Paper RAG Assistant

This project retrieves arXiv research papers with SQLite FTS5 plus FAISS/HNSW hybrid search, then uses GPT-4o-mini through an adaptive abstract/full-text agent. Full-text context remains capped at 5,000 tokens.

## Structure

- `api.py` — FastAPI backend (`/health`, `/ask`, `/metrics`)
- `app.py` — Streamlit frontend
- `rag.py`, `agent.py`, `paper_tools.py` — retrieval, adaptive answering, and PDF chunking
- `metrics.py` — Prometheus metrics
- `grafana/research_rag_overview.json` — importable dashboard
- `ingest.py`, `ingest_vector.py` — optional ingestion tools

## Windows PowerShell setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
$env:OPENAI_API_KEY = "your-openai-api-key"
```

Run ingestion only when the local database or vector index is missing:

```powershell
python ingest.py
python ingest_vector.py
```

Start the services in separate PowerShell windows:

```powershell
uvicorn api:app --reload
streamlit run app.py
```

- Streamlit: http://localhost:8501
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

The API uses GPT-4o-mini only. It does not persist prompts, answers, paper titles, API keys, or user questions in product traces.

## Future milestones

Docker, a Prometheus runtime, a Grafana runtime, PostgreSQL, Qdrant, authentication, and cloud deployment are intentionally out of scope for this first product layer.
