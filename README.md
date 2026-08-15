# AI Research Paper RAG Assistant

This project retrieves arXiv research papers with SQLite FTS5 plus FAISS/HNSW hybrid search, then uses GPT-4o-mini through an adaptive abstract/full-text agent. Full-text context remains capped at 5,000 tokens.

## Problem Description

The AI Research Paper Assistant helps students, researchers, and practitioners discover and understand recent AI/ML research without manually reading thousands of papers. It retrieves relevant arXiv papers, synthesizes answers from titles and abstracts, and escalates to selected full-text PDFs when abstracts are insufficient for detailed questions.

### Dataset

The project uses approximately 7,701 arXiv AI/ML papers from 2025–2026, including titles, abstracts, authors, categories, dates, and arXiv links. [Source dataset](https://www.kaggle.com/datasets/shree0910/arxiv-aiml-research-papers-20252026)

## Structure

- `api.py` — FastAPI backend (`/health`, `/ask`, `/metrics`)
- `app.py` — Streamlit frontend
- `research_assistant/` — retrieval, adaptive answering, PDF chunking, metrics, and project paths
- `scripts/` — ingestion, evaluation, and RRF tuning commands
- `tests/` — API, Streamlit, and path/ingestion checks
- `data/source/` — source arXiv dataset; `data/indexes/` and `data/evaluation/` — local generated data
- `config/` — RRF configuration; `artifacts/` — local PDF cache and traces
- `grafana/research_rag_overview.json` — importable dashboard
- `ingest.py`, `ingest_vector.py` — compatibility entry points for optional ingestion

## Windows PowerShell setup

```powershell
# Install uv first: https://docs.astral.sh/uv/getting-started/installation/
uv sync
Copy-Item .env.example .env
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:API_TIMEOUT_SECONDS = "180"
```

Run ingestion only when the local database or vector index is missing:

```powershell
uv run python ingest.py
uv run python ingest_vector.py
```

Run checks:

```powershell
uv run python -m unittest discover -s tests -v
```

Run the retrieval baseline (after local ingestion):

```powershell
uv run python scripts/evaluate.py
```

`RAG_*_PATH` settings can be repository-relative or absolute. RRF precedence is
built-in defaults, then `config/rrf_config.json`, then `RAG_RRF_*` environment
variables. For Hugging Face downloads through SOCKS, set
`HTTPS_PROXY=socks5://host:port` before vector ingestion.

The opt-in paid RAG baseline and real UI-to-API smoke check need an
`OPENAI_API_KEY`; start the API, then run `make baseline-rag` or `make live-smoke`.

Start the services in separate PowerShell windows:

```powershell
uv run uvicorn api:app --reload
uv run streamlit run app.py
```

- Streamlit: http://localhost:8501
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

The API uses GPT-4o-mini only. It does not persist prompts, answers, paper titles, API keys, or user questions in product traces.

## Future milestones

Docker, the Prometheus/Grafana runtime, PostgreSQL, Qdrant, authentication, and cloud deployment are reserved for a future Docker milestone.
