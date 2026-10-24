# AI Research Paper RAG Assistant

This project retrieves arXiv research papers with hybrid search — SQLite FTS5 keyword search plus Qdrant vector search (FAISS/HNSW as an offline fallback), fused with reciprocal-rank fusion — then uses GPT-4o-mini through an adaptive abstract/full-text agent. Full-text context remains capped at 5,000 tokens. A custom web UI and the API are served by one FastAPI process, and Prometheus metrics feed a Grafana Cloud dashboard.

## Problem Description

The AI Research Paper Assistant helps students, researchers, and practitioners discover and understand recent AI/ML research without manually reading thousands of papers. It retrieves relevant arXiv papers, synthesizes answers from titles and abstracts, and escalates to selected full-text PDFs when abstracts are insufficient for detailed questions.

### Dataset

The project uses approximately 7,701 arXiv AI/ML papers from 2025–2026, including titles, abstracts, authors, categories, dates, and arXiv links. [Source dataset](https://www.kaggle.com/datasets/shree0910/arxiv-aiml-research-papers-20252026)

## Architecture

| Part | Technology |
|---|---|
| Web UI | HTML/CSS/JS in `static/` (dark mode, citations, feedback), served by FastAPI at `/` |
| API | FastAPI: `/ask`, `/feedback`, `/health`, `/metrics` (`api.py`) |
| Keyword search | SQLite FTS5 (`data/indexes/arxiv.db`) |
| Vector search | Qdrant Cloud free tier (`SEARCH_BACKEND=sqlite_qdrant`), or local FAISS HNSW (`sqlite_faiss`) |
| Embeddings | `all-MiniLM-L6-v2` via fastembed (ONNX, no PyTorch) |
| LLM | GPT-4o-mini, abstract-first with full-text PDF escalation |
| Monitoring | Prometheus metrics → Grafana Alloy → Grafana Cloud dashboard ([docs/monitoring.md](docs/monitoring.md)) |
| Tracing | Optional LangSmith traces of every agent step, search and LLM call ([docs/tracing.md](docs/tracing.md)) |

## Structure

- `api.py` — FastAPI backend and web UI host
- `static/` — web UI (`index.html`, `styles.css`, `app.js`, vendored `marked` + `DOMPurify`)
- `research_assistant/` — retrieval, embeddings, adaptive answering, PDF chunking, metrics, and project paths
- `scripts/` — ingestion (SQLite, vectors, Qdrant), evaluation, and RRF tuning commands
- `tests/` — API, embeddings, retrieval, and path/ingestion checks
- `data/source/` — source arXiv dataset; `data/indexes/` and `data/evaluation/` — local generated data
- `config/` — RRF configuration; `artifacts/` — local model cache, PDF cache, and traces
- `grafana/research_rag_overview.json` — importable 12-panel dashboard; `monitoring/alloy.alloy` — metrics shipper config
- `ingest.py`, `ingest_vector.py`, `ingest_qdrant.py` — ingestion entry points

## Windows PowerShell setup

```powershell
# Install uv first: https://docs.astral.sh/uv/getting-started/installation/
uv sync
Copy-Item .env.example .env
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:API_TIMEOUT_SECONDS = "180"
```

Run ingestion only when the local database, vector index, or Qdrant collection is missing:

```powershell
uv run python ingest.py          # SQLite FTS5 catalogue
uv run python ingest_vector.py   # embeddings + FAISS index
uv run python ingest_qdrant.py   # upload embeddings to Qdrant (needs QDRANT_URL / QDRANT_API_KEY)
```

Run checks:

```powershell
uv run python -m unittest discover -s tests -v
```

Run the retrieval baseline (after local ingestion):

```powershell
uv run python -m scripts.evaluate
```

`RAG_*_PATH` settings can be repository-relative or absolute. RRF precedence is
built-in defaults, then `config/rrf_config.json`, then `RAG_RRF_*` environment
variables. The embedding model (~90 MB) downloads to `artifacts/models/` on first use;
for downloads through SOCKS, set `HTTPS_PROXY=socks5://host:port`.

The opt-in paid RAG baseline and real end-to-end smoke check need an
`OPENAI_API_KEY`; start the API, then run `make baseline-rag` or `make live-smoke`.

Start the app:

```powershell
uv run uvicorn api:app --reload
```

- Web UI: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

The API uses GPT-4o-mini only. It does not persist prompts, answers, paper titles, API keys, or user questions locally. If you turn on LangSmith tracing (off by default), questions, retrieved abstracts and answers are sent to LangSmith.

## Monitoring

See [docs/monitoring.md](docs/monitoring.md) to stream metrics to a free Grafana Cloud stack and import
the dashboard (requests, errors, latency, tokens, cost, answer stages, full-text routing, judge
outcomes, and user feedback).

## Cloud deployment

A free public build runs on Streamlit Community Cloud from `deploy/streamlit_app.py`; see
[deploy/README.md](deploy/README.md).

## Paper Scout features (Streamlit app)

Everything is free for visitors and works without an account. Personal data (history, review deck,
collections) stays in the visitor's browser (localStorage); shares and feedback go to `SHARE_BACKEND`.

| Feature | Where |
|---|---|
| Follow-up suggestions (from the same answer call) | pills under each answer |
| Explain simply / Go deeper (same sources, citations kept) | action row |
| Copy, Markdown with references, BibTeX (stable keys, de-duplicated) | More |
| Question history (last 50, reopens instantly, inline "Clear history") | sidebar → Recent |
| Featured answers gallery (zero model calls) | empty state |
| Study this: flashcards + quiz; wrong answers go to review; harder set | action row |
| Spaced review (FSRS), daily cap, topics, streak, JSON backup/import | sidebar → Review |
| Anki `.apkg` and CSV export | Study this / Review |
| Concept map (Graphviz) with explore pills, SVG + Mermaid export | action row |
| Share links (`?a=<id>`), read-only page, optional study set | action row |
| Upload your own PDFs (session only) + "Include my papers" | next to the composer |
| Literature review report (.md and .docx) | More → Write a review |
| Listen (browser speech, speed and voice) | action row |
| Collections with export (.md + .bib) | Save, sidebar → Collections |
| Feedback with reasons | 👍 / 👎 |

Feature code lives in `app/features/` (one module per feature) and `app/storage/local.py`; the plan and
mockups are in [PLAN.md](PLAN.md) and [docs/mockups/features.html](docs/mockups/features.html).

### Environment variables for the extra features

| Variable | Default | Purpose |
|---|---|---|
| `EXTRAS_LLM_PROVIDER` | `openai` | `openai`, `gemini` or `groq` for follow-up fallback, rewrites, study sets, concept maps, reviews |
| `EXTRAS_LLM_MODEL` | per provider | `gpt-4o-mini`, `gemini-2.5-flash`, `llama-3.3-70b-versatile` |
| `GEMINI_API_KEY`, `GROQ_API_KEY` | – | keys for the free-tier providers |
| `SHARE_BACKEND` | `memory` | `supabase` (persistent) or `memory` (this server process only) |
| `SUPABASE_URL`, `SUPABASE_ANON_KEY` | – | Supabase project (tables and RLS policies in [PLAN.md](PLAN.md)) |
| `APP_URL` | page URL | base URL of share links, e.g. `https://paprly.streamlit.app` |
| `REVIEW_DAILY_CAP` | `20` | spaced-review cards per day |

See `.streamlit/secrets.toml.example` for a ready-to-paste secrets block.

### Rebuilding the featured answers

```powershell
uv run python scripts/build_featured.py   # calls the real pipeline; writes data/featured.json
```

### Tests

```powershell
uv run python -m pytest -q   # unit tests + streamlit.testing AppTest smoke tests (no network)
```
