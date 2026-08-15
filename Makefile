.DEFAULT_GOAL := test

test:
	uv run python -m unittest discover -s tests -v

ingest:
	uv run python ingest.py

ingest-vectors:
	uv run python ingest_vector.py

evaluate:
	uv run python scripts/evaluate.py

baseline-rag:
	uv run python scripts/baseline_rag.py

live-smoke:
	uv run python scripts/live_smoke.py

api:
	uv run uvicorn api:app

app:
	uv run streamlit run app.py
