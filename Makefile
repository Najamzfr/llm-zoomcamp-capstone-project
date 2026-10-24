.DEFAULT_GOAL := test

test:
	uv run python -m unittest discover -s tests -v

ingest:
	uv run python ingest.py

ingest-vectors:
	uv run python ingest_vector.py

ingest-qdrant:
	uv run python ingest_qdrant.py

evaluate:
	uv run python -m scripts.evaluate

baseline-rag:
	uv run python -m scripts.baseline_rag

live-smoke:
	uv run python -m scripts.live_smoke

api:
	uv run uvicorn api:app

run: api
