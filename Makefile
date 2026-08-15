.DEFAULT_GOAL := test

test:
	uv run python -m unittest discover -s tests -v

ingest:
	uv run python ingest.py

ingest-vectors:
	uv run python ingest_vector.py

evaluate:
	uv run python scripts/evaluate.py

api:
	uv run uvicorn api:app

app:
	uv run streamlit run app.py
