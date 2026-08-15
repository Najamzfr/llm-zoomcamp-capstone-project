"""Opt-in real Streamlit-client to FastAPI `/ask` smoke check."""

import os

from app import _ask, _health


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running the live smoke check.")
    if not _health():
        raise RuntimeError("FastAPI health check failed; start `make api` first.")
    result = _ask("What research papers discuss neural network retrieval?")
    if not result.get("answer") or not result.get("retrieved_papers"):
        raise RuntimeError("The API returned no answer or retrieved papers.")
    print("Live Streamlit-client to API smoke check passed.")


if __name__ == "__main__":
    main()
