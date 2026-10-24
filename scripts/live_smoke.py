"""Opt-in real end-to-end check against a running FastAPI server (`make api`)."""

import os

import httpx

from research_assistant.config import settings


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running the live smoke check.")
    with httpx.Client(base_url=settings.api_base_url, trust_env=False) as client:
        try:
            client.get("/health", timeout=5).raise_for_status()
        except httpx.HTTPError as error:
            raise RuntimeError("FastAPI health check failed; start `make api` first.") from error
        response = client.post(
            "/ask",
            json={"question": "What research papers discuss neural network retrieval?"},
            timeout=settings.api_timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("answer") or not result.get("retrieved_papers"):
            raise RuntimeError("The API returned no answer or retrieved papers.")
        if not client.get("/").text.lstrip().startswith("<!doctype html>"):
            raise RuntimeError("The web UI is not being served at /.")
    print("Live API and UI smoke check passed.")


if __name__ == "__main__":
    main()
