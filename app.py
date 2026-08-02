"""Streamlit frontend for the FastAPI research-paper assistant."""

import os
import math

import httpx
import streamlit as st


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _api_timeout_seconds():
    try:
        value = float(os.environ.get("API_TIMEOUT_SECONDS", "180"))
    except (TypeError, ValueError):
        return 180.0
    return value if math.isfinite(value) and value > 0 else 180.0


API_TIMEOUT_SECONDS = _api_timeout_seconds()


def _health():
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _ask(question):
    response = httpx.post(
        f"{API_BASE_URL}/ask", json={"question": question}, timeout=API_TIMEOUT_SECONDS
    )
    if response.is_error:
        try:
            detail = response.json().get("detail", "The backend could not answer.")
        except ValueError:
            detail = "The backend could not answer."
        raise RuntimeError(detail)
    return response.json()


def main():
    st.set_page_config(page_title="Research Paper RAG", page_icon="📄")
    st.title("AI Research Paper Assistant")
    st.write("Ask a question and get a cited answer grounded in retrieved arXiv papers.")

    with st.sidebar:
        st.subheader("Backend")
        st.success("Healthy") if _health() else st.error("Unavailable")
        st.caption(API_BASE_URL)

    question = st.text_area("Research question", height=120, placeholder="Compare the methods used for...")
    if not st.button("Ask", type="primary"):
        return
    if not question.strip():
        st.warning("Enter a research question first.")
        return

    try:
        with st.spinner("Searching papers and composing an answer…"):
            result = _ask(question)
    except (httpx.HTTPError, RuntimeError):
        st.error("The backend could not answer right now. Check that FastAPI is running.")
        return

    st.markdown(result.get("answer", ""))
    st.subheader("Retrieved papers")
    for paper in result.get("retrieved_papers", []):
        title = paper.get("title", "Untitled paper")
        url = paper.get("url")
        st.markdown(f"- [{title}]({url})" if url else f"- {title}")
        details = " | ".join(str(value) for value in (paper.get("published"), paper.get("category")) if value)
        if details:
            st.caption(details)

    with st.expander("Run Details"):
        st.write("Answer stage:", result.get("answer_stage", "unknown"))
        st.json(result.get("trajectory", []))
        usage = result.get("usage", {})
        st.write(
            f"Tokens: {usage.get('input_tokens', 0)} input / "
            f"{usage.get('output_tokens', 0)} output / "
            f"{usage.get('total_tokens', 0)} total"
        )
        st.write(f"Estimated cost: ${usage.get('estimated_cost_usd', 0):.8f}")


if __name__ == "__main__":
    main()
