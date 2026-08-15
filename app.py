import math

import httpx
import streamlit as st
from research_assistant.config import settings

API_BASE_URL = settings.api_base_url
DEFAULT_API_TIMEOUT_SECONDS = settings.api_timeout_seconds


def _api_timeout_seconds():
    return (
        DEFAULT_API_TIMEOUT_SECONDS
        if math.isfinite(DEFAULT_API_TIMEOUT_SECONDS) and DEFAULT_API_TIMEOUT_SECONDS > 0
        else 300.0
    )


API_TIMEOUT_SECONDS = _api_timeout_seconds()
HTTP_CLIENT = httpx.Client(base_url=API_BASE_URL, trust_env=False)


def _health():
    try:
        response = HTTP_CLIENT.get("/health", timeout=5)
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _ask(question):
    try:
        response = HTTP_CLIENT.post(
            "/ask",
            json={"question": question},
            timeout=API_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException as error:
        raise RuntimeError(
            f"The request exceeded {API_TIMEOUT_SECONDS:g} seconds. "
            "The first request can be slower while the embedding model loads."
        ) from error
    except httpx.ConnectError as error:
        raise RuntimeError(
            f"Cannot connect to FastAPI at {API_BASE_URL}. Start it with: "
            "uvicorn api:app"
        ) from error
    except httpx.RequestError as error:
        raise RuntimeError("The connection to FastAPI failed.") from error

    if response.is_error:
        try:
            detail = response.json().get("detail", "The backend could not answer.")
        except ValueError:
            detail = "The backend could not answer."
        raise RuntimeError(f"FastAPI returned {response.status_code}: {detail}")
    return response.json()


def main():
    st.set_page_config(page_title="Research Paper RAG", page_icon="📄")
    st.title("AI Research Paper Assistant")
    st.write("Ask a question and get a cited answer grounded in retrieved arXiv papers.")

    with st.sidebar:
        st.subheader("FastAPI")
        st.success("Connected") if _health() else st.error("Unavailable")
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
    except RuntimeError as error:
        st.error(str(error))
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
