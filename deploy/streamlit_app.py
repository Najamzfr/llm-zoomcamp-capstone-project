"""Streamlit Community Cloud entry point: runs the research agent in-process (no FastAPI server).

Deploy settings: main file `deploy/streamlit_app.py`, Python 3.12, secrets as in deploy/README.md.
Locally: `uv run --with streamlit streamlit run deploy/streamlit_app.py`.
"""

import os
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_secrets_into_env():
    """Expose root-level Streamlit secrets as environment variables before the app config loads."""
    try:
        secrets = dict(st.secrets)
    except Exception:  # no secrets file locally; .env is used instead
        return
    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)
    os.environ.setdefault("SEARCH_BACKEND", "sqlite_qdrant")


_load_secrets_into_env()

from api import AskLimiter, _response  # noqa: E402  (reuse the API's response shape and limiter)
from research_assistant.config import settings  # noqa: E402
from research_assistant.tracing import send_feedback  # noqa: E402

STAGE_LABELS = {
    "abstract": "From abstracts",
    "full_text": "From full text",
    "abstract_fallback": "Abstract fallback",
}
EXAMPLES = [
    "How is retrieval-augmented generation evaluated?",
    "Recent methods for reducing LLM hallucinations",
    "Compare approaches to efficient fine-tuning",
]

st.set_page_config(page_title="Paper Scout", page_icon="📄", layout="centered")
st.markdown(
    """
    <style>
      .block-container { max-width: 860px; padding-top: 2.2rem; }
      .ps-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 0.2rem; }
      .ps-mark { width: 40px; height: 40px; border-radius: 11px; background: #4f46e5; color: #fff;
                 display: grid; place-items: center; font-size: 20px; }
      .ps-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }
      .ps-tag { opacity: 0.7; margin: 0 0 1.2rem; }
      .ps-badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
                  font-weight: 600; background: rgba(79, 70, 229, 0.12); color: #6366f1; }
      .ps-paper { border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 12px; padding: 12px 14px;
                  height: 100%; }
      .ps-paper-meta { font-size: 0.75rem; opacity: 0.7; display: flex; gap: 8px; margin-bottom: 4px; }
      .ps-num { font-weight: 700; color: #6366f1; }
      .ps-paper a { font-weight: 600; text-decoration: none; }
      .ps-paper a:hover { text-decoration: underline; }
    </style>
    <div class="ps-brand"><div class="ps-mark">📄</div><p class="ps-title">Paper Scout</p></div>
    <p class="ps-tag">Answers grounded in 7,701 recent arXiv AI/ML papers</p>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the paper index…")
def get_agent():
    if not settings.database_path.exists():
        from scripts.ingest import ingest

        ingest()  # ~1 s: builds the SQLite FTS5 catalogue from data/source/data.csv
    from research_assistant.agent import AdaptiveResearchAgent

    return AdaptiveResearchAgent(trace_path=None)


@st.cache_resource
def get_limiter():
    # Shared by all visitors of this app instance; protects the OpenAI budget.
    return AskLimiter(settings.ask_rate_per_minute, settings.ask_daily_limit)


if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not configured. Add it to the app's secrets.")
    st.stop()

cols = st.columns(len(EXAMPLES))
for column, example in zip(cols, EXAMPLES):
    if column.button(example, use_container_width=True):
        st.session_state["question"] = example
        st.session_state["submit"] = True

with st.form("ask"):
    question = st.text_area(
        "What do you want to know?",
        key="question",
        height=110,
        max_chars=settings.max_question_length,
        placeholder="e.g. How are LLM agents evaluated in recent papers?",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted or st.session_state.pop("submit", False):
    question = (st.session_state.get("question") or "").strip()
    if not question:
        st.warning("Type a question first.")
    elif not get_limiter().allow():
        st.warning("The question limit was reached. Please wait a minute and try again.")
    else:
        try:
            with st.spinner("Searching papers and composing a cited answer… detailed questions can take a minute."):
                result = get_agent().run(question)
            response = _response(result, question, uuid.uuid4().hex)
            response["trace_run_id"] = result.get("trace_run_id")
            st.session_state["result"] = response
            st.session_state.pop("rated", None)
        except Exception as error:  # show a friendly message instead of a stack trace
            st.session_state.pop("result", None)
            st.error(f"The research assistant failed to answer ({type(error).__name__}). Please try again.")

result = st.session_state.get("result")
if result:
    stage = result.get("answer_stage", "unknown")
    st.markdown(
        f"#### Answer &nbsp;<span class='ps-badge'>{STAGE_LABELS.get(stage, stage)}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(result.get("answer", ""))

    rating = st.feedback("thumbs", key=f"fb-{result.get('request_id')}")
    if rating is not None and not st.session_state.get("rated"):
        st.session_state["rated"] = True
        send_feedback(result.get("trace_run_id"), "up" if rating == 1 else "down")
        st.toast("Thanks for the feedback!")

    st.markdown("#### Sources")
    papers = result.get("retrieved_papers", [])
    for row in range(0, len(papers), 2):
        for column, (index, paper) in zip(st.columns(2), list(enumerate(papers))[row : row + 2]):
            meta = " · ".join(str(v) for v in (paper.get("category"), (paper.get("published") or "")[:10]) if v)
            title = paper.get("title") or "Untitled paper"
            link = f"<a href='{paper['url']}' target='_blank' rel='noopener'>{title}</a>" if paper.get("url") else title
            column.markdown(
                f"<div class='ps-paper'><div class='ps-paper-meta'><span class='ps-num'>[{index + 1}]</span>"
                f"<span>{meta}</span></div>{link}</div>",
                unsafe_allow_html=True,
            )

    with st.expander("Run details"):
        usage = result.get("usage", {})
        a, b, c, d = st.columns(4)
        a.metric("Stage", STAGE_LABELS.get(stage, stage))
        b.metric("Input tokens", f"{usage.get('input_tokens', 0):,}")
        c.metric("Output tokens", f"{usage.get('output_tokens', 0):,}")
        d.metric("Est. cost", f"${usage.get('estimated_cost_usd', 0) or 0:.5f}")
        for step in result.get("trajectory", []):
            st.markdown(f"- `{step.get('name')}` {step.get('status') or ''} {step.get('outcome') or ''}")
        if stage == "full_text":
            st.caption("In full-text answers, [n] refers to the numbered PDF excerpts, not the source list.")

st.caption("SQLite FTS5 + Qdrant hybrid search · fastembed · GPT-4o-mini")
