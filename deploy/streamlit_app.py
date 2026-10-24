"""Streamlit Community Cloud entry point: runs the research agent in-process (no FastAPI server).

Deploy settings: main file `deploy/streamlit_app.py`, Python 3.12, secrets as in deploy/README.md.
Locally: `uv run --with "streamlit>=1.50" streamlit run deploy/streamlit_app.py`.
Theme (fonts, colours, light/dark): `.streamlit/config.toml`.
"""

import html
import os
import sys
import time
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

REPO_URL = "https://github.com/Najamzfr/llm-zoomcamp-capstone-project"
STAGE_LABELS = {
    "abstract": "From abstracts",
    "full_text": "From full text",
    "abstract_fallback": "Abstract fallback",
}
EXAMPLES = [
    "How is retrieval-augmented generation evaluated?",
    "Recent methods for reducing LLM hallucinations",
    "Compare approaches to efficient fine-tuning",
    "Which benchmarks are used to evaluate LLM agents?",
]
USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/auto_awesome:"

st.set_page_config(page_title="Paper Scout", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
      .block-container { max-width: 860px; padding-top: 2.5rem; padding-bottom: 6rem; }
      footer { visibility: hidden; }

      /* Hero (empty state) */
      .ps-hero { text-align: center; padding: 2.2rem 0 1.4rem; }
      .ps-logo { width: 56px; height: 56px; margin: 0 auto 14px; border-radius: 16px; display: grid;
                 place-items: center; font-size: 26px; color: #fff;
                 background: linear-gradient(135deg, #6366f1, #8b5cf6);
                 box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35); }
      .ps-title { font-size: 2.4rem; line-height: 1.15; font-weight: 700; letter-spacing: -0.03em; margin: 0;
                  background: linear-gradient(135deg, #6366f1 10%, #a855f7 90%);
                  -webkit-background-clip: text; background-clip: text; color: transparent; }
      .ps-sub { opacity: 0.72; font-size: 1.05rem; margin: 0.4rem 0 1.1rem; }
      .ps-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
      .ps-chip { font-size: 0.8rem; padding: 5px 12px; border-radius: 999px;
                 border: 1px solid rgba(128, 128, 128, 0.25); background: rgba(128, 128, 128, 0.06); }
      .ps-try { text-align: center; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
                opacity: 0.6; margin: 1.6rem 0 0.4rem; }

      @media (max-width: 640px) { .ps-hero { padding-top: 3.6rem; } .ps-title { font-size: 2rem; } }

      /* Answer header */
      .ps-meta { display: flex; align-items: center; gap: 8px; margin-bottom: 0.35rem; font-size: 0.8rem; }
      .ps-badge { font-weight: 600; padding: 2px 10px; border-radius: 999px;
                  background: rgba(99, 102, 241, 0.14); color: #6366f1; }
      .ps-badge.full_text { background: rgba(22, 163, 74, 0.14); color: #16a34a; }
      .ps-badge.abstract_fallback { background: rgba(217, 119, 6, 0.14); color: #d97706; }
      .ps-time { opacity: 0.6; }

      /* Source cards */
      .ps-section { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
                    opacity: 0.6; margin: 1rem 0 0.5rem; }
      .ps-sources { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 10px; }
      a.ps-card { display: flex; flex-direction: column; gap: 6px; padding: 12px 14px; border-radius: 12px;
                  border: 1px solid rgba(128, 128, 128, 0.22); background: rgba(128, 128, 128, 0.04);
                  color: inherit !important; text-decoration: none !important;
                  transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease; }
      a.ps-card:hover { transform: translateY(-2px); border-color: #6366f1;
                        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.16); }
      .ps-card-top { display: flex; align-items: center; gap: 6px; font-size: 0.72rem; }
      .ps-num { display: grid; place-items: center; width: 22px; height: 22px; border-radius: 7px;
                font-weight: 700; background: rgba(99, 102, 241, 0.14); color: #6366f1; }
      .ps-cat { font-family: ui-monospace, Consolas, monospace; padding: 1px 6px; border-radius: 5px;
                background: rgba(128, 128, 128, 0.12); }
      .ps-date { margin-left: auto; opacity: 0.6; }
      .ps-card-title { font-weight: 600; font-size: 0.9rem; line-height: 1.35; }
      .ps-open { font-size: 0.75rem; color: #6366f1; margin-top: auto; }

      /* Compact stats */
      .ps-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; margin: 0.3rem 0 0.6rem; }
      .ps-stat { padding: 8px 10px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2);
                 background: rgba(128, 128, 128, 0.05); }
      .ps-stat-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.6; }
      .ps-stat-value { font-size: 1.05rem; font-weight: 600; font-variant-numeric: tabular-nums; }

      /* Sidebar */
      .ps-side-brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.1rem; }
      .ps-side-brand span { display: grid; place-items: center; width: 32px; height: 32px; border-radius: 9px;
                            color: #fff; background: linear-gradient(135deg, #6366f1, #8b5cf6); }
    </style>
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


state = st.session_state
state.setdefault("messages", [])
state.setdefault("rated", set())
state.setdefault("stats", {"questions": 0, "cost": 0.0})


def _queue_example():
    state["pending"] = state.get("example_pick")
    state["example_pick"] = None


def describe(event, data):
    """Turn an agent progress event into a short, human sentence."""
    if event == "searched":
        return f"Found {data.get('papers', 0)} candidate papers with keyword + vector search"
    if event == "routed":
        if data.get("needs_full_text"):
            return "Abstracts aren't detailed enough — reading the full papers"
        return "The abstracts cover this — answering from them"
    if event == "fetching_full_text":
        return f"Downloading and chunking {data.get('papers', 0)} PDFs"
    if event == "full_text_ready":
        return f"Ranked {data.get('chunks', 0)} relevant excerpts ({data.get('tokens', 0):,} tokens)"
    if event == "full_text_failed":
        return "Full text unavailable — falling back to the abstracts"
    if event == "drafted":
        return "Drafted a cited answer"
    if event == "checked":
        status = data.get("status")
        if status == "pass":
            return "Answer verified against the evidence"
        if status == "needs_full_text":
            return "Verifier wants more detail — escalating to full text"
        return "Some details aren't fully supported — noted in the answer"
    return event


def source_cards(papers):
    cards = []
    for number, paper in enumerate(papers, start=1):
        title = html.escape(paper.get("title") or "Untitled paper")
        category = html.escape(str(paper.get("category") or ""))
        date = html.escape(str(paper.get("published") or "")[:10])
        url = paper.get("url") or ""
        href = html.escape(url, quote=True) if url.startswith(("https://", "http://")) else "#"
        cards.append(
            f"<a class='ps-card' href='{href}' target='_blank' rel='noopener noreferrer'>"
            f"<div class='ps-card-top'><span class='ps-num'>{number}</span>"
            + (f"<span class='ps-cat'>{category}</span>" if category else "")
            + (f"<span class='ps-date'>{date}</span>" if date else "")
            + f"</div><div class='ps-card-title'>{title}</div>"
            "<div class='ps-open'>Open on arXiv ↗</div></a>"
        )
    return "<div class='ps-sources'>" + "".join(cards) + "</div>"


def stat_grid(items):
    return "<div class='ps-stats'>" + "".join(
        f"<div class='ps-stat'><div class='ps-stat-label'>{html.escape(label)}</div>"
        f"<div class='ps-stat-value'>{html.escape(value)}</div></div>"
        for label, value in items
    ) + "</div>"


def render_assistant(message):
    kind = message.get("kind")
    if kind == "warning":
        st.warning(message["content"], icon=":material/hourglass_top:")
        return
    if kind == "error":
        st.error(message["content"], icon=":material/error:")
        return

    result = message["result"]
    stage = result.get("answer_stage", "unknown")
    stage_text = html.escape(STAGE_LABELS.get(stage, stage))
    st.markdown(
        f"<div class='ps-meta'><span class='ps-badge {html.escape(stage)}'>{stage_text}</span>"
        f"<span class='ps-time'>{message.get('seconds', 0):.0f} s</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(result.get("answer", ""))  # LLM text: Markdown only, no raw HTML

    request_id = result.get("request_id")
    rating = st.feedback("thumbs", key=f"fb-{request_id}")
    if rating is not None and request_id not in state["rated"]:
        state["rated"].add(request_id)
        send_feedback(result.get("trace_run_id"), "up" if rating == 1 else "down")
        st.toast("Thanks for the feedback!", icon=":material/favorite:")

    papers = result.get("retrieved_papers", [])
    if papers:
        st.markdown("<div class='ps-section'>Sources</div>" + source_cards(papers), unsafe_allow_html=True)

    with st.expander("Research steps & usage", icon=":material/insights:"):
        usage = result.get("usage", {})
        st.markdown(
            stat_grid(
                [
                    ("Stage", STAGE_LABELS.get(stage, stage)),
                    ("Input tokens", f"{usage.get('input_tokens', 0):,}"),
                    ("Output tokens", f"{usage.get('output_tokens', 0):,}"),
                    ("Est. cost", f"${usage.get('estimated_cost_usd', 0) or 0:.4f}"),
                ]
            ),
            unsafe_allow_html=True,
        )
        for step in message.get("steps", []):
            st.markdown(f":material/check_small: {step}")
        if stage == "full_text":
            st.caption("In full-text answers, [n] refers to the numbered PDF excerpts, not the source cards.")


def answer(question):
    """Run the agent with a live progress panel and append the reply to the conversation."""
    if not get_limiter().allow():
        state["messages"].append(
            {"role": "assistant", "kind": "warning",
             "content": "The question limit for this demo was reached. Please wait a minute and try again."}
        )
        return
    steps = []
    started = time.perf_counter()
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.status("Researching…", expanded=True) as status:

            def on_progress(event, data):
                line = describe(event, data)
                steps.append(line)
                status.update(label=line)
                st.markdown(f":material/check_small: {line}")

            try:
                result = get_agent().run(question, on_progress=on_progress)
            except Exception as error:  # friendly message instead of a stack trace
                status.update(label="Something went wrong", state="error", expanded=False)
                state["messages"].append(
                    {"role": "assistant", "kind": "error",
                     "content": f"The research assistant failed to answer ({type(error).__name__}). Please try again."}
                )
                return
            seconds = time.perf_counter() - started
            stage = STAGE_LABELS.get(result.get("answer_stage"), "Done")
            status.update(label=f"Done in {seconds:.0f} s · {stage}", state="complete", expanded=False)

    response = _response(result, question, uuid.uuid4().hex)
    response["trace_run_id"] = result.get("trace_run_id")
    state["messages"].append(
        {"role": "assistant", "result": response, "steps": steps, "seconds": seconds}
    )
    state["stats"]["questions"] += 1
    state["stats"]["cost"] += float(response.get("usage", {}).get("estimated_cost_usd") or 0)


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("<div class='ps-side-brand'><span>📄</span>Paper Scout</div>", unsafe_allow_html=True)
    st.caption("Research assistant over 7,701 arXiv AI/ML papers (2025–2026).")
    if st.button("New chat", icon=":material/edit_square:", width="stretch", type="primary"):
        state["messages"] = []
        st.rerun()
    st.divider()
    st.markdown(
        stat_grid(
            [
                ("Questions", str(state["stats"]["questions"])),
                ("Session cost", f"${state['stats']['cost']:.4f}"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Hybrid search: SQLite FTS5 keywords + Qdrant vectors, fused with RRF\n"
        "2. GPT-4o-mini decides if the abstracts are enough\n"
        "3. If not, it reads the most relevant PDF excerpts\n"
        "4. A verifier checks the answer against the evidence"
    )
    st.caption("Each question is answered independently; the assistant has no chat memory.")
    st.divider()
    st.markdown(f":material/code: [Source on GitHub]({REPO_URL})")
    st.caption("FastAPI · Qdrant · fastembed · GPT-4o-mini · LangSmith")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not configured. Add it to the app's secrets.")
    st.stop()

# ---------- conversation ----------
if not state["messages"]:
    st.markdown(
        """
        <div class="ps-hero">
          <div class="ps-logo">📄</div>
          <div class="ps-title">Paper Scout</div>
          <div class="ps-sub">Ask anything about recent AI/ML research — get a cited answer in seconds.</div>
          <div class="ps-chips">
            <span class="ps-chip">🔎 Hybrid keyword + vector search</span>
            <span class="ps-chip">📑 Reads full PDFs when needed</span>
            <span class="ps-chip">✅ Every claim cited</span>
          </div>
          <div class="ps-try">Try one of these</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.pills(
        "Example questions",
        EXAMPLES,
        key="example_pick",
        on_change=_queue_example,
        label_visibility="collapsed",
        width="stretch",
    )

for message in state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            render_assistant(message)

typed = st.chat_input(
    "Ask about methods, benchmarks or trends…", max_chars=settings.max_question_length
)
question = (typed or state.pop("pending", None) or "").strip()
if question:
    state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)
    answer(question)
    st.rerun()
