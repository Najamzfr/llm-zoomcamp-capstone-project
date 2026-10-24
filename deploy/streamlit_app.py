"""Streamlit Community Cloud entry point: runs the research agent in-process (no FastAPI server).

Deploy settings: main file `deploy/streamlit_app.py`, Python 3.12, secrets as in deploy/README.md.
Locally: `uv run --with "streamlit>=1.50" streamlit run deploy/streamlit_app.py`.
Theme tokens and fonts: `.streamlit/config.toml`. Formatting helpers: research_assistant/presentation.py.
"""

import html
import os
import sqlite3
import sys
import time
import uuid
from contextlib import closing
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
from research_assistant.presentation import (  # noqa: E402
    abstract_excerpt,
    bibtex,
    clean_excerpt,
    format_date,
    link_citations,
    period_start,
    short_title,
    shorten,
    split_short_answer,
)
from research_assistant.tracing import send_feedback  # noqa: E402

SAMPLES = [
    ("How is retrieval-augmented generation evaluated?", "Survey · metrics and benchmarks"),
    ("Compare approaches to efficient fine-tuning", "Comparison table"),
    ("Recent methods for reducing LLM hallucinations", "Survey · 2026 papers"),
    ("Which benchmarks are used to evaluate LLM agents?", "List with sources"),
]
PERIODS = ["All dates", "Since Jan 2026", "Last month"]

st.set_page_config(page_title="Paper Scout", page_icon=":material/article:", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root { --paper:#f7f6f3; --surface:#ffffff; --sunk:#efede8; --line:#e2dfd8; --ink:#1f1d1a;
        --ink2:#5b574f; --accent:#1d4ed8; --accent-soft:#e3eafc; --verified:#1a7f4b; }
.block-container { max-width: 1180px; padding-top: 2.6rem; padding-bottom: 2rem; }
footer, [data-testid="stStatusWidget"], [class*="viewerBadge"] { visibility: hidden; }
h1, h2, h3 { font-family: 'Newsreader', Georgia, serif !important; font-weight: 500 !important; letter-spacing: -0.01em; }

/* Empty state */
.ps-lead { font-size: 1.15rem; line-height: 1.6; color: var(--ink2); max-width: 64ch; margin: -0.4rem 0 1.4rem; }
[class*="st-key-sample"] button { background: var(--surface); border: 1px solid var(--line); min-height: 92px;
  justify-content: flex-start; text-align: left; padding: 14px 16px; align-items: flex-start; }
[class*="st-key-sample"] button:hover { border-color: var(--ink2); background: var(--surface); }
[class*="st-key-sample"] button p, [class*="st-key-sample"] button div { font-size: 1rem; line-height: 1.45; text-align: left;
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[class*="st-key-sample"] button > div, [class*="st-key-sample"] [data-testid="stMarkdownContainer"] { width: 100%; text-align: left; justify-content: flex-start; }
[data-testid="stChatInput"] > div { background: var(--surface) !important; border: 1px solid var(--line); }
[data-testid="stChatInput"] textarea { color: var(--ink); }

/* Question header */
.q-title { font-family: 'Newsreader', Georgia, serif; font-size: 30px; line-height: 1.2; font-weight: 500;
           color: var(--ink); margin: 0.6rem 0 0.35rem; scroll-margin-top: 80px; }
.q-meta { color: var(--ink2); font-size: 14px; display: flex; flex-wrap: wrap; gap: 6px 14px; margin-bottom: 1rem; }
.q-meta .ok { color: var(--verified); font-weight: 500; }
.ps-sep { border: none; border-top: 1px solid var(--line); margin: 2.2rem 0 1rem; }

/* Live progress */
.steps { border: 1px solid var(--line); border-radius: 8px; background: var(--surface); max-width: 720px; }
.step { display: flex; align-items: center; gap: 14px; padding: 12px 16px; border-top: 1px solid var(--line); font-size: 15px; }
.step:first-child { border-top: none; }
.step .dot { width: 14px; height: 14px; border-radius: 50%; flex: none; border: 2px solid var(--line); }
.step.done .dot { background: var(--verified); border-color: var(--verified); }
.step.active .dot { border-color: var(--accent); border-right-color: transparent; animation: ps-spin 0.9s linear infinite; }
.step.pending { color: var(--ink2); }
.step .count { margin-left: auto; color: var(--ink2); font-size: 13.5px; text-align: right; }
.skeleton { height: 14px; border-radius: 7px; background: var(--sunk); margin: 14px 0; max-width: 720px; }
@keyframes ps-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .step.active .dot { animation: none; } }

/* Answer */
[class*="st-key-tldr"] { background: var(--sunk); border-radius: 8px; padding: 14px 18px; }
[class*="st-key-tldr"] p { font-size: 16px; line-height: 1.6; margin: 0; }
[class*="st-key-prose"] p, [class*="st-key-prose"] li { font-family: 'Newsreader', Georgia, serif; font-size: 18px; line-height: 1.6; max-width: 66ch; }
[class*="st-key-prose"] h4 { font-family: 'IBM Plex Sans', sans-serif !important; font-size: 16px; font-weight: 600 !important; margin-top: 1.2rem; }
[class*="st-key-prose"] table { font-size: 14.5px; border-collapse: collapse; width: 100%; }
[class*="st-key-prose"] th { text-align: left; font-weight: 600; color: var(--ink2); }
[class*="st-key-prose"] th, [class*="st-key-prose"] td { border: none; border-bottom: 1px solid var(--line); padding: 9px 12px 9px 0; vertical-align: top; }
a.cite { font: 600 12px 'IBM Plex Sans', sans-serif; color: var(--accent) !important; background: var(--accent-soft);
         border-radius: 4px; padding: 1px 5px; margin: 0 1px; text-decoration: none !important; vertical-align: 2px; cursor: pointer; }
a.cite:hover { background: var(--accent); color: #fff !important; }

/* Sources */
.src-label { color: var(--ink2); font-size: 14px; margin: 0.2rem 0 0.5rem; }
.src { border: 1px solid var(--line); border-radius: 8px; padding: 11px 13px; margin-bottom: 8px; background: var(--surface);
       scroll-margin-top: 90px; transition: background 0.2s, border-color 0.2s; }
.src.is-target { border-color: var(--accent); background: var(--accent-soft); }
.src-top { font-size: 13px; color: var(--ink2); }
.src-top b { color: var(--accent); margin-right: 6px; }
.src-title { font-weight: 600; font-size: 14.5px; line-height: 1.35; margin: 4px 0; color: var(--ink); }
.src-quote { font-family: 'Newsreader', Georgia, serif; font-style: italic; font-size: 14.5px; color: var(--ink2); margin-bottom: 4px; }
.src a { font-size: 13.5px; color: var(--accent) !important; }
.src-row { display: none; gap: 8px; overflow-x: auto; padding: 10px 0 4px; }
.src-chip { flex: none; border: 1px solid var(--line); border-radius: 8px; padding: 7px 11px; background: var(--surface);
            font-size: 14px; color: var(--ink) !important; text-decoration: none !important; white-space: nowrap; }
.src-chip b { color: var(--accent); margin-right: 4px; }
.src-chip.is-target { border-color: var(--accent); background: var(--accent-soft); }

/* Actions */
.ps-copy { display: inline-flex; align-items: center; height: 40px; padding: 0 14px; border: 1px solid var(--line);
           border-radius: 8px; background: var(--surface); font-size: 14px; cursor: pointer; user-select: none; }
.ps-copy:hover { border-color: var(--ink2); }
.ps-raw { display: none; }

/* Sidebar */
.ps-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem; }
.ps-mark { width: 30px; height: 30px; border-radius: 6px; background: var(--ink); color: #fff; display: grid; place-items: center;
           font: 500 19px 'Newsreader', Georgia, serif; }
.ps-name { font: 500 22px 'Newsreader', Georgia, serif; color: var(--ink); }
.ps-side-label { color: var(--ink2); font-size: 14px; margin: 0.8rem 0 0.2rem; }
[class*="st-key-hist"] button { justify-content: flex-start; text-align: left; padding: 6px 8px; }
[class*="st-key-hist"] button > div, [class*="st-key-hist"] [data-testid="stMarkdownContainer"] { width: 100%; text-align: left; justify-content: flex-start; }
[class*="st-key-hist"] button p { font-size: 14.5px; line-height: 1.4; }
.ps-scope { font-size: 14px; line-height: 1.55; color: var(--ink2); }
.ps-scope b { color: var(--ink); }

/* Filters above the composer */
[class*="st-key-filters"] { gap: 8px; flex-wrap: wrap; }
[class*="st-key-filters"] [data-testid="stButtonGroup"] > div { flex-wrap: wrap; }
[class*="st-key-filters"] p { font-size: 14px; }

/* Medium widths: the 3:1 rail would be too narrow, so stack it under the answer as a card grid. */
@media (max-width: 1050px) {
  [data-testid="stHorizontalBlock"]:has([class*="st-key-rail"]) { flex-direction: column; }
  [data-testid="stHorizontalBlock"]:has([class*="st-key-rail"]) > [data-testid="stColumn"] { width: 100% !important; flex: 1 1 auto !important; }
  [class*="st-key-rail"] .src-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; }
}
@media (max-width: 640px) {
  [class*="st-key-rail"] { display: none; }
  .src-row { display: flex; }
  .q-title { font-size: 25px; }
  [class*="st-key-prose"] p, [class*="st-key-prose"] li { font-size: 17px; }
  .block-container { padding-top: 3.4rem; }
}
</style>
"""
st.html(CSS)

# Delegated click handling on the page: citation chips highlight and scroll to their
# source card (the visible one: rail on desktop, chip row on phones); "Copy answer" copies text.
CLICK_JS = """
<script>
(() => {  // own scope: Streamlit re-runs this snippet on every rerun
const doc = document;
if (!doc.__paperScout) {
  doc.__paperScout = true;
  doc.addEventListener('click', (event) => {
    const cite = event.target.closest('a.cite');
    if (cite) {
      event.preventDefault();
      const key = cite.dataset.src;
      doc.querySelectorAll('.is-target').forEach((el) => el.classList.remove('is-target'));
      const targets = [...doc.querySelectorAll('[data-src="' + key + '"]')]
        .filter((el) => !el.classList.contains('cite') && el.offsetParent !== null);
      targets.forEach((el) => el.classList.add('is-target'));
      if (targets[0]) targets[0].scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
      return;
    }
    const copy = event.target.closest('[data-copy]');
    if (copy) {
      const source = doc.getElementById(copy.dataset.copy);
      const text = source ? source.textContent : '';
      const done = () => { const old = copy.textContent; copy.textContent = 'Copied'; setTimeout(() => { copy.textContent = old; }, 1500); };
      const fallback = () => {
        const area = doc.createElement('textarea'); area.value = text; doc.body.appendChild(area);
        area.select(); try { doc.execCommand('copy'); done(); } catch (e) {} area.remove();
      };
      const clip = navigator.clipboard;
      if (clip && clip.writeText) { clip.writeText(text).then(done, fallback); } else { fallback(); }
    }
  }, true);
}
const scrollTarget = "__SCROLL__";
// Repeat briefly: the chat-input layout re-pins the page to the bottom while a rerun settles.
if (scrollTarget) [300, 900, 1600].forEach((delay) => setTimeout(() => {
  const el = doc.getElementById(scrollTarget); if (el) el.scrollIntoView({ block: 'start' });
}, delay));
})();
</script>
"""


@st.cache_resource(show_spinner="Preparing the paper index…")
def ensure_index():
    if not settings.database_path.exists():
        from scripts.ingest import ingest

        ingest()  # ~1 s: builds the SQLite FTS5 catalogue from data/source/data.csv
    with closing(sqlite3.connect(settings.database_path)) as connection:
        total = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
        first, last = connection.execute("SELECT min(submitted_date), max(submitted_date) FROM documents").fetchone()
        categories = [row[0] for row in connection.execute(
            "SELECT primary_category FROM documents GROUP BY 1 ORDER BY count(*) DESC LIMIT 6")]
    return {"total": total, "first": first, "last": last, "categories": categories}


@st.cache_resource(show_spinner=False)
def get_agent():
    ensure_index()
    from research_assistant.agent import AdaptiveResearchAgent

    return AdaptiveResearchAgent(trace_path=None)


@st.cache_resource
def get_limiter():
    # Shared by all visitors of this app instance; protects the OpenAI budget.
    return AskLimiter(settings.ask_rate_per_minute, settings.ask_daily_limit)


state = st.session_state
state.setdefault("messages", [])
state.setdefault("history", [])
state.setdefault("rated", {})
corpus = ensure_index()
TOTAL = f"{corpus['total']:,}"


def month_label(value):
    return format_date(value).split(" ", 1)[-1] if value else ""


def ask(question):
    """Queue a question; it is answered on the next run so the page re-renders without the welcome block."""
    question = (question or "").strip()
    if not question:
        return
    filters = {}
    if state.get("cat_filter"):
        filters["categories"] = list(state["cat_filter"])
    since = {"Since Jan 2026": "2026-01-01", "Last month": period_start("Last month", corpus["last"])}.get(
        state.get("period_filter") or "All dates")
    if since:
        filters["since"] = since
    state["messages"].append({"id": uuid.uuid4().hex[:8], "question": question, "status": "queued", "filters": filters})
    if question in state["history"]:
        state["history"].remove(question)
    state["history"].append(question)


# ---------------------------------------------------------------- progress view
class Progress:
    """Named research steps with counts, re-rendered into one placeholder as events arrive."""

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.steps = [{"key": "search", "label": f"Searching {TOTAL} papers by keyword and meaning", "count": "", "state": "active"}]
        self.read_full = 0
        self.verified = None
        self.render()

    def _step(self, key):
        return next((step for step in self.steps if step["key"] == key), None)

    def start(self, key, label, count=""):
        for step in self.steps:
            if step["state"] == "active":
                step["state"] = "done"
        existing = self._step(key)
        if existing:
            existing.update(label=label, count=count, state="active")
        else:
            self.steps.append({"key": key, "label": label, "count": count, "state": "active"})

    def finish(self, key, label, count=""):
        step = self._step(key) or {"key": key}
        if step not in self.steps:
            self.steps.append(step)
        step.update(label=label, count=count, state="done")

    def on_event(self, event, data):
        if event == "searched":
            self.finish("search", f"Searched {TOTAL} papers by keyword and meaning", f"{data.get('candidates', 0)} matches")
            self.start("rank", "Ranking the most relevant papers")
        elif event == "routed":
            self.finish("rank", "Ranked the most relevant papers", "abstracts suffice" if not data.get("needs_full_text") else "")
            if data.get("needs_full_text"):
                self.start("read", "Reading the full text")
            else:
                self.start("write", "Writing a cited answer")
        elif event == "fetching_full_text":
            titles = ", ".join(short_title(title, 2) for title in data.get("titles", [])[:3])
            self.read_full = data.get("papers", 0)
            self.start("read", f"Reading full text of {self.read_full} papers", html.escape(titles))
        elif event == "full_text_ready":
            self.finish("read", f"Read full text of {self.read_full} papers", f"{data.get('chunks', 0)} excerpts")
            self.start("write", "Writing a cited answer")
        elif event == "full_text_failed":
            self.read_full = 0
            self.finish("read", "Full text unavailable", "using abstracts")
            self.start("write", "Writing a cited answer")
        elif event == "drafted":
            self.finish("write", "Wrote a cited answer")
            self.start("check", "Checking each claim against its source")
        elif event == "checked":
            status = data.get("status")
            self.verified = status == "pass"
            label = {"pass": "verified", "needs_full_text": "needs full text"}.get(status, "partly supported")
            self.finish("check", "Checked each claim against its source", label)
        self.render()

    def render(self):
        rows = list(self.steps)
        if not self._step("check"):
            rows.append({"label": "Checking each claim against its source", "count": "–", "state": "pending"})
        body = "".join(
            f"<div class='step {row['state']}'><span class='dot'></span><span>{html.escape(row['label'])}</span>"
            f"<span class='count'>{row['count']}</span></div>"
            for row in rows
        )
        skeleton = "".join(f"<div class='skeleton' style='width:{w}%'></div>" for w in (92, 84, 61))
        self.placeholder.html(f"<div class='steps'>{body}</div>{skeleton}")


def run_question(message):
    started = time.perf_counter()
    if not get_limiter().allow():
        message.update(status="limited")
        return
    progress = Progress(st.empty())
    try:
        result = get_agent().run(message["question"], on_progress=progress.on_event, filters=message["filters"] or None)
    except Exception as error:  # friendly message instead of a stack trace
        message.update(status="error", error=type(error).__name__)
        return
    response = _response(result, message["question"], uuid.uuid4().hex)
    raw = {paper["paper_id"]: paper for paper in result.get("papers", [])}
    excerpts = result.get("excerpts") or {}
    for paper in response["retrieved_papers"]:
        source = raw.get(paper["paper_id"], {})
        paper["authors"] = source.get("authors", "")
        paper["excerpt"] = clean_excerpt(excerpts.get(paper["paper_id"]), 150) or abstract_excerpt(source.get("text"), 150)
    message.update(
        status="done",
        result=response,
        trace_run_id=result.get("trace_run_id"),
        seconds=time.perf_counter() - started,
        read_full=progress.read_full if result.get("answer_stage") == "full_text" else 0,
        verified=progress.verified,
        steps=[f"{s['label']}{(' · ' + s['count']) if s['count'] else ''}" for s in progress.steps],
    )


# ---------------------------------------------------------------- answer view
def source_html(prefix, papers):
    rail, chips = [], []
    for number, paper in enumerate(papers, start=1):
        key = f"{prefix}-{number}"
        title = html.escape(paper.get("title") or "Untitled paper")
        url = paper.get("url") or ""
        href = html.escape(url, quote=True) if url.startswith(("https://", "http://")) else ""
        meta = " · ".join(html.escape(v) for v in (paper.get("category") or "", format_date(paper.get("published"))) if v)
        quote = f"<div class='src-quote'>“{html.escape(paper['excerpt'])}”</div>" if paper.get("excerpt") else ""
        link = f"<a href='{href}' target='_blank' rel='noopener noreferrer'>Open on arXiv</a>" if href else ""
        rail.append(f"<div class='src' id='{key}' data-src='{key}'><div class='src-top'><b>{number}</b>{meta}</div>"
                    f"<div class='src-title'>{title}</div>{quote}{link}</div>")
        chips.append(f"<a class='src-chip' data-src='{key}' href='{href or '#'}' target='_blank' rel='noopener noreferrer'>"
                     f"<b>{number}</b>{html.escape(short_title(paper.get('title')))}</a>")
    return "".join(rail), "<div class='src-row'>" + "".join(chips) + "</div>"


def render_message(message):
    prefix = f"m{message['id']}"
    st.html(f"<div class='q-title' id='q-{message['id']}'>{html.escape(message['question'])}</div>")
    status = message["status"]
    if status == "queued":
        # Phase 1: draw the researching view and finish the run, which clears the previous page
        # (Streamlit only removes leftover elements when a run ends). Phase 2 does the work.
        Progress(st.empty())
        state["start_research"] = message["id"]
        return
    if status == "running":
        run_question(message)
        st.rerun()
    if status == "limited":
        st.warning("This demo has reached its question limit. Please wait a minute and try again.")
        return
    if status == "error":
        st.error(f"The research assistant could not answer ({message.get('error')}). Please try again.")
        return

    result = message["result"]
    papers = result.get("retrieved_papers", [])
    meta = [f"{len(papers)} papers"]
    if message.get("read_full"):
        meta.append(f"{message['read_full']} read in full")
    verified = ("<span class='ok'>All claims verified</span>" if message.get("verified")
                else "<span>Some claims not fully verified</span>")
    filters = message.get("filters") or {}
    scope = ", ".join(filters.get("categories", [])) + (f" since {format_date(filters['since'])}" if filters.get("since") else "")
    st.html(f"<div class='q-meta'><span>{' · '.join(meta)}</span>{verified}<span>{message.get('seconds', 0):.0f} s</span>"
            + (f"<span>Filtered: {html.escape(scope.strip())}</span>" if scope.strip() else "") + "</div>")

    short, rest = split_short_answer(result.get("answer", ""))
    rail_html, chip_row = source_html(prefix, papers)
    left, right = st.columns([3, 1], gap="large")
    with left:
        if short:
            with st.container(key=f"tldr-{prefix}"):
                st.markdown(f"**Short answer.** {link_citations(short, prefix, len(papers))}", unsafe_allow_html=True)
        st.html(chip_row)
        with st.container(key=f"prose-{prefix}"):
            st.markdown(link_citations(rest, prefix, len(papers)), unsafe_allow_html=True)

        raw_answer = html.escape(result.get("answer", ""))
        rated = state["rated"].get(message["id"])
        with st.container(horizontal=True, key=f"actions-{prefix}"):
            st.html(f"<span class='ps-copy' role='button' tabindex='0' data-copy='raw-{prefix}'>Copy answer</span>"
                    f"<pre class='ps-raw' id='raw-{prefix}'>{raw_answer}</pre>", width="content")
            st.download_button("Export citations (BibTeX)", bibtex(papers), file_name="paper-scout-sources.bib",
                               mime="application/x-bibtex", key=f"bib-{prefix}")
            if rated:
                st.caption("Thanks for the feedback.")
            else:
                for label, value in (("Helpful", "up"), ("Not helpful", "down")):
                    if st.button(label, key=f"{value}-{prefix}"):
                        state["rated"][message["id"]] = value
                        send_feedback(message.get("trace_run_id"), value)
                        st.rerun()
        with st.expander("Details"):
            usage = result.get("usage", {})
            st.markdown(
                f"Tokens: {usage.get('input_tokens', 0):,} in · {usage.get('output_tokens', 0):,} out · "
                f"Estimated cost: ${usage.get('estimated_cost_usd', 0) or 0:.4f}"
            )
            for step in message.get("steps", []):
                st.markdown(f"- {step}")
            st.caption("Each question is answered on its own; there is no chat memory.")
    with right:
        with st.container(key=f"rail-{prefix}"):
            st.html(f"<div class='src-label'>Sources</div><div class='src-grid'>{rail_html}</div>")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.html("<div class='ps-brand'><div class='ps-mark'>P</div><div class='ps-name'>Paper Scout</div></div>")
    if st.button("New question", type="primary" if not state["messages"] else "secondary", width="stretch"):
        state["messages"] = []
        st.rerun()
    if state["history"]:
        st.html("<div class='ps-side-label'>Recent</div>")
        open_questions = {m["question"]: m["id"] for m in state["messages"]}
        for index, question in enumerate(reversed(state["history"][-8:])):
            if st.button(shorten(question, 60), key=f"hist-{index}", type="tertiary", width="stretch"):
                if question in open_questions:
                    state["scroll_to"] = f"q-{open_questions[question]}"
                else:
                    ask(question)
                st.rerun()
    st.divider()
    st.html(
        f"<div class='ps-scope'><b>{TOTAL}</b> arXiv papers<br>{html.escape(', '.join(corpus['categories'][:3]))} and more<br>"
        f"{month_label(corpus['first'])} – {month_label(corpus['last'])}<br>"
        "Each question is answered on its own; there is no chat memory.</div>"
    )

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not configured. Add it to the app's secrets.")
    st.stop()

# ---------------------------------------------------------------- main
if not state["messages"]:
    st.markdown(f"# Ask {TOTAL} AI/ML papers a question.")
    st.html(
        f"<p class='ps-lead'>Answers are built from arXiv papers published {month_label(corpus['first'])} – "
        f"{month_label(corpus['last'])}. Every claim links to the paper it came from.</p>"
    )
    columns = st.columns(2)
    for index, (question, hint) in enumerate(SAMPLES):
        with columns[index % 2]:
            if st.button(f"{question}  \n:gray[{hint}]", key=f"sample-{index}", width="stretch"):
                ask(question)
                st.rerun()

for position, message in enumerate(state["messages"]):
    if position:
        st.html("<hr class='ps-sep'>")
    render_message(message)

with st.container(key="filters", horizontal=True):
    st.pills("Category", corpus["categories"], selection_mode="multi", key="cat_filter",
             label_visibility="collapsed", help="Only search these arXiv categories (none selected = all)")
    st.segmented_control("Period", PERIODS, default=PERIODS[0], key="period_filter", label_visibility="collapsed",
                         help=f"Relative to the newest paper ({format_date(corpus['last'])})")

typed = st.chat_input(
    "Ask a follow-up as a new question…" if state["messages"] else "Ask about a method, benchmark or trend…",
    max_chars=settings.max_question_length,
)
if typed:
    ask(typed)
    st.rerun()

st.html(CLICK_JS.replace("__SCROLL__", state.pop("scroll_to", "") or ""), unsafe_allow_javascript=True)

if state.get("start_research"):
    message_id = state.pop("start_research")
    for message in state["messages"]:
        if message["id"] == message_id and message["status"] == "queued":
            message["status"] = "running"
    st.rerun()
