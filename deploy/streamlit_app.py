"""Streamlit Community Cloud entry point: runs the research agent in-process (no FastAPI server).

Deploy settings: main file `deploy/streamlit_app.py`, Python 3.12, secrets as in deploy/README.md.
Locally: `uv run streamlit run deploy/streamlit_app.py`.
Theme tokens and fonts: `.streamlit/config.toml`. Formatting helpers: research_assistant/presentation.py.
Extra features (follow-ups, exports, study, review, sharing…): app/features/. Plan: PLAN.md.

UI rules: never show corpus counts, index dates, category codes, model names, cost or research
steps; sources open only in the side panel.
"""

import copy
import html
import json
import os
import sqlite3
import sys
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

from api import AskLimiter  # noqa: E402  (shared rate limiter)
from app.features import export, followups, history  # noqa: E402
from app.features.llm_extras import LLMExtrasError, Rewrite, llm_json  # noqa: E402
from app.features.records import build_record, byline, new_id, sources_context, variant_record  # noqa: E402
from app.storage import local  # noqa: E402
from research_assistant.config import settings  # noqa: E402
from research_assistant.presentation import link_citations, period_start, shorten, split_short_answer  # noqa: E402
from research_assistant.tracing import send_feedback  # noqa: E402

FEATURED_PATH = ROOT / "data" / "featured.json"
SAMPLES = [
    ("How is retrieval-augmented generation evaluated?", "Metrics, benchmarks and judges"),
    ("Compare approaches to efficient fine-tuning", "Comparison table"),
    ("What methods reduce hallucinations in LLMs?", "Survey of recent work"),
    ("Which benchmarks are used to evaluate LLM agents?", "List with sources"),
]
# Research areas are shown by name, never by their catalogue codes.
AREA_NAMES = {
    "cs.CV": "Vision", "cs.LG": "Machine learning", "cs.CL": "Language", "cs.RO": "Robotics",
    "cs.AI": "AI", "cs.NE": "Neural and evolutionary", "cs.IR": "Retrieval", "stat.ML": "Statistics",
    "cs.CR": "Security", "cs.HC": "Human–computer interaction", "eess.IV": "Imaging", "cs.SE": "Software",
}
PERIODS = ["Any time", "Last 6 months", "Last month"]
VARIANTS = {
    "simpler": (
        "Simpler version",
        "Explain the answer for a smart beginner: short sentences, one everyday analogy, and a quick "
        "plain-language explanation of any technical term. Keep each [n] citation next to the claim it "
        "supports. Start with \"Short answer:\" and two sentences, then at most three short sections with "
        "#### headings. Return {\"answer_md\": \"...\"}.",
    ),
    "deeper": (
        "Deeper version",
        "Go deeper than the answer: explain the mechanisms, the trade-offs, the key equations if the "
        "sources contain them (LaTeX between $ signs), and the open questions. Keep each [n] citation next "
        "to the claim it supports. Start with \"Short answer:\" and two or three sentences, then sections "
        "with #### headings. Return {\"answer_md\": \"...\"}.",
    ),
}

st.set_page_config(page_title="Paper Scout", page_icon=":material/article:", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root { --paper:#f7f6f3; --surface:#ffffff; --sunk:#efede8; --line:#e2dfd8; --ink:#1f1d1a;
        --ink2:#5b574f; --accent:#1d4ed8; --accent-soft:#e3eafc; --verified:#1a7f4b; --bad:#b42318; }
.block-container { max-width: 940px; padding-top: 2.6rem; padding-bottom: 2rem; }
footer, [data-testid="stStatusWidget"], [class*="viewerBadge"] { visibility: hidden; }
h1, h2, h3 { font-family: 'Newsreader', Georgia, serif !important; font-weight: 500 !important; letter-spacing: -0.01em; }
button:focus-visible, a:focus-visible, [role="button"]:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

/* Empty state */
.ps-lead { font-size: 1.15rem; line-height: 1.6; color: var(--ink2); max-width: 64ch; margin: -0.4rem 0 1.4rem; }
[class*="st-key-gallery_"] button { background: var(--surface); border: 1px solid var(--line); min-height: 100px;
  justify-content: flex-start; text-align: left; padding: 14px 16px; align-items: flex-start; }
[class*="st-key-gallery_"] button:hover { border-color: var(--ink2); background: var(--surface); }
[class*="st-key-gallery_"] button p, [class*="st-key-gallery_"] button div { font-size: 1rem; line-height: 1.45; text-align: left;
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[class*="st-key-gallery_"] button > div, [class*="st-key-gallery_"] [data-testid="stMarkdownContainer"] { width: 100%; text-align: left; justify-content: flex-start; }
[data-testid="stChatInput"] > div { background: var(--surface) !important; border: 1px solid var(--line); }
[data-testid="stChatInput"] textarea { color: var(--ink); }

/* Question header */
.q-title { font-family: 'Newsreader', Georgia, serif; font-size: 30px; line-height: 1.2; font-weight: 500;
           color: var(--ink); margin: 0.6rem 0 0.35rem; scroll-margin-top: 80px; }
.q-kicker { color: var(--ink2); font-size: 14px; margin-top: 0.6rem; }
.q-kicker a { color: var(--accent) !important; }
.q-meta { color: var(--ink2); font-size: 14px; display: flex; flex-wrap: wrap; gap: 6px 14px; margin-bottom: 1rem; min-height: 4px; }
.q-meta .ok { color: var(--verified); font-weight: 500; }
.ps-sep { border: none; border-top: 1px solid var(--line); margin: 2.2rem 0 1rem; }

/* Busy state */
.ps-busy { display: flex; align-items: center; gap: 12px; color: var(--ink2); font-size: 15px; margin: 6px 0 4px; }
.ps-busy .dot { width: 14px; height: 14px; border-radius: 50%; border: 2px solid var(--accent); border-right-color: transparent;
                animation: ps-spin 0.9s linear infinite; flex: none; }
.skeleton { height: 14px; border-radius: 7px; background: var(--sunk); margin: 14px 0; max-width: 720px; }
@keyframes ps-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .ps-busy .dot { animation: none; } }

/* Answer */
[class*="st-key-tldr_"] { background: var(--sunk); border-radius: 8px; padding: 14px 18px; }
[class*="st-key-tldr_"] p { font-size: 16px; line-height: 1.6; margin: 0; }
[class*="st-key-prose_"] p, [class*="st-key-prose_"] li { font-family: 'Newsreader', Georgia, serif; font-size: 18px; line-height: 1.6; max-width: 66ch; }
[class*="st-key-prose_"] h4 { font-family: 'IBM Plex Sans', sans-serif !important; font-size: 16px; font-weight: 600 !important; margin-top: 1.2rem; }
[class*="st-key-prose_"] table { font-size: 14.5px; border-collapse: collapse; width: 100%; }
[class*="st-key-prose_"] th { text-align: left; font-weight: 600; color: var(--ink2); }
[class*="st-key-prose_"] th, [class*="st-key-prose_"] td { border: none; border-bottom: 1px solid var(--line); padding: 9px 12px 9px 0; vertical-align: top; }
a.cite { font: 600 12px 'IBM Plex Sans', sans-serif; color: var(--accent) !important; background: #e3eafc;
         border-radius: 4px; padding: 1px 5px; margin: 0 1px; text-decoration: none !important; vertical-align: 2px; cursor: pointer; }
a.cite:hover { background: var(--accent); color: #fff !important; }

/* Follow-ups */
[class*="st-key-followups_"] [data-testid="stButtonGroup"] > div { flex-wrap: wrap; }
[class*="st-key-followups_"] button { height: auto !important; min-height: 38px; padding: 7px 14px !important; }
[class*="st-key-followups_"] button p, [class*="st-key-followups_"] button div { white-space: normal !important; text-align: left; font-size: 14.5px;
  line-height: 1.35; overflow: visible !important; text-overflow: clip !important; }
[class*="st-key-followups_"] label p { color: var(--ink2); font-size: 14px; }

/* Action row */
[class*="st-key-actions_"] { gap: 6px; flex-wrap: wrap; align-items: center; border-top: 1px solid var(--line); padding-top: 10px; margin-top: 4px; }
[class*="st-key-actions_"] button { min-height: 40px; }
[class*="st-key-actions_"] [data-testid="stFeedback"] { margin-left: auto; }
.ps-btn { display: inline-flex; align-items: center; height: 40px; padding: 0 13px; border: 1px solid var(--line); border-radius: 8px;
          background: var(--surface); font: 400 14px 'IBM Plex Sans', sans-serif; color: var(--ink); cursor: pointer; white-space: nowrap; }
.ps-btn:hover { border-color: var(--ink2); }
.ps-raw { display: none; }
.ps-note { color: var(--ink2); font-size: 13.5px; }
.only-narrow { display: none; }
[class*="st-key-narrow_"] { display: none; }

/* Sources side panel */
.ps-drawer { position: fixed; top: 0; right: 0; height: 100vh; width: min(380px, 100vw); background: var(--surface);
             border-left: 1px solid var(--line); box-shadow: -14px 0 32px rgba(31,29,26,0.10); z-index: 1000100;
             transform: translateX(105%); visibility: hidden; transition: transform 0.22s ease, visibility 0s linear 0.22s;
             overflow-y: auto; padding: 16px 16px 40px; box-sizing: border-box; }
.ps-drawer-src { display: none !important; }
.ps-drawer.open { transform: none; visibility: visible; transition: transform 0.22s ease; }
@media (prefers-reduced-motion: reduce) { .ps-drawer, .ps-drawer.open { transition: none; } }
.ps-drawer-h { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; position: sticky; top: -16px;
               background: var(--surface); padding: 6px 0; }
.ps-drawer-h span { font-weight: 600; font-size: 15px; }
.ps-drawer-x { border: 1px solid var(--line); background: var(--surface); border-radius: 8px; height: 36px; padding: 0 12px; cursor: pointer; font-size: 14px; color: var(--ink); }
.src { border: 1px solid var(--line); border-radius: 8px; padding: 11px 13px; margin-bottom: 8px; background: var(--surface);
       scroll-margin-top: 70px; transition: background 0.2s, border-color 0.2s; }
.src.is-target { border-color: var(--accent); background: #e3eafc; }
.src-top { font-size: 13px; color: var(--ink2); }
.src-top b { color: var(--accent); margin-right: 6px; }
.src-tag { font-size: 12px; background: var(--sunk); border-radius: 4px; padding: 0 6px; margin-right: 6px; color: var(--ink2); }
.src-title { font-weight: 600; font-size: 14.5px; line-height: 1.35; margin: 4px 0; color: var(--ink); }
.src-quote { font-family: 'Newsreader', Georgia, serif; font-style: italic; font-size: 14.5px; color: var(--ink2); margin-bottom: 4px; }
.src a { font-size: 13.5px; color: var(--accent) !important; }

/* Sidebar */
.ps-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem; }
.ps-mark { width: 30px; height: 30px; border-radius: 6px; background: var(--ink); color: #fff; display: grid; place-items: center;
           font: 500 19px 'Newsreader', Georgia, serif; }
.ps-name { font: 500 22px 'Newsreader', Georgia, serif; color: var(--ink); }
.ps-side-label { color: var(--ink2); font-size: 14px; margin: 0.8rem 0 0.2rem; }
[class*="st-key-hist_"] button { justify-content: flex-start; text-align: left; padding: 6px 8px; }
[class*="st-key-hist_"] button > div, [class*="st-key-hist_"] [data-testid="stMarkdownContainer"] { width: 100%; text-align: left; justify-content: flex-start; }
[class*="st-key-hist_"] button p { font-size: 14.5px; line-height: 1.4; }
.ps-scope { font-size: 14px; line-height: 1.55; color: var(--ink2); }

/* Filters above the composer */
[class*="st-key-filters"] { gap: 8px; flex-wrap: wrap; }
[class*="st-key-filters"] [data-testid="stButtonGroup"] > div { flex-wrap: wrap; }
[class*="st-key-filters"] p { font-size: 14px; }

@media (max-width: 640px) {
  .q-title { font-size: 25px; }
  [class*="st-key-prose_"] p, [class*="st-key-prose_"] li { font-size: 17px; }
  .block-container { padding-top: 3.4rem; }
  /* Phone: secondary actions move into "More" (their copies there end in "_m"). */
  [class*="st-key-actions_"] [class*="st-key-deeper_"]:not([class$="_m"]):not([class*="_m "]) { display: none !important; }
  [class*="st-key-narrow_"] { display: flex; }
}
</style>
"""
st.html(CSS)

# Delegated click handling: [n] chips and "Sources · n" open that answer's side panel and highlight
# the card; "Copy answer" copies the raw Markdown (with a hint when the clipboard is blocked).
CLICK_JS = """
export default function (component) {
const doc = document;
if (!doc.__paperScout) {
  doc.__paperScout = true;
  // Each answer renders its source cards into a hidden template; the panel itself lives on <body>
  // because Streamlit's containers break position: fixed.
  const host = () => {
    let el = doc.getElementById('ps-drawer-host');
    if (!el) {
      el = doc.createElement('aside');
      el.id = 'ps-drawer-host'; el.className = 'ps-drawer'; el.setAttribute('aria-label', 'Sources'); el.setAttribute('aria-hidden', 'true');
      doc.body.appendChild(el);
    }
    return el;
  };
  const closeAll = () => { const el = doc.getElementById('ps-drawer-host'); if (el) { el.classList.remove('open'); el.setAttribute('aria-hidden', 'true'); } };
  const openDrawer = (prefix, key) => {
    const template = doc.getElementById('drawer-' + prefix);
    if (!template) return;
    const drawer = host();
    if (drawer.dataset.prefix !== prefix || !drawer.classList.contains('open')) { drawer.innerHTML = template.innerHTML; drawer.dataset.prefix = prefix; }
    drawer.classList.add('open'); drawer.setAttribute('aria-hidden', 'false');
    drawer.querySelectorAll('.is-target').forEach((el) => el.classList.remove('is-target'));
    const card = key ? drawer.querySelector('[data-src="' + key + '"]') : null;
    if (card) { card.classList.add('is-target'); setTimeout(() => card.scrollIntoView({ behavior: 'smooth', block: 'center' }), 60); }
    else { const close = drawer.querySelector('[data-close-drawer]'); if (close) close.focus({ preventScroll: true }); }
  };
  doc.addEventListener('click', (event) => {
    const cite = event.target.closest('a.cite');
    if (cite) {
      event.preventDefault();
      const key = cite.dataset.src || '';
      openDrawer(key.slice(0, key.lastIndexOf('-')), key);
      return;
    }
    const opener = event.target.closest('[data-open-sources]');
    if (opener) {
      const drawer = doc.getElementById('ps-drawer-host');
      if (drawer && drawer.classList.contains('open') && drawer.dataset.prefix === opener.dataset.openSources) closeAll();
      else openDrawer(opener.dataset.openSources, null);
      return;
    }
    if (event.target.closest('[data-close-drawer]')) { closeAll(); return; }
    const copy = event.target.closest('[data-copy]');
    if (copy) {
      const source = doc.getElementById(copy.dataset.copy);
      const text = source ? source.textContent : '';
      const say = (label) => { const old = copy.dataset.label || copy.textContent; copy.dataset.label = old; copy.textContent = label; setTimeout(() => { copy.textContent = old; }, 2200); };
      const fallback = () => {
        const area = doc.createElement('textarea'); area.value = text; doc.body.appendChild(area); area.select();
        let ok = false; try { ok = doc.execCommand('copy'); } catch (e) {} area.remove();
        say(ok ? 'Copied' : 'Copy blocked: use "Show as text"');
      };
      const clip = navigator.clipboard;
      if (clip && clip.writeText) { clip.writeText(text).then(() => say('Copied'), fallback); } else { fallback(); }
    }
  }, true);
  doc.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeAll(); });
}
const scrollTarget = (component.data || {}).scroll || "";
// Repeat briefly: the chat-input layout re-pins the page to the bottom while a rerun settles.
if (scrollTarget) [300, 900, 1600].forEach((delay) => setTimeout(() => {
  const el = doc.getElementById(scrollTarget); if (el) el.scrollIntoView({ block: 'start' });
}, delay));
}
"""


@st.cache_resource
def page_script():
    """Registered once: re-registering a component on every run logs a warning."""
    return st.components.v2.component("paper_scout_page", js=CLICK_JS)


@st.cache_resource(show_spinner="Getting ready…")
def ensure_index():
    if not settings.database_path.exists():
        from scripts.ingest import ingest

        ingest()  # ~1 s: builds the SQLite FTS5 catalogue from data/source/data.csv
    with closing(sqlite3.connect(settings.database_path)) as connection:
        last = connection.execute("SELECT max(submitted_date) FROM documents").fetchone()[0]
        areas = [row[0] for row in connection.execute(
            "SELECT primary_category FROM documents GROUP BY 1 ORDER BY count(*) DESC LIMIT 8")]
    return {"last": last, "areas": [area for area in areas if area in AREA_NAMES][:6]}


@st.cache_resource(show_spinner=False)
def get_agent():
    ensure_index()
    from research_assistant.agent import AdaptiveResearchAgent

    return AdaptiveResearchAgent(trace_path=None)


@st.cache_resource
def get_limiter():
    # Shared by all visitors of this app instance; protects the model budget.
    return AskLimiter(settings.ask_rate_per_minute, settings.ask_daily_limit)


@st.cache_data
def load_featured():
    try:
        return json.loads(FEATURED_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


state = st.session_state
state.setdefault("messages", [])
state.setdefault("panel", {})  # message id -> open feature panel name
state.setdefault("errors", {})  # message id -> last feature error
corpus = ensure_index()


def ask(question):
    """Queue a question; it is answered on the next run so the page re-renders without the welcome block."""
    question = (question or "").strip()
    if not question:
        return
    filters = {}
    if state.get("area_filter"):
        filters["categories"] = list(state["area_filter"])
    since = period_start(state.get("period_filter") or PERIODS[0], corpus["last"])
    if since:
        filters["since"] = since
    state["messages"].append({"id": new_id(), "question": question, "status": "queued", "filters": filters})


def open_record(record):
    """Show a stored answer (history, gallery, collection) instantly, without a model call."""
    for message in state["messages"]:
        if message["id"] == record["id"]:
            state["scroll_to"] = f"q-{record['id']}"
            return
    record = copy.deepcopy(record)
    record.setdefault("kind", "answer")
    state["messages"].append({"id": record["id"], "question": record["question"], "status": "done", "record": record})
    state["scroll_to"] = f"q-{record['id']}"


def find_message(message_id):
    return next((message for message in state["messages"] if message["id"] == message_id), None)


# ---------------------------------------------------------------- research run
def busy_html():
    skeleton = "".join(f"<div class='skeleton' style='width:{w}%'></div>" for w in (92, 84, 61))
    return ("<div role='status' aria-busy='true'><div class='ps-busy'><span class='dot'></span>"
            f"Researching your question…</div>{skeleton}</div>")


def run_question(message):
    if not os.environ.get("OPENAI_API_KEY"):
        message.update(status="error", error="unconfigured")
        return
    if not get_limiter().allow():
        message.update(status="limited")
        return
    st.html(busy_html())
    checked = {}

    def on_progress(event, data):
        if event == "checked":
            checked["ok"] = data.get("status") == "pass"

    try:
        result = get_agent().run(message["question"], on_progress=on_progress, filters=message["filters"] or None)
    except Exception:  # friendly message instead of a stack trace; details stay in the server log
        message.update(status="error", error="failed")
        return
    record = build_record(result, message["question"], verified=checked.get("ok"), record_id=message["id"])
    message.update(status="done", record=record, trace_run_id=result.get("trace_run_id"))
    history.add(record)


def make_variant(message, kind):
    """Explain simply / Go deeper: rewrite from the same sources (no new search), as a new message."""
    parent = message["record"]
    label, instruction = VARIANTS[kind]
    try:
        with st.spinner(f"Writing a {label.lower()}…"):
            data = llm_json(instruction, Rewrite, sources_context(parent), feature=kind, answer_id=parent["id"])
    except LLMExtrasError as error:
        state["errors"][message["id"]] = str(error)
        return
    for existing in state["messages"]:  # already open: just jump to it
        record = existing.get("record") or {}
        if record.get("parent_id") == parent["id"] and record.get("kind") == kind:
            state["scroll_to"] = f"q-{existing['id']}"
            return
    record = variant_record(parent, kind, data["answer_md"])
    state["messages"].append({"id": record["id"], "question": parent["question"], "status": "done", "record": record})
    state["scroll_to"] = f"q-{record['id']}"


# ---------------------------------------------------------------- answer view
def drawer_html(prefix, sources):
    cards = []
    for source in sources:
        key = f"{prefix}-{source['n']}"
        url = source.get("url") or ""
        href = html.escape(url, quote=True) if url.startswith(("https://", "http://")) else ""
        tag = "<span class='src-tag'>Your upload</span>" if source.get("upload") else ""
        page = f" · p. {source['page']}" if source.get("upload") and source.get("page") else ""
        quote = f"<div class='src-quote'>“{html.escape(source['excerpt'])}”</div>" if source.get("excerpt") else ""
        link = f"<a href='{href}' target='_blank' rel='noopener noreferrer'>Open paper ↗</a>" if href else ""
        cards.append(f"<div class='src' data-src='{key}'><div class='src-top'><b>{source['n']}</b>{tag}"
                     f"{html.escape(byline(source))}{page}</div><div class='src-title'>{html.escape(source['title'])}</div>"
                     f"{quote}{link}</div>")
    return (f"<div class='ps-drawer-src' id='drawer-{prefix}' hidden>"
            "<div class='ps-drawer-h'><span>Sources</span>"
            "<button class='ps-drawer-x' data-close-drawer type='button'>Close ✕</button></div>"
            + "".join(cards) + "</div>")


def send_rating(message_id):
    message = find_message(message_id)
    value = state.get(f"feedback_{message_id}")
    if message is None or value is None:
        return
    send_feedback(message.get("trace_run_id"), "up" if value == 1 else "down")


def pick_followup(message_id):
    key = f"followups_{message_id}"
    question = state.get(key)
    if question:
        ask(question)
    state[key] = None


def render_more(message, record, prefix):
    mid = message["id"]
    with st.popover("More", key=f"more_{mid}"):
        st.html(f"<span class='ps-btn' role='button' tabindex='0' data-copy='raw-{prefix}'>Copy answer</span>"
                f"<pre class='ps-raw' id='raw-{prefix}'>{html.escape(record['answer_md'])}</pre>")
        st.download_button("Download Markdown", export.answer_markdown(record), file_name=export.markdown_filename(record["question"]),
                           mime="text/markdown", key=f"md_{mid}", width="stretch")
        st.download_button("Download BibTeX", export.bibtex(record["sources"]) or "% No citable sources\n",
                           file_name=export.markdown_filename(record["question"])[:-3] + ".bib",
                           mime="application/x-bibtex", key=f"bib_{mid}", width="stretch")
        with st.expander("Show as text"):
            st.code(record["answer_md"], language="markdown", wrap_lines=True)
        with st.container(key=f"narrow_{mid}", horizontal=True):  # phone-only overflow of the action row
            if record.get("kind") == "answer" and st.button("Go deeper", key=f"deeper_{mid}_m"):
                make_variant(message, "deeper")
                st.rerun()


def render_actions(message, record, prefix):
    mid = message["id"]
    with st.container(horizontal=True, key=f"actions_{mid}"):
        st.html(f"<button class='ps-btn' type='button' data-open-sources='{prefix}'>Sources · {len(record['sources'])}</button>",
                width="content")
        if record.get("kind") == "answer":
            if st.button("Explain simply", key=f"simpler_{mid}"):
                make_variant(message, "simpler")
                st.rerun()
            if st.button("Go deeper", key=f"deeper_{mid}"):  # moves into "More" on phones
                make_variant(message, "deeper")
                st.rerun()
        render_more(message, record, prefix)
        st.feedback("thumbs", key=f"feedback_{mid}", on_change=send_rating, args=(mid,))
    if state["errors"].get(mid):
        st.error(state["errors"].pop(mid))


def render_message(message):
    mid = message["id"]
    prefix = f"m{mid}"
    status = message["status"]
    record = message.get("record") or {}
    kind = record.get("kind", "answer")
    if kind in VARIANTS:
        parent = find_message(record.get("parent_id"))
        of = (f"<a href='#q-{parent['id']}'>{html.escape(parent['question'])}</a>" if parent
              else html.escape(record["question"]))
        st.html(f"<div class='q-kicker'>{VARIANTS[kind][0]} of: {of}</div>"
                f"<div class='q-title' id='q-{mid}'>{VARIANTS[kind][0]}</div>")
    else:
        st.html(f"<div class='q-title' id='q-{mid}'>{html.escape(message['question'])}</div>")
    if status == "queued":
        # Phase 1: draw the researching view and finish the run, which clears the previous page
        # (Streamlit only removes leftover elements when a run ends). Phase 2 does the work.
        st.html(busy_html())
        state["start_research"] = mid
        return
    if status == "running":
        run_question(message)
        st.rerun()
    if status == "limited":
        st.warning("Lots of people are asking right now. Please wait a minute, then ask again.")
        return
    if status == "error":
        if message.get("error") == "unconfigured":
            st.error("The assistant isn't configured on this server yet. Please try again later.")
        else:
            st.error("Something went wrong while researching this question. Please ask it again.")
        return

    st.html("<div class='q-meta'>" + ("<span class='ok'>Checked against sources</span>" if record.get("verified") else "")
            + "</div>")
    sources = record["sources"]
    short, rest = split_short_answer(record["answer_md"])
    if short:
        with st.container(key=f"tldr_{mid}"):
            st.markdown(f"**Short answer.** {link_citations(short, prefix, len(sources))}", unsafe_allow_html=True)
    with st.container(key=f"prose_{mid}"):
        st.markdown(link_citations(rest, prefix, len(sources)), unsafe_allow_html=True)

    if kind == "answer":
        questions = followups.ensure(record)
        if questions:
            st.pills("Ask next", questions, key=f"followups_{mid}", on_change=pick_followup, args=(mid,))
    render_actions(message, record, prefix)
    st.html(drawer_html(prefix, sources))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    local.sync()
    st.html("<div class='ps-brand'><div class='ps-mark'>P</div><div class='ps-name'>Paper Scout</div></div>")
    if st.button("New question", type="primary" if not state["messages"] else "secondary", width="stretch"):
        state["messages"] = []
        state["panel"] = {}
        st.rerun()
    saved = history.items()
    if saved:
        st.html("<div class='ps-side-label'>Recent</div>")
        for item in reversed(saved[-12:]):
            if st.button(shorten(item["question"], 60), key=f"hist_{item['id']}", type="tertiary", width="stretch"):
                open_record(item)
                st.rerun()
        if state.get("confirm_clear"):
            st.caption(f"Delete all {len(saved)} saved questions from this browser?")
            with st.container(horizontal=True):
                if st.button("Delete history", key="clear_yes", type="primary"):
                    history.clear()
                    state["confirm_clear"] = False
                    st.rerun()
                if st.button("Cancel", key="clear_no"):
                    state["confirm_clear"] = False
                    st.rerun()
        elif st.button("Clear history", key="clear_history", type="tertiary"):
            state["confirm_clear"] = True
            st.rerun()
    if local.loaded() and not local.available():
        st.caption("This browser blocks storage, so history won't be kept after you leave.")
    st.divider()
    st.html("<div class='ps-scope'>Answers come from recent AI and machine-learning research. "
            "Every claim links to its source. Each question is answered on its own.</div>")

# ---------------------------------------------------------------- main
if not state["messages"]:
    st.markdown("# Ask the research a question.")
    st.html("<p class='ps-lead'>Answers are built from recent AI and machine-learning papers. "
            "Every claim links to the paper it came from.</p>")
    featured = load_featured()
    columns = st.columns(2)
    if featured:
        for index, record in enumerate(featured):
            with columns[index % 2]:
                if st.button(f"{record['question']}  \n:gray[{record.get('teaser', '')}]", key=f"gallery_{record['id']}", width="stretch"):
                    open_record(record)
                    st.rerun()
    else:
        for index, (question, hint) in enumerate(SAMPLES):
            with columns[index % 2]:
                if st.button(f"{question}  \n:gray[{hint}]", key=f"gallery_sample{index}", width="stretch"):
                    ask(question)
                    st.rerun()

for position, message in enumerate(state["messages"]):
    if position:
        st.html("<hr class='ps-sep'>")
    render_message(message)

with st.container(key="filters", horizontal=True):
    st.pills("Research area", corpus["areas"], format_func=lambda code: AREA_NAMES.get(code, code), selection_mode="multi",
             key="area_filter", label_visibility="collapsed", help="Only search these research areas (none selected = all)")
    st.segmented_control("Period", PERIODS, default=PERIODS[0], key="period_filter", label_visibility="collapsed",
                         help="Only search recent papers")

typed = st.chat_input(
    "Ask a follow-up as a new question…" if state["messages"] else "Ask about a method, benchmark or trend…",
    max_chars=settings.max_question_length,
)
if typed:
    ask(typed)
    st.rerun()

page_script()(key="page_script", data={"scroll": state.pop("scroll_to", "") or "", "run": new_id()}, height=0)

if state.get("start_research"):
    message_id = state.pop("start_research")
    message = find_message(message_id)
    if message and message["status"] == "queued":
        message["status"] = "running"
    st.rerun()

if local.pending():  # flush browser-storage writes made during this run
    st.rerun()
