"""Shareable answer links (F10) and the storage backend shared with feedback (F15).

SHARE_BACKEND = supabase | memory (default memory: process memory, fine for local use; links die on
restart). Supabase is called through its REST API (PostgREST) with the anon key; see PLAN.md for the
tables, RLS policies and the get_share() function that only returns one share by id.

A snapshot is immutable and contains only {id, question, answer_md, sources (n, title, authors,
year, url, excerpt), created_at} and optionally a study set — never usage, model names, category
codes, history or user identifiers.
"""

import copy
import os
import re

import requests
import streamlit as st

from .records import new_id, now_iso, without_upload_text

MAX_ANSWER_BYTES = 10 * 1024
MAX_SHARES_PER_SESSION = 20
ID_PATTERN = re.compile(r"^[A-Za-z0-9]{8}$")
CATEGORY_CODE = re.compile(r"\b(?:cs|stat|eess|math|q-bio|q-fin|physics|econ)\.[A-Z]{2}\b")
SOURCE_FIELDS = ("n", "title", "authors", "year", "url", "excerpt")
CARD_FIELDS = ("id", "front", "back", "src", "difficulty")
QUIZ_FIELDS = ("id", "question", "options", "answer_index", "explanation", "src")


class ShareError(RuntimeError):
    """User-facing reason a share couldn't be created or loaded."""


# ------------------------------------------------------------------ backends
class MemoryBackend:
    def __init__(self):
        self.shares = {}
        self.feedback = []

    def save_share(self, share_id, payload):
        self.shares[share_id] = copy.deepcopy(payload)

    def load_share(self, share_id):
        return copy.deepcopy(self.shares.get(share_id))

    def save_feedback(self, payload):
        self.feedback.append(copy.deepcopy(payload))


class SupabaseBackend:
    def __init__(self, url, key, timeout=10):
        self.url = url.rstrip("/")
        self.headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.timeout = timeout

    def _post(self, path, body, prefer="return=minimal"):
        response = requests.post(f"{self.url}/rest/v1/{path}", json=body, timeout=self.timeout,
                                 headers={**self.headers, "Prefer": prefer})
        response.raise_for_status()
        return response

    def save_share(self, share_id, payload):
        self._post("shares", {"id": share_id, "payload": payload})

    def load_share(self, share_id):
        response = self._post("rpc/get_share", {"share_id": share_id}, prefer="return=representation")
        return response.json() or None

    def save_feedback(self, payload):
        self._post("feedback", {"payload": payload})


@st.cache_resource
def _memory_backend():
    return MemoryBackend()


def backend():
    """The configured backend, or None when Supabase is selected but not configured."""
    name = (os.getenv("SHARE_BACKEND") or "memory").strip().lower()
    if name == "supabase":
        url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY")
        return SupabaseBackend(url, key) if url and key else None
    return _memory_backend()


# ------------------------------------------------------------------ snapshots
def _clean(text):
    return CATEGORY_CODE.sub("", str(text or "")).strip()


def sanitize_study(study):
    if not study:
        return None
    return {
        "cards": [{key: card.get(key) for key in CARD_FIELDS} for card in study.get("cards", [])],
        "quiz": [{key: question.get(key) for key in QUIZ_FIELDS} for question in study.get("quiz", [])],
    }


def snapshot(record, study=None):
    """Whitelisted, size-checked payload for a share. Raises ShareError for empty/oversized answers."""
    answer = _clean(record.get("answer_md"))
    if not answer:
        raise ShareError("There's no answer to share yet.")
    if len(answer.encode("utf-8")) > MAX_ANSWER_BYTES:
        raise ShareError("This answer is too long to share as a link. Download it as Markdown instead.")
    payload = {
        "question": _clean(record.get("question")),
        "answer_md": answer,
        "sources": [{key: _clean(source.get(key)) if isinstance(source.get(key), str) else source.get(key)
                     for key in SOURCE_FIELDS} for source in without_upload_text(record.get("sources", []))],
        "created_at": now_iso(),
    }
    cleaned_study = sanitize_study(study)
    if cleaned_study:
        payload["study"] = cleaned_study
    return payload


def app_url():
    configured = os.getenv("APP_URL")
    if configured:
        return configured.rstrip("/")
    try:
        return str(st.context.url).split("?")[0].rstrip("/")
    except Exception:
        return "http://localhost:8501"


def create(record, study=None):
    """Save a snapshot and return its public URL."""
    state = st.session_state
    if state.get("shares_made", 0) >= MAX_SHARES_PER_SESSION:
        raise ShareError("You've shared a lot of answers in this session. Please try again later.")
    store = backend()
    if store is None:
        raise ShareError("Sharing isn't set up on this server yet.")
    payload = snapshot(record, study)
    share_id = new_id(8)
    payload["id"] = share_id
    try:
        store.save_share(share_id, payload)
    except Exception as error:
        raise ShareError("Couldn't save the link right now. Please try again in a minute.") from error
    state["shares_made"] = state.get("shares_made", 0) + 1
    return f"{app_url()}/?a={share_id}"


def load(share_id):
    """The stored snapshot, or None when the id is malformed, unknown or the backend is down."""
    if not share_id or not ID_PATTERN.match(share_id):
        return None
    store = backend()
    if store is None:
        return None
    try:
        payload = store.load_share(share_id)
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload.get("answer_md"):
        return None
    return payload


def as_record(payload):
    """A snapshot rendered like a stored answer (read-only)."""
    sources = [dict(source, evidence=source.get("excerpt", "")) for source in payload.get("sources", [])]
    return {"id": payload["id"], "question": payload["question"], "answer_md": payload["answer_md"],
            "followups": [], "sources": sources, "created_at": payload.get("created_at"), "kind": "shared",
            "verified": None}


def render_panel(record, mid, study=None):
    state = st.session_state
    st.html("<div class='panel-h'><b>Share this answer</b></div>")
    include = False
    if study:
        include = st.checkbox("Include the study set", value=True, key=f"share_{mid}_study")
    if st.button("Create share link", key=f"share_{mid}_create", type="primary"):
        try:
            state[f"share_{mid}_url"] = create(record, study if include else None)
            state["copy_text"] = state[f"share_{mid}_url"]
            st.toast("Link copied")
        except ShareError as error:
            st.error(str(error))
    url = state.get(f"share_{mid}_url")
    if url:
        st.code(url, language="text", wrap_lines=True)
    st.caption("Anyone with the link sees this question, the answer and its sources — never your history.")
