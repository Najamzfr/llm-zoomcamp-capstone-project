"""Feedback with reasons (F15): thumbs, and on thumbs-down a reason + optional note.

Rows go to SHARE_BACKEND as {answer_id, question, rating, reason, text, created_at} — no user
identifiers. The rating is also attached to the LangSmith trace when tracing is on.
"""

import streamlit as st

from research_assistant.tracing import send_feedback

from . import share
from .records import now_iso

REASONS = ["Wrong", "Missing source", "Too vague", "Off-topic", "Other"]
MAX_TEXT = 500


def row(record, rating, reason=None, text=""):
    return {"answer_id": record["id"], "question": record["question"], "rating": rating, "reason": reason,
            "text": (text or "").strip()[:MAX_TEXT], "created_at": now_iso()}


def submit(record, rating, reason=None, text="", trace_run_id=None):
    """Store one feedback row; returns False when the backend is unavailable (the UI still thanks)."""
    send_feedback(trace_run_id, rating)
    store = share.backend()
    if store is None:
        return False
    try:
        store.save_feedback(row(record, rating, reason, text))
        return True
    except Exception:
        return False


def on_thumbs(record, mid, trace_run_id):
    value = st.session_state.get(f"feedback_{mid}")
    if value == 1:
        submit(record, "up", trace_run_id=trace_run_id)
        st.session_state.setdefault("feedback_done", set()).add(mid)
        st.toast("Thanks, noted")


def render_reasons(record, mid, trace_run_id):
    """Reason pills under the action row after a thumbs-down (until sent)."""
    state = st.session_state
    if state.get(f"feedback_{mid}") != 0 or mid in state.get("feedback_done", set()):
        return
    with st.container(border=True, key=f"feedback_{mid}_box"):
        reason = st.pills("What went wrong?", REASONS, key=f"feedback_{mid}_reason")
        text = st.text_input("Anything else? (optional)", max_chars=MAX_TEXT, key=f"feedback_{mid}_text")
        if st.button("Send feedback", key=f"feedback_{mid}_send", type="primary"):
            submit(record, "down", reason, text, trace_run_id)
            state.setdefault("feedback_done", set()).add(mid)
            st.toast("Thanks, noted")
            st.rerun()
