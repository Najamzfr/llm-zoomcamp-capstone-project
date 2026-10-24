"""Question history kept in the browser (F4): the last 50 answers, newest last."""

from app.storage import local

LIMIT = 50


def items():
    return local.get(local.HISTORY)


def add(record):
    """Store an answer record; asking the same question again replaces the older entry."""
    entry = {key: record.get(key) for key in ("id", "question", "answer_md", "followups", "sources", "created_at", "verified")}
    kept = [item for item in items() if item.get("question", "").strip().lower() != record["question"].strip().lower()]
    local.put(local.HISTORY, (kept + [entry])[-LIMIT:])


def find(record_id):
    return next((item for item in items() if item.get("id") == record_id), None)


def clear():
    local.put(local.HISTORY, [])
