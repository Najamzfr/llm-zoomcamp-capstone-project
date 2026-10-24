"""Optional LangSmith tracing. Everything here is a no-op unless LANGSMITH_TRACING=true."""

import os
from contextlib import nullcontext

try:
    import langsmith
    from langsmith import traceable as _traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:  # tracing is optional
    langsmith = None
    _traceable = None
    get_current_run_tree = None

TAGS = ["paper-scout"]


def tracing_enabled():
    return (
        langsmith is not None
        and os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


def drop_self(inputs):
    """Keep method inputs readable in LangSmith by hiding the bound instance."""
    return {key: value for key, value in inputs.items() if key != "self"}


def traceable(**kwargs):
    """`langsmith.traceable`, or an identity decorator when langsmith is not installed."""
    if _traceable is None:
        return lambda function: function
    kwargs.setdefault("process_inputs", drop_self)
    kwargs.setdefault("tags", TAGS)
    return _traceable(**kwargs)


def span(name, run_type="chain", inputs=None):
    """Context manager that groups child runs (e.g. one LLM call) under a named step."""
    if not tracing_enabled():
        return nullcontext()
    return langsmith.trace(name=name, run_type=run_type, inputs=inputs or {}, tags=TAGS)


def wrap_client(client):
    """Trace OpenAI calls (prompts, tokens, latency) when tracing is on."""
    if not tracing_enabled():
        return client
    from langsmith.wrappers import wrap_openai

    return wrap_openai(client, chat_name=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"))


def current_run_id():
    if not tracing_enabled() or get_current_run_tree is None:
        return None
    run = get_current_run_tree()
    return str(run.id) if run is not None else None


def add_run_metadata(metadata):
    if not tracing_enabled() or get_current_run_tree is None:
        return
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata(metadata)


def send_feedback(run_id, rating):
    """Attach the UI's thumbs up/down to the traced run. Never raises."""
    if not run_id or not tracing_enabled():
        return False
    try:
        langsmith.Client().create_feedback(
            run_id, key="user_rating", score=1 if rating == "up" else 0
        )
        return True
    except Exception:
        return False
