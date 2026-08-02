"""Low-cardinality Prometheus metrics for the RAG service."""

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge, Histogram


def _collector(factory, name, documentation, label_names=()):
    """Reuse a collector when Streamlit or a dev server reloads this module."""
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing
    try:
        return factory(name, documentation, label_names)
    except ValueError:
        return REGISTRY._names_to_collectors[name]


REQUESTS = _collector(
    Counter, "rag_requests", "Total HTTP requests.", ("endpoint", "status_class")
)
LATENCY = _collector(
    Histogram,
    "rag_request_latency_seconds",
    "HTTP request latency in seconds.",
    ("endpoint",),
)
ACTIVE_REQUESTS = _collector(
    Gauge, "rag_active_requests", "Requests currently in progress."
)
INPUT_TOKENS = _collector(
    Counter, "rag_input_tokens", "Input tokens consumed by model calls."
)
OUTPUT_TOKENS = _collector(
    Counter, "rag_output_tokens", "Output tokens consumed by model calls."
)
ESTIMATED_COST = _collector(
    Counter, "rag_estimated_cost_usd", "Estimated model cost in USD."
)
ANSWER_STAGES = _collector(
    Counter, "rag_answer_stages", "Answers by stage.", ("answer_stage",)
)
FULL_TEXT_ATTEMPTS = _collector(
    Counter, "rag_full_text_retrieval_attempts", "Full-text retrieval attempts."
)
FULL_TEXT_SUCCESSES = _collector(
    Counter, "rag_full_text_retrieval_successes", "Successful full-text retrievals."
)
JUDGE_OUTCOMES = _collector(
    Counter, "rag_judge_outcomes", "Judge outcomes.", ("outcome",)
)
FULL_TEXT_ROUTING = _collector(
    Counter,
    "rag_full_text_routing",
    "Answers routed to abstract or full-text retrieval.",
    ("decision",),
)


def observe_request(endpoint, status_code, duration_seconds):
    status_class = f"{int(status_code) // 100}xx"
    REQUESTS.labels(endpoint, status_class).inc()
    LATENCY.labels(endpoint).observe(duration_seconds)


def observe_result(result):
    stage = str(result.get("answer_stage", "unknown"))
    ANSWER_STAGES.labels(stage if stage in {"abstract", "full_text", "abstract_fallback"} else "unknown").inc()
    FULL_TEXT_ROUTING.labels("full_text" if stage == "full_text" else "abstract").inc()
    usage = result.get("usage") or {}
    INPUT_TOKENS.inc(float(usage.get("input_tokens", 0) or 0))
    OUTPUT_TOKENS.inc(float(usage.get("output_tokens", 0) or 0))
    ESTIMATED_COST.inc(float(usage.get("estimated_cost_usd", 0) or 0))
    for event in result.get("trajectory", []):
        if event.get("name") == "retrieve_full_text":
            FULL_TEXT_ATTEMPTS.inc()
            if event.get("success"):
                FULL_TEXT_SUCCESSES.inc()
        if event.get("name") in {"evaluate_answer", "evaluate_final_answer"}:
            JUDGE_OUTCOMES.labels(event.get("outcome", "unknown")).inc()


__all__ = [
    "ACTIVE_REQUESTS",
    "CONTENT_TYPE_LATEST",
    "observe_request",
    "observe_result",
]
