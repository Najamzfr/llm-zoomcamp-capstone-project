import time
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator
from prometheus_client import generate_latest

from research_assistant import metrics

MAX_QUESTION_LENGTH = 2_000
_agent = None
_agent_lock = Lock()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        if len(value) > MAX_QUESTION_LENGTH:
            raise ValueError("question is too long")
        return value


def _get_agent():
    global _agent
    if _agent is not None:
        return _agent
    with _agent_lock:
        if _agent is None:
            from research_assistant.agent import AdaptiveResearchAgent

            _agent = AdaptiveResearchAgent(trace_path=None)
    return _agent


def _judge_outcome(event):
    status = event.get("status")
    if status == "pass":
        return "pass"
    if status in {"insufficient", "fail"}:
        return "fail"
    if event.get("needs_full_text") is False:
        return "pass"
    if event.get("needs_full_text") is True or status == "needs_full_text":
        return "unknown"
    return "unknown"


def _compact_trajectory(trajectory):
    compact = []
    for event in trajectory or []:
        name = event.get("name", "unknown")
        item = {"name": name}
        if name == "search_papers":
            item["retrieved_count"] = len(event.get("retrieved_paper_ids", []))
        elif name == "retrieve_full_text":
            item.update(
                {
                    "selected_count": len(event.get("arguments", {}).get("paper_ids", [])),
                    "chunk_count": len(event.get("retrieved_chunk_ids", [])),
                    "context_tokens": event.get("context_tokens", 0),
                    "downloaded_count": len(event.get("downloaded_paper_ids", [])),
                    "success": "error" not in event,
                }
            )
        elif name in {
            "evaluate_abstract_sufficiency",
            "evaluate_answer",
            "evaluate_final_answer",
        }:
            decision = event.get("result", {})
            item["status"] = decision.get("status")
            item["needs_full_text"] = decision.get("needs_full_text")
            item["outcome"] = _judge_outcome(decision)
        compact.append(item)
    return compact


def _paper_response(paper):
    return {
        "title": paper.get("title", ""),
        "paper_id": paper.get("paper_id", ""),
        "published": paper.get("submitted_date"),
        "category": paper.get("category"),
        "url": paper.get("arxiv_url") or paper.get("pdf_url"),
    }


def _response(result, question):
    papers = result.get("papers", [])
    return {
        "question": question,
        "answer": result.get("answer", ""),
        "answer_stage": result.get("answer_stage", "unknown"),
        "paper_names": [paper.get("title", "") for paper in papers],
        "retrieved_papers": [_paper_response(paper) for paper in papers],
        "trajectory": _compact_trajectory(result.get("trajectory")),
        "usage": result.get("usage", {}),
    }


@asynccontextmanager
async def lifespan(_app):
    yield


app = FastAPI(title="Research Paper RAG API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    started = time.perf_counter()
    metrics.ACTIVE_REQUESTS.inc()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        metrics.ACTIVE_REQUESTS.dec()
        metrics.observe_request(
            request.url.path, status_code, time.perf_counter() - started
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: AskRequest):
    try:
        result = _get_agent().run(payload.question)
        response = _response(result, payload.question)
        metrics.observe_result(response)
        return response
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=503, detail="The research assistant is unavailable.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="The research assistant failed to answer.") from error


@app.get("/metrics")
async def prometheus_metrics():
    return Response(generate_latest(), media_type=metrics.CONTENT_TYPE_LATEST)


__all__ = ["AskRequest", "app"]
