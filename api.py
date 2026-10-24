import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from prometheus_client import generate_latest

from research_assistant.config import settings
from research_assistant import metrics
from research_assistant.tracing import send_feedback

MAX_QUESTION_LENGTH = settings.max_question_length
STATIC_DIR = Path(__file__).resolve().parent / "static"
METRIC_PATHS = {"/ask", "/feedback", "/health"}
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


class AskLimiter:
    """Global in-memory cap on /ask to protect the OpenAI budget."""

    def __init__(self, per_minute, per_day):
        self.per_minute = per_minute
        self.per_day = per_day
        self._recent = deque()
        self._day = None
        self._day_count = 0
        self._lock = Lock()

    def allow(self, now=None):
        now = time.time() if now is None else now
        day = int(now // 86400)
        with self._lock:
            if day != self._day:
                self._day, self._day_count = day, 0
            while self._recent and now - self._recent[0] >= 60:
                self._recent.popleft()
            if self.per_day > 0 and self._day_count >= self.per_day:
                return False
            if self.per_minute > 0 and len(self._recent) >= self.per_minute:
                return False
            self._recent.append(now)
            self._day_count += 1
            return True


_ask_limiter = AskLimiter(settings.ask_rate_per_minute, settings.ask_daily_limit)


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]


class FeedbackStore:
    """Remembers recent answer ids so each answer can be rated once."""

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self._items = OrderedDict()
        self._lock = Lock()

    def add(self, run_id=None):
        request_id = uuid.uuid4().hex
        with self._lock:
            self._items[request_id] = {"rated": False, "run_id": run_id}
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)
        return request_id

    def rate(self, request_id):
        """Return ('ok' | 'unknown' | 'rated', LangSmith run id or None)."""
        with self._lock:
            item = self._items.get(request_id)
            if item is None:
                return "unknown", None
            if item["rated"]:
                return "rated", None
            item["rated"] = True
            return "ok", item["run_id"]


_feedback_store = FeedbackStore()


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


def _response(result, question, request_id=None):
    papers = result.get("papers", [])
    return {
        "request_id": request_id,
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


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    if request.url.path not in METRIC_PATHS:
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


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: AskRequest):
    if not _ask_limiter.allow():
        raise HTTPException(
            status_code=429,
            detail="Too many questions right now. Please try again later.",
        )
    try:
        result = _get_agent().run(payload.question)
        request_id = _feedback_store.add(result.get("trace_run_id"))
        response = _response(result, payload.question, request_id)
        metrics.observe_result(response)
        return response
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=503, detail="The research assistant is unavailable.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="The research assistant failed to answer.") from error


@app.post("/feedback", status_code=204)
async def feedback(payload: FeedbackRequest, background_tasks: BackgroundTasks):
    outcome, run_id = _feedback_store.rate(payload.request_id)
    if outcome == "unknown":
        raise HTTPException(status_code=404, detail="Unknown or expired answer.")
    if outcome == "rated":
        raise HTTPException(status_code=409, detail="This answer was already rated.")
    metrics.observe_feedback(payload.rating)
    if run_id:
        background_tasks.add_task(send_feedback, run_id, payload.rating)
    return Response(status_code=204)


@app.get("/metrics")
async def prometheus_metrics():
    return Response(generate_latest(), media_type=metrics.CONTENT_TYPE_LATEST)


__all__ = ["AskRequest", "FeedbackRequest", "app"]
