"""Small no-network API checks; run with ``python -m unittest test_api``."""

import os

os.environ["LANGSMITH_TRACING"] = "false"  # never send test runs to LangSmith

import unittest

from fastapi.testclient import TestClient

import api


class StubAgent:
    def __init__(self):
        self.questions = []

    def run(self, question):
        self.questions.append(question)
        return {
            "answer": "stub answer",
            "answer_stage": "abstract",
            "papers": [
                {
                    "title": "Stub paper",
                    "paper_id": "1234.5678",
                    "submitted_date": "2025-01-01",
                    "category": "cs.AI",
                    "arxiv_url": "https://arxiv.org/abs/1234.5678",
                }
            ],
            "trajectory": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "estimated_cost_usd": 0.0,
            },
        }


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_agent = api._agent
        cls.stub_agent = StubAgent()
        api._agent = cls.stub_agent
        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        api._agent = cls.original_agent

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_rate_limiter_blocks_after_per_minute_cap(self):
        limiter = api.AskLimiter(per_minute=2, per_day=0)
        self.assertTrue(limiter.allow(now=1000.0))
        self.assertTrue(limiter.allow(now=1001.0))
        self.assertFalse(limiter.allow(now=1002.0))
        self.assertTrue(limiter.allow(now=1061.0))

    def test_rate_limiter_daily_cap(self):
        limiter = api.AskLimiter(per_minute=0, per_day=1)
        self.assertTrue(limiter.allow(now=100.0))
        self.assertFalse(limiter.allow(now=200.0))
        self.assertTrue(limiter.allow(now=100.0 + 86400))

    def test_validation(self):
        self.assertEqual(self.client.post("/ask", json={"question": "  "}).status_code, 422)
        self.assertEqual(self.client.post("/ask", json={"question": "x" * 2001}).status_code, 422)

    def test_successful_ask_uses_stub_agent(self):
        response = self.client.post("/ask", json={"question": "  compare methods  "})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["question"], "compare methods")
        self.assertEqual(body["answer"], "stub answer")
        self.assertEqual(body["answer_stage"], "abstract")
        self.assertEqual(body["paper_names"], ["Stub paper"])
        self.assertEqual(body["retrieved_papers"][0]["paper_id"], "1234.5678")
        self.assertEqual(body["retrieved_papers"][0]["title"], "Stub paper")
        self.assertEqual(body["retrieved_papers"][0]["category"], "cs.AI")
        self.assertEqual(body["retrieved_papers"][0]["published"], "2025-01-01")
        self.assertEqual(body["retrieved_papers"][0]["url"], "https://arxiv.org/abs/1234.5678")
        self.assertEqual(body["usage"]["input_tokens"], 1)
        self.assertEqual(body["usage"]["output_tokens"], 2)
        self.assertEqual(body["usage"]["total_tokens"], 3)
        self.assertEqual(self.stub_agent.questions[-1], "compare methods")

    def test_ui_is_served(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn("/static/app.js", page.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)

    def test_ask_returns_request_id_and_feedback_flow(self):
        request_id = self.client.post("/ask", json={"question": "rate me"}).json()["request_id"]
        self.assertRegex(request_id, r"^[0-9a-f]{32}$")
        first = self.client.post("/feedback", json={"request_id": request_id, "rating": "up"})
        self.assertEqual(first.status_code, 204)
        again = self.client.post("/feedback", json={"request_id": request_id, "rating": "down"})
        self.assertEqual(again.status_code, 409)
        self.assertIn('rag_feedback_total{rating="up"}', self.client.get("/metrics").text)

    def test_feedback_validation(self):
        unknown = self.client.post("/feedback", json={"request_id": "missing", "rating": "up"})
        self.assertEqual(unknown.status_code, 404)
        invalid = self.client.post("/feedback", json={"request_id": "x", "rating": "meh"})
        self.assertEqual(invalid.status_code, 422)

    def test_feedback_store_is_bounded(self):
        store = api.FeedbackStore(max_size=2)
        oldest = store.add()
        store.add()
        store.add()
        self.assertEqual(store.rate(oldest), ("unknown", None))

    def test_feedback_store_returns_trace_run_id_once(self):
        store = api.FeedbackStore()
        request_id = store.add("run-123")
        self.assertEqual(store.rate(request_id), ("ok", "run-123"))
        self.assertEqual(store.rate(request_id), ("rated", None))

    def test_metrics(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("rag_requests", response.text)
        self.assertNotIn('endpoint="/metrics"', response.text)
        self.client.get("/static/styles.css")
        self.assertNotIn('endpoint="/static', self.client.get("/metrics").text)


if __name__ == "__main__":
    unittest.main()
