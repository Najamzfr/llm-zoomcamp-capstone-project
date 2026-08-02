"""Small no-network API checks; run with ``python -m unittest test_api``."""

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
        self.assertEqual(body["retrieved_papers"][0]["url"], "https://arxiv.org/abs/1234.5678")
        self.assertEqual(body["usage"]["total_tokens"], 3)
        self.assertEqual(self.stub_agent.questions, ["compare methods"])

    def test_metrics(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("rag_requests", response.text)
        self.assertNotIn('endpoint="/metrics"', response.text)


if __name__ == "__main__":
    unittest.main()
