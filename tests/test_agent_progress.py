import os

os.environ["LANGSMITH_TRACING"] = "false"  # never send test runs to LangSmith

import unittest
from unittest.mock import patch

from research_assistant.agent import AdaptiveResearchAgent
from research_assistant.rag import empty_usage

PAPERS = [
    {"paper_id": f"p{i}", "title": f"Paper {i}", "text": "abstract", "submitted_date": "2026-01-01", "rank": i}
    for i in range(1, 4)
]


class StubSearch:
    def search(self, query, k=5):
        return list(PAPERS)


class StubPaperTools:
    def retrieve(self, query, papers):
        chunks = [{"chunk_id": "p1-p1-c1", "paper_id": "p1", "page": 1, "text": "evidence"}]
        return chunks, "[1] Paper 1 (page 1)\nevidence", ["p1"]


def make_agent(router, check):
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        agent = AdaptiveResearchAgent(hybrid_search=StubSearch(), paper_tools=StubPaperTools())
    decisions = {
        "route_abstracts_or_full_text": router,
        "check_abstract_answer": check,
        "check_full_text_answer": {"status": "pass"},
    }
    agent._json_call = lambda instructions, prompt, step="json_decision": (decisions[step], empty_usage("gpt-4o-mini"))
    agent._abstract_answer = lambda question, papers: ("abstract answer [1]", empty_usage("gpt-4o-mini"))
    agent._text_call = lambda instructions, prompt, max_output_tokens: ("full answer [1]", empty_usage("gpt-4o-mini"))
    return agent


class AgentProgressTests(unittest.TestCase):
    def run_agent(self, router, check=None):
        events = []
        agent = make_agent(router, check or {"status": "pass"})
        result = agent.run("question", on_progress=lambda event, data: events.append((event, data)))
        return result, events

    def test_abstract_path_events(self):
        result, events = self.run_agent({"needs_full_text": False, "reason": "enough"})
        self.assertEqual([name for name, _ in events], ["searched", "routed", "drafted", "checked"])
        self.assertEqual(events[0][1], {"papers": 3})
        self.assertFalse(events[1][1]["needs_full_text"])
        self.assertEqual(result["answer_stage"], "abstract")

    def test_full_text_path_events(self):
        result, events = self.run_agent(
            {"needs_full_text": True, "reason": "numbers", "selected_paper_ids": ["p1"]}
        )
        self.assertEqual(
            [name for name, _ in events],
            ["searched", "routed", "fetching_full_text", "full_text_ready", "drafted", "checked"],
        )
        self.assertEqual(events[3][1]["chunks"], 1)
        self.assertEqual(events[3][1]["downloaded"], 1)
        self.assertEqual(result["answer_stage"], "full_text")

    def test_failing_callback_does_not_break_answer(self):
        def explode(event, data):
            raise RuntimeError("ui broke")

        agent = make_agent({"needs_full_text": False}, {"status": "pass"})
        self.assertEqual(agent.run("question", on_progress=explode)["answer"], "abstract answer [1]")

    def test_without_callback(self):
        agent = make_agent({"needs_full_text": False}, {"status": "pass"})
        self.assertEqual(agent.run("question")["answer_stage"], "abstract")


if __name__ == "__main__":
    unittest.main()
