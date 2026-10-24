"""Streamlit AppTest smoke tests for deploy/streamlit_app.py (no model calls: the gallery uses stored answers)."""

import os

os.environ["LANGSMITH_TRACING"] = "false"

import json
import re
import unittest
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "deploy" / "streamlit_app.py")
FEATURED = json.loads((ROOT / "data" / "featured.json").read_text(encoding="utf-8"))
INTERNALS = re.compile(r"\b(cs|stat|eess)\.[A-Z]{2}\b|arxiv|\b\d{1,3},\d{3} papers|tokens|\$\d|gpt-|cost", re.IGNORECASE)


def visible_text(app):
    parts = [element.value for element in app.markdown] + [element.label for element in app.button]
    parts += [getattr(element, "body", "") or "" for element in app.get("html")]
    # Tags and attributes are dropped: link targets (paper URLs) aren't visible copy.
    return "\n".join(re.sub(r"<[^>]+>", " ", str(part)) for part in parts)


def start():
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    return app


def open_featured(app, index=0):
    app.button(key=f"gallery_{FEATURED[index]['id']}").click().run()
    return FEATURED[index]["id"]


class EmptyStateTests(unittest.TestCase):
    def test_gallery_renders_without_internals(self):
        app = start()
        self.assertFalse(app.exception)
        gallery = [button for button in app.button if (button.key or "").startswith("gallery_")]
        self.assertEqual(len(gallery), len(FEATURED))
        self.assertIsNone(INTERNALS.search(visible_text(app)))

    def test_gallery_answer_renders_with_zero_model_calls(self):
        app = start()
        with mock.patch("app.features.llm_extras._complete", side_effect=AssertionError("no model call expected")):
            record_id = open_featured(app)
        self.assertFalse(app.exception)
        text = visible_text(app)
        self.assertIn(FEATURED[0]["question"], text)
        self.assertIn(f"Sources · {len(FEATURED[0]['sources'])}", text)
        self.assertEqual(app.pills(key=f"followups_{record_id}").options, FEATURED[0]["followups"])
        self.assertIsNone(INTERNALS.search(text))


class ActionTests(unittest.TestCase):
    def test_explain_simply_adds_a_linked_version(self):
        app = start()
        record_id = open_featured(app)
        reply = '{"answer_md": "Short answer: It checks retrieval and answers separately [1].\\n\\n#### Why\\nLike an exam [2]."}'
        with mock.patch("app.features.llm_extras._complete", return_value=reply):
            app.button(key=f"simpler_{record_id}").click().run()
        self.assertFalse(app.exception)
        text = visible_text(app)
        self.assertIn("Simpler version of:", text)
        self.assertIn("checks retrieval", text)

    def test_followup_pill_queues_a_new_question(self):
        app = start()
        record_id = open_featured(app)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):  # stop before any real research
            app.pills(key=f"followups_{record_id}").set_value(FEATURED[0]["followups"][0]).run()
        self.assertIn(FEATURED[0]["followups"][0], [message["question"] for message in app.session_state["messages"]])

    def test_history_records_opened_answers_offline(self):
        app = start()
        app.session_state["ps_local"]["ps_history_v1"] = [dict(FEATURED[1], id="hist0001")]
        app.run()
        app.button(key="hist_hist0001").click().run()
        self.assertFalse(app.exception)
        self.assertIn(FEATURED[1]["question"], visible_text(app))
        app.button(key="clear_history").click().run()
        app.button(key="clear_yes").click().run()
        self.assertEqual(app.session_state["ps_local"]["ps_history_v1"], [])


if __name__ == "__main__":
    unittest.main()
