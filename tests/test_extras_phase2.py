"""Phase 2 unit tests: FSRS review deck, Anki/CSV export, concept map, study sets, share snapshots."""

import os

os.environ["LANGSMITH_TRACING"] = "false"

import csv
import io
import json
import re
import sqlite3
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import streamlit as st

from app.features import concept_map, export, llm_extras, review, share, study
from app.storage import local
from tests.test_extras import record

NOW = datetime(2026, 10, 25, 9, 0, tzinfo=timezone.utc)
STUDY_REPLY = json.dumps({
    "cards": [{"id": f"c{i}", "front": f"What is idea {i}?", "back": f"Idea {i} explained.", "src": [1, 9],
               "difficulty": "medium"} for i in range(1, 9)],
    "quiz": [{"id": f"q{i}", "question": f"Question {i}?", "options": ["A", "B", "C", "D"], "answer_index": 2,
              "explanation": "Because C [1].", "src": [2]} for i in range(1, 6)],
})
MAP_REPLY = json.dumps({
    "nodes": [{"id": "n1", "label": "Low-rank adaptation of weights", "src": [1]}, {"id": "n2", "label": "Adapters", "src": [2]},
              {"id": "n3", "label": "Fine-tuning cost", "src": []}, {"id": "n4", "label": "Accuracy", "src": [7]}],
    "edges": [{"from": "n1", "to": "n3", "label": "reduces the overall"}, {"from": "n2", "to": "n3", "label": "reduces"},
              {"from": "n1", "to": "n9", "label": "bad"}],
})


def reset_storage():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def cards():
    return [{"id": f"rec00001-sc{i}", "front": f"Front {i}", "back": f"Back {i}", "src": [1]} for i in range(1, 4)]


class ReviewTests(unittest.TestCase):
    def setUp(self):
        reset_storage()

    def test_add_is_idempotent_and_keeps_source_meta(self):
        self.assertEqual(review.add_cards(cards(), record(), NOW), 3)
        self.assertEqual(review.add_cards(cards(), record(), NOW), 0)
        entry = review.deck()["rec00001-sc1"]
        self.assertEqual(entry["topic"], "Compare LoRA and adapters?")
        self.assertEqual(entry["src_meta"][0]["title"], "LoRA: Low-Rank Adaptation of Large Language Models")
        self.assertEqual(len(review.due_cards(NOW)), 3)
        self.assertTrue(local.pending())

    def test_grading_moves_the_due_date(self):
        review.add_cards(cards(), record(), NOW)
        again = review.grade("rec00001-sc1", "Again", NOW)
        easy = review.grade("rec00001-sc2", "Easy", NOW)
        self.assertGreater(easy, again)
        self.assertGreater(easy - NOW, timedelta(days=1))
        self.assertLess(again - NOW, timedelta(hours=1))
        self.assertEqual(review.deck()["rec00001-sc2"]["due"], easy.isoformat())
        self.assertEqual([card_id for card_id, _ in review.due_cards(NOW)], ["rec00001-sc3"])
        previews = review.preview("rec00001-sc3", NOW)
        self.assertLess(previews["Again"], previews["Hard"])
        self.assertLessEqual(previews["Good"], previews["Easy"])

    def test_daily_cap_and_streak(self):
        review.add_cards(cards(), record(), NOW)
        with mock.patch.dict(os.environ, {"REVIEW_DAILY_CAP": "2"}):
            self.assertEqual(review.due_count(NOW), 2)
            review.grade("rec00001-sc1", "Good", NOW)
            review.grade("rec00001-sc2", "Good", NOW)
            self.assertEqual(review.due_count(NOW), 0)
        local.put(local.REVIEW_META, {"days": {"2026-10-22": 3, "2026-10-23": 1, "2026-10-24": 5, "2026-10-25": 2}})
        self.assertEqual(review.streak(NOW), 4)
        self.assertEqual(review.streak(NOW + timedelta(days=1)), 4)  # today not reviewed yet
        self.assertEqual(review.streak(NOW + timedelta(days=3)), 0)

    def test_days_follow_the_visitor_time_zone(self):
        late = datetime(2026, 10, 24, 22, 30, tzinfo=timezone.utc)
        self.assertEqual(review.local_day(late).isoformat(), "2026-10-24")
        st.session_state["ps_tz_minutes"] = 300  # UTC+5
        self.assertEqual(review.local_day(late).isoformat(), "2026-10-25")
        self.assertEqual(review.when(late + timedelta(minutes=10), late), "in 10 min")
        self.assertEqual(review.when(late + timedelta(days=3), late), "28 Oct 2026")

    def test_mistake_card_and_topics(self):
        question = {"id": "x-q1", "question": "Which is cheaper?", "options": ["A", "B", "C", "D"], "answer_index": 1,
                    "explanation": "B trains fewer weights.", "src": [2]}
        card = review.mistake_card(question)
        self.assertEqual(card["back"], "B. B trains fewer weights.")
        review.add_cards([card], record(), NOW)
        self.assertEqual(review.topics(), ["Compare LoRA and adapters?"])

    def test_backup_round_trip(self):
        review.add_cards(cards(), record(), NOW)
        local.put(local.COLLECTIONS, {"Thesis": [{"id": "a1", "question": "Q"}]})
        dump = review.backup_json()
        reset_storage()
        self.assertEqual(review.restore(dump), (3, 1))
        self.assertEqual(len(review.deck()), 3)
        self.assertEqual(review.restore(dump)[0], 0)
        with self.assertRaises(ValueError):
            review.restore('{"app": "other"}')


class AnkiCsvTests(unittest.TestCase):
    def test_apkg_opens_with_matching_card_count(self):
        reset_storage()
        review.add_cards(cards(), record(), NOW)
        data = export.anki_package(list(review.deck().values()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "deck.apkg"
            path.write_bytes(data)
            with zipfile.ZipFile(path) as archive:
                archive.extract("collection.anki2", folder)
            connection = sqlite3.connect(Path(folder) / "collection.anki2")
            notes = connection.execute("SELECT flds, tags FROM notes").fetchall()
            card_count = connection.execute("SELECT count(*) FROM cards").fetchone()[0]
            decks = json.loads(connection.execute("SELECT decks FROM col").fetchone()[0])
            connection.close()
        self.assertEqual(len(notes), 3)
        self.assertEqual(card_count, 3)
        self.assertIn(str(export.DECK_ID), decks)
        front, back, source = notes[0][0].split("\x1f")
        self.assertTrue(front.startswith("Front"))
        self.assertIn("LoRA", source)
        self.assertIn("2021", source)
        self.assertIn("compare-lora-and-adapters", notes[0][1])

    def test_ids_are_stable(self):
        self.assertEqual(export.DECK_ID, export._stable_id("Paper Scout deck"))
        self.assertGreater(export.DECK_ID, 1 << 30)

    def test_csv_has_header_and_rows(self):
        study_cards = [dict(card, src_meta=review.src_meta(record(), [1, 2])) for card in cards()]
        rows = list(csv.reader(io.StringIO(export.cards_csv(study_cards))))
        self.assertEqual(rows[0], ["front", "back", "source"])
        self.assertEqual(len(rows), 4)
        self.assertIn("https://arxiv.org/abs/2106.09685v2", rows[1][2])


class GeneratedFeatureTests(unittest.TestCase):
    def setUp(self):
        llm_extras._cached.clear()

    def test_study_set_ids_and_sources(self):
        with mock.patch.object(llm_extras, "_complete", return_value=STUDY_REPLY) as call:
            first = study.generate(record())
            again = study.generate(record())
            hard = study.generate(record(), "hard")
        self.assertEqual(call.call_count, 2)  # standard is cached; hard is a separate cache entry
        self.assertEqual(first["cards"][0]["id"], "rec00001-sc1")
        self.assertEqual(hard["cards"][0]["id"], "rec00001-hc1")
        self.assertEqual(first["cards"][0]["src"], [1])  # out-of-range source 9 dropped
        self.assertEqual(first, again)
        self.assertIn("mechanism", call.call_args_list[1].args[0][1]["content"])
        positions = {question["answer_index"] for question in first["quiz"]}
        self.assertGreater(len(positions), 1)  # options shuffled per question
        for question in first["quiz"]:
            self.assertEqual(question["options"][question["answer_index"]], "C")

    def test_concept_map_dot_and_mermaid(self):
        with mock.patch.object(llm_extras, "_complete", return_value=MAP_REPLY):
            cmap = concept_map.generate(record())
        self.assertEqual(cmap["nodes"][0]["label"], "Low-rank adaptation of weights")
        self.assertEqual(len(cmap["edges"]), 2)  # edge to unknown node dropped
        self.assertEqual(cmap["nodes"][3]["src"], [])  # source 7 doesn't exist
        dot = concept_map.to_dot(cmap)
        self.assertIn("rankdir=LR", dot)
        self.assertEqual(dot.count("dashed"), 2)  # Fine-tuning cost and Accuracy
        self.assertIn('"n1" -> "n3" [label=" reduces the overall "]', dot)
        mermaid = export.mermaid(cmap)
        self.assertTrue(mermaid.startswith("flowchart LR"))
        self.assertIn('n0["Low-rank adaptation of weights"]', mermaid)


class ShareTests(unittest.TestCase):
    def setUp(self):
        reset_storage()

    def test_snapshot_is_sanitised(self):
        rec = record()
        rec.update(usage={"input_tokens": 1200, "estimated_cost_usd": 0.01}, model="gpt-4o-mini", answer_stage="full_text")
        rec["answer_md"] += " See cs.CL work."
        rec["sources"][0]["category"] = "cs.CL"
        rec["sources"][0]["evidence"] = "long abstract"
        payload = share.snapshot(rec, {"cards": [{"id": "c1", "front": "F", "back": "B", "src": [1], "difficulty": "easy",
                                                  "src_meta": [{"title": "x"}]}], "quiz": []})
        self.assertEqual(set(payload), {"question", "answer_md", "sources", "created_at", "study"})
        self.assertEqual(set(payload["sources"][0]), set(share.SOURCE_FIELDS))
        self.assertNotIn("src_meta", payload["study"]["cards"][0])
        text = json.dumps(payload)
        for forbidden in ("cs.CL", "tokens", "cost", "gpt-", "full_text", "evidence", "history", "session"):
            self.assertNotIn(forbidden, text)
        self.assertIsNone(re.search(r"\b\d{1,3},\d{3} papers", text))

    def test_snapshot_limits(self):
        rec = record()
        with self.assertRaises(share.ShareError):
            share.snapshot(dict(rec, answer_md="  "))
        with self.assertRaises(share.ShareError):
            share.snapshot(dict(rec, answer_md="x" * (share.MAX_ANSWER_BYTES + 1)))

    def test_create_load_and_rate_limit(self):
        share._memory_backend.clear()
        with mock.patch.dict(os.environ, {"SHARE_BACKEND": "memory", "APP_URL": "https://paprly.streamlit.app/"}):
            url = share.create(record())
            share_id = url.rsplit("=", 1)[1]
            self.assertTrue(url.startswith("https://paprly.streamlit.app/?a="))
            self.assertRegex(share_id, r"^[A-Za-z0-9]{8}$")
            loaded = share.load(share_id)
            self.assertEqual(loaded["question"], "Compare LoRA and adapters?")
            self.assertEqual(share.as_record(loaded)["kind"], "shared")
            self.assertIsNone(share.load("missing1"))
            self.assertIsNone(share.load("../etc"))
            st.session_state["shares_made"] = share.MAX_SHARES_PER_SESSION
            with self.assertRaises(share.ShareError):
                share.create(record())

    def test_supabase_backend_requests(self):
        backend = share.SupabaseBackend("https://x.supabase.co/", "anon")
        response = mock.Mock(status_code=201, json=lambda: {"id": "abc"})
        response.raise_for_status = mock.Mock()
        with mock.patch.object(share.requests, "post", return_value=response) as post:
            backend.save_share("abcdefgh", {"q": 1})
            self.assertEqual(backend.load_share("abcdefgh"), {"id": "abc"})
            backend.save_feedback({"rating": "down"})
        urls = [call.args[0] for call in post.call_args_list]
        self.assertEqual(urls, ["https://x.supabase.co/rest/v1/shares", "https://x.supabase.co/rest/v1/rpc/get_share",
                                "https://x.supabase.co/rest/v1/feedback"])
        self.assertEqual(post.call_args_list[0].kwargs["headers"]["Prefer"], "return=minimal")
        with mock.patch.dict(os.environ, {"SHARE_BACKEND": "supabase", "SUPABASE_URL": "", "SUPABASE_ANON_KEY": ""}):
            self.assertIsNone(share.backend())


if __name__ == "__main__":
    unittest.main()
