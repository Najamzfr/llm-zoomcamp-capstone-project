"""Unit tests for the extra features in app/features (no network: the extras LLM is mocked)."""

import os

os.environ["LANGSMITH_TRACING"] = "false"  # never send test runs to LangSmith

import json
import re
import unittest
from unittest import mock

from app.features import export, llm_extras
from app.features.records import build_record, byline, sources_context, variant_record
from research_assistant.presentation import slug, split_followups, strip_for_speech

RESULT = {
    "answer": "Short answer: LoRA adapts low-rank matrices [1]; adapters add layers [2].\n\n"
              "#### Details\nMore here [1][2].\n\nFollow-ups:\n- How does rank affect accuracy?\n"
              "- - Do adapters slow inference?\n- When is full fine-tuning better?\n\n"
              "The available full-text excerpts may not completely resolve every requested detail.",
    "papers": [
        {"paper_id": "2106.09685v2", "title": "LoRA: Low-Rank Adaptation of Large Language Models",
         "authors": "Edward Hu; Yelong Shen; Phillip Wallis", "submitted_date": "2021-06-17",
         "arxiv_url": "https://arxiv.org/abs/2106.09685v2", "category": "cs.CL",
         "text": "Title: LoRA\n\nAbstract: We propose Low-Rank Adaptation, which freezes the weights. " * 5},
        {"paper_id": "1902.00751", "title": "Parameter-Efficient Transfer Learning for NLP",
         "authors": "Neil Houlsby, Andrei Giurgiu", "submitted_date": "2019-02-02",
         "arxiv_url": "https://arxiv.org/abs/1902.00751", "category": "cs.LG", "text": "Abstract: Adapters."},
    ],
    "excerpts": {},
}


def record():
    return build_record(RESULT, "Compare LoRA and adapters?", verified=True, record_id="rec00001")


class FollowupTests(unittest.TestCase):
    def test_split_followups_keeps_trailing_note(self):
        answer, questions = split_followups(RESULT["answer"])
        self.assertEqual(questions, ["How does rank affect accuracy?", "Do adapters slow inference?",
                                     "When is full fine-tuning better?"])
        self.assertNotIn("Follow-ups", answer)
        self.assertTrue(answer.endswith("requested detail."))
        self.assertEqual(split_followups("No list here."), ("No list here.", []))

    def test_slug_and_speech(self):
        self.assertEqual(slug("How is RAG evaluated? (2026)"), "how-is-rag-evaluated-2026")
        self.assertEqual(slug("¿¿"), "answer")
        spoken = strip_for_speech("**Short answer:** LoRA wins [1][2].\n\n| a | b |\n|---|---|\n#### Why\n- It is cheap [3].")
        self.assertNotIn("[", spoken)
        self.assertNotIn("|", spoken)
        self.assertNotIn("#", spoken)
        self.assertIn("LoRA wins.", spoken)


class RecordTests(unittest.TestCase):
    def test_build_record(self):
        rec = record()
        self.assertEqual(rec["id"], "rec00001")
        self.assertEqual(len(rec["followups"]), 3)
        self.assertEqual([s["n"] for s in rec["sources"]], [1, 2])
        self.assertEqual(rec["sources"][0]["year"], "2021")
        self.assertEqual(byline(rec["sources"][0]), "Edward Hu et al. · 2021")
        self.assertLessEqual(len(rec["sources"][0]["evidence"]), 1501)
        context = sources_context(rec)
        self.assertIn("[1] LoRA", context)
        self.assertIn("ANSWER:", context)

    def test_variant_shares_sources(self):
        parent = record()
        child = variant_record(parent, "simpler", "Short answer: easy [1].")
        self.assertEqual(child["sources"], parent["sources"])
        self.assertEqual(child["parent_id"], "rec00001")
        self.assertNotEqual(child["id"], parent["id"])


class ExportTests(unittest.TestCase):
    def test_bibtex_keys_and_dedup(self):
        sources = record()["sources"]
        bib = export.bibtex(sources + [dict(sources[0])])  # duplicate paper
        self.assertEqual(bib.count("@misc{"), 2)
        self.assertIn("@misc{hu2021lora,", bib)
        self.assertIn("@misc{houlsby2019parameterefficient,", bib)
        self.assertIn("author = {Edward Hu and Yelong Shen and Phillip Wallis}", bib)
        self.assertIn("eprint = {2106.09685v2}", bib)

    def test_bibtex_key_collisions_get_suffixes(self):
        same = {"authors": "Ann Lee", "year": "2026", "title": "The Study"}
        bib = export.bibtex([dict(same, paper_id="1"), dict(same, paper_id="2")])
        self.assertIn("{lee2026study,", bib)
        self.assertIn("{lee2026studyb,", bib)
        self.assertEqual(export.bibtex_key({"title": "", "authors": ""}), "anonndpaper")

    def test_markdown_export(self):
        rec = record()
        text = export.answer_markdown(rec)
        self.assertTrue(text.startswith("# Compare LoRA and adapters?"))
        self.assertIn("## References", text)
        self.assertIn("1. LoRA: Low-Rank Adaptation of Large Language Models. Edward Hu, Yelong Shen, Phillip Wallis (2021). "
                      "https://arxiv.org/abs/2106.09685v2", text)
        self.assertEqual(export.markdown_filename("How is RAG evaluated?"), "how-is-rag-evaluated.md")
        self.assertLessEqual(len(export.markdown_filename("word " * 40)), 63)


class LLMJsonTests(unittest.TestCase):
    def setUp(self):
        llm_extras._cached.clear()

    def test_valid_reply_is_cached(self):
        with mock.patch.object(llm_extras, "_complete", return_value='{"answer_md": "Short answer: yes it works [1]."}') as call:
            first = llm_extras.llm_json("p", llm_extras.Rewrite, "ctx", feature="simpler", answer_id="a1")
            second = llm_extras.llm_json("p", llm_extras.Rewrite, "ctx", feature="simpler", answer_id="a1")
        self.assertEqual(first, second)
        self.assertEqual(call.call_count, 1)

    def test_invalid_json_retries_once(self):
        replies = iter(["not json", '{"followups": ["Why?", "How?"]}'])
        with mock.patch.object(llm_extras, "_complete", side_effect=lambda messages: next(replies)) as call:
            data = llm_extras.llm_json("p", llm_extras.Followups, "ctx", feature="followups", answer_id="a2")
        self.assertEqual(data["followups"], ["Why?", "How?"])
        self.assertEqual(call.call_count, 2)
        retry_messages = call.call_args_list[1].args[0]
        self.assertIn("invalid", retry_messages[-1]["content"])

    def test_two_invalid_replies_raise_friendly_error(self):
        with mock.patch.object(llm_extras, "_complete", return_value='{"answer_md": ""}'):
            with self.assertRaises(llm_extras.LLMExtrasError) as caught:
                llm_extras.llm_json("p", llm_extras.Rewrite, "ctx", feature="deeper", answer_id="a3")
        self.assertIn("try again", str(caught.exception))

    def test_cache_key_depends_on_params(self):
        self.assertNotEqual(llm_extras.cache_key("study", "a", {"difficulty": "hard"}), llm_extras.cache_key("study", "a"))

    def test_provider_switch(self):
        with mock.patch.dict(os.environ, {"EXTRAS_LLM_PROVIDER": "groq", "EXTRAS_LLM_MODEL": ""}):
            config = llm_extras.provider_config()
        self.assertEqual(config["name"], "groq")
        self.assertIn("groq.com", config["base_url"])
        with mock.patch.dict(os.environ, {"EXTRAS_LLM_PROVIDER": "nope"}):
            self.assertEqual(llm_extras.provider_config()["name"], "openai")

    def test_missing_key_is_a_friendly_error(self):
        with mock.patch.dict(os.environ, {"EXTRAS_LLM_PROVIDER": "gemini", "GEMINI_API_KEY": ""}):
            with self.assertRaises(llm_extras.LLMExtrasError):
                llm_extras._complete([])


class FeaturedFileTests(unittest.TestCase):
    def test_featured_answers_have_no_internals(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "featured.json")
        featured = json.load(open(path, encoding="utf-8"))
        self.assertGreaterEqual(len(featured), 6)
        for item in featured:
            self.assertTrue(item["teaser"])
            self.assertTrue(item["sources"])
            visible = " ".join([item["question"], item["teaser"], item["answer_md"], *item["followups"]])
            self.assertIsNone(re.search(r"\b(cs|stat|eess)\.[A-Z]{2}\b|arxiv|corpus", visible, re.IGNORECASE), item["id"])


if __name__ == "__main__":
    unittest.main()
