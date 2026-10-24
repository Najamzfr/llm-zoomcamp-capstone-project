import os

os.environ["LANGSMITH_TRACING"] = "false"  # never send test runs to LangSmith

import csv
import tempfile
import unittest
from pathlib import Path

from research_assistant.presentation import (
    abstract_excerpt,
    bibtex,
    clean_excerpt,
    format_date,
    link_citations,
    period_start,
    short_title,
    split_short_answer,
)
from research_assistant.rag import HybridSearch
from scripts.ingest import REQUIRED_COLUMNS, ingest


class PresentationTests(unittest.TestCase):
    def test_split_short_answer_variants(self):
        for lead in ("Short answer:", "**Short answer:**", "**Short answer.**", "#### Short answer"):
            short, rest = split_short_answer(f"{lead} LoRA wins [1].\n\n#### Details\nMore [2].")
            self.assertEqual(short, "LoRA wins [1].")
            self.assertTrue(rest.startswith("#### Details"))
        self.assertEqual(split_short_answer("No lead here."), ("", "No lead here."))

    def test_link_citations_chips_and_escaping(self):
        out = link_citations("A [1][2], B [3, 1], bad [9], link [1](http://x) <b>x</b>", "m0", 3)
        self.assertEqual(out.count('class="cite"'), 4)
        self.assertIn('href="#m0-2"', out)
        self.assertIn("[9]", out)
        self.assertIn("[1](http://x)", out)
        self.assertIn("&lt;b&gt;", out)

    def test_excerpt_title_and_dates(self):
        text = "Title:\nT\n\nAbstract:\n" + "word " * 100
        excerpt = abstract_excerpt(text, limit=40)
        self.assertTrue(excerpt.endswith("…"))
        self.assertLessEqual(len(excerpt), 41)
        self.assertEqual(short_title("TLoRA+: A Low-Rank Method"), "TLoRA+")
        self.assertEqual(short_title("Dynamic expert pruning for adaptation"), "Dynamic expert pruning…")
        self.assertEqual(format_date("2026-04-05"), "5 Apr 2026")
        self.assertEqual(period_start("Last month", "2026-04-29"), "2026-03-30")
        self.assertIsNone(period_start("All", "2026-04-29"))

    def test_clean_excerpt_skips_pdf_debris(self):
        noisy = ("Preprint. Under review. Outputs Inputs Pretrained Matrix Figure 1: Structural view. "
                 "IEEE TRANSACTIONS ON SIGNAL PROCESSING 6. We show that adaptive rank allocation improves "
                 "accuracy under a fixed parameter budget. More text follows here.")
        self.assertEqual(clean_excerpt(noisy),
                         "We show that adaptive rank allocation improves accuracy under a fixed parameter budget.")
        self.assertEqual(clean_excerpt("Table 3 | 0.91 | 0.88 | 12.4"), "")

    def test_bibtex(self):
        bib = bibtex([{"paper_id": "2604.12345v1", "title": "A {Study}", "authors": "Ann Lee, Bo Chen",
                       "submitted_date": "2026-04-01", "category": "cs.CL"}])
        self.assertIn("@misc{arxiv260412345,", bib)
        self.assertIn("author = {Ann Lee and Bo Chen}", bib)
        self.assertIn("title = {{A Study}}", bib)
        self.assertIn("eprint = {2604.12345v1}", bib)


class StubVector:
    def search(self, query, k=5):
        return [{"paper_id": pid, "score": 1.0, "rank": i} for i, pid in enumerate(["b", "a", "c"], 1)]


class HybridFilterTests(unittest.TestCase):
    def test_category_and_date_filters(self):
        rows = [
            ("a", "Neural search", "cs.CL", "2026-04-20"),
            ("b", "Neural ranking", "cs.CV", "2026-04-21"),
            ("c", "Neural retrieval", "cs.CL", "2025-10-01"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            csv_path, db_path = Path(directory) / "p.csv", Path(directory) / "p.db"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                for pid, title, cat, day in rows:
                    writer.writerow({**{c: "" for c in REQUIRED_COLUMNS}, "paper_id": pid, "title": title,
                                     "abstract": "neural", "num_authors": "1", "primary_category": cat,
                                     "submitted_date": day})
            ingest(csv_path, db_path)
            search = HybridSearch(db_path=db_path, vector_search=StubVector(), candidate_k=5)
            self.assertEqual(len(search.search("neural", 5)), 3)
            only_cl = search.search("neural", 5, filters={"categories": ["cs.CL"]})
            self.assertEqual({p["paper_id"] for p in only_cl}, {"a", "c"})
            recent_cl = search.search("neural", 5, filters={"categories": ["cs.CL"], "since": "2026-01-01"})
            self.assertEqual([p["paper_id"] for p in recent_cl], ["a"])
            self.assertEqual(search.last_candidate_count, 1)


if __name__ == "__main__":
    unittest.main()
