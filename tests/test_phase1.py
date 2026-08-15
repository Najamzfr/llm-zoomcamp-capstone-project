import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_assistant.config import PROJECT_ROOT, _path
from research_assistant.rag import HybridSearch, SQLiteFTSBackend, build_hybrid_search, build_search_backend
from scripts.ingest import REQUIRED_COLUMNS, ingest


class StubVectorSearch:
    def search(self, query, k=5):
        return [{"paper_id": "two", "score": 0.9, "rank": 1}]


class PhaseOneTests(unittest.TestCase):
    def _write_csv(self, path):
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(
                [
                    {"paper_id": "one", "title": "Neural search", "abstract": "ranking", "authors": "A", "num_authors": "1", "first_author": "A", "primary_category": "cs.IR", "all_categories": "cs.IR", "submitted_date": "2026-01-01", "doi": "", "arxiv_url": "", "pdf_url": ""},
                    {"paper_id": "two", "title": "Vector retrieval", "abstract": "embeddings", "authors": "B", "num_authors": "1", "first_author": "B", "primary_category": "cs.IR", "all_categories": "cs.IR", "submitted_date": "2026-01-02", "doi": "", "arxiv_url": "", "pdf_url": ""},
                ]
            )

    def test_ingest_creates_parent_directory_and_sqlite_backend_searches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "papers.csv"
            db_path = root / "new" / "indexes" / "papers.db"
            self._write_csv(csv_path)
            self.assertEqual(ingest(csv_path, db_path)["rows"], 2)
            self.assertTrue(db_path.is_file())
            self.assertEqual(SQLiteFTSBackend(db_path).search("neural", 1)[0]["paper_id"], "one")

    def test_hybrid_backend_preserves_rrf_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path, db_path = root / "papers.csv", root / "papers.db"
            self._write_csv(csv_path)
            ingest(csv_path, db_path)
            backend = HybridSearch(
                db_path=db_path,
                vector_search=StubVectorSearch(),
                candidate_k=2,
                rrf_constant=0,
                text_weight=0,
                vector_weight=1,
            )
            self.assertEqual(backend.search("neural", 2)[0]["paper_id"], "two")

    def test_backend_factory_accepts_only_local_backends(self):
        self.assertIsInstance(build_search_backend("sqlite_fts"), SQLiteFTSBackend)
        with self.assertRaisesRegex(ValueError, "SEARCH_BACKEND"):
            build_search_backend("qdrant")

    def test_relative_config_paths_are_rooted_and_absolute_paths_are_preserved(self):
        self.assertEqual(_path("UNSET_TEST_PATH", "data/test.db"), PROJECT_ROOT / "data/test.db")
        absolute = Path(tempfile.gettempdir()) / "absolute-test.db"
        with patch.dict(os.environ, {"TEST_PATH": str(absolute)}):
            self.assertEqual(_path("TEST_PATH", "data/test.db"), absolute)

    def test_rrf_environment_values_override_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path, db_path = root / "papers.csv", root / "papers.db"
            config_path = root / "rrf.json"
            self._write_csv(csv_path)
            ingest(csv_path, db_path)
            config_path.write_text(
                '{"candidate_k": 2, "rrf_constant": 9, "text_weight": 0.9, "vector_weight": 0.1}',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"RAG_RRF_CONSTANT": "7"}):
                backend = build_hybrid_search(
                    db_path=db_path,
                    config_path=config_path,
                    vector_search=StubVectorSearch(),
                )
            self.assertEqual(backend.candidate_k, 2)
            self.assertEqual(backend.rrf_constant, 7)
            self.assertEqual(backend.text_weight, 0.9)


if __name__ == "__main__":
    unittest.main()
