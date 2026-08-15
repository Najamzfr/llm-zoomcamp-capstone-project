import csv
import tempfile
import unittest
from pathlib import Path

from research_assistant.rag import HybridSearch, SQLiteFTSBackend, build_search_backend
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


if __name__ == "__main__":
    unittest.main()
