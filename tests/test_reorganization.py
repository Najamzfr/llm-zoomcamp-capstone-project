import os

os.environ["LANGSMITH_TRACING"] = "false"  # never send test runs to LangSmith

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from ingest import ingest
from research_assistant import paths
from scripts.ingest import REQUIRED_COLUMNS


class ReorganizationTests(unittest.TestCase):
    def test_project_paths_are_rooted_and_source_exists(self):
        self.assertEqual(paths.SOURCE_DATA_PATH.parent.name, "source")
        self.assertEqual(paths.DATABASE_PATH.parent.name, "indexes")
        self.assertEqual(paths.CONFIG_PATH.parent.name, "config")
        self.assertTrue(paths.SOURCE_DATA_PATH.is_file())
        self.assertTrue(paths.CONFIG_PATH.is_file())

    def test_root_ingest_entry_point_accepts_explicit_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "papers.csv"
            database_path = Path(directory) / "papers.db"
            with data_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerow({"paper_id": "1", "title": "Test paper", "abstract": "Test abstract", "authors": "Test Author", "num_authors": "1", "first_author": "Test Author", "primary_category": "cs.AI", "all_categories": "cs.AI", "submitted_date": "2026-01-01", "doi": "", "arxiv_url": "https://arxiv.org/abs/1", "pdf_url": "https://arxiv.org/pdf/1"})

            result = ingest(data_path, database_path)
            self.assertEqual(result["rows"], 1)
            with sqlite3.connect(database_path) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM documents").fetchone()[0], 1)
            connection.close()
