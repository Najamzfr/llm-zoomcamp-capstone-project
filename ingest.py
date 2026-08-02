"""Reproducible CSV -> SQLite FTS5 ingestion for the paper catalogue."""

import csv
import sqlite3
import time
from pathlib import Path

REQUIRED_COLUMNS = (
    "paper_id", "title", "abstract", "authors", "num_authors",
    "first_author", "primary_category", "all_categories", "submitted_date",
    "doi", "arxiv_url", "pdf_url",
)


def clean_row(row):
    cleaned = {key: (value or "").strip() for key, value in row.items()}
    cleaned["num_authors"] = int(cleaned.get("num_authors") or 0)
    return cleaned


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        rows = []
        seen = set()
        for raw in reader:
            row = clean_row(raw)
            if not row["paper_id"] or not (row["title"] or row["abstract"]):
                continue
            if row["paper_id"] in seen:
                continue
            seen.add(row["paper_id"])
            rows.append(row)
    return rows


def create_table(conn):
    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute(
        """
        CREATE VIRTUAL TABLE documents USING fts5(
            paper_id UNINDEXED, title, text,
            authors UNINDEXED, num_authors UNINDEXED,
            first_author UNINDEXED, primary_category UNINDEXED,
            all_categories UNINDEXED, submitted_date UNINDEXED,
            doi UNINDEXED, arxiv_url UNINDEXED, pdf_url UNINDEXED
        )
        """
    )


def ingest(data_path="data.csv", db_path="arxiv.db", logger=None):
    started = time.perf_counter()
    rows = load_rows(data_path)
    if logger:
        logger.event("ingestion.loaded", rows=len(rows), data_path=str(data_path))

    with sqlite3.connect(db_path) as conn:
        create_table(conn)
        conn.executemany(
            """
            INSERT INTO documents (
                paper_id, title, text, authors, num_authors, first_author,
                primary_category, all_categories, submitted_date,
                doi, arxiv_url, pdf_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["paper_id"], row["title"],
                    f"Title:\n{row['title']}\n\nAbstract:\n{row['abstract']}",
                    row["authors"], row["num_authors"], row["first_author"],
                    row["primary_category"], row["all_categories"],
                    row["submitted_date"], row["doi"],
                    row["arxiv_url"], row["pdf_url"],
                )
                for row in rows
            ],
        )

    result = {"rows": len(rows), "database": str(db_path), "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
    if logger:
        logger.event("ingestion.completed", **result)
    return result


if __name__ == "__main__":
    print(ingest())