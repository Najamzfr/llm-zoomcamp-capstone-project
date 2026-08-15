"""Build exact embeddings and an HNSW approximate-nearest-neighbor index."""

import time

import faiss
import numpy as np
from research_assistant.config import settings
from research_assistant.paths import SOURCE_DATA_PATH
from scripts.ingest import load_rows
from sentence_transformers import SentenceTransformer

VECTOR_MODEL = settings.embedding_model
EXACT_INDEX_PATH = settings.vector_metadata_path
ANN_INDEX_PATH = settings.vector_index_path
HNSW_M = settings.hnsw_m
HNSW_EF_CONSTRUCTION = settings.hnsw_ef_construction
HNSW_EF_SEARCH = settings.hnsw_ef_search


def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m {seconds}s"


def document_text(row):
    return (
        f"Title: {row['title']}\n"
        f"Primary category: {row['primary_category']}\n"
        f"Categories: {row['all_categories']}\n"
        f"Abstract: {row['abstract']}\n"
        f"Authors: {row['authors']}"
    )


def ingest_vectors(data_path=SOURCE_DATA_PATH):
    started = time.perf_counter()
    EXACT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANN_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Stage 1/4: Loading papers...", flush=True)
    rows = load_rows(data_path)
    documents = [document_text(row) for row in rows]
    print(f"Loaded {len(rows):,} papers.", flush=True)

    print(f"\nStage 2/4: Loading {VECTOR_MODEL}...", flush=True)
    model = SentenceTransformer(VECTOR_MODEL)
    encode = getattr(model, "encode_document", model.encode)

    print(f"\nStage 3/4: Embedding {len(documents):,} papers...", flush=True)
    embeddings = encode(
        documents,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    print("\nStage 4/4: Building exact and HNSW indexes...", flush=True)
    paper_ids = np.asarray([row["paper_id"] for row in rows])
    np.savez_compressed(
        EXACT_INDEX_PATH,
        embeddings=embeddings,
        paper_ids=paper_ids,
        model=np.asarray(VECTOR_MODEL),
    )

    dimensions = embeddings.shape[1]
    ann = faiss.IndexHNSWFlat(dimensions, HNSW_M)
    ann.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    ann.hnsw.efSearch = HNSW_EF_SEARCH
    ann.add(embeddings)
    faiss.write_index(ann, ANN_INDEX_PATH)

    elapsed = time.perf_counter() - started
    result = {
        "rows": len(rows),
        "dimensions": dimensions,
        "embedding_model": VECTOR_MODEL,
        "exact_index": EXACT_INDEX_PATH,
        "ann_index": ANN_INDEX_PATH,
        "hnsw_m": HNSW_M,
        "ef_construction": HNSW_EF_CONSTRUCTION,
        "ef_search": HNSW_EF_SEARCH,
        "elapsed_seconds": round(elapsed, 2),
    }
    print(f"Completed in {format_time(elapsed)}")
    print(f"Exact index: {EXACT_INDEX_PATH.stat().st_size / 1024**2:.2f} MB")
    print(f"ANN index: {ANN_INDEX_PATH.stat().st_size / 1024**2:.2f} MB")
    return result


def main():
    result = ingest_vectors()
    print("\nSummary:")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
