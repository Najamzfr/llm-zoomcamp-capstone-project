"""Upload the existing vector embeddings to a Qdrant collection (no re-embedding)."""

import sys

import numpy as np
from research_assistant.config import settings

BATCH_SIZE = 256


def ingest_qdrant(metadata_path=settings.vector_metadata_path, client=None, force=False):
    if client is None:
        if not settings.qdrant_url:
            raise SystemExit("Set QDRANT_URL (and QDRANT_API_KEY) in .env first.")
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=60,
        )
    from qdrant_client.models import Distance, PointStruct, VectorParams

    data = np.load(metadata_path, allow_pickle=False)
    embeddings, paper_ids = data["embeddings"], data["paper_ids"]
    collection = settings.qdrant_collection

    if not force and client.collection_exists(collection):
        existing = client.count(collection).count
        if existing == len(paper_ids):
            print(f"{collection} already has {existing:,} points; skipping (use --force to reload).")
            return {"collection": collection, "points": existing, "skipped": True}
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

    for start in range(0, len(paper_ids), BATCH_SIZE):
        stop = start + BATCH_SIZE
        client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=int(index),
                    vector=embeddings[index].tolist(),
                    payload={"paper_id": str(paper_ids[index])},
                )
                for index in range(start, min(stop, len(paper_ids)))
            ],
        )
        print(f"Uploaded {min(stop, len(paper_ids)):,}/{len(paper_ids):,}", flush=True)
    return {"collection": collection, "points": len(paper_ids), "dimensions": embeddings.shape[1]}


def main():
    print(ingest_qdrant(force="--force" in sys.argv))


if __name__ == "__main__":
    main()
