"""Tune RRF on validation data only and save the selected configuration."""

import csv
import json
from itertools import product

from rag import VectorSearch, retrieve


VALIDATION_PATH = "ground_truth.csv"
CONFIG_PATH = "rrf_config.json"
K = 5
MAX_CANDIDATES = 50


def load_ground_truth(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def relevant_ids(record):
    return {value.strip() for value in record["relevant_paper_ids"].split(";") if value.strip()}


def cache_rankings(records):
    vector = VectorSearch()
    cached = []
    for number, record in enumerate(records, start=1):
        question = record["question"]
        cached.append({
            "relevant": relevant_ids(record),
            "text": [paper["paper_id"] for paper in retrieve(question, k=MAX_CANDIDATES)],
            "vector": [paper["paper_id"] for paper in vector.search(question, k=MAX_CANDIDATES)],
        })
        print(f"Caching retrieval results: {number}/{len(records)}", end="\r")
    print()
    return cached


def fused_ids(record, candidate_k, rrf_constant, text_weight):
    scores = {}
    for rank, paper_id in enumerate(record["text"][:candidate_k], start=1):
        scores[paper_id] = scores.get(paper_id, 0.0) + text_weight / (rrf_constant + rank)
    for rank, paper_id in enumerate(record["vector"][:candidate_k], start=1):
        scores[paper_id] = scores.get(paper_id, 0.0) + (1 - text_weight) / (rrf_constant + rank)
    return sorted(scores, key=scores.get, reverse=True)[:K]


def score_config(cached, candidate_k, rrf_constant, text_weight):
    precision = hit_rate = mrr = 0.0
    for record in cached:
        ranked = fused_ids(record, candidate_k, rrf_constant, text_weight)
        relevance = [paper_id in record["relevant"] for paper_id in ranked]
        precision += sum(relevance) / K
        hit_rate += int(any(relevance))
        mrr += next((1 / rank for rank, value in enumerate(relevance, 1) if value), 0)

    count = len(cached)
    metrics = {
        "precision_at_5": precision / count,
        "recall_at_5": precision / count,
        "hit_rate_at_5": hit_rate / count,
        "mrr_at_5": mrr / count,
    }
    # Balanced objective; recall equals precision because each query has five labels.
    objective = (
        metrics["precision_at_5"]
        + metrics["hit_rate_at_5"]
        + metrics["mrr_at_5"]
    ) / 3
    return objective, metrics


def main():
    records = load_ground_truth(VALIDATION_PATH)
    cached = cache_rankings(records)
    best = None

    grid = product(
        [10, 20, 30, 50],
        [10, 30, 60, 90],
        [round(value / 10, 1) for value in range(1, 10)],
    )
    for candidate_k, rrf_constant, text_weight in grid:
        objective, metrics = score_config(cached, candidate_k, rrf_constant, text_weight)
        result = {
            "candidate_k": candidate_k,
            "rrf_constant": rrf_constant,
            "text_weight": text_weight,
            "vector_weight": round(1 - text_weight, 1),
            "objective": objective,
            **metrics,
        }
        if best is None or result["objective"] > best["objective"]:
            best = result

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(best, file, indent=2)

    print("Best validation configuration:")
    print(json.dumps(best, indent=2))
    print(f"Saved to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
