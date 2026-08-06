import csv
import time

from research_assistant.paths import EVALUATION_DIR
from research_assistant.rag import ExactVectorSearch, VectorSearch, build_hybrid_search, retrieve

GROUND_TRUTH_PATH = EVALUATION_DIR / "ground_truth_test.csv"
K = 5


def load_ground_truth(path=GROUND_TRUTH_PATH):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def relevant_ids(record):
    return {
        paper_id.strip()
        for paper_id in record["relevant_paper_ids"].split(";")
        if paper_id.strip()
    }


def metrics(records, search):
    precision = recall = hit_rate = mrr = 0.0
    for record in records:
        expected = relevant_ids(record)
        results = search(record["question"])
        relevance = [paper["paper_id"] in expected for paper in results]
        precision += sum(relevance) / len(results) if results else 0
        recall += sum(relevance) / len(expected) if expected else 0
        hit_rate += int(any(relevance))
        mrr += next(
            (1 / rank for rank, value in enumerate(relevance, start=1) if value),
            0,
        )
    count = len(records)
    return {
        "precision": precision / count,
        "recall": recall / count,
        "hit_rate": hit_rate / count,
        "mrr": mrr / count,
    }


def ann_quality(records, exact, ann):
    recall = exact_ms = ann_ms = 0.0
    for record in records:
        question = record["question"]
        started = time.perf_counter()
        exact_ids = {paper["paper_id"] for paper in exact.search(question, K)}
        exact_ms += (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        ann_ids = {paper["paper_id"] for paper in ann.search(question, K)}
        ann_ms += (time.perf_counter() - started) * 1000
        recall += len(exact_ids & ann_ids) / len(exact_ids) if exact_ids else 0

    count = len(records)
    return {
        "ann_recall_at_5": recall / count,
        "exact_latency_ms": exact_ms / count,
        "ann_latency_ms": ann_ms / count,
    }


def print_table(results):
    print("\n+----------------+-------------+----------+------------+-------+")
    print("| Search method  | Precision@5 | Recall@5 | Hit Rate@5 | MRR@5 |")
    print("+----------------+-------------+----------+------------+-------+")
    for name, values in results.items():
        print(
            f"| {name:<14} | {values['precision']:>11.3f} | "
            f"{values['recall']:>8.3f} | {values['hit_rate']:>10.3f} | "
            f"{values['mrr']:>5.3f} |"
        )
    print("+----------------+-------------+----------+------------+-------+")


def main():
    records = load_ground_truth()
    if not records:
        raise ValueError("Ground-truth file is empty.")

    exact = ExactVectorSearch()
    ann = VectorSearch()
    hybrid = build_hybrid_search(vector_search=ann)
    results = {
        "Text (FTS5)": metrics(records, lambda q: retrieve(q, K)),
        "Exact vector": metrics(records, lambda q: exact.search(q, K)),
        "HNSW ANN": metrics(records, lambda q: ann.search(q, K)),
        "Hybrid (RRF)": metrics(records, lambda q: hybrid.search(q, K)),
    }
    quality = ann_quality(records, exact, ann)

    print(f"Questions: {len(records)}")
    print_table(results)
    print(f"ANN Recall@5 vs exact: {quality['ann_recall_at_5']:.3f}")
    print(f"Exact latency/query: {quality['exact_latency_ms']:.2f} ms")
    print(f"ANN latency/query: {quality['ann_latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
