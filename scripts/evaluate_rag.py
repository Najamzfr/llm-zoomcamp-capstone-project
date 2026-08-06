"""Generate and judge abstract-based RAG answers. No command-line parsing."""

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from research_assistant.paths import EVALUATION_DIR
from research_assistant.rag import (
    MODEL,
    ResearchAssistant,
    build_hybrid_search,
    calculate_usage,
    load_papers,
    merge_usages,
)

GROUND_TRUTH_PATH = EVALUATION_DIR / "ground_truth_test.csv"
ANSWERS_PATH = EVALUATION_DIR / "rag_answers.csv"
EVALUATIONS_PATH = EVALUATION_DIR / "rag_evaluations.csv"
LIMIT = 10  # Use None for the complete evaluation.
WORKERS = 4
RUN_GENERATION = True
RUN_JUDGE = True

JUDGE_PROMPT = """
You are evaluating a research-paper RAG answer. The reference evidence contains
abstracts independently labelled relevant to the question. Mark the answer
"good" only when it directly answers the question, captures the important
evidence, stays consistent with the abstracts, and does not invent material
claims. Concise synthesis is preferred over repeating every abstract. Return
JSON only: {"reasoning": "...", "score": "good" or "bad"}
""".strip()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, records):
    if not records:
        raise ValueError("There are no records to save.")
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def ground_truth_ids(record):
    return [
        paper_id.strip()
        for paper_id in record["relevant_paper_ids"].split(";")
        if paper_id.strip()
    ]


def reference_evidence(record):
    ids = ground_truth_ids(record)
    papers = load_papers(ids)
    found = {paper["paper_id"] for paper in papers}
    missing = [paper_id for paper_id in ids if paper_id not in found]
    if missing:
        raise ValueError(f"Ground-truth papers missing from arxiv.db: {missing}")
    evidence = "\n\n".join(
        f"[{paper['rank']}] {paper['title']} ({paper['paper_id']})\n{paper['text']}"
        for paper in papers
    )
    return papers, evidence


def parallel_map(records, function):
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for number, result in enumerate(pool.map(function, records), start=1):
            results.append(result)
            print(f"Processed: {number}/{len(records)}", end="\r", flush=True)
    print()
    return results


def json_call(instructions, prompt):
    from openai import OpenAI

    client = OpenAI()
    usages = []
    last_error = None
    for _ in range(3):
        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "developer", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=400,
        )
        usages.append(calculate_usage(response, MODEL))
        try:
            match = re.search(r"\{.*\}", response.output_text, flags=re.DOTALL)
            if not match:
                raise json.JSONDecodeError("No JSON object", response.output_text, 0)
            return json.loads(match.group(0)), merge_usages(usages)
        except json.JSONDecodeError as error:
            last_error = error
    raise ValueError(
        "Judge did not return valid JSON after three attempts."
    ) from last_error


def generate_answers():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before generating answers.")
    records = read_csv(GROUND_TRUTH_PATH)
    if LIMIT is not None:
        records = records[:LIMIT]
    hybrid = build_hybrid_search()

    def generate(record):
        reference_papers, evidence = reference_evidence(record)
        assistant = ResearchAssistant(hybrid_search=hybrid, search_method="hybrid")
        retrieved = assistant.search(record["question"])
        answer = assistant.llm(assistant.build_prompt(record["question"], retrieved))
        return {
            "question": record["question"],
            "reference_paper_ids": ";".join(
                paper["paper_id"] for paper in reference_papers
            ),
            "reference_evidence": evidence,
            "retrieved_paper_ids": ";".join(paper["paper_id"] for paper in retrieved),
            "answer_llm": answer,
            "usage": json.dumps(assistant.last_usage),
        }

    answers = parallel_map(records, generate)
    write_csv(ANSWERS_PATH, answers)
    print(f"Saved {len(answers)} answers to {ANSWERS_PATH}")
    print_usage(answers)
    return answers


def judge_answers():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before judging answers.")
    answers = read_csv(ANSWERS_PATH)

    def judge(record):
        result, usage = json_call(
            JUDGE_PROMPT,
            f"Question:\n{record['question']}\n\n"
            f"Reference evidence:\n{record['reference_evidence']}\n\n"
            f"RAG answer:\n{record['answer_llm']}",
        )
        score = result.get("score")
        if score not in {"good", "bad"}:
            raise ValueError(f"Invalid judge score: {score}")
        return {
            "question": record["question"],
            "score": score,
            "reasoning": result.get("reasoning", ""),
            "usage": json.dumps(usage),
        }

    evaluations = parallel_map(answers, judge)
    write_csv(EVALUATIONS_PATH, evaluations)
    good = sum(record["score"] == "good" for record in evaluations)
    print(f"Good answers: {good}/{len(evaluations)} = {good / len(evaluations):.1%}")
    print(f"Saved judgments to {EVALUATIONS_PATH}")
    print_usage(evaluations)
    return evaluations


def print_usage(records):
    usages = [json.loads(record["usage"]) for record in records]
    total = merge_usages(usages)
    print(
        f"Tokens: {total['total_tokens']:,} | "
        f"Estimated cost: ${total['estimated_cost_usd']:.6f}"
    )


def main():
    if RUN_GENERATION:
        generate_answers()
    if RUN_JUDGE:
        judge_answers()


if __name__ == "__main__":
    main()
