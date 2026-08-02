"""Generate and judge adaptive-agent answers and tool trajectories."""

import json
import os

from agent import AdaptiveResearchAgent
from evaluate_rag import (
    json_call,
    read_csv,
    reference_evidence,
    write_csv,
)
from paper_tools import MAX_CHUNKS, MAX_FULL_TEXT_TOKENS, MAX_PAPERS
from rag import merge_usages

GROUND_TRUTH_PATH = "ground_truth_test.csv"
ANSWERS_PATH = "agent_answers.csv"
EVALUATIONS_PATH = "agent_evaluations.csv"
LIMIT = 10  # Use None for the complete evaluation.
RUN_GENERATION = True
RUN_JUDGE = True

JUDGE_PROMPT = f"""
Evaluate both the adaptive research agent's final answer and its tool
trajectory.

Answer quality:
- directly answers the question;
- is supported by the reference abstracts and any retrieved full-text evidence;
- contains no invented material claims.

Trajectory quality:
- the initial search query is relevant;
- duplicate or unnecessary searches are avoided;
- full-text escalation occurs only when abstracts lack required detail;
- no more than {MAX_PAPERS} papers, {MAX_CHUNKS} chunks, or
  {MAX_FULL_TEXT_TOKENS} full-text context tokens are used;
- selected papers and chunk queries address the missing information;
- one escalation cycle is the maximum.

Return JSON only with keys "answer_reasoning", "answer_score",
"trajectory_reasoning", and "trajectory_score". Scores must be "good" or "bad".
""".strip()


def generate_answers():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running agent evaluation.")
    records = read_csv(GROUND_TRUTH_PATH)
    if LIMIT is not None:
        records = records[:LIMIT]
    agent = AdaptiveResearchAgent(trace_path="evaluation_agent_traces.jsonl")
    answers = []

    for number, record in enumerate(records, start=1):
        reference_papers, evidence = reference_evidence(record)
        result = agent.run(record["question"])
        deterministic = trajectory_metrics(
            result["trajectory"],
            {paper["paper_id"] for paper in reference_papers},
        )
        answers.append(
            {
                "question": record["question"],
                "reference_paper_ids": ";".join(
                    paper["paper_id"] for paper in reference_papers
                ),
                "reference_evidence": evidence,
                "answer_agent": result["answer"],
                "answer_stage": result["answer_stage"],
                "full_text_evidence": result["full_text_evidence"],
                "tool_calls": json.dumps(result["trajectory"], ensure_ascii=False),
                "deterministic_metrics": json.dumps(deterministic),
                "usage_by_stage": json.dumps(result["usage_by_stage"]),
                "usage": json.dumps(result["usage"]),
            }
        )
        print(f"Processed: {number}/{len(records)}", end="\r", flush=True)
    print()

    write_csv(ANSWERS_PATH, answers)
    print(f"Saved {len(answers)} agent answers to {ANSWERS_PATH}")
    print_usage(answers)
    return answers


def trajectory_metrics(trajectory, expected_ids):
    searches = [call for call in trajectory if call["name"] == "search_papers"]
    queries = [call["arguments"]["query"].strip().lower() for call in searches]
    retrieved = searches[0].get("retrieved_paper_ids", []) if searches else []
    full_text_calls = [
        call for call in trajectory if call["name"] == "retrieve_full_text"
    ]
    successful_full_text = [call for call in full_text_calls if "error" not in call]
    downloaded = {
        paper_id
        for call in successful_full_text
        for paper_id in call.get("downloaded_paper_ids", [])
    }
    selected = {
        paper_id
        for call in successful_full_text
        for paper_id in call.get("arguments", {}).get("paper_ids", [])
    }
    retrieved_chunks = {
        chunk_id
        for call in successful_full_text
        for chunk_id in call.get("retrieved_chunk_ids", [])
    }
    context_tokens = max(
        [call.get("context_tokens", 0) for call in successful_full_text] or [0]
    )
    return {
        "search_calls": len(searches),
        "duplicate_searches": len(queries) - len(set(queries)),
        "retrieval_precision_at_5": (
            len(set(retrieved) & expected_ids) / len(retrieved) if retrieved else 0
        ),
        "retrieval_recall_at_5": (
            len(set(retrieved) & expected_ids) / len(expected_ids)
            if expected_ids
            else 0
        ),
        "full_text_escalations": len(full_text_calls),
        "papers_downloaded": len(downloaded),
        "papers_selected": len(selected),
        "chunks_retrieved": len(retrieved_chunks),
        "context_tokens": context_tokens,
        "within_cost_controls": (
            len(full_text_calls) <= 1
            and len(selected) <= MAX_PAPERS
            and len(retrieved_chunks) <= MAX_CHUNKS
            and context_tokens <= MAX_FULL_TEXT_TOKENS
        ),
    }


def judge_answers():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before judging agent answers.")
    answers = read_csv(ANSWERS_PATH)
    evaluations = []

    for number, record in enumerate(answers, start=1):
        result, usage = json_call(
            JUDGE_PROMPT,
            f"Question:\n{record['question']}\n\n"
            f"Reference abstracts:\n{record['reference_evidence']}\n\n"
            f"Full-text evidence used:\n{record['full_text_evidence'] or 'None'}\n\n"
            f"Agent answer:\n{record['answer_agent']}\n\n"
            f"Tool trajectory:\n{record['tool_calls']}\n\n"
            f"Deterministic metrics:\n{record['deterministic_metrics']}",
        )
        answer_score = result.get("answer_score")
        trajectory_score = result.get("trajectory_score")
        if answer_score not in {"good", "bad"}:
            raise ValueError(f"Invalid answer score: {answer_score}")
        if trajectory_score not in {"good", "bad"}:
            raise ValueError(f"Invalid trajectory score: {trajectory_score}")
        evaluations.append(
            {
                "question": record["question"],
                "answer_score": answer_score,
                "answer_reasoning": result.get("answer_reasoning", ""),
                "trajectory_score": trajectory_score,
                "trajectory_reasoning": result.get("trajectory_reasoning", ""),
                "answer_stage": record["answer_stage"],
                "deterministic_metrics": record["deterministic_metrics"],
                "usage": json.dumps(usage),
            }
        )
        print(f"Judged: {number}/{len(answers)}", end="\r", flush=True)
    print()

    write_csv(EVALUATIONS_PATH, evaluations)
    answer_good = sum(row["answer_score"] == "good" for row in evaluations)
    trajectory_good = sum(row["trajectory_score"] == "good" for row in evaluations)
    print(f"Good answers: {answer_good}/{len(evaluations)}")
    print(f"Good trajectories: {trajectory_good}/{len(evaluations)}")
    print(f"Saved judgments to {EVALUATIONS_PATH}")
    print_usage(evaluations)
    return evaluations


def print_usage(records):
    total = merge_usages([json.loads(record["usage"]) for record in records])
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