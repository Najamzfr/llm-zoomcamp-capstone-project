"""Opt-in paid RAG baseline; never called by the default test suite."""

import json
import os

from research_assistant.config import PROJECT_ROOT
from research_assistant.rag import merge_usages
from scripts.evaluate_rag import generate_answers, judge_answers


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running the RAG baseline.")
    answers = generate_answers()
    evaluations = judge_answers()
    usage = merge_usages(
        json.loads(record["usage"]) for record in answers + evaluations
    )
    good = sum(record["score"] == "good" for record in evaluations)
    report = PROJECT_ROOT / "docs" / "rag-baseline.md"
    report.write_text(
        "# RAG baseline\n\n"
        f"- Questions: {len(evaluations)}\n"
        f"- Judge good rate: {good / len(evaluations):.1%}\n"
        f"- Model: {usage['model']}\n"
        f"- Tokens: {usage['total_tokens']:,}\n"
        f"- Estimated cost: ${usage['estimated_cost_usd']:.6f}\n",
        encoding="utf-8",
    )
    print(f"Saved {report}")


if __name__ == "__main__":
    main()
