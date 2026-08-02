import sqlite3

from agent import AdaptiveResearchAgent
from openai import OpenAIError


def main():
    query = input("Prompt: ").strip()
    if not query:
        print("Please enter a research question.")
        return

    try:
        result = AdaptiveResearchAgent().run(query)
    except (
        sqlite3.Error,
        FileNotFoundError,
        OpenAIError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Could not answer the question: {error}")
        return

    print(f"\nAnswer:\n{result['answer']}\n\nRetrieved papers:")
    for paper in result["papers"]:
        print(f"[{paper['rank']}] {paper['title']}")
        print(
            f"    {paper['submitted_date']} | {paper['category']} | {paper['arxiv_url']}"
        )

    usage = result["usage"]
    print(f"\nAnswer source: {result['answer_stage']}")
    print(
        f"Input: {usage['input_tokens']} | Output: {usage['output_tokens']} | "
        f"Total: {usage['total_tokens']} | "
        f"Estimated cost: ${usage['estimated_cost_usd']:.8f}"
    )


if __name__ == "__main__":
    main()
