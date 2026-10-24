"""Build data/featured.json: curated questions answered by the real pipeline (F5).

Run: uv run python scripts/build_featured.py   (needs OPENAI_API_KEY and the search backend
configured as for the app). The app renders these answers on the empty state with no API calls.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.features.records import build_record  # noqa: E402
from research_assistant.presentation import shorten, split_short_answer  # noqa: E402

OUTPUT = ROOT / "data" / "featured.json"
QUESTIONS = [
    "How is retrieval-augmented generation evaluated?",
    "Compare approaches to efficient fine-tuning of large language models",
    "What methods reduce hallucinations in large language models?",
    "Which benchmarks are used to evaluate LLM agents?",
    "How do vision-language models handle spatial reasoning?",
    "What are the main approaches to robot learning from demonstrations?",
    "How are diffusion models made faster at inference time?",
]


def teaser(record):
    short, rest = split_short_answer(record["answer_md"])
    first = (short or rest).split(". ")[0]
    import re

    return shorten(re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", first).strip(), 95)


def main():
    from scripts.ingest import ingest
    from research_assistant.config import settings

    if not settings.database_path.exists():
        ingest()
    from research_assistant.agent import AdaptiveResearchAgent

    agent = AdaptiveResearchAgent(trace_path=None)
    featured = []
    for index, question in enumerate(QUESTIONS, start=1):
        print(f"[{index}/{len(QUESTIONS)}] {question}")
        verified = {}
        result = agent.run(question, on_progress=lambda event, data: verified.update(
            ok=data.get("status") == "pass") if event == "checked" else None)
        record = build_record(result, question, verified=verified.get("ok"), record_id=f"feat{index:02d}")
        record["teaser"] = teaser(record)
        featured.append(record)
    OUTPUT.write_text(json.dumps(featured, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(featured)} answers to {OUTPUT}")


if __name__ == "__main__":
    main()
