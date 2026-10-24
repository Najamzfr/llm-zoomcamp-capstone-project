"""Suggested follow-up questions (F1).

They normally come free with the answer (the answer prompt ends with a "Follow-ups:" list, parsed by
presentation.split_followups). Records without them get one cheap cached llm_json call.
"""

from .llm_extras import Followups, LLMExtrasError, llm_json
from .records import sources_context

PROMPT = (
    "Suggest three short follow-up questions (at most 12 words each) that a reader of this answer "
    "would naturally ask next. Make them specific to the answer's content. Do not mention papers, "
    "sources, a corpus or a dataset. Return {\"followups\": [three strings]}."
)


def ensure(record):
    """Return the record's follow-ups, generating them once if the answer had none."""
    if record.get("followups"):
        return record["followups"]
    try:
        data = llm_json(PROMPT, Followups, sources_context(record), feature="followups", answer_id=record["id"])
    except LLMExtrasError:
        return []
    record["followups"] = [question.strip() for question in data["followups"] if question.strip()][:3]
    return record["followups"]
