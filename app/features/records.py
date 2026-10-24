"""The stored answer record shared by history, the gallery, shares, collections and exports.

    {id, question, answer_md, followups, sources: [{n, title, authors, year, url, paper_id,
     excerpt, evidence, upload?, page?}], created_at, kind, parent_id?, verified}

`evidence` is the text the answer was built from (abstract or best full-text excerpt, capped),
so every extra feature can work from the answer and its sources alone (rule R6).
"""

import secrets
import string
from datetime import datetime, timezone

from research_assistant.presentation import abstract_excerpt, clean_excerpt, shorten, split_followups

EVIDENCE_CHARS = 1500
ID_ALPHABET = string.ascii_letters + string.digits


def new_id(length=8):
    return "".join(secrets.choice(ID_ALPHABET) for _ in range(length))


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _evidence(text):
    text = text or ""
    if "Abstract:" in text:
        text = text.split("Abstract:", 1)[1]
    return shorten(text, EVIDENCE_CHARS)


def build_record(result, question, verified=None, record_id=None):
    """Record from an AdaptiveResearchAgent result (papers, excerpts, answer)."""
    answer, followups = split_followups(result.get("answer", ""))
    excerpts = result.get("excerpts") or {}
    sources = []
    for number, paper in enumerate(result.get("papers", []), start=1):
        chunk = excerpts.get(paper.get("paper_id"))
        upload = bool(paper.get("upload"))
        sources.append({
            "n": number,
            "title": paper.get("title") or "Untitled paper",
            "authors": paper.get("authors") or "",
            "year": str(paper.get("submitted_date") or "")[:4],
            "url": paper.get("arxiv_url") or paper.get("pdf_url") or "",
            "paper_id": paper.get("paper_id") or "",
            "excerpt": (clean_excerpt(chunk, 150) if chunk else "") or (
                clean_excerpt(paper.get("text"), 150) if upload else abstract_excerpt(paper.get("text"), 150)),
            "evidence": _evidence(chunk or paper.get("text")),
            **({"upload": True, "page": paper.get("page")} if upload else {}),
        })
    return {
        "id": record_id or new_id(),
        "question": question,
        "answer_md": answer,
        "followups": followups,
        "sources": sources,
        "created_at": now_iso(),
        "kind": "answer",
        "verified": verified,
    }


def variant_record(parent, kind, answer_md):
    """A 'Simpler version' / 'Deeper version' of `parent` that shares its sources."""
    return {
        "id": new_id(),
        "question": parent["question"],
        "answer_md": answer_md,
        "followups": [],
        "sources": parent["sources"],
        "created_at": now_iso(),
        "kind": kind,
        "parent_id": parent["id"],
        "verified": None,
    }


def sources_context(record, with_answer=True):
    """Numbered evidence (plus the answer) for extra-feature prompts — nothing else (R6)."""
    parts = []
    if with_answer:
        parts.append(f"QUESTION: {record['question']}\n\nANSWER:\n{record['answer_md']}")
    for source in record.get("sources", []):
        byline = ", ".join(value for value in (first_author(source.get("authors")), source.get("year")) if value)
        parts.append(f"[{source['n']}] {source['title']}{f' ({byline})' if byline else ''}\n{source.get('evidence') or source.get('excerpt') or ''}")
    return "\n\n".join(parts)


def author_list(authors):
    """Author names from 'A; B; C' (the catalogue format) or 'A, B, C'."""
    text = str(authors or "")
    separator = ";" if ";" in text else ","
    return [name.strip() for name in text.split(separator) if name.strip()]


def first_author(authors):
    names = author_list(authors)
    if not names:
        return ""
    return names[0] + (" et al." if len(names) > 1 else "")


def byline(source):
    """'Chen et al. · 2026' for source cards and exports."""
    return " · ".join(value for value in (first_author(source.get("authors")), source.get("year")) if value)
