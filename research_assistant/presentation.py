"""Pure formatting helpers for answer views (citation chips, short answer, excerpts, follow-ups)."""

import html
import re
from datetime import date, timedelta

SHORT_ANSWER = re.compile(r"^\s*(?:#+\s*)?\**\s*short answer\s*[:.]?\s*\**\s*[:.]?\s*", re.IGNORECASE)
CITATION = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\](?!\()")


def split_short_answer(markdown):
    """Return (short_answer, rest). short_answer is '' when the answer has no such lead."""
    text = (markdown or "").strip()
    match = SHORT_ANSWER.match(text)
    if not match:
        return "", text
    body = text[match.end():]
    short, _, rest = body.partition("\n\n")
    return short.strip(), rest.strip()


def link_citations(markdown, prefix, source_count):
    """Escape raw HTML in model output, then turn [1], [2][3] or [1, 2] into citation chips.

    Numbers outside 1..source_count are left as plain text.
    """
    escaped = html.escape(markdown or "", quote=False)

    def chips(match):
        numbers = [int(part) for part in re.split(r"\s*,\s*", match.group(1))]
        if not all(1 <= number <= source_count for number in numbers):
            return match.group(0)
        return "".join(
            f'<a class="cite" href="#{prefix}-{number}" data-src="{prefix}-{number}">{number}</a>'
            for number in numbers
        )

    return CITATION.sub(chips, escaped)


def abstract_excerpt(paper_text, limit=170):
    """First words of a paper's abstract from the stored 'Title: … Abstract: …' text."""
    text = paper_text or ""
    if "Abstract:" in text:
        text = text.split("Abstract:", 1)[1]
    return shorten(text, limit)


NOISE = re.compile(r"figure|table|arxiv|preprint|transactions|submitted|copyright|http|@|\bet al\b", re.IGNORECASE)


def clean_excerpt(text, limit=170):
    """First readable prose sentence of a PDF chunk (skips headers, captions and layout debris).

    Returns '' when nothing looks like a real sentence, so callers can fall back to the abstract.
    """
    flat = " ".join((text or "").split())
    for sentence in re.split(r"(?<=[.!?])\s+", flat):
        words = sentence.split()
        if len(words) < 8 or NOISE.search(sentence):
            continue
        letters = sum(char.isalpha() or char.isspace() for char in sentence)
        uppercase = sum(char.isupper() for char in sentence)
        if letters / len(sentence) < 0.88 or uppercase / max(len(sentence), 1) > 0.2:
            continue
        if not sentence[0].isupper() or not sentence.rstrip().endswith((".", "!", "?")):
            continue
        return shorten(sentence, limit)
    return ""


def shorten(text, limit=170):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",.;:")
    return cut + "…"


def short_title(title, words=3):
    """Compact label for small chips: text before a colon, else the first few words."""
    title = " ".join((title or "").split())
    if ":" in title and len(title.split(":", 1)[0]) <= 28:
        return title.split(":", 1)[0]
    parts = title.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def period_start(period, corpus_end):
    """ISO start date for a period label relative to the newest paper; None for 'All'."""
    months = {"Last 3 months": 3, "Last month": 1, "Last 6 months": 6}.get(period)
    if not months:
        return None
    end = date.fromisoformat(str(corpus_end)[:10])
    return (end - timedelta(days=30 * months)).isoformat()


def format_date(value):
    """'2026-04-15' -> '15 Apr 2026' (portable; avoids the non-Windows %-d directive)."""
    try:
        parsed = date.fromisoformat(str(value)[:10])
    except ValueError:
        return str(value or "")[:10]
    return f"{parsed.day} {parsed.strftime('%b %Y')}"


FOLLOWUPS = re.compile(r"^\s*(?:#+\s*)?\**\s*follow[- ]?ups?(?: questions)?\s*:?\s*\**\s*:?\s*$", re.IGNORECASE | re.MULTILINE)


def split_followups(markdown, limit=3):
    """Return (answer_without_block, [questions]) for an answer that ends with a 'Follow-ups:' list.

    Text after the list (e.g. a limitation note appended by the agent) is kept in the answer.
    """
    text = markdown or ""
    match = FOLLOWUPS.search(text)
    if not match:
        return text.strip(), []
    lines = text[match.end():].lstrip("\n").split("\n")
    questions, consumed = [], 0
    for line in lines:
        item = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$", line)
        if item:
            questions.append(re.sub(r"^(?:[-*•]\s*)+", "", item.group(1)).strip("* ").strip())
            consumed += 1
        elif not line.strip() and not questions:
            consumed += 1
        else:
            break
    tail = "\n".join(lines[consumed:]).strip()
    head = text[:match.start()].rstrip()
    answer = (head + ("\n\n" + tail if tail else "")).strip()
    return answer, [question for question in questions if question][:limit]


def slug(text, limit=60):
    """Lower-case ASCII file-name slug: 'How is RAG evaluated?' -> 'how-is-rag-evaluated'."""
    ascii_text = (text or "").encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return value[:limit].rstrip("-") or "answer"


def strip_for_speech(markdown):
    """Plain text for read-aloud: no citations, tables, headings marks, emphasis or links."""
    text = CITATION.sub("", markdown or "")
    text = re.sub(r"^\s*\|.*\|\s*$", "", text, flags=re.MULTILINE)  # tables are unreadable aloud
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`>]", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"\n{2,}", "\n", text).strip()
