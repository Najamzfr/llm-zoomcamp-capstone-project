"""Exports built from answer records: Markdown, BibTeX, Anki (.apkg), CSV, Mermaid and SVG."""

import csv
import hashlib
import io
import re
import tempfile
import unicodedata
from pathlib import Path

from research_assistant.presentation import slug

from .records import author_list, first_author

STOPWORDS = {"a", "an", "the", "on", "of", "for", "in", "to", "and", "with", "towards", "toward", "via",
             "from", "by", "is", "are", "at", "as", "do", "does", "can", "how", "what", "why", "when"}
APP_NAME = "Paper Scout"


def _ascii(text):
    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()


# ------------------------------------------------------------------ Markdown
def reference_line(source):
    names = author_list(source.get("authors"))
    author_text = ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")
    parts = [source.get("title") or "Untitled"]
    if author_text:
        parts.append(author_text)
    if source.get("year"):
        parts.append(f"({source['year']})")
    line = ". ".join(parts[:2]) + (f" {parts[2]}" if len(parts) > 2 else "")
    if source.get("upload"):
        return line + ". Your upload."
    return line + (f". {source['url']}" if source.get("url") else ".")


def answer_markdown(record):
    lines = [f"# {record['question']}", "", record["answer_md"].strip(), "", "## References", ""]
    lines += [f"{source['n']}. {reference_line(source)}" for source in record.get("sources", [])]
    return "\n".join(lines).strip() + "\n"


def markdown_filename(question):
    return slug(question)[:60] + ".md"


# ------------------------------------------------------------------ BibTeX
def bibtex_key(source):
    """Stable key like vaswani2017attention (first author's surname + year + first title word)."""
    first = first_author(source.get("authors")).replace(" et al.", "")
    surname = re.sub(r"[^a-z]", "", _ascii(first.split()[-1] if first.split() else "").lower()) or "anon"
    year = re.sub(r"\D", "", str(source.get("year") or ""))[:4] or "nd"
    words = [re.sub(r"[^a-z0-9]", "", word) for word in _ascii(source.get("title")).lower().split()]
    word = next((w for w in words if w and w not in STOPWORDS), "paper")
    return f"{surname}{year}{word}"


def _bib(value):
    return str(value or "").replace("{", "").replace("}", "").strip()


def bibtex(sources):
    """De-duplicated @misc/@article entries; colliding keys get a, b, c suffixes."""
    seen, used, entries = set(), {}, []
    for source in sources:
        identity = (source.get("paper_id") or source.get("url") or source.get("title") or "").lower()
        if not identity or identity in seen or source.get("upload"):
            continue
        seen.add(identity)
        key = bibtex_key(source)
        count = used.get(key, 0)
        used[key] = count + 1
        if count:
            key += "abcdefghijklmnopqrstuvwxyz"[min(count, 25)]
        authors = " and ".join(author_list(source.get("authors")))
        paper_id = str(source.get("paper_id") or "")
        fields = {
            "title": "{" + _bib(source.get("title")) + "}",
            "author": _bib(authors),
            "year": source.get("year") or "",
            "eprint": paper_id if re.match(r"^\d{4}\.\d{4,5}", paper_id) else "",
            "archivePrefix": "arXiv" if re.match(r"^\d{4}\.\d{4,5}", paper_id) else "",
            "url": source.get("url") or "",
        }
        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields.items() if value)
        entries.append(f"@misc{{{key},\n{body}\n}}")
    return "\n\n".join(entries) + ("\n" if entries else "")


# ------------------------------------------------------------------ Anki and CSV
def _stable_id(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (1 << 30) + (1 << 30)


DECK_ID = _stable_id(f"{APP_NAME} deck")
MODEL_ID = _stable_id(f"{APP_NAME} model v1")


def source_label(card, sources_by_n):
    """'Title (2026) https://…' for the cited sources of a card."""
    labels = []
    for number in card.get("src") or []:
        source = sources_by_n.get(number) or sources_by_n.get(str(number))
        if source:
            label = source.get("title", "")
            if source.get("year"):
                label += f" ({source['year']})"
            if source.get("url"):
                label += f" {source['url']}"
            labels.append(label)
    return "; ".join(labels)


def card_rows(cards):
    """[(front, back, source, tag)] from study cards or deck entries carrying `src_meta`."""
    rows = []
    for card in cards:
        meta = {item["n"]: item for item in card.get("src_meta") or []}
        rows.append((card["front"], card["back"], source_label(card, meta), slug(card.get("topic") or "", 40)))
    return rows


def anki_package(cards, deck_name=APP_NAME):
    """Bytes of a .apkg deck with a stable deck id and a Front/Back/Source note model."""
    import genanki

    model = genanki.Model(
        MODEL_ID,
        f"{APP_NAME} card",
        fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Source"}],
        templates=[{
            "name": "Card 1",
            "qfmt": "{{Front}}",
            "afmt": "{{FrontSide}}<hr id=answer>{{Back}}<div style='margin-top:12px;font-size:13px;color:#5b574f'>{{Source}}</div>",
        }],
    )
    deck = genanki.Deck(DECK_ID, deck_name)
    for front, back, source, tag in card_rows(cards):
        note = genanki.Note(
            model=model,
            fields=[_html(front), _html(back), _html(source)],
            tags=[tag] if tag else [],
            guid=genanki.guid_for(front, back),
        )
        deck.add_note(note)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "deck.apkg"
        genanki.Package(deck).write_to_file(str(path))
        return path.read_bytes()


def _html(text):
    import html

    return html.escape(str(text or ""))


def cards_csv(cards):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["front", "back", "source"])
    for front, back, source, _tag in card_rows(cards):
        writer.writerow([front, back, source])
    return buffer.getvalue()


# ------------------------------------------------------------------ concept map
def mermaid(concept_map):
    lines = ["flowchart LR"]
    ids = {node["id"]: f"n{index}" for index, node in enumerate(concept_map["nodes"])}
    for node in concept_map["nodes"]:
        label = node["label"].replace('"', "'")
        lines.append(f'  {ids[node["id"]]}["{label}"]')
    for edge in concept_map.get("edges", []):
        source, target = ids.get(edge["from"]), ids.get(edge["to"])
        if source and target:
            label = (edge.get("label") or "").replace('"', "'").replace("|", "/")
            lines.append(f"  {source} -->|{label}| {target}" if label else f"  {source} --> {target}")
    return "\n".join(lines) + "\n"


def svg(dot_source):
    """SVG bytes from DOT via the Graphviz binary, or None when it isn't installed."""
    try:
        import graphviz

        return graphviz.Source(dot_source).pipe(format="svg")
    except Exception:  # graphviz package or `dot` binary missing
        return None
