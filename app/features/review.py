"""Spaced review (F7) with FSRS (py-fsrs), stored in the browser under "ps_deck_v1".

Deck entry: {card_id: {front, back, src, src_meta, fsrs_state, due, topic, answer_id, added}}.
`src_meta` (title, year, link per cited source) is copied into each card so review works without
the original answer. Review stats (per-day counts, streak) live under "ps_review_meta_v1".
"""

import copy
import hashlib
import html
import json
import os
from datetime import datetime, timedelta, timezone

import streamlit as st
from fsrs import Card, Rating, Scheduler

from app.storage import local
from research_assistant.presentation import format_date, shorten

RATINGS = {"Again": Rating.Again, "Hard": Rating.Hard, "Good": Rating.Good, "Easy": Rating.Easy}
BACKUP_VERSION = 1


def daily_cap():
    try:
        return max(1, int(os.getenv("REVIEW_DAILY_CAP", "20")))
    except ValueError:
        return 20


def scheduler():
    return Scheduler(enable_fuzzing=False)


def utcnow():
    return datetime.now(timezone.utc)


def _parse(value):
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return utcnow()
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def local_day(moment):
    """Calendar day in the visitor's time zone (daily cap and streak roll over at their midnight)."""
    return (moment + timedelta(minutes=local.tz_minutes())).date()


def when(due, now):
    """'in 10 min' for soon, otherwise a local date such as '28 Oct 2026'."""
    delta = due - now
    if delta < timedelta(hours=20):
        return "in " + _interval(delta)
    return format_date(local_day(due).isoformat())


def _fsrs_id(card_id):
    return int(hashlib.sha256(card_id.encode("utf-8")).hexdigest()[:12], 16)


# ------------------------------------------------------------------ deck
def deck():
    return local.get(local.DECK)


def meta():
    return local.get(local.REVIEW_META)


def src_meta(record, numbers):
    by_n = {source["n"]: source for source in record.get("sources", [])}
    return [{"n": n, "title": by_n[n]["title"], "year": by_n[n].get("year", ""), "url": by_n[n].get("url", "")}
            for n in numbers or [] if n in by_n]


def add_cards(cards, record, now=None):
    """Add study cards from `record`'s answer; returns how many were new."""
    now = now or utcnow()
    current = dict(deck())
    added = 0
    for card in cards:
        card_id = str(card["id"])
        if card_id in current:
            continue
        fresh = Card(card_id=_fsrs_id(card_id), due=now)
        current[card_id] = {
            "front": card["front"], "back": card["back"], "src": list(card.get("src") or []),
            "src_meta": card.get("src_meta") or src_meta(record, card.get("src")),
            "fsrs_state": fresh.to_dict(), "due": fresh.due.isoformat(),
            "topic": record["question"], "answer_id": record["id"], "added": now.isoformat(),
        }
        added += 1
    if added:
        local.put(local.DECK, current)
    return added


def mistake_card(question):
    """Turn a wrongly answered quiz question into a review card."""
    correct = question["options"][question["answer_index"]]
    return {"id": f"{question['id']}-review", "front": question["question"],
            "back": f"{correct.rstrip('.')}. {question['explanation']}".strip(), "src": question.get("src") or []}


def topics():
    seen = []
    for entry in deck().values():
        if entry.get("topic") and entry["topic"] not in seen:
            seen.append(entry["topic"])
    return seen


def reviewed_today(now=None):
    return int(meta().get("days", {}).get(local_day(now or utcnow()).isoformat(), 0))


def due_cards(now=None, topic=None):
    now = now or utcnow()
    cards = [(card_id, entry) for card_id, entry in deck().items()
             if _parse(entry["due"]) <= now and (topic is None or entry.get("topic") == topic)]
    return sorted(cards, key=lambda item: item[1]["due"])


def due_count(now=None):
    """Cards to review today, after the daily cap."""
    return min(len(due_cards(now)), max(0, daily_cap() - reviewed_today(now)))


def preview(card_id, now=None):
    """{rating label: next due datetime} without changing the card."""
    now = now or utcnow()
    state = Card.from_dict(deck()[card_id]["fsrs_state"])
    return {label: scheduler().review_card(copy.deepcopy(state), rating, now)[0].due for label, rating in RATINGS.items()}


def grade(card_id, label, now=None):
    """Apply an Again/Hard/Good/Easy grade; returns the card's next due datetime."""
    now = now or utcnow()
    current = dict(deck())
    entry = dict(current[card_id])
    card, _log = scheduler().review_card(Card.from_dict(entry["fsrs_state"]), RATINGS[label], now)
    entry.update(fsrs_state=card.to_dict(), due=card.due.isoformat())
    current[card_id] = entry
    local.put(local.DECK, current)
    _count_review(now)
    return card.due


def _count_review(now):
    stats = copy.deepcopy(meta())
    today = local_day(now).isoformat()
    days = stats.setdefault("days", {})
    days[today] = int(days.get(today, 0)) + 1
    for day in sorted(days)[:-30]:  # keep a month of counts
        days.pop(day)
    local.put(local.REVIEW_META, stats)


def streak(now=None):
    """Consecutive days with at least one review, ending today (or yesterday)."""
    days = {day for day, count in meta().get("days", {}).items() if count}
    current = local_day(now or utcnow())
    if current.isoformat() not in days:
        current -= timedelta(days=1)
    length = 0
    while current.isoformat() in days:
        length += 1
        current -= timedelta(days=1)
    return length


def remove_topic(topic):
    local.put(local.DECK, {card_id: entry for card_id, entry in deck().items() if entry.get("topic") != topic})


# ------------------------------------------------------------------ backup
def backup_json():
    return json.dumps({"app": "paper-scout", "version": BACKUP_VERSION, "exported_at": utcnow().isoformat(),
                       "deck": deck(), "collections": local.get(local.COLLECTIONS), "review_meta": meta()},
                      ensure_ascii=False, indent=1)


def restore(raw):
    """Merge a backup into this browser's data; returns (cards added, collections merged)."""
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("app") != "paper-scout":
        raise ValueError("This file isn't a Paper Scout backup.")
    incoming = data.get("deck") or {}
    if not isinstance(incoming, dict):
        raise ValueError("The backup's deck is damaged.")
    required = {"front", "back", "fsrs_state", "due"}
    merged = dict(deck())
    added = 0
    for card_id, entry in incoming.items():
        if isinstance(entry, dict) and required <= entry.keys() and card_id not in merged:
            merged[card_id] = entry
            added += 1
    local.put(local.DECK, merged)
    collections = dict(local.get(local.COLLECTIONS))
    merged_collections = 0
    for name, items in (data.get("collections") or {}).items():
        if isinstance(items, list):
            known = {item.get("id") for item in collections.get(name, [])}
            collections[name] = collections.get(name, []) + [item for item in items if item.get("id") not in known]
            merged_collections += 1
    local.put(local.COLLECTIONS, collections)
    stats = copy.deepcopy(meta())
    for day, count in ((data.get("review_meta") or {}).get("days") or {}).items():
        stats.setdefault("days", {})[day] = max(int(count), int(stats.get("days", {}).get(day, 0)))
    local.put(local.REVIEW_META, stats)
    return added, merged_collections


# ------------------------------------------------------------------ review screen
def _sources_line(entry):
    links = []
    for source in entry.get("src_meta") or []:
        label = html.escape(shorten(source["title"], 70)) + (f" ({html.escape(source['year'])})" if source.get("year") else "")
        url = source.get("url") or ""
        if url.startswith(("https://", "http://")):
            label = f"<a href='{html.escape(url, quote=True)}' target='_blank' rel='noopener noreferrer'>{label}</a>"
        links.append(f"<b>{source['n']}</b> {label}")
    return "<div class='rv-src'>Sources: " + " · ".join(links) + "</div>" if links else ""


def render_screen():
    """Full review screen (replaces the answers while state['view'] == 'review')."""
    state = st.session_state
    now = utcnow()
    st.markdown("# Review")
    days = streak(now)
    budget = max(0, daily_cap() - reviewed_today(now))
    topic_options = ["All topics"] + topics()
    topic = st.pills("Topic", topic_options, default="All topics", key="review_topic", format_func=lambda t: shorten(t, 48))
    chosen = None if topic in (None, "All topics") else topic
    queue = due_cards(now, chosen)
    line = [f"{min(len(queue), budget)} due today"] + ([f"{days}-day review streak"] if days else [])
    st.html(f"<div class='q-meta'>{' · '.join(line)}</div>")

    if not deck():
        st.info("Your review deck is empty. Open an answer, choose **Study this**, then **Add all to review**.")
    elif budget == 0 and queue:
        st.success(f"That's {daily_cap()} reviews for today. The rest will be here tomorrow.")
    elif not queue:
        upcoming = sorted(_parse(entry["due"]) for entry in deck().values())
        nxt = f" The next card is due {when(upcoming[0], now)}." if upcoming else ""
        st.success("All caught up." + nxt)
    else:
        card_id, entry = queue[0]
        shown = state.get("review_show") == card_id
        back = (f"<div class='fc-back'>{html.escape(entry['back'])}</div>{_sources_line(entry)}" if shown else "")
        st.html(f"<div class='fc-card'><div><div class='fc-topic'>{html.escape(shorten(entry.get('topic', ''), 80))}</div>"
                f"<div class='fc-front'>{html.escape(entry['front'])}</div>{back}</div></div>")
        if not shown:
            if st.button("Show answer", key="review_show_btn", type="primary"):
                state["review_show"] = card_id
                st.rerun()
        else:
            nexts = preview(card_id, now)
            with st.container(horizontal=True, key="review_grades"):
                for label in RATINGS:
                    if st.button(f"{label}  \n:gray[{_interval(nexts[label] - now)}]", key=f"review_grade_{label}"):
                        due = grade(card_id, label, now)
                        state["review_show"] = None
                        state["review_last"] = f"Next review {when(due, now)}"
                        st.rerun()
    if state.get("review_last"):
        st.caption(state.pop("review_last"))

    st.divider()
    st.markdown("#### Your deck")
    total = len(deck())
    st.caption(f"{total} card{'s' if total != 1 else ''}. Your deck lives in this browser — download a backup to move it.")
    from . import export  # local import keeps review importable without genanki for tests

    cards = list(deck().values())
    with st.container(horizontal=True):
        st.download_button("Download backup (.json)", backup_json(), file_name="paper-scout-backup.json",
                           mime="application/json", key="review_backup")
        if cards:
            st.download_button("Download Anki deck", export.anki_package(cards), file_name="paper-scout-review.apkg",
                               mime="application/octet-stream", key="review_anki")
            st.download_button("Download CSV", export.cards_csv(cards), file_name="paper-scout-review.csv",
                               mime="text/csv", key="review_csv")
    upload = st.file_uploader("Import a backup (.json)", type=["json"], key=f"review_import_{state.get('import_nonce', 0)}")
    if upload is not None:
        try:
            added, merged = restore(upload.getvalue().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            st.error(f"Couldn't import that file: {error} Choose a .json file made with “Download backup”.")
        else:
            state["import_nonce"] = state.get("import_nonce", 0) + 1
            st.toast(f"Imported {added} cards and {merged} collections")
            st.rerun()
    if chosen and st.button(f"Remove this topic's cards", key="review_remove_topic", type="tertiary"):
        remove_topic(chosen)
        st.rerun()


def _interval(delta):
    minutes = max(1, round(delta.total_seconds() / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours = round(minutes / 60)
    if hours < 24:
        return f"{hours} h"
    days = round(hours / 24)
    return f"{days} day" + ("s" if days != 1 else "")
