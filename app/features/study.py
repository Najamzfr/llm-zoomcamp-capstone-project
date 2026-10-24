"""'Study this' (F6): flashcards and a quiz generated from one answer and its sources.

One llm_json call per answer and level (standard / hard), cached by llm_extras. Wrong quiz answers
become review cards automatically; "Add all to review" copies the flashcards into the deck.
"""

import html
import random

import streamlit as st

from research_assistant.presentation import link_citations

from . import export, review
from .llm_extras import LLMExtrasError, StudySet, llm_json
from .records import sources_context

PROMPT = (
    "Create a study set that tests understanding of this answer: 8 to 10 flashcards and 5 multiple-choice "
    "quiz questions. Flashcard fronts are a question or a term (at most 20 words); backs answer it in at most "
    "40 words. Each quiz question has exactly 4 plausible options with one correct answer (answer_index 0-3) "
    "and an explanation of at most 35 words. Give every card and question `src`: the numbers of the sources "
    "its content comes from. Use ids c1, c2… for cards and q1, q2… for questions. Mix easy, medium and hard."
)
HARD = (
    " Make every item hard: prefer mechanism questions (how and why something works) and comparison "
    "questions (how two approaches differ, when one beats the other). Set difficulty to \"hard\"."
)
LEVELS = {"standard": "Standard set", "hard": "Harder set"}


def generate(record, level="standard"):
    """Validated study set with ids unique to this answer and level; src numbers kept in range."""
    prompt = PROMPT + (HARD if level == "hard" else "")
    data = llm_json(prompt, StudySet, sources_context(record), feature="study", answer_id=record["id"],
                    params={"difficulty": level})
    valid = {source["n"] for source in record.get("sources", [])}
    tag = "h" if level == "hard" else "s"
    for index, card in enumerate(data["cards"], start=1):
        card["id"] = f"{record['id']}-{tag}c{index}"
        card["src"] = [n for n in card.get("src") or [] if n in valid]
        card["src_meta"] = review.src_meta(record, card["src"])
        card["topic"] = record["question"]
    for index, question in enumerate(data["quiz"], start=1):
        question["id"] = f"{record['id']}-{tag}q{index}"
        question["src"] = [n for n in question.get("src") or [] if n in valid]
        # Models tend to put the right answer first; shuffle the options (stable per question).
        order = list(range(len(question["options"])))
        random.Random(question["id"]).shuffle(order)
        question["answer_index"] = order.index(question["answer_index"])
        question["options"] = [question["options"][i] for i in order]
    return data


def _chips(text, src, prefix, count):
    marks = "".join(f"[{n}]" for n in src)
    return link_citations(f"{text} {marks}".strip(), prefix, count)


def render_panel(record, mid, prefix, readonly_study=None):
    """The panel under an answer. `readonly_study` is a study set attached to a shared answer."""
    state = st.session_state
    level = state.get(f"study_{mid}_level", "standard")
    try:
        if readonly_study and level == "standard":
            data = readonly_study
        else:
            with st.spinner("Writing flashcards and a quiz…"):
                data = generate(record, level)
    except LLMExtrasError as error:
        st.error(str(error))
        return
    state.setdefault("study_sets", {})[mid] = data
    count = len(record.get("sources", []))
    st.html(f"<div class='panel-h'><b>Study this</b><span class='ps-note'>{LEVELS[level]}</span></div>")
    cards_tab, quiz_tab = st.tabs(["Flashcards", "Quiz"])
    with cards_tab:
        _flashcards(record, data["cards"], mid, prefix, count, level)
    with quiz_tab:
        _quiz(record, data["quiz"], mid, prefix, count, level)


def _flashcards(record, cards, mid, prefix, count, level):
    state = st.session_state
    key = f"study_{mid}_{level}_card"
    index = min(state.get(key, 0), len(cards) - 1)
    shown = state.get(f"{key}_show", False)
    card = cards[index]
    back = (f"<div class='fc-back'>{_chips(html.escape(card['back']), card['src'], prefix, count)}</div>" if shown else "")
    st.markdown(f"<div class='fc-card'><div><span class='fc-diff'>{card['difficulty']}</span>"
                f"<div class='fc-front'>{html.escape(card['front'])}</div>{back}</div></div>", unsafe_allow_html=True)
    with st.container(horizontal=True, key=f"study_{mid}_nav"):
        if st.button("← Prev", key=f"study_{mid}_prev", disabled=index == 0):
            state[key], state[f"{key}_show"] = index - 1, False
            st.rerun()
        if st.button("Hide answer" if shown else "Show answer", key=f"study_{mid}_flip", type="primary"):
            state[f"{key}_show"] = not shown
            st.rerun()
        st.html(f"<span class='fc-count'>{index + 1} / {len(cards)}</span>", width="content")
        if st.button("Next →", key=f"study_{mid}_next", disabled=index >= len(cards) - 1):
            state[key], state[f"{key}_show"] = index + 1, False
            st.rerun()
    with st.container(horizontal=True, key=f"study_{mid}_tools"):
        if st.button("Add all to review", key=f"study_{mid}_addall"):
            added = review.add_cards(cards, record)
            st.toast(f"Added {added} cards to your review deck" if added else "These cards are already in your review deck")
        other = "standard" if level == "hard" else "hard"
        if st.button(LEVELS[other], key=f"study_{mid}_level_btn"):
            state[f"study_{mid}_level"] = other
            st.rerun()
        base = export.slug(record["question"], 50)
        st.download_button("Download Anki deck", export.anki_package(cards), file_name=f"{base}.apkg",
                           mime="application/octet-stream", key=f"study_{mid}_{level}_anki")
        st.download_button("Download CSV", export.cards_csv(cards), file_name=f"{base}.csv", mime="text/csv",
                           key=f"study_{mid}_{level}_csv")


def _answered(mid, level, question, record):
    """Radio callback: score once, and send a wrong answer to the review deck."""
    state = st.session_state
    choice = state.get(f"study_{mid}_{level}_{question['id']}")
    answers = state.setdefault(f"study_{mid}_{level}_answers", {})
    if choice is None or question["id"] in answers:
        return
    answers[question["id"]] = choice
    if choice != question["answer_index"]:
        card = review.mistake_card(question)
        if review.add_cards([card], record):
            st.toast("Added to your review deck")


def _quiz(record, quiz, mid, prefix, count, level):
    state = st.session_state
    answers = state.setdefault(f"study_{mid}_{level}_answers", {})
    position = min(state.get(f"study_{mid}_{level}_q", 0), len(quiz))
    if position >= len(quiz):
        score = sum(1 for question in quiz if answers.get(question["id"]) == question["answer_index"])
        st.markdown(f"#### Your score: {score} / {len(quiz)}")
        missed = len(quiz) - score
        st.caption("Questions you missed are in your review deck." if missed else "Perfect score.")
        if st.button("Try again", key=f"study_{mid}_{level}_retry"):
            for question in quiz:
                answers.pop(question["id"], None)
                state.pop(f"study_{mid}_{level}_{question['id']}", None)
            state[f"study_{mid}_{level}_q"] = 0
            st.rerun()
        return
    question = quiz[position]
    st.caption(f"Question {position + 1} of {len(quiz)}")
    st.radio(question["question"], range(4), format_func=lambda i: question["options"][i], index=None,
             key=f"study_{mid}_{level}_{question['id']}", disabled=question["id"] in answers,
             on_change=_answered, args=(mid, level, question, record))
    if question["id"] in answers:
        right = answers[question["id"]] == question["answer_index"]
        correct = html.escape(question["options"][question["answer_index"]].rstrip("."))
        verdict = "✓ Correct." if right else f"✕ Not quite — the answer is “{correct}”."
        st.markdown(f"<div class='verdict {'good' if right else 'bad'}'><b>{verdict}</b>"
                    f"<p>{_chips(html.escape(question['explanation']), question['src'], prefix, count)}</p></div>",
                    unsafe_allow_html=True)
        label = "See your score" if position == len(quiz) - 1 else "Next question"
        if st.button(label, key=f"study_{mid}_{level}_nextq", type="primary"):
            state[f"study_{mid}_{level}_q"] = position + 1
            st.rerun()
