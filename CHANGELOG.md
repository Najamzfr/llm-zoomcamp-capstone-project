# Changelog

## Phase 3 — bigger bets
- Upload your own paper: up to 3 PDFs (20 MB, ~60 pages) or a PDF link, chunked (800/100 tokens),
  embedded into an in-memory Qdrant collection for the session; "Include my papers" fuses them with
  search results via RRF and cites them as "Your upload". Cleared on New question; never stored.
  Clear errors for scanned, oversized or password-protected PDFs; links to private networks are refused.
- Write a review: structured literature review (overview, themes, methods table, timeline, open
  problems, cited references only) with .md and .docx downloads.
- Listen: browser Web Speech player (play/pause/stop, 0.8–1.5×, voice picker); hidden when unsupported.
- Collections: save answers into named collections; rename, remove, delete, export .md + combined .bib;
  included in the JSON backup.
- Feedback with reasons: thumbs-down asks Wrong / Missing source / Too vague / Off-topic / Other plus an
  optional note; rows go to SHARE_BACKEND without user identifiers.

## Phase 2 — core study features
- Study this: 8–10 flashcards and a 5-question quiz per answer (one cached call); options shuffled;
  each card and explanation cites its sources. Harder set (mechanism/comparison questions).
- Wrong quiz answers become review cards automatically ("Added to your review deck").
- Spaced review with FSRS (py-fsrs): Again/Hard/Good/Easy, next-due preview, daily cap
  (`REVIEW_DAILY_CAP`, default 20), topic filter, streak, JSON backup/import; sidebar "Review · n due".
- Anki `.apkg` (stable deck/model ids, Front/Back/Source, question tags) and CSV export of a study set
  or the whole deck.
- Concept map (Graphviz, dashed = unsupported concepts) with concept pills that ask follow-ups,
  SVG download (when Graphviz is installed) and Mermaid code.
- Share links (`?a=<id>`): sanitised immutable snapshot (optionally with the study set) stored in
  `SHARE_BACKEND` (supabase | memory); read-only page with Study this, Save to my review deck and
  Ask your own question; 20 shares per session, 10 KB answer cap, graceful "not available" page.

## Phase 1 — quick wins (Streamlit app)
- Answers end with three suggested follow-up questions (same model call); clicking one asks it.
- "Explain simply" and "Go deeper" rewrite an answer from the same sources, keeping its citations.
- More menu: Copy answer (with a "Show as text" fallback), Download Markdown (with references),
  Download BibTeX (stable keys such as `hu2021lora`, de-duplicated).
- Question history is kept in the browser (last 50) and reopens instantly; "Clear history" asks inline.
- Featured answers gallery on the empty state (`data/featured.json`, built by `scripts/build_featured.py`).
- Sources moved into a side panel opened by "Sources · n" or any citation chip.
- No internals in the UI: no corpus counts, index dates, category codes, cost, tokens or research steps.
- `llm_json()` gateway for the extra features: pydantic validation, one retry, 24 h cache,
  provider switch (`EXTRAS_LLM_PROVIDER` = openai | gemini | groq).
