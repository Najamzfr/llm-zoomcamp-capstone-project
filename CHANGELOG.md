# Changelog

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
