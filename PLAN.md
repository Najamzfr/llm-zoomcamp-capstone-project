# Paper Scout extras — implementation plan

Mockups: [docs/mockups/features.html](docs/mockups/features.html) (open in a browser; every
feature panel at desktop width and a 390 px phone frame).

## 0. Repo map (what exists today)

| Concern | Where |
|---|---|
| Streamlit entry (deployed on Streamlit Cloud, branch `prod`) | `deploy/streamlit_app.py` |
| "API client" | none — the Streamlit app runs `AdaptiveResearchAgent` in-process; `api._response()` shapes the result |
| FastAPI routes | `api.py`: `GET /`, `/health`, `/metrics`, `POST /ask`, `POST /feedback` (+ `static/` UI) |
| Answer schema | `{"request_id", "question", "answer" (Markdown with [n] citations, starts "Short answer:"), "answer_stage", "retrieved_papers": [{title, paper_id, published, category, url}], "trajectory", "usage"}`; agent result also has `papers` (full rows incl. abstract `text`, `authors`), `excerpts` {paper_id: best chunk}, `full_text_evidence`, `trace_run_id` |
| Citations | `[n]` = search rank of the paper (abstract and full-text answers share numbering) |
| session_state | `messages` [{id, question, status queued/running/done/limited/error, filters, result, trace_run_id, seconds, read_full, verified, steps}], `history` [question], `rated` {id: up/down}, `cat_filter`, `period_filter`, `scroll_to`, `start_research` |
| Config / secrets | `research_assistant/config.py` (`Settings`, env vars), Streamlit secrets copied to env by `_load_secrets_into_env()`, `.env.example` |
| Theme / CSS | `.streamlit/config.toml` (tokens paper `#f7f6f3`, surface `#fff`, sunk `#efede8`, line `#e2dfd8`, ink `#1f1d1a`, ink-2 `#5b574f`, accent `#1d4ed8`, verified `#1a7f4b`; IBM Plex Sans + Newsreader) and the `CSS` block in the entry file |
| Helpers to reuse | `research_assistant/presentation.py` (`split_short_answer`, `link_citations`, `clean_excerpt`, `shorten`, `short_title`, `bibtex`, `format_date`), `tracing.send_feedback`, `AskLimiter` |

## Conflicts with the current UI (resolved in favour of the new rules, no approval needed)

* **R2** — the current UI shows the paper count, "arXiv", category codes, the index date range,
  research steps with counts, and tokens/cost in "Details". All of these are removed:
  headline/lead/sidebar copy become count-free, category pills get plain names
  ("Vision", "Language"…), the progress view becomes a neutral "Finding and reading sources…"
  skeleton, the meta line keeps only "Checked against sources", "Details" is removed
  (cost/tokens stay in LangSmith and the server logs). Period filter becomes
  "Any time / Last 6 months / Last month" (relative, no dates shown).
* **R3** — sources are currently an always-visible right rail. They move into a slide-in side
  panel (same card design), closed by default, opened by **Sources · n** or any `[n]` chip.
* Feedback buttons "Helpful / Not helpful" are replaced by `st.feedback("thumbs")` (F15).
* Session-only "Recent" is replaced by browser-stored history (F4).

## Files touched

* `deploy/streamlit_app.py` — layout, action row, panels, views (review, collections, shared answer).
* `research_assistant/rag.py` — `ANSWER_FORMAT` asks for a trailing `Follow-ups:` block (F1).
* `research_assistant/agent.py` — `run(..., extra_search=None)` fusion hook for uploads (F11).
* `research_assistant/presentation.py` — `split_followups`, `strip_for_speech`, `slug`, stable BibTeX keys.
* `api.py` — `_response` strips follow-ups out of `answer` and returns `followups`.
* `deploy/requirements.txt`, `pyproject.toml`, `packages.txt` (new, `graphviz`), `README.md`,
  `deploy/README.md`, `.streamlit/secrets.toml.example` (new), `.env.example`.

## New modules (`app/`, importable from the repo root)

| Module | Contents |
|---|---|
| `app/features/llm_extras.py` | `llm_json(prompt, schema, context, *, feature, answer_id, params)`; provider switch openai/gemini/groq (OpenAI-compatible endpoints, so no new SDKs); pydantic validation + one retry; `st.cache_data(ttl=24h)` keyed by `sha256(feature+answer_id+params)`; pydantic models `Rewrite`, `StudySet`, `ConceptMap`, `LitReview`, `Followups` |
| `app/features/records.py` | the stored answer record `{id, question, answer_md, followups, sources[n,title,authors,year,url,paper_id,excerpt,evidence], created_at}` built from an agent result; `sources_context()` for R6 |
| `app/features/followups.py` | follow-up pills (F1) |
| `app/features/export.py` | Markdown, BibTeX (keys `vaswani2017attention`, de-duplicated), Anki `.apkg`, CSV, Mermaid, SVG |
| `app/features/study.py` | Study set UI: flashcards, quiz, mistakes → review, harder set |
| `app/features/review.py` | FSRS deck (py-fsrs), daily cap, topics, streak, backup/import |
| `app/features/concept_map.py` | ConceptMap → DOT (theme colours, dashed unsupported nodes), pills that ask follow-ups |
| `app/features/share.py` | snapshot sanitiser, backends `supabase` / `memory`, read-only view, rate limit |
| `app/features/uploads.py` | PDF ingest (PyMuPDF), 800/100-token chunks, fastembed, in-memory Qdrant, RRF fusion |
| `app/features/report.py` | literature review → Markdown and `.docx` |
| `app/features/tts.py` | Web Speech player (Play/Pause/Stop, speed, voice) |
| `app/features/collections.py` | collections view, rename/remove, combined `.md` + `.bib` |
| `app/features/feedback.py` | thumbs + reason pills, saved to `SHARE_BACKEND` |
| `app/storage/local.py` | one `st.components.v2` component that loads/saves versioned localStorage keys; safe when unavailable |
| `data/featured.json`, `scripts/build_featured.py` | gallery of 6–8 precomputed answers |

## New dependencies (all free / OSS)

`fsrs` (py-fsrs, MIT), `genanki` (MIT), `python-docx` (MIT), `graphviz` (MIT, SVG only; `packages.txt`
installs the `dot` binary on Streamlit Cloud), `pydantic` (already transitively installed; pinned
explicitly). Deliberate substitutions, see "Changed vs. brief" below.

## New env vars / secrets

`EXTRAS_LLM_PROVIDER` (openai|gemini|groq, default openai), `EXTRAS_LLM_MODEL` (default per
provider: gpt-4o-mini / gemini-2.5-flash / llama-3.3-70b-versatile), `GEMINI_API_KEY`,
`GROQ_API_KEY`, `SHARE_BACKEND` (supabase|memory, default memory), `SUPABASE_URL`,
`SUPABASE_ANON_KEY`, `APP_URL`, `REVIEW_DAILY_CAP` (20).

## Changed vs. brief (and why)

* **localStorage** via a 60-line `st.components.v2` component instead of `streamlit-local-storage`:
  one mount for all keys, no extra iframe per call, no async race. Same for Copy and Listen:
  v2 components run in the page, so the Clipboard and Speech APIs work without iframe permissions.
* **PDF extraction** with PyMuPDF (already a dependency, better text extraction) instead of `pypdf`.
* **Supabase** through its REST API with `requests` (already a dependency) instead of the
  `supabase` SDK; **ids** from `secrets` (8-char URL-safe alphabet) instead of `nanoid`.
* `SHARE_BACKEND=memory` (process memory) is the local/dev default; Google Sheets is not built.

## Phases

1. **Quick wins** — action row + side panel, R2 clean-up, follow-ups, simpler/deeper versions,
   Copy/Markdown/BibTeX, browser history, featured gallery. Commit "Phase 1".
2. **Core** — study set + quiz + mistakes, FSRS review, Anki/CSV, concept map, share links.
   Commit "Phase 2".
3. **Bigger bets** — uploads, literature review (.md/.docx), Listen, collections, feedback reasons.
   Commit "Phase 3".

Each phase: unit tests + `AppTest` smoke tests green, then a click-through in the browser pane.

## Supabase (when `SHARE_BACKEND=supabase`)

```sql
create table shares   (id text primary key, payload jsonb not null, created_at timestamptz default now());
create table feedback (id bigserial primary key, payload jsonb not null, created_at timestamptz default now());
alter table shares enable row level security;
alter table feedback enable row level security;
create policy "anon insert shares"   on shares   for insert to anon with check (true);
create policy "anon insert feedback" on feedback for insert to anon with check (true);
-- No select policies: rows can only be read one at a time through this function.
create function get_share(share_id text) returns jsonb language sql security definer
  set search_path = public as $$ select payload from shares where id = share_id $$;
grant execute on function get_share(text) to anon;
```
