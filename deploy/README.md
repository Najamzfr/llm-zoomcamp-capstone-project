# Deploy to Streamlit Community Cloud (free, no card)

The cloud build runs the same research agent in-process from `deploy/streamlit_app.py`, so no FastAPI
server is needed. Vectors come from Qdrant Cloud; the SQLite keyword index is rebuilt from
`data/source/data.csv` on first start (about 1 second). `deploy/requirements.txt` is used instead of
the repository's `uv.lock` because it sits next to the entry file.

## Steps
1. Push the branch to GitHub.
2. Sign in at https://share.streamlit.io with GitHub and click **Create app** → **Deploy a public app from GitHub**.
3. Repository `Najamzfr/llm-zoomcamp-capstone-project`, branch `phase3` (or `main` after merging),
   main file path `deploy/streamlit_app.py`. Choose an app URL, e.g. `paper-scout`.
4. **Advanced settings** → Python version **3.12**, and paste the secrets (TOML):
   ```toml
   OPENAI_API_KEY = "sk-..."
   QDRANT_URL = "https://xxxx.cloud.qdrant.io"
   QDRANT_API_KEY = "..."
   SEARCH_BACKEND = "sqlite_qdrant"
   RAG_ASK_RATE_PER_MINUTE = "5"
   RAG_ASK_DAILY_LIMIT = "100"
   # Optional tracing
   LANGSMITH_TRACING = "true"
   LANGSMITH_API_KEY = "lsv2_..."
   LANGSMITH_PROJECT = "paper-scout-cloud"
   ```
   For the extra features also add the block from `.streamlit/secrets.toml.example`
   (`EXTRAS_LLM_*`, `SHARE_BACKEND`, `SUPABASE_*`, `APP_URL`, `REVIEW_DAILY_CAP`).
   Persistent share links and feedback need a free Supabase project: run the SQL in `PLAN.md`
   (tables `shares` and `feedback`, RLS, `get_share()`) in its SQL editor first.
5. Click **Deploy**. The first build takes a few minutes; the app lives at
   `https://<your-app-name>.streamlit.app`.

## Notes
- `deploy/packages.txt` installs the Graphviz binary so concept maps can be downloaded as SVG.
- The public app spends your OpenAI credit, which is why the rate limits above are lower than local defaults.
  Also set a monthly limit in the OpenAI dashboard.
- Apps sleep after a period without visitors; the first visit wakes them (about a minute).
- Memory: search ~260 MB; full-text answers peak ~400 MB (PDF chunks are embedded 4 at a time).
- Prometheus/Grafana monitoring is for the local FastAPI app; on the cloud build use LangSmith traces
  and the 👍/👎 feedback recorded there.
- Run it locally: `uv run streamlit run deploy/streamlit_app.py`.
- Look and feel: answer-first layout with a short-answer box, comparison tables for "compare/vs"
  questions, citation chips that open the sources side panel, follow-up suggestions and an action row
  (see the features list in the main README and `PLAN.md`). The UI never shows corpus counts, index
  dates, category codes, model names, cost or research steps. Theme tokens and fonts live in
  `.streamlit/config.toml`; formatting helpers in `research_assistant/presentation.py`; feature code in
  `app/features/`. Requires Streamlit 1.50+ (uses `st.components.v2`).
- Citations: full-text excerpts are labelled with their paper's source number, so `[n]` always
  points at source card n.
