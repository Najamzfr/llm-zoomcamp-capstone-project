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
5. Click **Deploy**. The first build takes a few minutes; the app lives at
   `https://<your-app-name>.streamlit.app`.

## Notes
- The public app spends your OpenAI credit, which is why the rate limits above are lower than local defaults.
  Also set a monthly limit in the OpenAI dashboard.
- Apps sleep after a period without visitors; the first visit wakes them (about a minute).
- Memory: search ~260 MB; full-text answers peak ~400 MB (PDF chunks are embedded 4 at a time).
- Prometheus/Grafana monitoring is for the local FastAPI app; on the cloud build use LangSmith traces
  and the 👍/👎 feedback recorded there.
- Run it locally: `uv run --with "streamlit>=1.50" streamlit run deploy/streamlit_app.py`.
- Look and feel: chat layout with a live research-progress panel; fonts, colours and light/dark
  palettes live in `.streamlit/config.toml` (requires Streamlit 1.50+, pinned in `deploy/requirements.txt`).
