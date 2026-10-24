# Monitoring with Grafana Cloud (free)

The API exposes Prometheus metrics at `http://localhost:8000/metrics`. Grafana Alloy, a single
executable, scrapes them and pushes them to Grafana Cloud's free hosted Prometheus, where the
dashboard in `grafana/research_rag_overview.json` (12 panels) displays them. Nothing needs to be
installed locally besides Alloy, and no credit card is required.

## 1. Create the Grafana Cloud stack
1. Sign up at https://grafana.com and create a free stack.
2. In the Cloud Portal open your stack → **Prometheus** → **Details** (or *Send Metrics*).
3. Copy:
   - **Remote write endpoint**: ends in `/api/prom/push`
   - **Username / Instance ID**: a number
4. Create an access policy token with the `metrics:write` scope and copy it.

## 2. Put the credentials in `.env`
```
GRAFANA_CLOUD_URL=https://prometheus-prod-XX-....grafana.net/api/prom/push
GRAFANA_CLOUD_USER=1234567
GRAFANA_CLOUD_API_KEY=glc_...
ALLOY_PATH=D:/tools/alloy/alloy-windows-amd64.exe
```
`.env` is gitignored, so the token never reaches the repository.

## 3. Download Alloy and run it
1. From https://github.com/grafana/alloy/releases/latest download `alloy-windows-amd64.exe.zip`
   and extract it to `D:\tools\alloy\` (keeps it off the C: drive; no installer or service).
   Alternatively `winget install GrafanaLabs.Alloy` (installs to Program Files, which the script also
   finds). The installer adds an idle "Alloy" Windows service on port 12345; the script uses port
   12346 so both can coexist, or stop the service with `Stop-Service Alloy` (admin).
2. Start the API: `uv run uvicorn api:app`.
3. In a second PowerShell window, from the project root:
   ```powershell
   .\monitoring\run-alloy.ps1
   ```
   (If scripts are blocked: `Set-ExecutionPolicy -Scope Process RemoteSigned` first.)

Alloy's UI at http://127.0.0.1:12346 shows `prometheus.scrape.rag_api` and
`prometheus.remote_write.grafana_cloud` as healthy.

## 4. Import the dashboard
1. Grafana → **Dashboards** → **New** → **Import** → upload `grafana/research_rag_overview.json`.
2. When prompted for the Prometheus data source, pick your stack's `grafanacloud-<name>-prom`.
3. Ask a few questions in the app and rate some answers; the panels fill within about a minute.

## Panels
Requests/min, error rate, latency p50/p95, active requests, token usage, cost rate,
abstract vs full-text answers, full-text retrieval success, judge outcomes, routing decisions,
user feedback (👍/👎) and satisfaction %.

## Metrics reference
| Metric | Meaning |
|---|---|
| `rag_requests_total{endpoint,status_class}` | API requests by route and 2xx/4xx/5xx |
| `rag_request_latency_seconds` | Request latency histogram |
| `rag_active_requests` | Requests in progress |
| `rag_input_tokens_total`, `rag_output_tokens_total` | OpenAI tokens |
| `rag_estimated_cost_usd_total` | Estimated OpenAI cost |
| `rag_answer_stages_total{answer_stage}` | Abstract / full-text / fallback answers |
| `rag_feedback_total{rating}` | Thumbs up / down from the web UI |

Counters reset when the API restarts; Grafana's `rate()`/`increase()` handle that.
