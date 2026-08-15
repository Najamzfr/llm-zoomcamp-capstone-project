# Retrieval / RAG baseline

Recorded before Phase 1 configuration changes. The local backend is SQLite FTS5
plus FAISS HNSW, fused with reciprocal-rank fusion (RRF); GPT-4o-mini provides
abstract-first answers and optionally escalates to PDF chunks.

| Measure | Baseline |
| --- | ---: |
| Hybrid RRF Precision@5 | 0.442 |
| Hybrid RRF Recall@5 | 0.442 |
| Hybrid RRF Hit Rate@5 | 0.960 |
| Hybrid RRF MRR@5 | 0.734 |
| RRF objective | 0.712 |
| RRF candidates / constant | 50 / 90 |
| RRF text / vector weights | 0.8 / 0.2 |

These are the checked-in tuning results in `config/rrf_config.json`. Re-run
`uv run python scripts/evaluate.py` after ingesting the local indexes to record
text, exact-vector, and HNSW latency figures on the same ground-truth set.
