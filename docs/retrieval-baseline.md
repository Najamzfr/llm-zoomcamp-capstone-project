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
`uv run python -m scripts.evaluate` after ingesting the local indexes to record
text, exact-vector, and HNSW latency figures on the same ground-truth set.

## After switching embeddings to fastembed (ONNX, no PyTorch)

Same `all-MiniLM-L6-v2` model. Parity check on 300 random papers: cosine similarity between
fastembed and the stored sentence-transformers vectors was 1.00000 (mean and minimum), so the
FAISS index and the Qdrant collection were kept as-is. `uv run python -m scripts.evaluate`
(50 ground-truth questions):

| Search method | Precision@5 | Recall@5 | Hit Rate@5 | MRR@5 | Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text (FTS5) | 0.408 | 0.408 | 0.900 | 0.786 | 60.80 |
| Exact vector | 0.416 | 0.416 | 0.880 | 0.739 | 6.78 |
| HNSW ANN | 0.416 | 0.416 | 0.880 | 0.739 | 5.98 |
| Hybrid (RRF) | 0.456 | 0.456 | 0.920 | 0.781 | 104.50 |

ANN Recall@5 vs exact: 1.000.
