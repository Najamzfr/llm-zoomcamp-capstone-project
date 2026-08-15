"""One place for local runtime settings; environment variables override defaults."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _path(name, default):
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Settings:
    source_data_path: Path = _path("RAG_SOURCE_DATA_PATH", "data/source/data.csv")
    database_path: Path = _path("RAG_DATABASE_PATH", "data/indexes/arxiv.db")
    vector_metadata_path: Path = _path("RAG_VECTOR_METADATA_PATH", "data/indexes/vector_index.npz")
    vector_index_path: Path = _path("RAG_VECTOR_INDEX_PATH", "data/indexes/vector_index.bin")
    rrf_config_path: Path = _path("RAG_RRF_CONFIG_PATH", "config/rrf_config.json")
    trace_dir: Path = _path("RAG_TRACE_DIR", "artifacts/traces")
    paper_cache_dir: Path = _path("RAG_PAPER_CACHE_DIR", "artifacts/paper_cache")
    search_backend: str = os.getenv("SEARCH_BACKEND", "sqlite_faiss")
    search_method: str = os.getenv("RAG_SEARCH_METHOD", "hybrid")
    result_count: int = _int("RAG_RESULT_COUNT", 5)
    hnsw_ef_search: int = _int("RAG_HNSW_EF_SEARCH", 50)
    hnsw_m: int = _int("RAG_HNSW_M", 16)
    hnsw_ef_construction: int = _int("RAG_HNSW_EF_CONSTRUCTION", 200)
    rrf_candidate_k: int = _int("RAG_RRF_CANDIDATE_K", 20)
    rrf_constant: int = _int("RAG_RRF_CONSTANT", 60)
    rrf_text_weight: float = _float("RAG_RRF_TEXT_WEIGHT", 0.5)
    rrf_vector_weight: float = _float("RAG_RRF_VECTOR_WEIGHT", 0.5)
    chat_model: str = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
    api_timeout_seconds: float = _float("API_TIMEOUT_SECONDS", 300.0)
    max_question_length: int = _int("RAG_MAX_QUESTION_LENGTH", 2000)
    max_answer_tokens: int = _int("RAG_MAX_ANSWER_TOKENS", 400)
    max_router_tokens: int = _int("RAG_MAX_ROUTER_TOKENS", 300)
    max_full_text_answer_tokens: int = _int("RAG_MAX_FULL_TEXT_ANSWER_TOKENS", 700)
    max_papers: int = _int("RAG_MAX_PAPERS", 3)
    max_pdf_bytes: int = _int("RAG_MAX_PDF_BYTES", 25 * 1024 * 1024)
    max_pdf_pages: int = _int("RAG_MAX_PDF_PAGES", 100)
    max_chunks_per_pdf: int = _int("RAG_MAX_CHUNKS_PER_PDF", 500)
    download_timeout_seconds: int = _int("RAG_DOWNLOAD_TIMEOUT_SECONDS", 30)
    download_delay_seconds: float = _float("RAG_DOWNLOAD_DELAY_SECONDS", 1.0)
    chunk_tokens: int = _int("RAG_CHUNK_TOKENS", 600)
    chunk_overlap: int = _int("RAG_CHUNK_OVERLAP", 80)
    max_chunks: int = _int("RAG_MAX_CHUNKS", 8)
    max_chunks_per_paper: int = _int("RAG_MAX_CHUNKS_PER_PAPER", 4)
    max_full_text_tokens: int = _int("RAG_MAX_FULL_TEXT_TOKENS", 5000)


settings = Settings()
