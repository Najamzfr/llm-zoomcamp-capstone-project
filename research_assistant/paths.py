"""Repository-rooted paths for local project assets."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_PATH = PROJECT_ROOT / "data" / "source" / "data.csv"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
DATABASE_PATH = INDEX_DIR / "arxiv.db"
VECTOR_METADATA_PATH = INDEX_DIR / "vector_index.npz"
VECTOR_INDEX_PATH = INDEX_DIR / "vector_index.bin"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
CONFIG_PATH = PROJECT_ROOT / "config" / "rrf_config.json"
TRACE_DIR = PROJECT_ROOT / "artifacts" / "traces"
PAPER_CACHE_DIR = PROJECT_ROOT / "artifacts" / "paper_cache"
