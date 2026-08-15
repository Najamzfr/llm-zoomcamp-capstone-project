"""Repository-rooted paths for local project assets."""

from .config import PROJECT_ROOT, settings

SOURCE_DATA_PATH = settings.source_data_path
INDEX_DIR = settings.database_path.parent
DATABASE_PATH = settings.database_path
VECTOR_METADATA_PATH = settings.vector_metadata_path
VECTOR_INDEX_PATH = settings.vector_index_path
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
CONFIG_PATH = settings.rrf_config_path
TRACE_DIR = settings.trace_dir
PAPER_CACHE_DIR = settings.paper_cache_dir
