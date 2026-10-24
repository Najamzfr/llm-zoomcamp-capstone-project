"""Lightweight ONNX sentence embeddings (fastembed) shared by search and PDF retrieval."""

from functools import lru_cache

import numpy as np

from .config import settings


def _normalize(vectors):
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class Embedder:
    """Normalized query/document embeddings; same vectors as sentence-transformers."""

    def __init__(self, model_name=settings.embedding_model, cache_dir=settings.model_cache_dir, backend=None):
        if backend is None:
            from fastembed import TextEmbedding

            backend = TextEmbedding(model_name, cache_dir=str(cache_dir))
        self.model_name = model_name
        self.backend = backend

    def embed_documents(self, texts, batch_size=64):
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        return _normalize(list(self.backend.embed(texts, batch_size=batch_size)))

    def embed_query(self, text):
        embed = getattr(self.backend, "query_embed", self.backend.embed)
        return _normalize(next(iter(embed([text]))))


@lru_cache(maxsize=2)
def get_embedder(model_name=settings.embedding_model):
    return Embedder(model_name)
