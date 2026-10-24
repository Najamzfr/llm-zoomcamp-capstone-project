import os

os.environ["LANGSMITH_TRACING"] = "false"  # never send test runs to LangSmith

import unittest

import numpy as np

from research_assistant.embeddings import Embedder, get_embedder


class StubBackend:
    def __init__(self):
        self.calls = []

    def embed(self, texts, batch_size=256):
        self.calls.append(("embed", list(texts), batch_size))
        for index, _ in enumerate(texts):
            yield np.array([3.0, 4.0 + index], dtype="float64")

    def query_embed(self, texts):
        self.calls.append(("query_embed", list(texts)))
        for _ in texts:
            yield np.array([0.0, 2.0])


class EmbedderTests(unittest.TestCase):
    def test_documents_are_float32_and_unit_length(self):
        backend = StubBackend()
        vectors = Embedder("stub", backend=backend).embed_documents(["a", "b"], batch_size=8)
        self.assertEqual(vectors.shape, (2, 2))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], rtol=1e-6)
        self.assertEqual(backend.calls[0][2], 8)

    def test_query_uses_query_embed_and_is_normalized(self):
        backend = StubBackend()
        vector = Embedder("stub", backend=backend).embed_query("q")
        np.testing.assert_allclose(vector, [0.0, 1.0])
        self.assertEqual(backend.calls[0][0], "query_embed")

    def test_empty_documents(self):
        self.assertEqual(Embedder("stub", backend=StubBackend()).embed_documents([]).shape[0], 0)

    def test_get_embedder_is_cached(self):
        get_embedder.cache_clear()
        try:
            import research_assistant.embeddings as module

            original = module.Embedder
            module.Embedder = lambda name: object()
            self.assertIs(get_embedder("x"), get_embedder("x"))
        finally:
            module.Embedder = original
            get_embedder.cache_clear()


if __name__ == "__main__":
    unittest.main()
