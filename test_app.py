import unittest
from unittest.mock import patch

import httpx

import app


class AppClientTests(unittest.TestCase):
    @patch.object(app.HTTP_CLIENT, "post")
    def test_success(self, post):
        post.return_value = httpx.Response(
            200,
            json={"answer": "stub answer"},
            request=httpx.Request("POST", "http://localhost:8000/ask"),
        )

        self.assertEqual(app._ask("question")["answer"], "stub answer")

    @patch.object(app.HTTP_CLIENT, "post")
    def test_connection_error_is_specific(self, post):
        request = httpx.Request("POST", "http://localhost:8000/ask")
        post.side_effect = httpx.ConnectError("refused", request=request)

        with self.assertRaisesRegex(RuntimeError, "Cannot connect to FastAPI"):
            app._ask("question")

    @patch.object(app.HTTP_CLIENT, "post")
    def test_timeout_error_is_specific(self, post):
        request = httpx.Request("POST", "http://localhost:8000/ask")
        post.side_effect = httpx.ReadTimeout("timed out", request=request)

        expected = f"exceeded {app.API_TIMEOUT_SECONDS:g} seconds"
        with self.assertRaisesRegex(RuntimeError, expected):
            app._ask("question")

    @patch.object(app.HTTP_CLIENT, "post")
    def test_backend_error_keeps_safe_detail(self, post):
        post.return_value = httpx.Response(
            503,
            json={"detail": "The research assistant is unavailable."},
            request=httpx.Request("POST", "http://localhost:8000/ask"),
        )

        with self.assertRaisesRegex(RuntimeError, "FastAPI returned 503"):
            app._ask("question")


if __name__ == "__main__":
    unittest.main()