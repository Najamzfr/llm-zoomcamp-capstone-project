import os
import unittest
from unittest.mock import patch

from research_assistant import tracing

DISABLED = {"LANGSMITH_TRACING": "false", "LANGSMITH_API_KEY": ""}


class TracingTests(unittest.TestCase):
    def test_disabled_without_flag_or_key(self):
        with patch.dict(os.environ, DISABLED):
            self.assertFalse(tracing.tracing_enabled())
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": ""}):
            self.assertFalse(tracing.tracing_enabled())

    def test_enabled_with_flag_and_key(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "k"}):
            self.assertTrue(tracing.tracing_enabled())

    def test_helpers_are_noops_when_disabled(self):
        client = object()
        with patch.dict(os.environ, DISABLED):
            self.assertIs(tracing.wrap_client(client), client)
            self.assertIsNone(tracing.current_run_id())
            self.assertFalse(tracing.send_feedback("run", "up"))
            with tracing.span("step"):
                pass
            tracing.add_run_metadata({"a": 1})

    def test_decorated_functions_keep_their_results(self):
        @tracing.traceable(name="add", run_type="tool")
        def add(a, b):
            return a + b

        class Thing:
            @tracing.traceable(name="method")
            def double(self, value):
                return value * 2

        with patch.dict(os.environ, DISABLED):
            self.assertEqual(add(2, 3), 5)
            self.assertEqual(Thing().double(4), 8)

    def test_drop_self(self):
        self.assertEqual(tracing.drop_self({"self": 1, "query": "q"}), {"query": "q"})


if __name__ == "__main__":
    unittest.main()
