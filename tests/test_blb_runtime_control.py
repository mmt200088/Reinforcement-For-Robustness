from __future__ import annotations

from pathlib import Path
import tempfile
import unittest


class RuntimeControlTests(unittest.TestCase):
    def test_stop_file_is_consumed_and_state_resets(self):
        from rfr.search.runtime.control import (
            STOP_FLAG_FILENAME,
            consume_stop_flag,
            graceful_stop_requested,
            request_graceful_stop,
            reset_graceful_stop_state,
        )

        self.assertEqual(STOP_FLAG_FILENAME, "STOP_RL")
        with tempfile.TemporaryDirectory() as directory:
            stop_path = Path(directory) / STOP_FLAG_FILENAME
            reset_graceful_stop_state()
            self.assertFalse(graceful_stop_requested(stop_path))
            stop_path.touch()
            self.assertTrue(graceful_stop_requested(stop_path))
            consume_stop_flag(stop_path)
            reset_graceful_stop_state()
            self.assertFalse(stop_path.exists())
            self.assertFalse(graceful_stop_requested(stop_path))
            request_graceful_stop()
            self.assertTrue(graceful_stop_requested())


if __name__ == "__main__":
    unittest.main()
