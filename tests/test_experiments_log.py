import unittest
from unittest import mock

from tools import experiments_log


class ExperimentsLogTest(unittest.TestCase):
    def test_git_info_bounds_git_commands_with_timeout(self):
        calls = []

        def fake_check_output(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[:2] == ["git", "rev-parse"]:
                return b"abc123\n"
            if cmd[:2] == ["git", "status"]:
                return b""
            raise AssertionError(f"unexpected command: {cmd}")

        with mock.patch.object(experiments_log.subprocess, "check_output", fake_check_output):
            info = experiments_log._git_info()

        self.assertEqual(info["git_commit"], "abc123")
        self.assertFalse(info["git_dirty"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(kwargs.get("timeout") == 5 for _, kwargs in calls))


if __name__ == "__main__":
    unittest.main()
