import importlib.util
from pathlib import Path
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR_PATH = REPO_ROOT / "scripts" / "stage2_first10k_monitor.py"


def _load_monitor_module():
    spec = importlib.util.spec_from_file_location("stage2_first10k_monitor", MONITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Stage2First10kMonitorTest(unittest.TestCase):
    def test_read_jsonl_streams_lines_without_read_text(self):
        monitor = _load_monitor_module()
        original_read_text = Path.read_text

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "episodes.jsonl"
            path.write_text('{"episode": 0, "total_reward": 1.0}\n\nnot-json\n{"episode": 1}\n', encoding="utf-8")

            def fail_read_text(_path, *args, **kwargs):
                raise AssertionError("_read_jsonl should stream file lines")

            try:
                Path.read_text = fail_read_text
                rows = monitor._read_jsonl(path)
            finally:
                Path.read_text = original_read_text

        self.assertEqual([row["episode"] for row in rows], [0, 1])

    def test_gpu_stats_ignores_directory_path(self):
        monitor = _load_monitor_module()

        with tempfile.TemporaryDirectory() as td:
            summary = monitor._gpu_stats(Path(td))

        self.assertEqual(summary, {"samples": 0, "by_gpu": {}})


if __name__ == "__main__":
    unittest.main()
