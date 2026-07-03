import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "scripts" / "render_fusion_count_slots_eval_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("render_fusion_count_slots_eval_report", REPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FusionCountSlotsEvalReportTest(unittest.TestCase):
    def test_load_maps_scans_directory_without_path_glob_or_sidecar_reads(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecar = root / "map_summary.json"
            sidecar.write_text("{not-json", encoding="utf-8")
            (root / "_summary.json").write_text("{}", encoding="utf-8")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "block2_mrpc.json").write_text(
                json.dumps({"graph_key": "block2_mrpc", "options": []}),
                encoding="utf-8",
            )
            (root / "block1_mrpc.json").write_text(
                json.dumps({"graph_key": "block1_mrpc", "options": []}),
                encoding="utf-8",
            )

            original_read_text = Path.read_text

            def guarded_read_text(path, *args, **kwargs):
                if Path(path) == sidecar:
                    raise AssertionError("sidecar should not be opened as a map")
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(
                Path,
                "glob",
                side_effect=AssertionError("map loading should not use Path.glob"),
            ):
                with mock.patch.object(Path, "read_text", guarded_read_text):
                    maps = report._load_maps(root)

        self.assertEqual(list(maps), ["block1_mrpc", "block2_mrpc"])

    def test_option_lookup_indexes_graph_options_once(self):
        report = _load_report_module()

        class SinglePassOptions(list):
            def __init__(self, values):
                super().__init__(values)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise AssertionError("option lookup should reuse an index after first scan")
                return super().__iter__()

        options = SinglePassOptions(
            [
                {"option_id": 1, "fusion_count": 0},
                {"option_id": 2, "fusion_count": 1},
            ]
        )
        graph = {"options": options}

        first = report._option_by_id(graph, 1)
        second = report._option_by_id(graph, 2)

        self.assertEqual(first["fusion_count"], 0)
        self.assertEqual(second["fusion_count"], 1)
        self.assertEqual(options.iterations, 1)


if __name__ == "__main__":
    unittest.main()
