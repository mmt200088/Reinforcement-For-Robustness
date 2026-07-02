import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import report_fusion_count_map as report


class FusionCountMapReportTest(unittest.TestCase):
    def test_options_in_id_order_skips_sort_when_already_ordered(self):
        options = [
            {"option_id": 0, "fusion_count": 0},
            {"option_id": 1, "fusion_count": 1},
            {"option_id": 2, "fusion_count": 1},
        ]

        def fail_sorted(*_args, **_kwargs):
            raise AssertionError("ordered options should not be sorted again")

        original_sorted = getattr(report, "sorted", None)
        report.sorted = fail_sorted
        try:
            ordered = report._options_in_id_order(options)
        finally:
            if original_sorted is None:
                delattr(report, "sorted")
            else:
                report.sorted = original_sorted

        self.assertEqual([int(item["option_id"]) for item in ordered], [0, 1, 2])

    def test_options_in_id_order_sorts_unordered_options(self):
        options = [
            {"option_id": 2, "fusion_count": 1},
            {"option_id": 0, "fusion_count": 0},
            {"option_id": 1, "fusion_count": 1},
        ]

        ordered = report._options_in_id_order(options)

        self.assertEqual([int(item["option_id"]) for item in ordered], [0, 1, 2])

    def test_choose_option_scans_candidates_without_sorting(self):
        graph = {
            "graph_key": "blockX",
            "options": [
                {"option_id": 8, "fusion_count": 0, "total_bits": 1.0},
                {"option_id": 3, "fusion_count": 1, "total_bits": 99.0},
                {"option_id": 2, "fusion_count": 1, "total_bits": 100.0},
                {"option_id": 4, "fusion_count": 1, "total_bits": 0.0},
            ],
        }

        def fail_sorted(*_args, **_kwargs):
            raise AssertionError("_choose_option should not sort candidate lists")

        original_sorted = getattr(report, "sorted", None)
        report.sorted = fail_sorted
        try:
            option_id, count, clamped = report._choose_option(graph, 1)
        finally:
            if original_sorted is None:
                delattr(report, "sorted")
            else:
                report.sorted = original_sorted

        self.assertEqual(option_id, 2)
        self.assertEqual(count, 1)
        self.assertFalse(clamped)

    def test_load_maps_does_not_read_non_map_sidecars(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "block1_mrpc.json").write_text(
                json.dumps({
                    "graph_key": "block1_mrpc",
                    "options": [{"option_id": 0, "fusion_count": 0}],
                }),
                encoding="utf-8",
            )
            sidecar = root / "map_summary.json"
            sidecar.write_text("{not-json", encoding="utf-8")

            original_read_text = Path.read_text

            def guarded_read_text(path, *args, **kwargs):
                if Path(path) == sidecar:
                    raise AssertionError("sidecar should not be opened as a map")
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", guarded_read_text):
                graphs = report._load_maps(root)

            self.assertEqual(list(graphs), ["block1_mrpc"])


if __name__ == "__main__":
    unittest.main()
