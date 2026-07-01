import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import report_fusion_count_map as report


class FusionCountMapReportTest(unittest.TestCase):
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
