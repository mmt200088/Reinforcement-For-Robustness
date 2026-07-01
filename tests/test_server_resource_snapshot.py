import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "scripts" / "server_resource_snapshot.py"


def _load_snapshot_module():
    spec = importlib.util.spec_from_file_location("server_resource_snapshot", SNAPSHOT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ServerResourceSnapshotTest(unittest.TestCase):
    def test_parse_nvidia_smi_rows_normalizes_numeric_fields(self):
        snap = _load_snapshot_module()

        rows = snap.parse_nvidia_smi_csv(
            "0, NVIDIA A100-SXM4-40GB, 40960, 1024, 57\n"
            "1, NVIDIA A100-SXM4-40GB, 40960, 0, 0\n"
        )

        self.assertEqual(rows[0]["index"], 0)
        self.assertEqual(rows[0]["name"], "NVIDIA A100-SXM4-40GB")
        self.assertEqual(rows[0]["memory_total_mib"], 40960)
        self.assertEqual(rows[0]["memory_used_mib"], 1024)
        self.assertEqual(rows[0]["utilization_gpu_pct"], 57)
        self.assertEqual(rows[1]["utilization_gpu_pct"], 0)

    def test_cli_writes_json_and_markdown_from_offline_gpu_csv(self):
        snap = _load_snapshot_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            smi = root / "nvidia_smi.csv"
            smi.write_text(
                "0, NVIDIA A100-SXM4-40GB, 40960, 1024, 57\n"
                "1, NVIDIA A100-SXM4-40GB, 40960, 0, 0\n",
                encoding="utf-8",
            )
            out_json = root / "snapshot.json"
            out_md = root / "snapshot.md"

            rc = snap.main(
                [
                    "--root",
                    str(root),
                    "--nvidia-smi-csv",
                    str(smi),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["gpu_summary"]["gpu_count"], 2)
            self.assertEqual(payload["gpu_summary"]["idle_gpu_count"], 1)
            markdown = out_md.read_text(encoding="utf-8")
            self.assertIn("# Server Resource Snapshot", markdown)
            self.assertIn("GPU count: 2", markdown)
            self.assertIn("idle GPUs: 1", markdown)


if __name__ == "__main__":
    unittest.main()
