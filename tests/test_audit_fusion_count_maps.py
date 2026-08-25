import json
from pathlib import Path
import tempfile
import unittest


def _write_map(root: Path, graph_key: str, fusion_counts):
    payload = {
        "profile": root.name,
        "graph_key": graph_key,
        "options": [
            {
                "option_id": option_id,
                "fusion_count": fusion_count,
                "slots": {"slot": 30 - option_id},
            }
            for option_id, fusion_count in enumerate(fusion_counts)
        ],
    }
    (root / f"{graph_key}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_complete_profile(root: Path, *, block2_counts=(0, 1)):
    root.mkdir(parents=True)
    _write_map(root, f"block2_{root.name}", block2_counts)
    _write_map(root, "block4", (0, 1))
    for graph_key in ("block5_n1", "block5_n2", "block5_n4"):
        _write_map(root, graph_key, (0, 1))


class FusionCountMapAuditTest(unittest.TestCase):
    def test_complete_zero_one_profile_passes(self):
        from rfr.preparation.fusion.audit_maps import audit_profile_dir

        with tempfile.TemporaryDirectory() as td:
            profile_dir = Path(td) / "rte"
            _write_complete_profile(profile_dir)

            result = audit_profile_dir(profile_dir, max_allowed=1)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["missing_graph_keys"], [])
        self.assertEqual(result["violations"], [])

    def test_fusion_count_above_one_reports_exact_option_and_slots(self):
        from rfr.preparation.fusion.audit_maps import audit_profile_dir

        with tempfile.TemporaryDirectory() as td:
            profile_dir = Path(td) / "sst2"
            _write_complete_profile(profile_dir, block2_counts=(0, 1, 2))

            result = audit_profile_dir(profile_dir, max_allowed=1)

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["violations"], [{
            "graph_key": "block2_sst2",
            "option_id": 2,
            "fusion_count": 2,
            "slots": {"slot": 28},
        }])

    def test_missing_required_block_graphs_fail(self):
        from rfr.preparation.fusion.audit_maps import audit_profile_dir

        with tempfile.TemporaryDirectory() as td:
            profile_dir = Path(td) / "mrpc_large"
            profile_dir.mkdir()
            _write_map(profile_dir, "block4", (0, 1))

            result = audit_profile_dir(profile_dir, max_allowed=1)

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["missing_graph_keys"], [
            "block2_mrpc_large",
            "block5_n1",
            "block5_n2",
            "block5_n4",
        ])

    def test_cli_writes_combined_json_and_returns_nonzero_on_violation(self):
        from rfr.preparation.fusion.audit_maps import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clean = root / "rte"
            bad = root / "sst2_large"
            output = root / "audit.json"
            _write_complete_profile(clean)
            _write_complete_profile(bad, block2_counts=(0, 2))

            rc = main([
                "--profile-dir", str(clean),
                "--profile-dir", str(bad),
                "--max-allowed", "1",
                "--output-json", str(output),
            ])
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(rc, 1)
        self.assertEqual(payload["status"], "fail")
        self.assertEqual([item["profile"] for item in payload["profiles"]], [
            "rte",
            "sst2_large",
        ])
        self.assertEqual(payload["profiles"][1]["violations"][0]["fusion_count"], 2)


if __name__ == "__main__":
    unittest.main()
