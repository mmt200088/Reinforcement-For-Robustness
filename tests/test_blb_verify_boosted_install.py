import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class BoostedInstallVerifierTest(unittest.TestCase):
    def test_module_import_is_dependency_light(self):
        import scripts.blb_verify_boosted_install as verifier

        self.assertTrue(callable(verifier.verify_map))

    def test_verify_map_reuses_supplied_payload_without_rereading_file(self):
        import scripts.blb_verify_boosted_install as verifier

        payload = {
            "graph_key": "block2_mrpc",
            "block_idx": 2,
            "gelu_degree": 4,
            "attn_degree": 2,
            "options": [],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "block2_mrpc.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(Path, "read_text", side_effect=AssertionError("payload was reread")):
                checked, problems = verifier.verify_map(
                    path,
                    "mrpc",
                    "Rescale_optimizer",
                    12,
                    payload=payload,
                )

        self.assertEqual((checked, problems), (0, 0))

    def test_map_loop_passes_loaded_payload_to_verifier(self):
        import scripts.blb_verify_boosted_install as verifier

        payload = {
            "graph_key": "block2_mrpc",
            "block_idx": 2,
            "options": [{"option_id": 1, "fusion_count": 1, "boosted": True}],
        }
        with tempfile.TemporaryDirectory() as td:
            map_dir = Path(td)
            path = map_dir / "block2_mrpc.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            seen_payloads = []

            def fake_verify(path_arg, profile, ro_root, num_layers, *, payload=None):
                self.assertEqual(path_arg, path)
                self.assertEqual(profile, "mrpc")
                self.assertEqual(ro_root, "RO")
                self.assertEqual(num_layers, 12)
                seen_payloads.append(payload)
                return 1, 0

            checked, problems = verifier._verify_maps(
                map_dir,
                profile="mrpc",
                ro_root="RO",
                num_layers=12,
                verify_fn=fake_verify,
            )

        self.assertEqual((checked, problems), (1, 0))
        self.assertEqual(seen_payloads, [payload])

    def test_map_loop_discovers_maps_without_path_glob_or_sidecar_reads(self):
        import scripts.blb_verify_boosted_install as verifier

        map_payload = {
            "graph_key": "block2_mrpc",
            "block_idx": 2,
            "options": [{"option_id": 1, "fusion_count": 1, "boosted": True}],
        }
        with tempfile.TemporaryDirectory() as td:
            map_dir = Path(td)
            map_path = map_dir / "block2_mrpc.json"
            sidecar_path = map_dir / "map_summary.json"
            hidden_sidecar_path = map_dir / "_summary.json"
            map_path.write_text(json.dumps(map_payload), encoding="utf-8")
            sidecar_path.write_text(json.dumps({"not": "a fusion map"}), encoding="utf-8")
            hidden_sidecar_path.write_text(json.dumps({"not": "a fusion map"}), encoding="utf-8")
            (map_dir / "notes.txt").write_text("", encoding="utf-8")

            original_read_text = Path.read_text
            seen_paths = []

            def guarded_read_text(path, *args, **kwargs):
                if Path(path) in {sidecar_path, hidden_sidecar_path}:
                    raise AssertionError("sidecars should not be read as fusion maps")
                return original_read_text(path, *args, **kwargs)

            def fake_verify(path_arg, _profile, _ro_root, _num_layers, *, payload=None):
                seen_paths.append(path_arg)
                self.assertEqual(payload, map_payload)
                return 1, 0

            with (
                mock.patch.object(Path, "glob", side_effect=AssertionError("map discovery should not use Path.glob")),
                mock.patch.object(Path, "read_text", guarded_read_text),
            ):
                checked, problems = verifier._verify_maps(
                    map_dir,
                    profile="mrpc",
                    ro_root="RO",
                    num_layers=12,
                    verify_fn=fake_verify,
                )

        self.assertEqual((checked, problems), (1, 0))
        self.assertEqual(seen_paths, [map_path])
