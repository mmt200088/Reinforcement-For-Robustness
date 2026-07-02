import json
import tempfile
import unittest
from pathlib import Path
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

