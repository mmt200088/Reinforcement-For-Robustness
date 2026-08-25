import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class BLBRegistryArtifactConsistencyTests(unittest.TestCase):
    def test_exported_registry_files_match_embedded_payload(self):
        from rfr.preparation.fusion.export_action_registry import (
            build_registry_payload,
            write_registry_artifacts,
        )

        with tempfile.TemporaryDirectory() as td:
            payload = build_registry_payload(profile="mrpc", num_layers=2)
            paths = write_registry_artifacts(payload, td)

            current = json.loads(Path(paths["current_code_action_registry"]).read_text(encoding="utf-8"))
            full = json.loads(Path(paths["slot_registry_full"]).read_text(encoding="utf-8"))
            effective = json.loads(Path(paths["slot_registry_effective"]).read_text(encoding="utf-8"))

            self.assertEqual(current["slot_registry_full"], full)
            self.assertEqual(current["slot_registry_effective"], effective)

            l0b1 = [
                row for row in full
                if int(row.get("layer", -1)) == 0 and row.get("block") == "block1"
            ]
            first_input = [row for row in full if row.get("block") == "first_input"]
            self.assertTrue(l0b1)
            self.assertTrue(first_input)
            l0b1_k = [
                row for row in l0b1
                if row.get("field") == "output_truncation_k"
            ]
            l0b1_rescale = [
                row for row in l0b1
                if row.get("field") != "output_truncation_k"
            ]
            self.assertEqual(len(l0b1_k), 1)
            self.assertTrue(all(row["is_effective"] for row in l0b1_k))
            self.assertTrue(
                all(not row["is_effective"] for row in l0b1_rescale + first_input)
            )

            effective_ids = {row["slot_id"] for row in effective}
            self.assertTrue(all(row["slot_id"] in effective_ids for row in l0b1_k))
            self.assertFalse(
                any(
                    row["slot_id"] in effective_ids
                    for row in l0b1_rescale + first_input
                )
            )

    def test_registry_json_artifacts_stream_without_materializing_strings(self):
        from rfr.preparation.fusion import export_action_registry as registry

        slot_record = {
            "slot_id": "L0.B1.gelu_out_sf",
            "global_index": 0,
            "layer": 0,
            "block_index": 1,
            "block": "block1",
            "field": "gelu_out_sf",
            "kind": "F",
            "operation": "fresh",
            "location": "L0.B1.gelu_out_sf",
            "semantic_type": "fresh",
            "is_effective": True,
            "is_required": True,
            "all_max_action_index": 2,
            "value_type": "scaling_factor",
            "action_values": [28, 29, 30],
            "level_values": [28, 29, 30],
            "user_prompt_semantics": "GELU output fresh scaling factor.",
        }
        payload = {
            "profile": "mrpc",
            "num_layers": 1,
            "gelu_degree": [4],
            "attn_degree": [6],
            "expected_slots_per_layer": 1,
            "registry_hash": "hash",
            "summary": {
                "slot_count": 1,
                "required_slot_count": 1,
                "ineffective_or_compat_extra_count": 0,
                "required_count_by_layer": {"0": 1},
                "block_slot_counts_per_layer": {"block1": 1},
                "per_layer_slot_count": 1,
                "first_input_tail_slots": 0,
                "full_action_length": 1,
            },
            "slot_registry_full": [slot_record],
            "slot_registry_effective": [slot_record],
            "current_code_slot_check_markdown": "# Slot check\n",
            "action_index_mapping_markdown": "# Mapping\n",
        }

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(registry.json, "dumps", side_effect=AssertionError("stream JSON artifacts")):
                paths = registry.write_registry_artifacts(payload, td)

            current = json.loads(Path(paths["current_code_action_registry"]).read_text(encoding="utf-8"))
            full = json.loads(Path(paths["slot_registry_full"]).read_text(encoding="utf-8"))
            effective = json.loads(Path(paths["slot_registry_effective"]).read_text(encoding="utf-8"))

        self.assertEqual(current["slot_registry_full"], full)
        self.assertEqual(current["slot_registry_effective"], effective)


if __name__ == "__main__":
    unittest.main()
