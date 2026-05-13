import json
import tempfile
import unittest
from pathlib import Path


class BLBRegistryArtifactConsistencyTests(unittest.TestCase):
    def test_exported_registry_files_match_embedded_payload(self):
        from scripts.blb_export_action_registry import (
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
            self.assertTrue(all(not row["is_effective"] for row in l0b1 + first_input))

            effective_ids = {row["slot_id"] for row in effective}
            self.assertFalse(any(row["slot_id"] in effective_ids for row in l0b1 + first_input))


if __name__ == "__main__":
    unittest.main()
