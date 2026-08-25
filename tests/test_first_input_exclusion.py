import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class FirstInputExclusionTests(unittest.TestCase):
    def test_sequential_stage2_schedule_does_not_expose_first_input(self):
        source = (REPO_ROOT / "src/rfr/search/common/action_space.py").read_text(encoding="utf-8")

        self.assertNotIn('slot_field_names.append("__first_input_sf__")', source)
        self.assertNotIn("slot_dims.append(LEVELS_FIRST_INPUT)", source)
        self.assertNotIn("full_vec_offsets.append(fi_offset)", source)
        self.assertNotIn("includes_first_input", source)

    def test_action_slot_list_rejects_first_input_override(self):
        source = (REPO_ROOT / "src/rfr/search/common/action_io.py").read_text(encoding="utf-8")
        marker = 'if label == "L0.first_input.F" or field_name == "first_input_sf":'
        start = source.index(marker)
        end = source.index('        if kind == "K":', start)
        branch = source[start:end]

        self.assertIn("raise ValueError", branch)
        self.assertIn("first_input", branch)
        self.assertIn("deprecated", branch)

    def test_paean_action_grid_rejects_first_input_selector(self):
        source = (REPO_ROOT / "Paean" / "action_grid.py").read_text(encoding="utf-8")
        marker = 'if name in ("first_input", "firstinput"):'
        start = source.index(marker)
        end = source.index("    for layer_idx", start)
        branch = source[start:end]

        self.assertIn("raise ValueError", branch)
        self.assertIn("first_input", branch)
        self.assertIn("deprecated", branch)

    def test_final_eval_reports_do_not_emit_first_input_config(self):
        source = (REPO_ROOT / "Paean" / "blb_action_eval.py").read_text(encoding="utf-8")

        self.assertNotIn('"first_input_sf": int(decoded.first_input_sf)', source)
        self.assertNotIn('"path": "first_input.fresh"', source)
        self.assertNotIn("first_input_sf:", source)


if __name__ == "__main__":
    unittest.main()
