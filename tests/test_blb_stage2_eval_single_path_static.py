"""Static guard for the Stage-2 action-to-model cfg write-back path."""
from __future__ import annotations

import pathlib
import unittest


class Stage2EvalSinglePathStaticTest(unittest.TestCase):
    def test_executable_eval_paths_use_shared_optimizer_writeback_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        checked = [
            repo / "blb_stage2_rl" / "env.py",
            repo / "blb_stage2_rl" / "sequential_env.py",
            repo / "Paean" / "blb_action_eval.py",
        ]
        forbidden = [
            "apply_optimizer_output_to_cfg(",
            "sync_block2_qk_binding(",
            "sync_block2_aux_fresh_binding(",
            "sync_block4_v_mask_binding(",
            "sync_block5_aux_fresh_binding(",
            "_strip_layer_suffix(",
        ]
        offenders = []
        for path in checked:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(repo)} contains {token}")
            self.assertIn(
                "apply_optimizer_outputs_to_cfgs",
                text,
                f"{path.relative_to(repo)} must delegate optimizer write-back "
                "through the shared Stage-2 helper",
            )
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
