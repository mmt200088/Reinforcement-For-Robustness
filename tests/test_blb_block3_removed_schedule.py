"""C (2026-05-30): Stage-2 block 3 removed from the *decided* RL schedule.

Two layers of checks:

* ``Block3RemovedSourceTest`` -- torch-free source-text guarantees (always run,
  locally + CI). These pin the structural decisions of change C.
* ``Block3RemovedBehaviorTest`` -- behavioral checks that import
  ``blb_stage2_rl.action_space``. The package ``__init__`` imports torch, so
  these run in CI / on the server (torch installed) and skip cleanly on a
  torch-free dev box.
"""
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_SPACE = REPO_ROOT / "blb_stage2_rl" / "action_space.py"
SEQ_ENV = REPO_ROOT / "blb_stage2_rl" / "sequential_env.py"
BRIDGE = REPO_ROOT / "blb_rl_bridge.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class Block3RemovedSourceTest(unittest.TestCase):
    """Static (torch-free) guarantees about the block-3-removed schedule."""

    def test_block_order_tuples_exclude_block3(self):
        text = _read(ACTION_SPACE)
        self.assertIn("_LAYER0_BLOCK_ORDER: Tuple[int, ...] = (2, 4, 5)", text)
        self.assertIn("_LAYER_GE_1_BLOCK_ORDER: Tuple[int, ...] = (1, 2, 4, 5)", text)

    def test_horizon_formula_drops_block3(self):
        text = _read(ACTION_SPACE)
        # 3 (layer 0: B2,B4,B5) + (L-1)*4 (B1,B2,B4,B5) -> 47 for L=12.
        self.assertIn("return 3 + (L - 1) * 4", text)
        self.assertNotIn("return 4 + (L - 1) * 5", text)

    def test_block3_field_table_still_defined(self):
        # The legacy full action vector KEEPS block 3's slots (frozen at baseline);
        # only the decided schedule drops them. _BLOCK3_FIELDS must stay wired.
        text = _read(ACTION_SPACE)
        self.assertIn("_BLOCK3_FIELDS", text)
        self.assertIn("3: _BLOCK3_FIELDS", text)

    def test_sequential_env_freezes_block3_at_baseline(self):
        text = _read(SEQ_ENV)
        # _pending_full_vec is seeded with the all-max (== static_skeletons
        # baseline) vector so block 3 -- never written by a decided step -- stays
        # frozen at baseline. The old all-min seed must be gone.
        self.assertIn("make_all_max_action_vector(self.num_layers)", text)
        self.assertNotIn("empty_full_action_vec(self.num_layers)", text)

    def test_bridge_never_installs_block3_noise(self):
        text = _read(BRIDGE)
        # The install *call* must be gone (method name may survive only in a
        # comment). block3_cfgs stays in the signature for API compatibility.
        self.assertNotIn("self.handler.replace_layer_block3_noise(", text)
        self.assertIn("block3_cfgs", text)


class Block3RemovedBehaviorTest(unittest.TestCase):
    """Behavioral checks; need torch (blb_stage2_rl.__init__ imports it)."""

    def setUp(self):
        try:
            from blb_stage2_rl import action_space as A
        except Exception as exc:  # torch/transformers absent on a dev box
            self.skipTest(f"blb_stage2_rl.action_space unimportable: {exc}")
        self.A = A

    def test_step_schedule_has_47_steps_no_block3(self):
        sched = self.A.step_schedule(12)
        self.assertEqual(len(sched), 47)
        self.assertEqual(self.A.horizon_for_num_layers(12), 47)
        block_idxs = {s.block_idx for s in sched}
        self.assertNotIn(3, block_idxs)
        # Layer 0 has no block 1; every other decided block is present.
        self.assertEqual(block_idxs, {1, 2, 4, 5})

    def test_full_action_vector_keeps_block3_dims(self):
        # The legacy full vec is UNCHANGED -- block 3's slots still exist (frozen),
        # so make_all_max must still produce a full-width vector including them.
        self.assertGreater(len(self.A.block_dims(3)), 0)
        dims = self.A.action_dims_for_config(12)
        vec = self.A.make_all_max_action_vector(12)
        self.assertEqual(len(vec), len(dims))


if __name__ == "__main__":
    unittest.main()
