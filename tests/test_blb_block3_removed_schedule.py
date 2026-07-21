"""Stage-2 Block3 keeps baseline SFs while exposing a real truncation-K action.

Two layers of checks:

* ``Block3RuntimeSourceTest`` -- torch-free source-text guarantees (always run,
  locally + CI).
* ``Block3RuntimeBehaviorTest`` -- behavioral checks that import
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


class Block3RuntimeSourceTest(unittest.TestCase):
    """Static guarantees for baseline-owned SFs plus runtime-owned K."""

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

    def test_optimizer_requests_do_not_skip_block3(self):
        text = _read(ACTION_SPACE)
        start = text.index("def build_optimizer_requests(")
        end = text.index("\n    return out", start) + len("\n    return out")
        body = text[start:end]
        self.assertNotIn("if int(block_idx) == 3:", body)

    def test_bridge_installs_block3_noise(self):
        text = _read(BRIDGE)
        self.assertIn("self.handler.replace_layer_block3_noise(", text)
        self.assertIn('add("block3")', text)


class Block3RuntimeBehaviorTest(unittest.TestCase):
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

    def test_optimizer_requests_include_every_block3_layer(self):
        layers = 2
        decoded = self.A.action_vector_to_cfgs(
            self.A.make_all_max_action_vector(layers),
            self.A.load_max_sfs("mrpc"),
            num_layers=layers,
            gelu_degree=[4] * layers,
            attn_degree=[6] * layers,
        )
        requests = self.A.build_optimizer_requests("mrpc", decoded.cfgs_dict())
        block3_names = sorted(
            name for name, (block_name, _cfg) in requests.items()
            if block_name == "block3"
        )
        self.assertEqual(block3_names, ["block3_exp_n6_L0", "block3_exp_n6_L1"])

    def test_bridge_installs_and_restores_block3_per_layer(self):
        from blb_rl_bridge import BLBNoiseRLBridge

        class Handler:
            def __init__(self):
                self.install_calls = []
                self.restore_calls = []

            def replace_layer_block3_noise(self, **kwargs):
                self.install_calls.append(kwargs)

            def restore_layer_block3_noise(self, **kwargs):
                self.restore_calls.append(kwargs)

            def __getattr__(self, name):
                if name.startswith(("replace_layer_block", "restore_layer_block")):
                    return lambda **_kwargs: None
                raise AttributeError(name)

        handler = Handler()
        bridge = BLBNoiseRLBridge(handler)
        cfgs = {0: object(), 1: object()}

        bridge.apply(block3_cfgs=cfgs)

        self.assertEqual(len(handler.install_calls), 1)
        self.assertEqual(handler.install_calls[0]["layer_indices"], [0, 1])
        self.assertEqual(handler.install_calls[0]["cfg_per_layer"], cfgs)
        self.assertEqual(bridge.installed_layers(), {0: {"block3"}, 1: {"block3"}})

        bridge.clear()
        self.assertEqual(handler.restore_calls[0]["layer_indices"], [0, 1])


if __name__ == "__main__":
    unittest.main()
