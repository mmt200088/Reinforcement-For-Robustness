from types import SimpleNamespace
import unittest

import numpy as np

from blb_stage2_rl.action_space import K_LEVELS, load_max_sfs
from blb_stage2_rl.fusion_count_map import FusionCountMap
from blb_stage2_rl.sequential_env import BLBStage2SequentialEnv


class _CapturingBridge:
    def __init__(self):
        self.cfg = None
        self.invoker = SimpleNamespace(baselines={})

    def evaluate(self, *, config_name, block_name, cfg):
        self.cfg = cfg
        return SimpleNamespace(
            valid=True,
            total_bits=240,
            fusion_count=1,
            invalid_chain=None,
        )


class FixedFusionSequentialEnvTest(unittest.TestCase):
    def _env(self):
        bridge = _CapturingBridge()
        base = SimpleNamespace(
            num_layers=12,
            env_cfg=SimpleNamespace(profile="mrpc"),
            max_sfs=load_max_sfs("mrpc"),
            gelu_degree=np.asarray([4] * 12, dtype=int),
            attn_degree=np.asarray([6] * 12, dtype=int),
            rescale_bridge=bridge,
        )
        return BLBStage2SequentialEnv(
            base_env=base,
            fusion_map=FusionCountMap.load("mrpc"),
        ), bridge

    def test_block2_local_zero_uses_real_fusion_one_for_boost_and_bookkeeping(self):
        env, bridge = self._env()

        result = env.evaluate_step([0, 3])

        self.assertEqual(result["policy_option_index"], 0)
        self.assertEqual(result["map_option_id"], 1)
        self.assertEqual(result["fusion_choice"].fusion_count, 1)
        self.assertEqual(result["fusion_choice"].k_value, K_LEVELS[3])
        self.assertIsNotNone(result["boosted_field_values"])
        self.assertIn(K_LEVELS[3], result["boosted_field_values"].values())
        self.assertIs(result["block_cfg"], bridge.cfg)

    def test_fixed_eval_can_override_block2_to_control_fusion_zero(self):
        env, bridge = self._env()

        result = env.evaluate_step([0, 3], map_option_id_override=0)

        self.assertEqual(result["policy_option_index"], 0)
        self.assertEqual(result["map_option_id"], 0)
        self.assertEqual(result["fusion_choice"].fusion_count, 0)
        self.assertEqual(result["fusion_choice"].k_value, K_LEVELS[3])
        self.assertIsNone(result["boosted_field_values"])
        self.assertIs(result["block_cfg"], bridge.cfg)


if __name__ == "__main__":
    unittest.main()
