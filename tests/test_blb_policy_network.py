from __future__ import annotations

from dataclasses import fields
import importlib.util
import unittest


class ProductionPolicyNetworkTests(unittest.TestCase):
    def test_fixed_policy_network_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("blb_stage2_rl.policy_network")
        )

    def test_fixed_identity_and_architecture(self):
        from blb_stage2_rl.policy_network import (
            POLICY_ARCHITECTURE,
            POLICY_NETWORK_ID,
            POLICY_RL_VARIANT,
        )

        self.assertEqual(POLICY_NETWORK_ID, "shared_gtrxl_small_v1")
        self.assertEqual(
            POLICY_RL_VARIANT,
            "blb_v3_layerwise_robust_shared_gtrxl_small_v1",
        )
        self.assertEqual(
            POLICY_ARCHITECTURE,
            {"d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 256},
        )

    def test_checkpoint_requires_explicit_current_identity(self):
        from blb_stage2_rl.policy_network import (
            POLICY_NETWORK_ID,
            validate_checkpoint_policy_network,
        )

        validate_checkpoint_policy_network(
            {"policy_network_variant": POLICY_NETWORK_ID}
        )
        for checkpoint in (
            {},
            {"policy_network_variant": "shared_gtrxl_v1"},
            {"policy_network_variant": "separate_critic_mlp_v1"},
        ):
            with self.assertRaises(RuntimeError):
                validate_checkpoint_policy_network(checkpoint)

    def test_contract_binds_fixed_network_and_rejects_shape_drift(self):
        from blb_stage2_rl.policy_network import bind_policy_network_contract

        shape = {
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 256,
        }
        bound = bind_policy_network_contract(
            {"schema": "base"}, policy_shape=shape
        )
        self.assertEqual(bound["policy_network"]["policy_shape"], shape)
        self.assertEqual(
            bound["policy_network"]["variant"],
            "shared_gtrxl_small_v1",
        )
        with self.assertRaises(ValueError):
            bind_policy_network_contract(
                {"schema": "base"},
                policy_shape={**shape, "d_model": 256},
            )

    def test_sequential_policy_has_no_network_selector(self):
        from blb_stage2_rl.sequential_policy import SequentialPolicyConfig

        names = {field.name for field in fields(SequentialPolicyConfig)}
        self.assertNotIn("network_variant", names)
        self.assertNotIn("allow_custom_architecture", names)
        cfg = SequentialPolicyConfig(state_dim=8, max_step_dim=2)
        self.assertEqual(
            (cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff),
            (128, 4, 2, 256),
        )


if __name__ == "__main__":
    unittest.main()
