from __future__ import annotations

import copy
from pathlib import Path
import unittest


class PolicyNetworkVariantContractTest(unittest.TestCase):
    def test_all_stage2_policy_construction_paths_receive_requested_variant(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8"
        )
        legacy_start = source.index("# Checkpoint variant:")
        legacy_end = source.index("# Warmstart:", legacy_start)
        legacy_policy_setup = source[legacy_start:legacy_end]
        self.assertIn(
            'network_variant=getattr(train_cfg, "policy_network_variant", None)',
            legacy_policy_setup,
        )

    def test_ppo_diagnostics_schema_covers_network_ablation_evidence(self):
        from dataclasses import fields

        from blb_stage2_rl.diagnostics import PPOUpdateStats

        names = {field.name for field in fields(PPOUpdateStats)}
        self.assertTrue({
            "value_explained_variance_pre",
            "value_explained_variance_post",
            "actor_critic_shared_grad_cosine",
            "preclip_grad_norm_mean",
            "entropy_per_slot",
            "approx_kl_per_slot",
            "clip_fraction_per_slot",
            "raw_advantage_snr_per_slot",
        }.issubset(names))

    def test_registry_exposes_small_default_and_retained_large_ablations(self):
        from blb_stage2_rl.network_variants import (
            DEFAULT_POLICY_NETWORK_VARIANT,
            FRESH_POLICY_NETWORK_VARIANT,
            SUPPORTED_POLICY_NETWORK_VARIANTS,
            policy_network_variant_spec,
        )

        self.assertEqual(DEFAULT_POLICY_NETWORK_VARIANT, "shared_gtrxl_v1")
        self.assertEqual(FRESH_POLICY_NETWORK_VARIANT, "shared_gtrxl_small_v1")
        self.assertEqual(
            tuple(SUPPORTED_POLICY_NETWORK_VARIANTS),
            (
                "shared_gtrxl_small_v1",
                "shared_gtrxl_v1",
                "separate_critic_gtrxl_v1",
                "separate_critic_mlp_v1",
            ),
        )
        small = policy_network_variant_spec("shared_gtrxl_small_v1")
        self.assertTrue(small.shares_actor_trunk)
        self.assertEqual(
            small.architecture,
            {
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 2,
                "d_ff": 256,
            },
        )
        self.assertTrue(
            policy_network_variant_spec("shared_gtrxl_v1").shares_actor_trunk
        )
        self.assertFalse(
            policy_network_variant_spec(
                "separate_critic_gtrxl_v1"
            ).shares_actor_trunk
        )
        self.assertEqual(
            policy_network_variant_spec("separate_critic_mlp_v1").critic_kind,
            "mlp",
        )

    def test_normalization_is_strict_but_accepts_documented_short_aliases(self):
        from blb_stage2_rl.network_variants import normalize_policy_network_variant

        self.assertEqual(
            normalize_policy_network_variant("shared"), "shared_gtrxl_v1"
        )
        self.assertEqual(
            normalize_policy_network_variant("small"), "shared_gtrxl_small_v1"
        )
        self.assertEqual(
            normalize_policy_network_variant("separate-gtrxl"),
            "separate_critic_gtrxl_v1",
        )
        self.assertEqual(
            normalize_policy_network_variant("separate_mlp"),
            "separate_critic_mlp_v1",
        )
        with self.assertRaisesRegex(ValueError, "policy network variant"):
            normalize_policy_network_variant("larger_maybe")

    def test_shared_variant_preserves_legacy_algorithm_contract_exactly(self):
        from blb_stage2_rl.network_variants import bind_policy_network_contract

        legacy = {
            "schema_version": "stage2_layerwise_algorithm_contract_v5",
            "algorithm_revision": "v10",
            "rl_variant": "blb_v3_layerwise_robust_gtrxl_v1",
            "policy": {"state_dim": 136, "horizon": 12},
        }
        original = copy.deepcopy(legacy)

        bound = bind_policy_network_contract(
            legacy,
            "shared_gtrxl_v1",
            policy_shape={"d_model": 256, "n_layers": 4},
        )

        self.assertEqual(bound, original)
        self.assertEqual(legacy, original)
        self.assertNotIn("policy_network", bound)

    def test_ablation_variants_receive_distinct_persisted_contracts(self):
        from blb_stage2_rl.network_variants import bind_policy_network_contract

        base = {
            "schema_version": "stage2_layerwise_algorithm_contract_v5",
            "algorithm_revision": "v10",
            "rl_variant": "blb_v3_layerwise_robust_gtrxl_v1",
        }
        gtrxl = bind_policy_network_contract(
            base,
            "separate_critic_gtrxl_v1",
            policy_shape={"d_model": 256, "n_layers": 4},
        )
        mlp = bind_policy_network_contract(
            base,
            "separate_critic_mlp_v1",
            policy_shape={"hidden_dims": [512, 512, 256]},
        )
        small = bind_policy_network_contract(
            base,
            "shared_gtrxl_small_v1",
            policy_shape={
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 2,
                "d_ff": 256,
            },
        )

        self.assertNotEqual(gtrxl["rl_variant"], base["rl_variant"])
        self.assertNotEqual(mlp["rl_variant"], base["rl_variant"])
        self.assertNotEqual(small["rl_variant"], base["rl_variant"])
        self.assertNotEqual(gtrxl["rl_variant"], mlp["rl_variant"])
        self.assertEqual(
            gtrxl["policy_network"]["variant"],
            "separate_critic_gtrxl_v1",
        )
        self.assertNotIn("architecture", gtrxl["policy_network"])
        self.assertEqual(
            mlp["policy_network"]["variant"],
            "separate_critic_mlp_v1",
        )
        self.assertNotIn("architecture", mlp["policy_network"])
        self.assertEqual(
            small["policy_network"]["architecture"],
            {
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 2,
                "d_ff": 256,
            },
        )
        self.assertEqual(base["rl_variant"], "blb_v3_layerwise_robust_gtrxl_v1")

    def test_checkpoint_guard_accepts_legacy_shared_and_rejects_cross_variant_resume(self):
        from blb_stage2_rl.network_variants import (
            policy_network_variant_from_checkpoint,
            validate_checkpoint_policy_network_variant,
        )

        legacy_shared = {"rl_variant": "blb_v3_layerwise_robust_gtrxl_v1"}
        self.assertEqual(
            policy_network_variant_from_checkpoint(legacy_shared),
            "shared_gtrxl_v1",
        )
        validate_checkpoint_policy_network_variant(
            legacy_shared, "shared_gtrxl_v1"
        )

        separate = {
            "rl_variant": "blb_v3_layerwise_robust_separate_critic_gtrxl_v1",
            "policy_network_variant": "separate_critic_gtrxl_v1",
        }
        validate_checkpoint_policy_network_variant(
            separate, "separate_critic_gtrxl_v1"
        )
        with self.assertRaisesRegex(RuntimeError, "policy network variant"):
            validate_checkpoint_policy_network_variant(
                separate, "shared_gtrxl_v1"
            )
        with self.assertRaisesRegex(RuntimeError, "policy network variant"):
            validate_checkpoint_policy_network_variant(
                legacy_shared, "separate_critic_mlp_v1"
            )
        with self.assertRaisesRegex(RuntimeError, "policy network variant"):
            validate_checkpoint_policy_network_variant(
                legacy_shared, "shared_gtrxl_small_v1"
            )

        small = {
            "rl_variant": "blb_v3_layerwise_robust_shared_gtrxl_small_v1",
            "policy_network_variant": "shared_gtrxl_small_v1",
        }
        validate_checkpoint_policy_network_variant(
            small, "shared_gtrxl_small_v1"
        )
        with self.assertRaisesRegex(RuntimeError, "policy network variant"):
            validate_checkpoint_policy_network_variant(
                small, "shared_gtrxl_v1"
            )


if __name__ == "__main__":
    unittest.main()
