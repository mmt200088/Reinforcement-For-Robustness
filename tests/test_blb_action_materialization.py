"""Runtime contracts for canonical Stage-2 action materialization."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

try:
    from function_handler import make_block3_default_config
    from blb_stage2_rl.optimizer_cost import (
        configure_truncation_backend,
        materialize_decoded_action,
        materialized_config_fingerprint,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - local macOS may be torch-free.
    make_block3_default_config = None  # type: ignore
    configure_truncation_backend = None  # type: ignore
    materialize_decoded_action = None  # type: ignore
    materialized_config_fingerprint = None  # type: ignore
    _IMPORT_ERROR = exc


@unittest.skipUnless(_IMPORT_ERROR is None, f"runtime imports unavailable: {_IMPORT_ERROR!r}")
class ActionMaterializationTests(unittest.TestCase):
    @staticmethod
    def _cfg(*, x_sf: int = 20, k: int = 8):
        return make_block3_default_config(
            degree=2,
            N=8192,
            x_fresh_sf=int(x_sf),
            inv_2n_sf=18,
            square_rescale_sfs=(16, 14),
            output_truncation_k=int(k),
        )

    @staticmethod
    def _valid_output(*, include_compact: bool = True):
        raw = {"result": {"valid": True}}
        if include_compact:
            raw["new_compact_config"] = {
                "cut_point_sf": [
                    {"i": 0, "name": "x", "type": "SOURCE", "sf": 30},
                ],
                "propagation_deltas": [],
                "effective_rotations": [],
            }
        return SimpleNamespace(
            valid=True,
            raw=raw,
            total_bits=100,
            fusion_count=0,
            invalid_chain=None,
        )

    def test_valid_but_incomplete_replan_fails_closed(self):
        cfg = self._cfg()
        result = materialize_decoded_action(
            action_indices=[0],
            decoded=SimpleNamespace(block3_cfgs={0: cfg}),
            cfgs_dict={"block3": {0: cfg}},
            outputs={"block3_exp_n2_L0": self._valid_output(include_compact=False)},
            signals=SimpleNamespace(any_invalid=False),
            profile="mrpc",
            invoker_baselines={"block3_exp_n2": ([0, 2, 3, 4], [], [])},
        )

        self.assertFalse(result.model_ready)
        self.assertFalse(result.optimizer_invalid)
        self.assertEqual(result.failure_reason, "replan_config_not_fully_applied")
        self.assertEqual(
            result.replan_application["missing_compact_config_count"], 1,
        )
        self.assertEqual(result.final_config_fingerprint, "")

    def test_missing_optimizer_output_fails_closed_before_model_install(self):
        cfg = self._cfg()
        result = materialize_decoded_action(
            action_indices=[0],
            decoded=SimpleNamespace(block3_cfgs={0: cfg}),
            cfgs_dict={"block3": {0: cfg}},
            outputs={},
            signals=SimpleNamespace(any_invalid=False),
            profile="mrpc",
            invoker_baselines={"block3_exp_n2": ([0, 2, 3, 4], [], [])},
        )

        self.assertFalse(result.model_ready)
        self.assertEqual(result.failure_reason, "optimizer_output_set_mismatch")
        self.assertEqual(
            result.replan_application["missing_optimizer_outputs"],
            ["block3_exp_n2_L0"],
        )
        self.assertEqual(result.final_config_fingerprint, "")

    def test_missing_optimizer_baseline_skeleton_fails_closed(self):
        cfg = self._cfg()
        result = materialize_decoded_action(
            action_indices=[0],
            decoded=SimpleNamespace(block3_cfgs={0: cfg}),
            cfgs_dict={"block3": {0: cfg}},
            outputs={"block3_exp_n2_L0": self._valid_output()},
            signals=SimpleNamespace(any_invalid=False),
            profile="mrpc",
            invoker_baselines={},
        )

        self.assertFalse(result.model_ready)
        self.assertEqual(result.failure_reason, "replan_config_not_fully_applied")
        self.assertEqual(
            result.replan_application["missing_baseline_skeletons"],
            ["block3_exp_n2_L0"],
        )
        self.assertFalse(
            result.replan_application["model_uses_replan_config"],
        )

    def test_complete_replan_is_model_ready_and_preserves_block3_k(self):
        cfg = self._cfg(k=8)
        result = materialize_decoded_action(
            action_indices=[0],
            decoded=SimpleNamespace(block3_cfgs={0: cfg}),
            cfgs_dict={"block3": {0: cfg}},
            outputs={"block3_exp_n2_L0": self._valid_output()},
            signals=SimpleNamespace(any_invalid=False),
            profile="mrpc",
            invoker_baselines={"block3_exp_n2": ([0, 2, 3, 4], [], [])},
        )

        self.assertTrue(result.model_ready, result.replan_application)
        self.assertEqual(cfg.output_truncation_k, 8)
        self.assertEqual(cfg.x_fresh.scaling_factor, 30)
        self.assertRegex(result.final_config_fingerprint, r"^[0-9a-f]{64}$")

    def test_final_config_identity_includes_boosted_sf_and_k(self):
        cfg = self._cfg(x_sf=20, k=8)
        base = {"block3": {0: cfg}}
        boosted = copy.deepcopy(base)
        boosted["block3"][0].x_fresh.scaling_factor = 31
        different_k = copy.deepcopy(base)
        different_k["block3"][0].output_truncation_k = 13

        base_fp = materialized_config_fingerprint(base)
        self.assertNotEqual(base_fp, materialized_config_fingerprint(boosted))
        self.assertNotEqual(base_fp, materialized_config_fingerprint(different_k))

    def test_truncation_backend_is_explicit_and_part_of_identity(self):
        legacy = {"block3": {0: self._cfg()}}
        stochastic = copy.deepcopy(legacy)
        configure_truncation_backend(
            stochastic,
            backend="stochastic_ring",
            ring_bits=43,
            source_fractional_bits=24,
        )

        cfg = stochastic["block3"][0]
        self.assertEqual(cfg.output_truncation_mode, "stochastic_ring")
        self.assertEqual(cfg.output_truncation_ring_bits, 43)
        self.assertEqual(cfg.output_truncation_source_fractional_bits, 24)
        self.assertNotEqual(
            materialized_config_fingerprint(legacy),
            materialized_config_fingerprint(stochastic),
        )

    def test_stochastic_backend_rejects_fractional_width_without_signed_headroom(self):
        with self.assertRaisesRegex(ValueError, "smaller than ring_bits"):
            configure_truncation_backend(
                {"block3": {0: self._cfg()}},
                backend="stochastic_ring",
                ring_bits=24,
                source_fractional_bits=24,
            )

    def test_stochastic_backend_rejects_target_k_above_source_before_install(self):
        with self.assertRaisesRegex(ValueError, "target truncation K"):
            configure_truncation_backend(
                {"block3": {0: self._cfg(k=13)}},
                backend="stochastic_ring",
                ring_bits=43,
                source_fractional_bits=8,
            )

    def test_real_mrpc_all_max_materializes_all_blocks_and_preserves_block3_k(self):
        from blb_stage2_rl.action_space import (
            load_max_sfs,
            make_all_max_action_vector,
        )
        from blb_stage2_rl.optimizer_cost import materialize_action_for_model
        from rescale_optimizer_bridge import InProcessInvoker, RescaleOptimizerBridge

        invoker = InProcessInvoker.from_profile(
            rescale_optimizer_root="Rescale_optimizer",
            profile="mrpc",
        )
        result = materialize_action_for_model(
            make_all_max_action_vector(num_layers=12),
            profile="mrpc",
            num_layers=12,
            max_sfs=load_max_sfs("mrpc"),
            rescale_bridge=RescaleOptimizerBridge(invoker=invoker),
            gelu_degree=4,
            attn_degree=4,
            invoker_baselines=invoker.baselines,
        )

        self.assertTrue(result.model_ready, result.replan_application)
        self.assertFalse(result.optimizer_invalid)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.replan_application["expected_config_count"], 59)
        self.assertEqual(result.replan_application["applied_config_count"], 59)
        self.assertTrue(result.replan_application["model_uses_replan_config"])
        self.assertEqual(len(result.decoded.block3_cfgs), 12)
        self.assertEqual(
            {cfg.output_truncation_k for cfg in result.decoded.block3_cfgs.values()},
            {13},
        )
        self.assertEqual(
            sum(name.startswith("block3_exp_n4_L") for name in result.outputs),
            12,
        )
        self.assertRegex(result.final_config_fingerprint, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
