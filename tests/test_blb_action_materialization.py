"""Runtime contracts for canonical Stage-2 action materialization."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import importlib
import sys
import types
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


_MISSING = object()


@contextmanager
def _stubbed_action_space():
    package = importlib.import_module("blb_stage2_rl")
    module_name = "blb_stage2_rl.action_space"
    module_before = sys.modules.get(module_name, _MISSING)
    attribute_before = package.__dict__.get("action_space", _MISSING)

    bridge = types.ModuleType("blb_rl_bridge")
    for name in (
        "Block1ActionSpec",
        "Block2ActionSpec",
        "Block3ActionSpec",
        "Block4ActionSpec",
        "Block5ActionSpec",
        "build_block1_cfg_from_action",
        "build_block2_cfg_from_action",
        "build_block3_cfg_from_action",
        "build_block4_cfg_from_action",
        "build_block5_cfg_from_action",
    ):
        setattr(bridge, name, type(name, (), {}))
    handler = types.ModuleType("function_handler")
    handler.NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N = {}
    for name in (
        "Block1NoiseConfig",
        "Block2NoiseConfig",
        "Block3NoiseConfig",
        "Block4NoiseConfig",
        "Block5NoiseConfig",
    ):
        setattr(handler, name, type(name, (), {}))

    dependency_names = ("blb_rl_bridge", "function_handler")
    dependencies_before = {
        name: sys.modules.get(name, _MISSING)
        for name in dependency_names
    }
    sys.modules["blb_rl_bridge"] = bridge
    sys.modules["function_handler"] = handler
    sys.modules.pop(module_name, None)
    package.__dict__.pop("action_space", None)
    try:
        yield importlib.import_module(module_name)
    finally:
        if module_before is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = module_before
        if attribute_before is _MISSING:
            package.__dict__.pop("action_space", None)
        else:
            package.action_space = attribute_before
        for name, module in dependencies_before.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class ActionVectorBoundsTests(unittest.TestCase):
    @staticmethod
    def _first_k_offset(action_space):
        return next(
            offset
            for offset, (_block_idx, _field_name, kind) in enumerate(
                action_space.per_layer_field_offsets()
            )
            if kind == "K"
        )

    def test_full_decode_rejects_negative_k_index(self):
        with _stubbed_action_space() as action_space:
            action = action_space.make_all_max_action_vector(num_layers=1)
            action[self._first_k_offset(action_space)] = -1

            with self.assertRaisesRegex(ValueError, "action index.*-1.*out of range"):
                action_space.action_vector_to_cfgs(
                    action,
                    max_sfs=action_space.MaxSFsTable(),
                    num_layers=1,
                )

    def test_full_decode_rejects_k_index_equal_to_level_count(self):
        with _stubbed_action_space() as action_space:
            action = action_space.make_all_max_action_vector(num_layers=1)
            action[self._first_k_offset(action_space)] = len(action_space.K_LEVELS)

            with self.assertRaisesRegex(
                ValueError,
                rf"action index.*{len(action_space.K_LEVELS)}.*out of range",
            ):
                action_space.action_vector_to_cfgs(
                    action,
                    max_sfs=action_space.MaxSFsTable(),
                    num_layers=1,
                )

    def test_full_decode_rejects_fractional_and_matrix_actions(self):
        with _stubbed_action_space() as action_space:
            action = action_space.make_all_max_action_vector(num_layers=1)
            fractional = action.astype(float)
            fractional[self._first_k_offset(action_space)] = -0.1

            with self.assertRaisesRegex(ValueError, "integer categorical indices"):
                action_space.action_vector_to_cfgs(
                    fractional,
                    max_sfs=action_space.MaxSFsTable(),
                    num_layers=1,
                )
            with self.assertRaisesRegex(ValueError, "one-dimensional"):
                action_space.action_vector_to_cfgs(
                    action.reshape(1, -1),
                    max_sfs=action_space.MaxSFsTable(),
                    num_layers=1,
                )

    def test_reporting_and_k_summaries_reject_invalid_full_vectors(self):
        with _stubbed_action_space() as action_space:
            action = action_space.make_all_max_action_vector(num_layers=1)
            negative_k = action.copy()
            negative_k[self._first_k_offset(action_space)] = -1
            negative_sf = action.copy()
            negative_sf[0] = -1

            with self.assertRaisesRegex(ValueError, "out of range"):
                action_space.describe_action_vector(
                    negative_k,
                    max_sfs=action_space.MaxSFsTable(),
                    num_layers=1,
                )
            with self.assertRaisesRegex(ValueError, "out of range"):
                action_space.avg_truncation_k_in_action(
                    negative_sf,
                    num_layers=1,
                )


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
        self.assertEqual(len(result.decoded.block1_cfgs), 12)
        self.assertEqual(result.decoded.block1_cfgs[0].output_truncation_k, 13)
        self.assertFalse(result.decoded.block1_cfgs[0].noise_enabled)
        self.assertTrue(
            all(
                cfg.noise_enabled
                for layer_idx, cfg in result.decoded.block1_cfgs.items()
                if layer_idx > 0
            )
        )
        self.assertEqual(len(result.decoded.block3_cfgs), 12)
        self.assertEqual(
            {cfg.output_truncation_k for cfg in result.decoded.block3_cfgs.values()},
            {13},
        )
        self.assertEqual(
            sum(name.startswith("block3_exp_n4_L") for name in result.outputs),
            12,
        )
        effective_rotation_count = sum(
            int(entry.get("count", 1) or 1)
            for output in result.outputs.values()
            for entry in (
                output.raw.get("new_compact_config", {}).get(
                    "effective_rotations", [],
                )
            )
        )
        enabled_rotation_flags = [
            (block_name, layer_idx, flag, int(count))
            for block_name, layer_cfgs in result.cfgs_dict.items()
            for layer_idx, cfg in layer_cfgs.items()
            for flag, count in getattr(
                cfg, "rotation_repeat_counts", {},
            ).items()
            if bool(getattr(cfg, flag, False))
        ]
        self.assertGreater(effective_rotation_count, 0)
        self.assertTrue(enabled_rotation_flags)
        self.assertTrue(any(count == 3 for *_, count in enabled_rotation_flags))
        self.assertRegex(result.final_config_fingerprint, r"^[0-9a-f]{64}$")

    def test_real_mrpc_k6_materializes_all_blocks_including_layer0(self):
        from blb_stage2_rl.action_space import (
            K_LEVELS,
            load_max_sfs,
            make_all_max_action_vector,
            per_layer_field_offsets,
        )
        from blb_stage2_rl.optimizer_cost import materialize_action_for_model
        from rescale_optimizer_bridge import InProcessInvoker, RescaleOptimizerBridge

        num_layers = 12
        action = make_all_max_action_vector(num_layers=num_layers)
        per_layer_fields = per_layer_field_offsets()
        k6_index = K_LEVELS.index(6)
        truncation_offsets = [
            field_offset
            for field_offset, (_block_idx, field_name, kind) in enumerate(
                per_layer_fields
            )
            if kind == "K" and field_name == "output_truncation_k"
        ]
        self.assertEqual(len(truncation_offsets), 5)
        for layer_idx in range(num_layers):
            layer_start = layer_idx * len(per_layer_fields)
            for field_offset in truncation_offsets:
                action[layer_start + field_offset] = k6_index

        invoker = InProcessInvoker.from_profile(
            rescale_optimizer_root="Rescale_optimizer",
            profile="mrpc",
        )
        result = materialize_action_for_model(
            action,
            profile="mrpc",
            num_layers=num_layers,
            max_sfs=load_max_sfs("mrpc"),
            rescale_bridge=RescaleOptimizerBridge(invoker=invoker),
            gelu_degree=4,
            attn_degree=4,
            invoker_baselines=invoker.baselines,
        )

        self.assertTrue(result.model_ready, result.replan_application)
        self.assertFalse(result.optimizer_invalid)
        self.assertIsNone(result.failure_reason)
        expected_count = result.replan_application["expected_config_count"]
        self.assertGreater(expected_count, 0)
        self.assertEqual(
            expected_count,
            result.replan_application["applied_config_count"],
        )
        self.assertTrue(result.replan_application["model_uses_replan_config"])
        for block_idx in range(1, 6):
            cfgs = getattr(result.decoded, f"block{block_idx}_cfgs")
            self.assertEqual(set(cfgs), set(range(num_layers)))
            self.assertEqual(
                {cfg.output_truncation_k for cfg in cfgs.values()},
                {6},
            )
        self.assertEqual(result.decoded.block1_cfgs[0].output_truncation_k, 6)
        self.assertEqual(result.decoded.block3_cfgs[0].output_truncation_k, 6)
        self.assertTrue(result.outputs)
        self.assertEqual(len(result.outputs), expected_count)
        self.assertEqual(
            sum(name.startswith("block3_exp_n4_L") for name in result.outputs),
            num_layers,
        )
        self.assertRegex(result.final_config_fingerprint, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
