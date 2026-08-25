"""Chain-integrity contract tests for the BLB Stage-2 RL pipeline.

The load-bearing flow this file pins down is:

    action -> Rescale_optimizer (new_compact_config)
           -> apply_optimizer_output_to_cfg (4 write classes + rotation flags)
           -> sync_block2_qk_binding (Q/K mirror)
           -> bridge.apply (cfg -> function_handler noise install)
           -> model forward (_sample_gaussian_for_point reads cfg LIVE)

A regression in any link silently corrupts training (model installs noise at
the RL-proposed SF instead of the optimizer-chosen SF). The tests synthesise
``new_compact_config`` dicts by hand rather than running the real optimizer so
they stay fast and deterministic.

Imports require torch + transformers (``function_handler`` pulls both); the
pattern matches ``test_blb_stage2_rl_regressions.py``.
"""
from __future__ import annotations

import unittest

try:
    import torch
    from rfr.search.runtime.model_handler import (
        NoisePoint,
        _make_block3_approximation_exponential,
        _sample_gaussian_for_point,
        make_block1_default_config,
        make_block2_default_config,
        make_block3_default_config,
        make_block4_default_config,
    )
    from rfr.preparation.rescale.bridge import (
        apply_optimizer_output_to_cfg,
        sync_block2_qk_binding,
    )
    from rfr.preparation.rescale.optimizer_cost import apply_optimizer_outputs_to_cfgs
    from blb_stage2_rl.sequential_policy import (
        BLBStage2SequentialPolicy,
        SequentialPolicyConfig,
    )
    from rfr.search.runtime.probe_runner import (
        _split_round_robin,
        _trial_seed,
        parse_device_ids,
    )
    _IMPORT_ERROR = None
except Exception as _exc:
    torch = None  # type: ignore
    NoisePoint = None  # type: ignore
    _sample_gaussian_for_point = None  # type: ignore
    make_block1_default_config = None  # type: ignore
    make_block2_default_config = None  # type: ignore
    make_block3_default_config = None  # type: ignore
    make_block4_default_config = None  # type: ignore
    _make_block3_approximation_exponential = None  # type: ignore
    apply_optimizer_output_to_cfg = None  # type: ignore
    sync_block2_qk_binding = None  # type: ignore
    apply_optimizer_outputs_to_cfgs = None  # type: ignore
    BLBStage2SequentialPolicy = None  # type: ignore
    SequentialPolicyConfig = None  # type: ignore
    _split_round_robin = None  # type: ignore
    _trial_seed = None  # type: ignore
    parse_device_ids = None  # type: ignore
    _IMPORT_ERROR = _exc

_SKIP_REASON = (
    f"torch / function_handler not importable: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None else ""
)
_TORCH_AVAILABLE = _IMPORT_ERROR is None


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class ApplyOptimizerOutputBlock1Test(unittest.TestCase):
    """Verify the four write classes against the real block1_mrpc skeleton."""


    BASELINE_SKELETON = [0, 2, 4, 5]

    def setUp(self):


        self.cfg = make_block1_default_config(
            N=8192,
            gelu_out_sf=28,
            wffn2_sf=20,
            mean_inv_d_sf=20,
            var_inv_d_sf=20,
            mean_rescale_sf=20,
            var_rescale_sf=20,
        )

    def test_fresh_writes_to_source_field(self):
        raw = {
            "new_compact_config": {
                "cut_point_sf": [
                    {"i": 0, "name": "gelu_out", "type": "SOURCE", "sf": 30},
                ],
                "propagation_deltas": [],
                "effective_rotations": [],
            },
            "result": {"valid": True},
        }
        overrides = apply_optimizer_output_to_cfg(
            self.cfg, output_raw=raw, block_idx=1, graph_key="block1_mrpc",
            baseline_skeleton=self.BASELINE_SKELETON,
        )
        self.assertEqual(self.cfg.gelu_out_fresh.scaling_factor, 30)
        self.assertTrue(any(
            e.cfg_attr == "gelu_out_fresh.scaling_factor" and e.source == "fresh"
            and e.old_value == 28 and e.new_value == 30
            for e in overrides
        ), f"expected fresh override entry; got {overrides}")

    def test_rescale_post_writes_to_rescale_field(self):

        raw = {
            "new_compact_config": {
                "cut_point_sf": [
                    {"i": 0, "name": "gelu_out", "type": "SOURCE", "sf": 30},
                    {"i": 2, "name": "ctpt_inv_d_1", "type": "CTPT_MUL",
                     "sf_pre": 70, "sf_post": 34, "drop": 36},
                ],
                "propagation_deltas": [],
                "effective_rotations": [],
            },
            "result": {"valid": True},
        }
        overrides = apply_optimizer_output_to_cfg(
            self.cfg, output_raw=raw, block_idx=1, graph_key="block1_mrpc",
            baseline_skeleton=self.BASELINE_SKELETON,
        )
        self.assertEqual(self.cfg.mean_result_rescale.scaling_factor, 34)
        self.assertTrue(any(
            e.cfg_attr == "mean_result_rescale.scaling_factor"
            and e.source == "rescale_post" and e.new_value == 34
            for e in overrides
        ), f"expected rescale_post override entry; got {overrides}")

    def test_fused_away_sets_rescale_to_none(self):


        raw = {
            "new_compact_config": {
                "cut_point_sf": [
                    {"i": 0, "name": "gelu_out", "type": "SOURCE", "sf": 30},
                    {"i": 2, "name": "ctpt_inv_d_1", "type": "CTPT_MUL",
                     "sf_pre": 70, "sf_post": 34, "drop": 36},

                ],
                "propagation_deltas": [],
                "effective_rotations": [],
            },
            "result": {"valid": True},
        }
        self.assertIsNotNone(self.cfg.var_result_rescale,
                             "precondition: var_result_rescale starts non-None")
        overrides = apply_optimizer_output_to_cfg(
            self.cfg, output_raw=raw, block_idx=1, graph_key="block1_mrpc",
            baseline_skeleton=self.BASELINE_SKELETON,
        )
        self.assertIsNone(self.cfg.var_result_rescale,
                          "fused-away rescale must become None")
        self.assertTrue(any(
            e.source == "rescale_fused_away" and e.new_value is None
            and e.cfg_attr == "var_result_rescale"
            for e in overrides
        ), f"expected fused_away override entry; got {overrides}")

    def test_propagation_delta_writes_to_encode_field(self):

        raw = {
            "new_compact_config": {
                "cut_point_sf": [],
                "propagation_deltas": [
                    {"node_id": 1, "name": "ctpt_ffn2",
                     "type": "CTPT_MUL", "delta": 22},
                ],
                "effective_rotations": [],
            },
            "result": {"valid": True},
        }
        overrides = apply_optimizer_output_to_cfg(
            self.cfg, output_raw=raw, block_idx=1, graph_key="block1_mrpc",
            baseline_skeleton=self.BASELINE_SKELETON,
        )
        self.assertEqual(self.cfg.wffn2_encode.scaling_factor, 22)
        self.assertTrue(any(
            e.cfg_attr == "wffn2_encode.scaling_factor"
            and e.source == "propagation_delta"
            and e.new_value == 22
            for e in overrides
        ), f"expected propagation_delta override entry; got {overrides}")

    def test_non_integer_delta_skipped(self):


        raw = {
            "new_compact_config": {
                "cut_point_sf": [],
                "propagation_deltas": [
                    {"node_id": 3, "name": "ctct_ext_square",
                     "type": "CTCT_MUL", "delta": "x2"},
                ],
                "effective_rotations": [],
            },
            "result": {"valid": True},
        }
        original_wffn2_sf = self.cfg.wffn2_encode.scaling_factor
        overrides = apply_optimizer_output_to_cfg(
            self.cfg, output_raw=raw, block_idx=1, graph_key="block1_mrpc",
            baseline_skeleton=self.BASELINE_SKELETON,
        )

        self.assertFalse(any(o.source == "propagation_delta" for o in overrides))
        self.assertEqual(self.cfg.wffn2_encode.scaling_factor, original_wffn2_sf)

    def test_invalid_result_short_circuits_no_op(self):
        raw = {
            "new_compact_config": {
                "cut_point_sf": [
                    {"i": 0, "name": "gelu_out", "type": "SOURCE", "sf": 99},
                ],
                "propagation_deltas": [
                    {"node_id": 1, "name": "ctpt_ffn2",
                     "type": "CTPT_MUL", "delta": 99},
                ],
                "effective_rotations": [],
            },
            "result": {"valid": False},
        }
        original_fresh_sf = self.cfg.gelu_out_fresh.scaling_factor
        original_wffn2_sf = self.cfg.wffn2_encode.scaling_factor
        overrides = apply_optimizer_output_to_cfg(
            self.cfg, output_raw=raw, block_idx=1, graph_key="block1_mrpc",
            baseline_skeleton=self.BASELINE_SKELETON,
        )

        self.assertEqual(self.cfg.gelu_out_fresh.scaling_factor, original_fresh_sf)
        self.assertEqual(self.cfg.wffn2_encode.scaling_factor, original_wffn2_sf)
        self.assertEqual(overrides, [])

    def test_missing_compact_config_no_op(self):

        raw = {"result": {"valid": True}}
        original_fresh_sf = self.cfg.gelu_out_fresh.scaling_factor
        overrides = apply_optimizer_output_to_cfg(
            self.cfg, output_raw=raw, block_idx=1, graph_key="block1_mrpc",
            baseline_skeleton=self.BASELINE_SKELETON,
        )
        self.assertEqual(self.cfg.gelu_out_fresh.scaling_factor, original_fresh_sf)
        self.assertEqual(overrides, [])


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class ApplyOptimizerOutputBlock2RotationTest(unittest.TestCase):
    """Rotation flags are set from effective_rotations + rotation_name_map."""

    BASELINE_SKELETON = [0, 2, 5, 7, 8]

    def setUp(self):
        self.cfg = make_block2_default_config(
            N=16384,
            inv_std_fresh_sf=30,
            x_centered_fresh_sf=30,
            gamma_sf=22,
            wk_sf=22,
            kt_mask1_sf=22,
            kt_mask2_sf=22,
            wq_sf=22,
            q_mask1_sf=22,
            q_mask2_sf=22,
            wv_sf=22,
            qkt_merge_mask_sf=22,
            gamma_rescale_sf=22,
            kt_mask2_rescale_sf=22,
            qkt_merge_mask_rescale_sf=22,

            rotation_after_gamma_rescale=False,
            rotation_after_kt_mask2_rescale=False,
        )

    def test_rotation_flag_toggled_on_via_name_map(self):
        raw = {
            "new_compact_config": {
                "cut_point_sf": [],
                "propagation_deltas": [],
                "effective_rotations": [
                    {"name": "rot_after_gama1", "depth": 31},
                ],
            },
            "result": {"valid": True},
        }
        rotation_name_map = {
            "rot_after_gama1": "rotation_after_gamma_rescale",
        }
        overrides = apply_optimizer_output_to_cfg(
            self.cfg, output_raw=raw, block_idx=2, graph_key="block2_mrpc",
            baseline_skeleton=self.BASELINE_SKELETON,
            rotation_name_map=rotation_name_map,
        )
        self.assertTrue(self.cfg.rotation_after_gamma_rescale)
        self.assertTrue(any(
            o.cfg_attr == "rotation_after_gamma_rescale"
            and o.source == "rotation_flag"
            and o.old_value is False and o.new_value is True
            for o in overrides
        ), f"expected rotation flag override; got {overrides}")

    def test_rotation_flag_resets_when_absent(self):

        self.cfg.rotation_after_gamma_rescale = True
        raw = {
            "new_compact_config": {
                "cut_point_sf": [],
                "propagation_deltas": [],
                "effective_rotations": [],
            },
            "result": {"valid": True},
        }
        rotation_name_map = {
            "rot_after_gama1": "rotation_after_gamma_rescale",
        }
        overrides = apply_optimizer_output_to_cfg(
            self.cfg, output_raw=raw, block_idx=2, graph_key="block2_mrpc",
            baseline_skeleton=self.BASELINE_SKELETON,
            rotation_name_map=rotation_name_map,
        )
        self.assertFalse(self.cfg.rotation_after_gamma_rescale,
                         "rotation flag must reset to False when optimizer "
                         "lists no effective rotations")
        self.assertTrue(any(
            o.cfg_attr == "rotation_after_gamma_rescale"
            and o.source == "rotation_flag"
            and o.old_value is True and o.new_value is False
            for o in overrides
        ), f"expected rotation flag reset override; got {overrides}")


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class DefaultOptimizerRotationBindingTest(unittest.TestCase):
    """Production materialization must not require an injected name map."""

    @staticmethod
    def _raw(*rotations):
        return {
            "new_compact_config": {
                "cut_point_sf": [],
                "propagation_deltas": [],
                "effective_rotations": list(rotations),
            },
            "result": {"valid": True},
        }

    def test_block2_shared_rotation_enables_all_qkv_model_branches(self):
        cfg = make_block2_default_config(
            wq_rescale_sf=30,
            wk_rescale_sf=30,
            wv_rescale_sf=30,
        )
        apply_optimizer_output_to_cfg(
            cfg,
            output_raw=self._raw({"name": "gs_rot", "count": 1}),
            block_idx=2,
            graph_key="block2_mrpc",
            baseline_skeleton=[0, 1, 2, 3],
        )

        for flag in (
            "rotation_after_wq_rescale",
            "rotation_after_wk_rescale",
            "rotation_after_wv_rescale",
        ):
            self.assertTrue(getattr(cfg, flag), flag)
            self.assertEqual(cfg.rotation_repeat_counts[flag], 1)

    def test_block4_rotation_count_three_is_preserved_for_runtime(self):
        cfg = make_block4_default_config(ln_square_rescale_sf=31)
        apply_optimizer_output_to_cfg(
            cfg,
            output_raw=self._raw({
                "name": "rot_pre_ctpt_invd_2",
                "count": 3,
            }),
            block_idx=4,
            graph_key="block4",
            baseline_skeleton=[0, 1, 2, 3],
        )

        flag = "rotation_after_ln_square_rescale"
        self.assertTrue(getattr(cfg, flag))
        self.assertEqual(cfg.rotation_repeat_counts[flag], 3)

    def test_unknown_effective_rotation_fails_closed_instead_of_disappearing(self):
        cfg = make_block4_default_config()
        with self.assertRaisesRegex(ValueError, "unmapped effective rotation"):
            apply_optimizer_output_to_cfg(
                cfg,
                output_raw=self._raw({"name": "future_rotation", "count": 1}),
                block_idx=4,
                graph_key="block4",
                baseline_skeleton=[0, 1, 2, 3],
            )


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class SyncBlock2QKBindingTest(unittest.TestCase):
    """Block 2 Q/K binding invariant must survive every cfg mutation site."""

    def setUp(self):
        self.cfg = make_block2_default_config(
            N=16384,
            inv_std_fresh_sf=30,
            x_centered_fresh_sf=30,
            gamma_sf=22,
            wk_sf=22,
            kt_mask1_sf=22,
            kt_mask2_sf=22,
            wq_sf=22,
            q_mask1_sf=22,
            q_mask2_sf=22,
            wv_sf=22,
            qkt_merge_mask_sf=22,
        )

    def test_wk_mutation_mirrors_onto_wq(self):

        self.cfg.wk_encode.scaling_factor = 28
        self.assertEqual(self.cfg.wq_encode.scaling_factor, 22,
                         "precondition: Q stays at pre-override SF")
        overrides = sync_block2_qk_binding(self.cfg)
        self.assertEqual(self.cfg.wq_encode.scaling_factor, 28,
                         "Q must mirror K after sync")
        self.assertTrue(any(
            o.cfg_attr == "wq_encode.scaling_factor"
            and o.source == "qk_binding_sync"
            and o.old_value == 22 and o.new_value == 28
            for o in overrides
        ), f"expected qk_binding_sync override; got {overrides}")

    def test_all_three_pairs_mirrored(self):
        self.cfg.wk_encode.scaling_factor = 24
        self.cfg.kt_mask1_encode.scaling_factor = 26
        self.cfg.kt_mask2_encode.scaling_factor = 30
        sync_block2_qk_binding(self.cfg)
        self.assertEqual(self.cfg.wq_encode.scaling_factor, 24)
        self.assertEqual(self.cfg.q_mask1_encode.scaling_factor, 26)
        self.assertEqual(self.cfg.q_mask2_encode.scaling_factor, 30)

    def test_no_op_when_already_synced(self):

        overrides = sync_block2_qk_binding(self.cfg)
        self.assertEqual(overrides, [],
                         "sync must be a no-op when Q already matches K")


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class ApplyOptimizerOutputsToCfgsSharedHelperTest(unittest.TestCase):
    """RL, fixed-action eval, and final eval must share one cfg write-back path."""

    def _out(self, raw):
        class _Output:
            valid = True

            def __init__(self, payload):
                self.raw = payload

        return _Output(raw)

    def test_shared_helper_applies_block2_bindings(self):
        cfg = make_block2_default_config(
            N=16384,
            inv_std_fresh_sf=18,
            x_centered_fresh_sf=12,
            wk_sf=20,
            wq_sf=10,
            kt_mask1_sf=21,
            q_mask1_sf=11,
            kt_mask2_sf=22,
            q_mask2_sf=12,
        )
        raw = {
            "new_compact_config": {
                "cut_point_sf": [
                    {"i": 0, "name": "inv_std", "type": "SOURCE", "sf": 31},
                ],
                "propagation_deltas": [
                    {"name": "ctpt_wq_wk", "delta": 28},
                    {"name": "ctpt_rotKT_mask1", "delta": 26},
                    {"name": "ctpt_rotKT_mask2", "delta": 24},
                ],
                "effective_rotations": [],
            },
            "result": {"valid": True},
        }
        diag = apply_optimizer_outputs_to_cfgs(
            profile="mrpc",
            cfgs_dict={"block2": {0: cfg}},
            opt_outputs={"block2_mrpc_L0": self._out(raw)},
            invoker_baselines={"block2_mrpc": ([0, 2, 4, 6], [], [])},
        )
        self.assertTrue(diag["model_uses_replan_config"], diag)
        self.assertEqual(cfg.inv_std_fresh.scaling_factor, 31)
        self.assertEqual(cfg.x_centered_fresh.scaling_factor, 31)
        self.assertEqual(cfg.wk_encode.scaling_factor, 28)
        self.assertEqual(cfg.wq_encode.scaling_factor, 28)
        self.assertEqual(cfg.kt_mask1_encode.scaling_factor, 26)
        self.assertEqual(cfg.q_mask1_encode.scaling_factor, 26)
        self.assertEqual(cfg.kt_mask2_encode.scaling_factor, 24)
        self.assertEqual(cfg.q_mask2_encode.scaling_factor, 24)

    def test_shared_helper_applies_block4_v_mask_binding(self):
        from rfr.search.runtime.model_handler import make_block4_default_config

        cfg = make_block4_default_config(
            N=16384,
            softmax_out_mask_sf=29,
            v_mask_sf=15,
        )
        raw = {
            "new_compact_config": {
                "cut_point_sf": [],
                "propagation_deltas": [
                    {"name": "ctpt_mask2", "delta": 27},
                ],
                "effective_rotations": [],
            },
            "result": {"valid": True},
        }
        diag = apply_optimizer_outputs_to_cfgs(
            profile="mrpc",
            cfgs_dict={"block4": {0: cfg}},
            opt_outputs={"block4_L0": self._out(raw)},
            invoker_baselines={"block4": ([0, 2, 5, 6], [], [])},
        )
        self.assertTrue(diag["model_uses_replan_config"], diag)
        self.assertEqual(cfg.softmax_out_mask_encode.scaling_factor, 27)
        self.assertEqual(cfg.v_mask_encode.scaling_factor, 27)

    def test_shared_helper_applies_block5_aux_fresh_binding(self):
        from rfr.search.runtime.model_handler import make_block5_default_config

        cfg = make_block5_default_config(
            gelu_degree=4,
            N=16384,
            inv_std_fresh_sf=16,
            x_centered_fresh_sf=18,
        )
        raw = {
            "new_compact_config": {
                "cut_point_sf": [
                    {"i": 0, "name": "x_mean", "type": "SOURCE", "sf": 32},
                ],
                "propagation_deltas": [],
                "effective_rotations": [],
            },
            "result": {"valid": True},
        }
        diag = apply_optimizer_outputs_to_cfgs(
            profile="mrpc",
            cfgs_dict={"block5": {0: cfg}},
            opt_outputs={"block5_n4_L0": self._out(raw)},
            invoker_baselines={"block5_n4": ([0, 1, 3, 4, 6, 7], [], [])},
        )
        self.assertTrue(diag["model_uses_replan_config"], diag)
        self.assertEqual(cfg.x_centered_fresh.scaling_factor, 32)
        self.assertEqual(cfg.inv_std_fresh.scaling_factor, 32)

    def test_shared_helper_preserves_block3_k_while_writing_sf(self):
        cfg = make_block3_default_config(
            degree=2,
            N=8192,
            x_fresh_sf=20,
            inv_2n_sf=18,
            square_rescale_sfs=(16, 14),
            output_truncation_k=8,
        )
        raw = {
            "new_compact_config": {
                "cut_point_sf": [
                    {"i": 0, "name": "x", "type": "SOURCE", "sf": 30},
                ],
                "propagation_deltas": [],
                "effective_rotations": [],
            },
            "result": {"valid": True},
        }
        diag = apply_optimizer_outputs_to_cfgs(
            profile="mrpc",
            cfgs_dict={"block3": {0: cfg}},
            opt_outputs={"block3_exp_n2_L0": self._out(raw)},
            invoker_baselines={"block3_exp_n2": ([0, 2, 3, 4], [], [])},
        )

        self.assertTrue(diag["model_uses_replan_config"], diag)
        self.assertEqual(cfg.x_fresh.scaling_factor, 30)
        self.assertEqual(cfg.output_truncation_k, 8)


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class Block3TruncationExecutionTest(unittest.TestCase):
    def test_block3_k_changes_post_polynomial_output(self):
        from rfr.search.runtime import model_handler as fh

        common = dict(
            degree=2,
            N=8192,
            x_fresh_sf=30,
            inv_2n_sf=22,
            square_rescale_sfs=(None, None),
        )
        no_k = make_block3_default_config(**common, output_truncation_k=None)
        k8 = make_block3_default_config(**common, output_truncation_k=8)
        k13 = make_block3_default_config(**common, output_truncation_k=13)
        x = torch.tensor([0.17391, -0.28137], dtype=torch.float64)
        original_sampler = fh._sample_gaussian_for_point
        fh._sample_gaussian_for_point = lambda reference, _point: torch.zeros_like(reference)
        try:
            raw = _make_block3_approximation_exponential(no_k)(x)
            out8 = _make_block3_approximation_exponential(k8)(x)
            out13 = _make_block3_approximation_exponential(k13)(x)
        finally:
            fh._sample_gaussian_for_point = original_sampler

        torch.testing.assert_close(out8, torch.trunc(raw * (2 ** 8)) / (2 ** 8))
        torch.testing.assert_close(out13, torch.trunc(raw * (2 ** 13)) / (2 ** 13))
        self.assertFalse(torch.equal(out8, out13))


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class SampleGaussianLiveReadTest(unittest.TestCase):
    """The optimizer override only changes ``cfg.<field>.scaling_factor``.

    For that mutation to actually affect noise installed in the model,
    ``_sample_gaussian_for_point`` MUST read ``point.scaling_factor`` at every
    forward call (not cache it at install time). This property is what makes
    the entire chain end-to-end correct.
    """

    def test_mutating_scaling_factor_changes_noise_variance(self):

        point = NoisePoint(distribution="encoding", scaling_factor=22, N=8192)
        reference = torch.zeros(50000)

        torch.manual_seed(0)
        sample_low_sf = _sample_gaussian_for_point(reference, point)
        var_low = float(sample_low_sf.var())


        point.scaling_factor = 30
        torch.manual_seed(0)
        sample_high_sf = _sample_gaussian_for_point(reference, point)
        var_high = float(sample_high_sf.var())

        self.assertGreater(var_low, var_high,
                           "higher SF must produce smaller noise variance")


        ratio = var_low / max(var_high, 1e-30)
        self.assertGreater(ratio, 100.0,
                           f"variance ratio {ratio:.2e} too small — sf may be "
                           "cached at install time, breaking optimizer override")

    def test_none_point_returns_zero_noise(self):
        reference = torch.zeros(100)
        sample = _sample_gaussian_for_point(reference, None)
        self.assertEqual(float(sample.abs().max()), 0.0,
                         "None NoisePoint must produce zero noise (fused-away path)")

    def test_distribution_string_case_insensitive(self):


        upper = NoisePoint(distribution="ENCODING", scaling_factor=22, N=8192)
        lower = NoisePoint(distribution="encoding", scaling_factor=22, N=8192)
        reference = torch.zeros(10000)
        torch.manual_seed(0)
        s_upper = _sample_gaussian_for_point(reference, upper)
        torch.manual_seed(0)
        s_lower = _sample_gaussian_for_point(reference, lower)

        self.assertAlmostEqual(float(s_upper.var()), float(s_lower.var()),
                               places=6)


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class SequentialPolicyInitTest(unittest.TestCase):
    """Pin the GTrXL per-slot actor init and warmstart-prior contract."""

    def _make_policy(self) -> "BLBStage2SequentialPolicy":
        cfg = SequentialPolicyConfig(
            state_dim=64,
            max_step_dim=13,
            max_num_levels=6,
            d_hidden=256,
            horizon=59,
        )
        return BLBStage2SequentialPolicy(cfg)

    def test_action_head_weight_is_small(self):
        """Every per-slot head must start near zero so external prior wins."""
        policy = self._make_policy()
        for idx, weight in enumerate(policy.slot_head_weight):
            weight_norm = float(weight.norm().item())
            self.assertLess(
                weight_norm, 0.05,
                f"slot_head_weight[{idx}] norm {weight_norm:.4f} too large; "
                "warmstart prior will be overwhelmed by random logits",
            )

    def test_action_head_bias_starts_zero(self):
        """Learned actor biases start zero; caller controls baseline prior."""
        policy = self._make_policy()
        for idx, bias in enumerate(policy.slot_head_bias):
            self.assertTrue(
                torch.allclose(bias, torch.zeros_like(bias)),
                f"slot_head_bias[{idx}] must start at 0",
            )
        self.assertTrue(torch.all(policy._preferred_per_slot_idx < 0))

    def test_value_head_init_standard(self):
        policy = self._make_policy()
        final_linear = policy.value_head[-1]
        self.assertTrue(
            torch.allclose(final_linear.bias, torch.zeros_like(final_linear.bias)),
            "value head final bias must start at 0",
        )
        weight_norm = float(final_linear.weight.norm().item())
        self.assertLess(weight_norm, 2.0,
                        f"value head final weight norm {weight_norm:.4f} unexpected")

    def test_warmstart_bias_dominates_random_state(self):
        """With +3.5 warmstart bias and random state, preferred index should
        retain a comfortable margin over alternatives (>2.0 logit units)."""
        policy = self._make_policy()
        max_step_dim = policy.cfg.max_step_dim
        preferred_idx = [4] * max_step_dim
        policy.apply_preferred_per_step_bias(preferred_idx, gain=3.5)

        torch.manual_seed(0)

        state = torch.randn(32, policy.cfg.state_dim)
        with torch.no_grad():
            logits, _value = policy.forward(state, baseline_prior_scale=3.5)

        preferred_logit = logits[:, :, 4]
        mean_other_logit = (
            logits[:, :, [0, 1, 2, 3, 5]].mean(dim=-1)
        )
        margin = (preferred_logit - mean_other_logit).mean().item()


        self.assertGreater(
            margin, 2.5,
            f"warmstart-bias margin {margin:.3f} too low; encoder noise is "
            "overwhelming the bias (regression of 2026-05-19 fix)",
        )


@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class ProbeRunnerHelpersTest(unittest.TestCase):
    """Pin the trial-split and seed-derivation rules behind the 2026-05-19
    two-GPU reward-probe parallelism. These are pure-Python, so they run on
    any machine where torch imports.
    """

    def test_split_round_robin_k5_n2(self):
        out = _split_round_robin(5, 2)
        self.assertEqual(out, [[0, 2, 4], [1, 3]])

    def test_split_round_robin_k5_n3(self):
        out = _split_round_robin(5, 3)
        self.assertEqual(out, [[0, 3], [1, 4], [2]])

    def test_split_round_robin_k4_n4_one_trial_per_gpu(self):
        out = _split_round_robin(4, 4)
        self.assertEqual(out, [[0], [1], [2], [3]])

    def test_split_round_robin_k0(self):
        out = _split_round_robin(0, 2)
        self.assertEqual(out, [[], []])

    def test_split_round_robin_single_worker(self):
        out = _split_round_robin(5, 1)
        self.assertEqual(out, [[0, 1, 2, 3, 4]])

    def test_trial_seed_deterministic(self):

        s1 = _trial_seed(12345, 3)
        s2 = _trial_seed(12345, 3)
        self.assertEqual(s1, s2)

    def test_trial_seed_independent_per_trial(self):

        base = 12345
        seeds = [_trial_seed(base, i) for i in range(5)]

        self.assertEqual(len(set(seeds)), 5,
                         f"trial seeds collided: {seeds}")

        self.assertGreater(max(seeds) - min(seeds), 10**9,
                           f"trial seeds too clustered: {seeds}")

    def test_parse_device_ids_basic(self):
        self.assertEqual(parse_device_ids("0,1"), [0, 1])
        self.assertEqual(parse_device_ids("0"), [0])
        self.assertEqual(parse_device_ids(""), [])
        self.assertEqual(parse_device_ids(None), [])

        self.assertEqual(parse_device_ids(" 0 , 1 "), [0, 1])

    def test_parse_device_ids_accepts_fire_tuple(self):


        self.assertEqual(parse_device_ids((0, 1)), [0, 1])
        self.assertEqual(parse_device_ids([0, 1]), [0, 1])
        self.assertEqual(parse_device_ids((0, 1, 2, 3)), [0, 1, 2, 3])
        self.assertEqual(parse_device_ids(0), [0])

    def test_parse_device_ids_accepts_parenthesized_fire_string(self):


        self.assertEqual(parse_device_ids("(0, 1)"), [0, 1])
        self.assertEqual(parse_device_ids("[0, 1]"), [0, 1])

    def test_parse_device_ids_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_device_ids("0,abc")


@unittest.skipUnless(
    _TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()
    and torch.cuda.device_count() >= 2,
    "two-GPU probe runner test requires >=2 visible CUDA devices",
)
class ProbeRunnerTwoGPUTest(unittest.TestCase):
    """End-to-end smoke: build a runner with two trivial models on two GPUs,
    run K trials, check we get K results back in correct order. Only runs
    when CUDA reports >=2 devices.

    Uses a stub model (nn.Linear) rather than full BERT to keep the test
    fast; the trial-split + thread-fan-out + ordering logic is what matters.
    """

    def _make_stub_setup(self, num_layers: int = 2):


        from rfr.search.runtime.probe_runner import ProbeRunner, ProbeWorker

        class _StubBatch:
            def __init__(self, device):
                self.input_ids = torch.zeros(2, 4, dtype=torch.long, device=device)
                self.attention_mask = torch.ones(2, 4, dtype=torch.long, device=device)
                self.labels = torch.zeros(2, dtype=torch.long, device=device)
                self.token_type_ids = None

        class _StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 2)

            def forward(self, *, input_ids, attention_mask, labels=None, token_type_ids=None):

                x = input_ids.float()
                logits = self.proj(x)
                class _Out:
                    pass
                out = _Out()
                out.logits = logits
                return out

        workers = []
        for d in (0, 1):
            device = torch.device(f"cuda:{d}")
            with torch.cuda.device(device):
                model = _StubModel().to(device).eval()


            class _NoopBridge:
                def apply(self, **_kw): pass
                def clear(self): pass
            workers.append(ProbeWorker(
                device=device,
                model=model,
                handler=None,
                bridge=_NoopBridge(),
                probe_batches=[_StubBatch(device)],
                is_regression=False,
                role=("primary" if d == 0 else "replica"),
            ))
        return ProbeRunner(workers)

    def test_runner_returns_results_in_trial_order(self):
        runner = self._make_stub_setup()
        results = runner.run_trials(k=5, base_seed=12345)
        self.assertEqual(len(results), 5,
                         f"expected 5 (loss, m1, m2) tuples; got {len(results)}")

        diag = runner.last_diagnostics
        self.assertEqual(diag.k, 5)
        self.assertEqual(len(diag.per_worker_seconds), 2)
        self.assertEqual(diag.per_worker_trial_counts, [3, 2],
                         "round-robin split of k=5 across 2 workers should be [3, 2]")
        self.assertEqual([str(d) for d in runner.devices], ["cuda:0", "cuda:1"])

    def test_runner_handles_k0_gracefully(self):
        runner = self._make_stub_setup()
        results = runner.run_trials(k=0, base_seed=0)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
