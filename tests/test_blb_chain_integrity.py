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
    from function_handler import (
        NoisePoint,
        _sample_gaussian_for_point,
        make_block1_default_config,
        make_block2_default_config,
    )
    from rescale_optimizer_bridge import (
        apply_optimizer_output_to_cfg,
        sync_block2_qk_binding,
    )
    _IMPORT_ERROR = None
except Exception as _exc:  # torch / transformers / function_handler missing
    torch = None  # type: ignore
    NoisePoint = None  # type: ignore
    _sample_gaussian_for_point = None  # type: ignore
    make_block1_default_config = None  # type: ignore
    make_block2_default_config = None  # type: ignore
    apply_optimizer_output_to_cfg = None  # type: ignore
    sync_block2_qk_binding = None  # type: ignore
    _IMPORT_ERROR = _exc

_SKIP_REASON = (
    f"torch / function_handler not importable: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None else ""
)
_TORCH_AVAILABLE = _IMPORT_ERROR is None


# ---------------------------------------------------------------------------
# apply_optimizer_output_to_cfg — Block 1 write classes
# ---------------------------------------------------------------------------
@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class ApplyOptimizerOutputBlock1Test(unittest.TestCase):
    """Verify the four write classes against the real block1_mrpc skeleton."""

    # block1_mrpc baseline_skeleton from
    # Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json.
    # Positions: [0=gelu_out_fresh, 1=mean_result_rescale,
    #             2=var_result_rescale, 3=fused-stub]
    BASELINE_SKELETON = [0, 2, 4, 5]

    def setUp(self):
        # Build a cfg with RL SFs intentionally different from baseline so any
        # write-back is visible. Both rescale fields enabled so we can also
        # verify the "fused away → None" path.
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
        # baseline_skeleton[1] = node 2 (ctpt_inv_d_1) → mean_result_rescale
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
        # baseline_skeleton[2] = node 4 (ctpt_inv_d_2 / var_result_rescale).
        # cut_point_sf has no entry for node 4 → var_result_rescale gets fused
        # away, cfg field must become None.
        raw = {
            "new_compact_config": {
                "cut_point_sf": [
                    {"i": 0, "name": "gelu_out", "type": "SOURCE", "sf": 30},
                    {"i": 2, "name": "ctpt_inv_d_1", "type": "CTPT_MUL",
                     "sf_pre": 70, "sf_post": 34, "drop": 36},
                    # node 4 deliberately absent
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
        # GRAPH_NODE_TO_CFG_ATTR[1]["ctpt_ffn2"] = "wffn2_encode"
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
        # "x2" / "x4" propagation deltas have no cfg correspondence; must be
        # silently skipped rather than crash.
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
        # No prop_delta override applied (non-integer delta)
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
        # cfg untouched, overrides empty
        self.assertEqual(self.cfg.gelu_out_fresh.scaling_factor, original_fresh_sf)
        self.assertEqual(self.cfg.wffn2_encode.scaling_factor, original_wffn2_sf)
        self.assertEqual(overrides, [])

    def test_missing_compact_config_no_op(self):
        # E.g. an invoker error path returns raw without ``new_compact_config``.
        raw = {"result": {"valid": True}}
        original_fresh_sf = self.cfg.gelu_out_fresh.scaling_factor
        overrides = apply_optimizer_output_to_cfg(
            self.cfg, output_raw=raw, block_idx=1, graph_key="block1_mrpc",
            baseline_skeleton=self.BASELINE_SKELETON,
        )
        self.assertEqual(self.cfg.gelu_out_fresh.scaling_factor, original_fresh_sf)
        self.assertEqual(overrides, [])


# ---------------------------------------------------------------------------
# apply_optimizer_output_to_cfg — Block 2 rotation flag handling
# ---------------------------------------------------------------------------
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
            # rotation flags off by default
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
        # Precondition: flag already on
        self.cfg.rotation_after_gamma_rescale = True
        raw = {
            "new_compact_config": {
                "cut_point_sf": [],
                "propagation_deltas": [],
                "effective_rotations": [],  # optimizer says no rotations
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


# ---------------------------------------------------------------------------
# sync_block2_qk_binding — Q-side mirrors K-side after override
# ---------------------------------------------------------------------------
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
        # Simulate the optimizer override touching only the K side.
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
        # Q already == K (setUp).
        overrides = sync_block2_qk_binding(self.cfg)
        self.assertEqual(overrides, [],
                         "sync must be a no-op when Q already matches K")


# ---------------------------------------------------------------------------
# _sample_gaussian_for_point reads scaling_factor LIVE
# ---------------------------------------------------------------------------
@unittest.skipUnless(_TORCH_AVAILABLE, _SKIP_REASON)
class SampleGaussianLiveReadTest(unittest.TestCase):
    """The optimizer override only changes ``cfg.<field>.scaling_factor``.

    For that mutation to actually affect noise installed in the model,
    ``_sample_gaussian_for_point`` MUST read ``point.scaling_factor`` at every
    forward call (not cache it at install time). This property is what makes
    the entire chain end-to-end correct.
    """

    def test_mutating_scaling_factor_changes_noise_variance(self):
        # Same NoisePoint reference; mutate sf between two samples.
        point = NoisePoint(distribution="encoding", scaling_factor=22, N=8192)
        reference = torch.zeros(50000)

        torch.manual_seed(0)
        sample_low_sf = _sample_gaussian_for_point(reference, point)
        var_low = float(sample_low_sf.var())

        # Higher SF ⇒ smaller variance (more bits ⇒ tighter precision).
        point.scaling_factor = 30
        torch.manual_seed(0)
        sample_high_sf = _sample_gaussian_for_point(reference, point)
        var_high = float(sample_high_sf.var())

        self.assertGreater(var_low, var_high,
                           "higher SF must produce smaller noise variance")
        # The two SF levels differ by 8 bits in the table; the ratio should be
        # well over 100x for a sanity check.
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
        # rescale_optimizer_bridge.apply_optimizer_output_to_cfg never alters
        # the distribution string; this guards against a future regression
        # where someone uppercases the field and breaks the variance lookup.
        upper = NoisePoint(distribution="ENCODING", scaling_factor=22, N=8192)
        lower = NoisePoint(distribution="encoding", scaling_factor=22, N=8192)
        reference = torch.zeros(10000)
        torch.manual_seed(0)
        s_upper = _sample_gaussian_for_point(reference, upper)
        torch.manual_seed(0)
        s_lower = _sample_gaussian_for_point(reference, lower)
        # Variance should be identical because lookup is .lower()-normalised.
        self.assertAlmostEqual(float(s_upper.var()), float(s_lower.var()),
                               places=6)


if __name__ == "__main__":
    unittest.main()
