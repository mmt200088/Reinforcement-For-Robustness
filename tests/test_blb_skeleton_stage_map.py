"""Skeleton-driven stage-map derivation tests (torch-free).

Pins the SSOT in ``blb_stage2_rl/skeleton_stage_map.py``: the ordered t_new
stages + active rescale slots must be DERIVED from the actual
``static_skeletons`` cut_point_sf, not hard-coded. This is the regression guard
for the 2026 skeleton regen that silently drifted block2 / block4 / block5_n1.

Imports ``skeleton_stage_map`` directly (it has no relative imports), so the
package ``__init__`` (which pulls torch) is bypassed and the test runs in the
torch-free lane. ``rescale_optimizer`` is torch-free too.
"""
from __future__ import annotations

import os
import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_RO_ROOT = _REPO_ROOT / "Rescale_optimizer"
_BLB_DIR = _REPO_ROOT / "blb_stage2_rl"

try:
    if str(_BLB_DIR) not in sys.path:
        sys.path.insert(0, str(_BLB_DIR))
    if str(_RO_ROOT) not in sys.path:
        sys.path.insert(0, str(_RO_ROOT))
    import skeleton_stage_map as ssm  # type: ignore
    from rescale_optimizer import ReplanSession  # type: ignore

    _IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover
    ssm = None  # type: ignore
    ReplanSession = None  # type: ignore
    _IMPORT_ERROR = _exc

_AVAILABLE = _IMPORT_ERROR is None
_SKIP = "" if _AVAILABLE else f"skeleton_stage_map / rescale_optimizer not importable: {_IMPORT_ERROR!r}"

# The CORRECT cfg-field sequence per graph (block2/4/5_n1 are the FIXED values
# the regen should produce; the rest reproduce the previously-correct maps).
_EXPECTED_T_NEW_CFG_FIELDS = {
    "block1_mrpc": ["gelu_out_fresh", "mean_result_rescale", "var_result_rescale"],
    "block2_mrpc": ["inv_std_fresh", "gamma_result_rescale",
                    "kt_mask1_result_rescale", "qkt_matmul_result_rescale"],
    "block3_exp_n2": ["x_fresh", "square_rescales", "square_rescales"],
    "block4": ["softmax_out_fresh", "softmax_v_matmul_rescale",
               "ln_mean_result_rescale", "ln_square_result_rescale"],
    "block5_n0": ["x_centered_fresh", "normalize_result_rescale", "wffn1_result_rescale"],
    "block5_n1": ["x_centered_fresh", "normalize_result_rescale", "gelu_coeff_mul_rescales"],
    "block5_n2": ["x_centered_fresh", "normalize_result_rescale",
                  "wffn1_result_rescale", "gelu_coeff_mul_rescales"],
    "block5_n4": ["x_centered_fresh", "normalize_result_rescale", "wffn1_result_rescale",
                  "gelu_power_rescales", "gelu_coeff_mul_rescales"],
}


@unittest.skipUnless(_AVAILABLE, _SKIP)
class SkeletonStageMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = ReplanSession.from_profile(profile="mrpc", root=str(_RO_ROOT))
        arch = {gk: cls.session.baselines[gk].archive_entry for gk in cls.session.baselines}
        cls.plans = ssm.build_stage_plans(arch)

    def test_t_new_length_matches_replan_session(self):
        # Derived stage count must equal RO's own baseline t length for every graph.
        for gk, plan in self.plans.items():
            tb = list(self.session.baselines[gk].t_baseline)
            self.assertEqual(
                len(plan.t_new_entries), len(tb),
                f"{gk}: derived {len(plan.t_new_entries)} stages != t_baseline {len(tb)}",
            )

    def test_no_unmapped_rescale_nodes(self):
        for gk, plan in self.plans.items():
            self.assertEqual(
                plan.unmapped_rescale_nodes, [],
                f"{gk}: unmapped rescale nodes {plan.unmapped_rescale_nodes} — "
                "skeleton has a node the SSOT table does not cover",
            )

    def test_derived_cfg_fields_match_expected(self):
        for gk, expected in _EXPECTED_T_NEW_CFG_FIELDS.items():
            self.assertIn(gk, self.plans, f"{gk} missing from derived plans")
            got = [e[0] for e in self.plans[gk].t_new_entries]
            self.assertEqual(got, expected, f"{gk}: derived cfg fields {got} != expected {expected}")

    def test_block2_active_rescales_follow_new_skeleton(self):
        # The 2026 regen: block2 rescales are gama1 / rotKT_mask1 / preprocess_qkt.
        active = set(self.plans["block2_mrpc"].active_rescale_rl_fields)
        self.assertEqual(
            active,
            {"gamma_rescale_sf", "kt_mask1_rescale_sf", "q_mask1_rescale_sf", "qkt_matmul_rescale_sf"},
        )
        # The pre-regen active slots must no longer be active.
        self.assertNotIn("kt_mask2_rescale_sf", active)
        self.assertNotIn("qkt_merge_mask_rescale_sf", active)

    def test_block4_active_rescales_follow_new_skeleton(self):
        active = set(self.plans["block4"].active_rescale_rl_fields)
        self.assertEqual(
            active,
            {"softmax_v_matmul_rescale_sf", "ln_mean_rescale_sf", "ln_square_rescale_sf"},
        )
        self.assertNotIn("ln_var_rescale_sf", active)

    def test_block5_n1_middle_rescale_is_normalize_not_gamma(self):
        active = set(self.plans["block5_n1"].active_rescale_rl_fields)
        self.assertIn("normalize_rescale_sf", active)
        self.assertNotIn("gamma_rescale_sf", active)

    def test_complete_chain_is_fully_mapped(self):
        # The SSOT must cover EVERY cut-point of every COMPLETE chain (not just
        # the nodes the current skeleton selects), so any skeleton subset — and
        # any future RO regen — maps without gaps. A new RO node fails here loudly.
        cfgs = ssm.load_profile_configs(str(_RO_ROOT), "mrpc")
        self.assertTrue(cfgs, "no per-graph configs found under configs/mrpc/")
        for graph_key, cfg in cfgs.items():
            bidx = ssm._block_idx_for_graph(graph_key)
            self.assertIsNotNone(bidx, f"cannot classify graph {graph_key}")
            missing = ssm.unmapped_full_chain_nodes(bidx, cfg)
            self.assertEqual(
                missing, [],
                f"{graph_key}: full-chain nodes {missing} are not in the SSOT node "
                "map — extend _NODE_MAP to keep the automation complete",
            )


# ---------------------------------------------------------------------------
# Proof that the BRIDGE actually DERIVES its t_new map from the live skeleton
# (skeleton_stage_map), not the static DEFAULT_CFG_TO_T_NEW_MAP fallback. Needs
# torch + the RO package, so it skips in the torch-free lane and runs on the
# server. This is the decisive auto-adapt check: it asserts the *derivation*
# output directly, independent of whether the static fallback also happens to be
# correct, so a future skeleton regen is provably handled automatically.
# ---------------------------------------------------------------------------
try:
    from rescale_optimizer_bridge import (  # type: ignore
        InProcessInvoker as _InProcessInvoker,
        _derive_t_new_table_from_invoker as _derive_t_new,
    )

    _BRIDGE_IMPORT_ERROR = None
except Exception as _be:  # pragma: no cover - torch / RO package missing
    _InProcessInvoker = None  # type: ignore
    _derive_t_new = None  # type: ignore
    _BRIDGE_IMPORT_ERROR = _be

_BRIDGE_AVAILABLE = _BRIDGE_IMPORT_ERROR is None
_BRIDGE_SKIP = "" if _BRIDGE_AVAILABLE else f"rescale_optimizer_bridge not importable: {_BRIDGE_IMPORT_ERROR!r}"


@unittest.skipUnless(_BRIDGE_AVAILABLE, _BRIDGE_SKIP)
class BridgeDerivesT_newFromSkeletonTest(unittest.TestCase):
    """The bridge's t_new map must be DERIVED from the real skeleton, so the
    auto-adaptation is genuine (not the static DEFAULT table being correct)."""

    @classmethod
    def setUpClass(cls):
        cls.table = _derive_t_new(
            _InProcessInvoker.from_profile(rescale_optimizer_root=str(_RO_ROOT), profile="mrpc")
        )

    def test_derivation_is_active_not_empty(self):
        self.assertTrue(
            self.table,
            "_derive_t_new_table_from_invoker returned EMPTY — the bridge would "
            "silently fall back to DEFAULT_CFG_TO_T_NEW_MAP and a skeleton regen "
            "would drift again. The skeleton-driven auto-adapt is NOT active.",
        )

    def test_derived_fields_follow_current_skeleton(self):
        def cfg_fields(gk):
            return [e.cfg_field for e in self.table[gk]]

        self.assertEqual(
            cfg_fields("block2_mrpc"),
            ["inv_std_fresh", "gamma_result_rescale",
             "kt_mask1_result_rescale", "qkt_matmul_result_rescale"],
        )
        self.assertEqual(
            cfg_fields("block4"),
            ["softmax_out_fresh", "softmax_v_matmul_rescale",
             "ln_mean_result_rescale", "ln_square_result_rescale"],
        )
        self.assertEqual(
            cfg_fields("block5_n1"),
            ["x_centered_fresh", "normalize_result_rescale", "gelu_coeff_mul_rescales"],
        )


if __name__ == "__main__":
    unittest.main()
