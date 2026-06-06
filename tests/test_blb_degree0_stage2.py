"""Degree-0 (ReLU) Stage-2: DISABLED at the entry, decode kept DORMANT.

Background
----------
GELU degree 0 = "replace GELU with ReLU"; its Stage-2 graph is ``block5_n0``
(LN tail + Wffn1 only — no polynomial GELU nodes). Stage-2 gained degree-0
support on 2026-06-02, but Stage-1 stopped sampling degree 0 (``f85c77e``) and
on **2026-06-06 Stage-2 disabled it too**: ``ALLOWED_GELU_DEGREES`` dropped 0,
so ``load_static_skeletons_baseline`` rejects a degree-0 layer loudly at the
bootstrap entry (both per-slot and fusion runners hit it). The block5_n0 decode
/ RO contract / handler are *kept dormant* (historical configs, manual eval, and
a one-line revert if degree 0 returns). This file pins BOTH facts:

* :class:`Degree0RescaleOptimizerContractTest` — torch-free. The dormant
  block5_n0 RO contract still holds: the real ``Rescale_optimizer`` accepts the
  degree-0 baseline payload and rejects a stray ``ctpt_gelu_coeff`` delta.

* :class:`Degree0DisabledAtEntryTest` — needs ``blb_stage2_rl`` (torch), so it is
  guarded + skips without torch. Asserts ``ALLOWED_GELU_DEGREES`` excludes 0 and
  that a degree-0 layer raises at ``load_static_skeletons_baseline``, while the
  dormant action-decode still composes a ``block5_n0`` request when fed degree 0
  directly (proving the decode is retained, not deleted).
"""
from __future__ import annotations

import os
import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_RO_ROOT = _REPO_ROOT / "Rescale_optimizer"


# ---------------------------------------------------------------------------
# Torch-free: real Rescale_optimizer contract for block5_n0
# ---------------------------------------------------------------------------
try:
    if str(_RO_ROOT) not in sys.path:
        sys.path.insert(0, str(_RO_ROOT))
    from rescale_optimizer import ReplanSession  # type: ignore

    _RO_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - environment without the package
    ReplanSession = None  # type: ignore
    _RO_IMPORT_ERROR = _exc

_RO_AVAILABLE = _RO_IMPORT_ERROR is None
_RO_SKIP = (
    "" if _RO_AVAILABLE else f"rescale_optimizer not importable: {_RO_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_RO_AVAILABLE, _RO_SKIP)
class Degree0RescaleOptimizerContractTest(unittest.TestCase):
    """Dormant block5_n0 RO contract — kept valid for re-enable / manual eval."""

    @classmethod
    def setUpClass(cls):
        cls.session = ReplanSession.from_profile(
            profile="mrpc",
            root=str(_RO_ROOT),
            include=["block5_n0", "block5_n1"],
        )

    def _replan(self, t_new, deltas):
        return self.session.replan(
            "block5_n0", t_new=t_new, delta_overrides=deltas, return_dict=True,
        )

    def test_block5_n0_is_loaded_from_profile(self):
        # Auto-discovery (include=None in production) must surface block5_n0; we
        # request it explicitly here but still assert the baseline is present.
        self.assertIn("block5_n0", self.session.baselines)
        rec = self.session.baselines["block5_n0"]
        # ReLU graph has a 3-slot t_new: [fresh, normalize_rescale, wffn1_rescale]
        self.assertEqual(len(list(rec.t_baseline)), 3,
                         f"block5_n0 t_baseline should have 3 slots; got {rec.t_baseline}")

    def test_degree0_baseline_payload_is_valid(self):
        # Exactly what default_block5_cfg_to_delta + DEFAULT_CFG_TO_T_NEW_MAP
        # produce for the block5_n0 baseline: NO ctpt_gelu_coeff delta.
        out = self._replan(
            [30, 30, 30],
            {"ctct_xmean_over_std": "x2", "ctpt_gamal": 20, "ctpt_wffn1": 22},
        )
        self.assertTrue(out.get("valid"),
                        f"block5_n0 baseline payload must be valid; got {out}")
        self.assertFalse(bool(out.get("invalid_chain")))

    def test_sending_gelu_coeff_to_n0_is_rejected(self):
        # ReLU has no polynomial coefficient node. If the bridge wrongly emits
        # ctpt_gelu_coeff for degree 0 the chain is no longer valid — this is
        # the regression default_block5_cfg_to_delta's degree>=1 gate prevents.
        out = self._replan(
            [30, 30, 30],
            {"ctct_xmean_over_std": "x2", "ctpt_gamal": 20, "ctpt_wffn1": 22,
             "ctpt_gelu_coeff": 20},
        )
        self.assertFalse(
            out.get("valid"),
            "block5_n0 must REJECT a ctpt_gelu_coeff delta (graph has no such "
            "node); default_block5_cfg_to_delta must gate it off for degree 0",
        )


# ---------------------------------------------------------------------------
# Needs blb_stage2_rl (package __init__ pulls torch) — guarded + skips.
# ---------------------------------------------------------------------------
try:
    from blb_stage2_rl.baseline_bootstrap import (  # type: ignore
        ALLOWED_GELU_DEGREES,
        load_static_skeletons_baseline,
    )

    _BB_IMPORT_ERROR = None
except Exception as _bb_exc:  # torch missing → package __init__ fails
    ALLOWED_GELU_DEGREES = None  # type: ignore
    load_static_skeletons_baseline = None  # type: ignore
    _BB_IMPORT_ERROR = _bb_exc

_BB_AVAILABLE = _BB_IMPORT_ERROR is None
_BB_SKIP = (
    "" if _BB_AVAILABLE else f"blb_stage2_rl.baseline_bootstrap not importable: {_BB_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_BB_AVAILABLE, _BB_SKIP)
class Degree0DisabledAtEntryTest(unittest.TestCase):
    """Degree 0 is rejected at the Stage-2 bootstrap entry; decode stays dormant."""

    def test_allowed_gelu_degrees_excludes_zero(self):
        self.assertNotIn(0, ALLOWED_GELU_DEGREES)
        self.assertEqual(tuple(ALLOWED_GELU_DEGREES), (1, 2, 4))

    def test_degree0_layer_rejected_at_bootstrap(self):
        # A degree-0 layer must now abort loudly at the universal Stage-2 entry
        # (both per-slot and fusion runners call load_static_skeletons_baseline).
        num_layers = 12
        gelu = [0] + [1] * (num_layers - 1)
        softmax = [2] * num_layers
        with self.assertRaises(ValueError) as ctx:
            load_static_skeletons_baseline(
                rescale_optimizer_root=str(_RO_ROOT),
                dataset="mrpc",
                num_layers=num_layers,
                gelu_per_layer=gelu,
                softmax_per_layer=softmax,
            )
        msg = str(ctx.exception)
        self.assertIn("0", msg)
        self.assertIn("disabled", msg.lower())

    def test_degree0_action_decode_still_builds_block5_n0_request(self):
        # Dormant decode: the action_space path (NOT gated by ALLOWED_GELU_DEGREES)
        # still composes a block5_n0 request when fed degree 0 directly, proving the
        # decode is retained for manual eval / a one-line re-enable — not deleted.
        from blb_stage2_rl.action_space import (
            action_vector_to_cfgs,
            build_optimizer_requests,
            load_max_sfs,
            make_all_max_action_vector,
        )

        decoded = action_vector_to_cfgs(
            make_all_max_action_vector(num_layers=1),
            load_max_sfs("mrpc"),
            num_layers=1,
            gelu_degree=[0],
            attn_degree=[2],
        )
        requests = build_optimizer_requests("mrpc", decoded.cfgs_dict())

        self.assertIn("block5_n0_L0", requests)
        self.assertNotIn("block5_n4_L0", requests)


if __name__ == "__main__":
    unittest.main()
