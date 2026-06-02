"""Degree-0 (ReLU) Stage-2 support contract tests.

Background
----------
Stage-1 may pick GELU degree 0 for a layer, which means "replace GELU with
ReLU" (``replace_layer_gelu`` installs ``nn.ReLU``). The corresponding Stage-2
graph is ``block5_n0`` (LN tail + Wffn1 only — no polynomial GELU nodes). Until
2026-06-02 Stage-2 had no degree-0 handling and would fail whenever a layer's
GELU degree was 0. This file pins the degree-0 wiring so it cannot silently
regress:

* :class:`Degree0RescaleOptimizerContractTest` — torch-free. Talks to the real
  ``Rescale_optimizer`` package directly and proves the cost path the bridge
  feeds for a degree-0 baseline is accepted (``valid=True``), and that sending
  the polynomial ``ctpt_gelu_coeff`` delta to ``block5_n0`` is rejected (which
  is exactly why ``default_block5_cfg_to_delta`` must gate it off for degree 0).
  Runs everywhere, including the torch-free local/CI lanes.

* :class:`Degree0BaselineExtractionTest` — needs the ``blb_stage2_rl`` package
  (its ``__init__`` pulls torch), so it is guarded + skips when torch is
  missing. Reads the REAL ``static_skeletons_mrpc.json`` and checks the
  degree-0 baseline extraction (graph_key, source ``inv_std`` → x_centered
  mapping, the two n0 rescales, no GELU-coeff field, no unmapped nodes).

The model-forward install path (``replace_layer_block5_noise`` ReLU branch)
needs a real BERT model and is exercised by the server smoke run, not here.
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
    """The block5_n0 graph must accept the bridge's degree-0 payload."""

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
class Degree0BaselineExtractionTest(unittest.TestCase):
    """Real-archive baseline extraction for a degree-0 (ReLU) layer."""

    def test_allowed_gelu_degrees_includes_zero(self):
        self.assertIn(0, ALLOWED_GELU_DEGREES)

    def test_block5_n0_baseline_extraction(self):
        if not (_RO_ROOT / "configs" / "mrpc" / "static_skeletons_mrpc.json").exists():
            self.skipTest("mrpc static_skeletons archive not present")
        num_layers = 12
        # Layer 0 = ReLU (degree 0); softmax kept at 2 (block3_exp_n2 has a
        # successful baseline, unlike n6).
        gelu = [0] + [1] * (num_layers - 1)
        softmax = [2] * num_layers
        base = load_static_skeletons_baseline(
            rescale_optimizer_root=str(_RO_ROOT),
            dataset="mrpc",
            num_layers=num_layers,
            gelu_per_layer=gelu,
            softmax_per_layer=softmax,
        )
        self.assertEqual(base.aggregate_invalid_block_count, 0)
        self.assertEqual(base.missing_block_layer, [])

        bl = base.per_block_layer[(5, 0)]
        self.assertEqual(bl.graph_key, "block5_n0")
        fields = bl.field_baseline_sfs
        # Source node is named "inv_std" in the n0 graph (not "x_mean"); the
        # baseline must still populate x_centered_fresh_sf from it.
        self.assertEqual(fields.get("x_centered_fresh_sf"), 30,
                         f"x_centered_fresh_sf missing/wrong for n0: {fields}")
        # The two n0 rescales (normalize + wffn1) must be present.
        self.assertIn("normalize_rescale_sf", fields)
        self.assertIn("wffn1_rescale_sf", fields)
        # ReLU has no polynomial GELU coefficient encode / rescale.
        self.assertNotIn("gelu_coeff_sf", fields)
        self.assertNotIn("gelu_coeff_mul_rescale_sf_0", fields)
        self.assertNotIn("gelu_power_rescale_sf_0", fields)
        # Everything in the n0 skeleton mapped to an RL field — no leftovers.
        self.assertEqual(bl.unmapped_rescale_nodes, [])
        self.assertEqual(bl.unmapped_propagation_nodes, [])


if __name__ == "__main__":
    unittest.main()
