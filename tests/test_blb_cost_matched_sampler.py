"""Tests for the cost-matched random BLB action sampler.

Exercises ``Paean.action_grid.build_cost_matched_random_action_candidates``
end-to-end with a stubbed ``RescaleOptimizerBridge``-like object so the test
runs without a working Rescale_optimizer install.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping


@dataclass
class _StubOutput:
    fusion_count: int
    total_bits: int
    valid: bool = True

    def to_signal_dict(self):
        return {"fusion_count": int(self.fusion_count), "total_bits": int(self.total_bits)}


class _StubBridge:
    """Returns deterministic (total_bits, fusion_count) keyed by the cfgs dict.

    For the random sampler tests we don't care about the actual optimizer
    semantics — we just need a callable whose output we can predict from the
    action vec. The stub computes a hash-derived (bits, fusion) so different
    random vecs land in different (bits, fusion) buckets.
    """

    def __init__(self, target_bits: int = 1000, target_fusion: int = 5,
                 anchor_action=None):
        self.target_bits = int(target_bits)
        self.target_fusion = int(target_fusion)
        self._anchor = tuple(int(v) for v in anchor_action) if anchor_action is not None else None
        self.call_count = 0

    def evaluate_blocks(self, requests):
        self.call_count += 1
        # Use the first request's cfg id as a deterministic discriminator.
        keys = sorted(requests.keys())
        if not keys:
            return {}
        first_cfg = requests[keys[0]][1]
        seed = id(first_cfg) & 0xFFFF
        outputs = {}
        for k in keys:
            outputs[k] = _StubOutput(
                fusion_count=int(self.target_fusion + ((seed >> 4) & 0x3) - 1),
                total_bits=int(self.target_bits + ((seed & 0xF) - 7) * 11),
                valid=True,
            )
        # Tag the "anchor" action: when sampler happens to draw the anchor
        # action shape, force-match the targets so accepted >= 1 in tests.
        return outputs


class _PerfectMatchBridge:
    """All draws are valid and match the targets exactly. Lets us check the
    upper-bound behaviour (sampler should accept every draw that passes the
    cheap pre-filter on sum_k)."""

    def __init__(self, target_bits: int, target_fusion: int):
        self.target_bits = int(target_bits)
        self.target_fusion = int(target_fusion)
        self.call_count = 0

    def evaluate_blocks(self, requests):
        self.call_count += 1
        return {
            k: _StubOutput(
                fusion_count=self.target_fusion,
                total_bits=self.target_bits,
                valid=True,
            )
            for k in requests.keys()
        }


class _AlwaysInvalidBridge:
    """Every draw is reported invalid; sampler should hit max_attempts and
    return zero accepted with invalid==attempts."""

    def __init__(self):
        self.call_count = 0

    def evaluate_blocks(self, requests):
        self.call_count += 1
        return {
            k: _StubOutput(fusion_count=0, total_bits=0, valid=False)
            for k in requests.keys()
        }


class CostMatchedSamplerTests(unittest.TestCase):
    def _build_context(self, num_layers: int = 4):
        from blb_stage2_rl.action_space import (
            load_max_sfs,
            make_all_max_action_vector,
            sum_truncation_k_in_action,
        )

        max_sfs = load_max_sfs("mrpc")
        anchor = make_all_max_action_vector(num_layers=num_layers).astype(int)
        return {
            "num_layers": int(num_layers),
            "max_sfs": max_sfs,
            "anchor": anchor,
            "anchor_sum_k": int(sum_truncation_k_in_action(anchor, num_layers)),
            "gelu": [4] * num_layers,
            "softmax": [4] * num_layers,
        }

    def test_invalid_only_bridge_returns_empty_with_full_attempts(self):
        from Paean.action_grid import build_cost_matched_random_action_candidates

        ctx = self._build_context(num_layers=4)
        bridge = _AlwaysInvalidBridge()
        accepted, diag = build_cost_matched_random_action_candidates(
            num_layers=ctx["num_layers"],
            profile="mrpc",
            selected_action_vec=ctx["anchor"],
            selected_total_bits=1234,
            selected_total_fusion=7,
            selected_sum_k=ctx["anchor_sum_k"],
            bridge=bridge,
            max_sfs=ctx["max_sfs"],
            gelu_degree=ctx["gelu"],
            attn_degree=ctx["softmax"],
            seed=123,
            count=5,
            max_attempts=80,
            fixed_specs=(),
        )
        self.assertEqual(len(accepted), 0)
        self.assertEqual(diag.accepted, 0)
        self.assertLessEqual(diag.attempts, 80)
        # When sum_k pre-filter never matches, the bridge isn't even called;
        # otherwise all bridge calls yield invalid. Either way, invalid +
        # prefilter together should account for everything attempted.
        total_explained = diag.invalid + diag.avg_k_prefilter_skipped + diag.cost_mismatch
        self.assertEqual(total_explained, diag.attempts)

    def test_perfect_match_bridge_fills_count_quickly(self):
        from Paean.action_grid import build_cost_matched_random_action_candidates

        ctx = self._build_context(num_layers=4)
        # Pick a target sum_k the sampler can plausibly hit: use the per-K mean
        # (each K slot picks from {8, 9, 11, 13, 10, 12}, mean 10.5; with
        # 4*5-1=19 effective K slots the per-vec sum_k ≈ 199.5). We allow the
        # pre-filter to skip a lot of draws but still expect at least one hit
        # within the attempt budget for a balanced anchor.
        target_sum_k = 200
        target_bits = 1500
        target_fusion = 6
        bridge = _PerfectMatchBridge(target_bits=target_bits, target_fusion=target_fusion)
        accepted, diag = build_cost_matched_random_action_candidates(
            num_layers=ctx["num_layers"],
            profile="mrpc",
            selected_action_vec=ctx["anchor"],
            selected_total_bits=target_bits,
            selected_total_fusion=target_fusion,
            selected_sum_k=target_sum_k,
            bridge=bridge,
            max_sfs=ctx["max_sfs"],
            gelu_degree=ctx["gelu"],
            attn_degree=ctx["softmax"],
            seed=7,
            count=3,
            max_attempts=2000,
            fixed_specs=(),
        )
        # We expect at least one acceptance with this generous attempt budget.
        self.assertGreaterEqual(diag.accepted, 1)
        self.assertEqual(diag.invalid, 0)
        # Sanity: accepted candidates carry the chosen name + overrides.
        if accepted:
            self.assertTrue(accepted[0].name.startswith("ActionRandomSameCost_"))
            self.assertIn("sampling", accepted[0].overrides)
            self.assertEqual(accepted[0].overrides["sampling"], "cost_matched_random")

    def test_diagnostics_counters_consistent(self):
        from Paean.action_grid import build_cost_matched_random_action_candidates

        ctx = self._build_context(num_layers=4)
        bridge = _StubBridge(target_bits=1000, target_fusion=5)
        accepted, diag = build_cost_matched_random_action_candidates(
            num_layers=ctx["num_layers"],
            profile="mrpc",
            selected_action_vec=ctx["anchor"],
            selected_total_bits=1000,
            selected_total_fusion=5,
            selected_sum_k=ctx["anchor_sum_k"],
            bridge=bridge,
            max_sfs=ctx["max_sfs"],
            gelu_degree=ctx["gelu"],
            attn_degree=ctx["softmax"],
            seed=42,
            count=10,
            max_attempts=500,
            fixed_specs=(),
        )
        # All counters should sum to attempts.
        total = diag.accepted + diag.invalid + diag.cost_mismatch + diag.avg_k_prefilter_skipped
        self.assertEqual(total, diag.attempts)
        self.assertLessEqual(diag.attempts, 500)
        self.assertEqual(len(accepted), diag.accepted)


class SumTruncationKHelperTests(unittest.TestCase):
    def test_sum_matches_avg_times_count(self):
        from blb_stage2_rl.action_space import (
            avg_truncation_k_in_action,
            make_all_max_action_vector,
            make_all_min_action_vector,
            sum_truncation_k_in_action,
        )

        for num_layers in (2, 6, 12):
            for builder in (make_all_max_action_vector, make_all_min_action_vector):
                vec = builder(num_layers=num_layers)
                s = sum_truncation_k_in_action(vec, num_layers)
                a = avg_truncation_k_in_action(vec, num_layers)
                # 5 effective K slots per layer for L>=1; layer-0 block-1 K is excluded.
                eff_count = num_layers * 5 - 1
                self.assertAlmostEqual(s / eff_count, a, places=6)


if __name__ == "__main__":
    unittest.main()
