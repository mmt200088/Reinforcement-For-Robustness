"""Torch-free smoke tests for the 4-sub-stage Stage-2 RL path.

Covers:
  * progressive re-baseline budget allocator (compute_substage_acc_threshold)
  * rank-key extraction from a synthetic record
  * substage_progress JSON round-trip
  * substage_subdir layout
  * splice_block_slots_into_base (preserving block-3 baseline + non-active blocks)

The package-level ``blb_stage2_rl/__init__.py`` imports torch via runner.py,
so we sidestep it by loading ``substage_runner.py`` directly with importlib
(matches the workaround the existing torch-free tests use).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from typing import Sequence


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SUBSTAGE_RUNNER_PATH = os.path.join(_REPO_ROOT, "blb_stage2_rl", "substage_runner.py")


def _load_substage_runner_module():
    """Bypass the package import (which pulls torch) and load substage_runner
    standalone. The functions we test are pure stdlib / numpy."""
    spec = importlib.util.spec_from_file_location(
        "blb_stage2_rl_substage_runner_smoke", _SUBSTAGE_RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    # The module references ``torch`` inside ``run_substage_via_runner`` only;
    # top-level ``try: import torch`` swallows ImportError.
    spec.loader.exec_module(module)
    return module


def _make_fake_spec(layer_idx: int, block_idx: int,
                     full_vec_offsets: Sequence[int]):
    """Construct a minimal stand-in for BlockStepSpec so we don't import
    blb_stage2_rl.action_space (which pulls torch indirectly via the package
    __init__)."""

    class _FakeSpec:
        def __init__(self):
            self.layer_idx = int(layer_idx)
            self.block_idx = int(block_idx)
            self.full_vec_offsets = tuple(int(x) for x in full_vec_offsets)

    return _FakeSpec()


class BudgetAllocatorTests(unittest.TestCase):
    def test_progressive_rebaseline_first_substage_uses_acc_orig(self):
        sr = _load_substage_runner_module()
        compute_substage_acc_threshold = sr.compute_substage_acc_threshold

        t = compute_substage_acc_threshold(
            acc_orig=0.86,
            acc_after_prev=0.86,
            stage2_limit_tolerance=0.04,
            num_substages=4,
            probe_size=256,
        )
        # Threshold = max(0.86 - 0.04/4, 0.86 - 0.04) - 1/256
        expected = max(0.86 - 0.01, 0.86 - 0.04) - 1.0 / 256
        self.assertAlmostEqual(t, expected, places=6)

    def test_progressive_rebaseline_carries_unused_budget_forward(self):
        sr = _load_substage_runner_module()
        compute_substage_acc_threshold = sr.compute_substage_acc_threshold

        # Sub-stage 1 only used 0.005 (instead of allowed 0.01). Sub-stage 2's
        # acc_after_prev is 0.855; its budget = max(0.855 - 0.01, 0.86 - 0.04).
        # The progressive arm 0.845 wins, so 0.005 of unused budget effectively
        # flows forward (next stage can drop to 0.845 instead of 0.84).
        t = compute_substage_acc_threshold(
            acc_orig=0.86,
            acc_after_prev=0.855,
            stage2_limit_tolerance=0.04,
            num_substages=4,
            probe_size=256,
        )
        expected = max(0.855 - 0.01, 0.86 - 0.04) - 1.0 / 256
        self.assertAlmostEqual(t, expected, places=6)
        # Sanity: 0.845 > 0.82 (hard floor 0.86 - 0.04)
        self.assertGreater(t + 1.0 / 256, 0.86 - 0.04)

    def test_hard_floor_caps_aggressive_progressive_arm(self):
        sr = _load_substage_runner_module()
        compute_substage_acc_threshold = sr.compute_substage_acc_threshold

        # acc_after_prev is below the hard floor — happens if a prior stage
        # accidentally degraded too much (validation slip). The hard floor
        # must kick in so the next stage doesn't keep dropping further.
        t = compute_substage_acc_threshold(
            acc_orig=0.86,
            acc_after_prev=0.80,
            stage2_limit_tolerance=0.04,
            num_substages=4,
            probe_size=256,
        )
        # max(0.80 - 0.01, 0.82) = 0.82; minus 1/256
        expected = max(0.80 - 0.01, 0.86 - 0.04) - 1.0 / 256
        self.assertAlmostEqual(t, expected, places=6)
        self.assertAlmostEqual(t, 0.82 - 1.0 / 256, places=6)

    def test_threshold_is_lower_when_probe_size_is_smaller(self):
        sr = _load_substage_runner_module()
        compute_substage_acc_threshold = sr.compute_substage_acc_threshold

        t_small = compute_substage_acc_threshold(
            acc_orig=0.86, acc_after_prev=0.86,
            stage2_limit_tolerance=0.04, num_substages=4, probe_size=64,
        )
        t_big = compute_substage_acc_threshold(
            acc_orig=0.86, acc_after_prev=0.86,
            stage2_limit_tolerance=0.04, num_substages=4, probe_size=256,
        )
        # Bigger probe → smaller guard → higher threshold
        self.assertGreater(t_big, t_small)


class RankKeyTests(unittest.TestCase):
    def test_rank_key_prefers_no_invalid_then_priority_then_cost_then_reward(self):
        sr = _load_substage_runner_module()
        _rank_key_from_record = sr._rank_key_from_record

        class _Rec:
            def __init__(self, invalid, priority, cost_rank, reward):
                self.invalid_steps = invalid
                self.terminal_priority = priority
                self.terminal_cost_rank_score = cost_rank
                self.total_reward = reward

        # No invalid > 1 invalid
        a = _rank_key_from_record(_Rec(0, 3, 0.5, 10.0))
        b = _rank_key_from_record(_Rec(1, 3, 0.5, 10.0))
        self.assertGreater(a, b)
        # P3 > P2 within "no invalid"
        c = _rank_key_from_record(_Rec(0, 2, 0.5, 10.0))
        self.assertGreater(a, c)
        # Higher cost_rank wins tie within same priority
        d = _rank_key_from_record(_Rec(0, 3, 0.7, 10.0))
        self.assertGreater(d, a)
        # Reward is the last tie-breaker
        e = _rank_key_from_record(_Rec(0, 3, 0.5, 20.0))
        self.assertGreater(e, a)


class ProgressIOTests(unittest.TestCase):
    def test_substage_progress_roundtrip(self):
        sr = _load_substage_runner_module()
        load_substage_progress = sr.load_substage_progress
        save_substage_progress = sr.save_substage_progress

        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_substage_progress(d))
            payload = {
                "block_order": [1, 2, 4, 5],
                "frozen_blocks": [3],
                "current_substage_idx": 2,
                "acc_orig": 0.862,
                "acc_after_each_substage": [0.862, 0.860, 0.857],
                "completed_substages": [
                    {"substage_index": 0, "active_block": 1, "acc_after": 0.860},
                    {"substage_index": 1, "active_block": 2, "acc_after": 0.857},
                ],
            }
            save_substage_progress(d, payload)
            loaded = load_substage_progress(d)
            self.assertEqual(loaded["block_order"], [1, 2, 4, 5])
            self.assertEqual(loaded["current_substage_idx"], 2)
            self.assertEqual(len(loaded["completed_substages"]), 2)

    def test_substage_subdir_naming(self):
        sr = _load_substage_runner_module()
        substage_subdir = sr.substage_subdir

        d = "/tmp/whatever"
        self.assertEqual(
            substage_subdir(d, 0, 1),
            os.path.join(d, "substage_1_block1"),
        )
        self.assertEqual(
            substage_subdir(d, 2, 4),
            os.path.join(d, "substage_3_block4"),
        )


class SpliceBlockSlotsTests(unittest.TestCase):
    def test_splice_overwrites_only_active_block_offsets(self):
        # Pure-numpy reimplementation of the helper would let us test it
        # standalone. The real helper does the same thing so we exercise the
        # actual function via a synthetic schedule.
        import numpy as np
        sr = _load_substage_runner_module()
        splice_block_slots_into_base = sr.splice_block_slots_into_base

        # Pretend full vec is 24 elements; block 2 occupies offsets {3,4,5}
        # at layer 0, {12,13,14} at layer 1.
        frozen_base = np.array([7] * 24, dtype=np.int64)
        validated_best = np.array(list(range(24)), dtype=np.int64)
        schedule = [
            _make_fake_spec(layer_idx=0, block_idx=2, full_vec_offsets=(3, 4, 5)),
            _make_fake_spec(layer_idx=1, block_idx=2, full_vec_offsets=(12, 13, 14)),
        ]
        out = splice_block_slots_into_base(
            frozen_base=frozen_base,
            validated_best=validated_best,
            active_block_idx=2,
            substage_schedule=schedule,
        )
        # Active offsets take from validated_best
        self.assertEqual(int(out[3]), 3)
        self.assertEqual(int(out[4]), 4)
        self.assertEqual(int(out[5]), 5)
        self.assertEqual(int(out[12]), 12)
        self.assertEqual(int(out[13]), 13)
        self.assertEqual(int(out[14]), 14)
        # Every other position stays at the frozen base (7)
        untouched_idx = [i for i in range(24) if i not in {3, 4, 5, 12, 13, 14}]
        for i in untouched_idx:
            self.assertEqual(int(out[i]), 7,
                             f"non-active offset {i} should remain frozen")


if __name__ == "__main__":
    unittest.main()
