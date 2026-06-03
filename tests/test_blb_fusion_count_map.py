"""Fusion-count map + NoiseOrder tests (torch-free).

Imports the torch-free modules directly (repo root + ``blb_stage2_rl`` on
``sys.path``) so the package ``__init__`` (which pulls torch) is bypassed — runs
on a torch-free box (local dev) and in the torch-free CI lane. Mirrors the import
trick in ``tests/test_blb_skeleton_stage_map.py``.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLB_DIR = _REPO_ROOT / "blb_stage2_rl"
_RO_ROOT = _REPO_ROOT / "Rescale_optimizer"
for _p in (str(_REPO_ROOT), str(_BLB_DIR), str(_RO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fusion_count_map as fcm
import fusion_enum

import noise_tables


class NoiseOrderTest(unittest.TestCase):
    def test_summed_installed_variance_sums_table_values(self):
        order = fcm.SummedInstalledVariance()
        pts = [
            fcm.InstalledNoisePoint(scaling_factor=30, distribution="fresh", N=8192),
            fcm.InstalledNoisePoint(scaling_factor=22, distribution="encoding", N=8192),
        ]
        T = noise_tables.NOISE_VARIANCE_TABLE_BY_N
        expected = T[8192][30]["fresh"] + T[8192][22]["encoding"]
        self.assertAlmostEqual(order.total_variance(pts), expected, places=18)

    def test_empty_plan_is_zero(self):
        self.assertEqual(fcm.SummedInstalledVariance().total_variance([]), 0.0)

    def test_name_is_stable(self):
        self.assertEqual(fcm.SummedInstalledVariance().name, "summed_installed_variance")


def _toy_payload():
    return {
        "profile": "mrpc",
        "graphs": {
            "block1_mrpc": {
                "graph_key": "block1_mrpc",
                "k_slot_index": 8,  # block1 K is the 9th (last) slot
                "block_num_slots": 9,
                "options": [
                    {
                        "option_id": 0,
                        "fusion_count": 1,
                        "tie_index": 0,
                        "total_variance": 1.0,
                        "total_bits": 100,
                        "slots": {"gelu_out_sf": 30},
                        "action_indices": [4, 4, 2, 2, 3, 3, 0, 0, 3],
                    },  # all-max = baseline
                    {
                        "option_id": 1,
                        "fusion_count": 2,
                        "tie_index": 0,
                        "total_variance": 2.0,
                        "total_bits": 90,
                        "slots": {"gelu_out_sf": 28},
                        "action_indices": [3, 4, 2, 2, 0, 3, 0, 0, 3],
                    },
                ],
            },
        },
        "max_num_options": 2,
    }


class FusionMapLoaderTest(unittest.TestCase):
    def _toy(self):
        return fcm.FusionCountMap.from_payload(_toy_payload())

    def test_baseline_option_is_zero(self):
        self.assertEqual(self._toy().baseline_option_id("block1_mrpc"), 0)

    def test_num_and_max_options(self):
        m = self._toy()
        self.assertEqual(m.num_options("block1_mrpc"), 2)
        self.assertEqual(m.max_num_options(), 2)

    def test_expand_overwrites_k_slot(self):
        # option 1, K slot (idx 8) overwritten with k_index=5; rest per action_indices
        out = self._toy().expand("block1_mrpc", option_id=1, k_index=5)
        self.assertEqual(list(out), [3, 4, 2, 2, 0, 3, 0, 0, 5])

    def test_from_payload_rejects_nonzero_baseline(self):
        bad = _toy_payload()
        bad["graphs"]["block1_mrpc"]["options"][0]["option_id"] = 7
        with self.assertRaises(AssertionError):
            fcm.FusionCountMap.from_payload(bad)


class GroupMinNoiseOptionsTest(unittest.TestCase):
    """Pure grouping/ordering core (torch-free).

    Scenario: baseline (A) has fusion_count=1, but a *lower* fusion_count=0
    config (C) also exists — baseline must still be forced to option 0. B shares
    A's fusion group but has higher variance (dropped). E duplicates D's installed
    plan (deduped). F ties D's variance with a different plan (both kept).
    """

    def _ec(self, indices, fc, var, bits, sig):
        return fusion_enum.EvaluatedConfig(
            action_indices=tuple(indices),
            fusion_count=fc,
            total_bits=bits,
            total_variance=var,
            installed_signature=sig,
            slots={},
        )

    def setUp(self):
        self.BASE = (4, 4, 3)
        self.evaluated = [
            self._ec(self.BASE, fc=1, var=1.0, bits=100, sig="A"),  # baseline
            self._ec((3, 4, 3), fc=1, var=2.0, bits=90, sig="B"),  # same fc, higher var -> dropped
            self._ec((4, 0, 3), fc=0, var=5.0, bits=80, sig="C"),  # lower fc than baseline
            self._ec((0, 4, 3), fc=2, var=3.0, bits=70, sig="D1"),  # f2 min
            self._ec((0, 4, 2), fc=2, var=3.0, bits=70, sig="D1"),  # dup install plan -> deduped
            self._ec((1, 4, 3), fc=2, var=3.0, bits=60, sig="F"),  # f2 min, distinct plan -> tie kept
        ]
        self.opts = fusion_enum.group_min_noise_options(self.evaluated, self.BASE)

    def test_baseline_is_option0_despite_lower_fusion_existing(self):
        self.assertEqual(self.opts[0]["action_indices"], list(self.BASE))
        self.assertEqual(self.opts[0]["option_id"], 0)
        self.assertEqual(self.opts[0]["fusion_count"], 1)

    def test_higher_variance_in_group_dropped_and_dups_removed(self):
        # kept: A(f1), C(f0), D1(f2), F(f2)  -> B dropped, one D deduped
        self.assertEqual(len(self.opts), 4)
        kept_indices = {tuple(o["action_indices"]) for o in self.opts}
        self.assertNotIn((3, 4, 3), kept_indices)  # B dropped (higher var in f1 group)
        self.assertEqual(sum(1 for i in kept_indices if i[:2] == (0, 4)), 1)  # one D deduped

    def test_fusion_tie_pairs(self):
        pairs = {(o["fusion_count"], o["tie_index"]) for o in self.opts}
        self.assertEqual(pairs, {(1, 0), (0, 0), (2, 0), (2, 1)})

    def test_lower_fusion_config_is_not_option0(self):
        # C (fusion 0) exists but baseline (fusion 1) is option 0; C ranks after.
        c = next(o for o in self.opts if o["fusion_count"] == 0)
        self.assertGreater(c["option_id"], 0)

    def test_missing_baseline_raises(self):
        with self.assertRaises(ValueError):
            fusion_enum.group_min_noise_options(self.evaluated, (9, 9, 9))


class ActiveRescalePremiseTest(unittest.TestCase):
    """Every RL block-type must derive >=1 active rescale RL field from the real
    skeleton archive. This is the premise the builder's loud guard enforces — if
    it is ever empty, the fusion-count map loses its rescale lever and fusion
    collapses to a single option (the 2026-06-03 server build hit exactly that via
    action_space's silently-empty __file__-relative cache; the builder now seeds
    the cache from the explicit ro_root)."""

    def test_seven_block_types_have_active_rescales(self):
        import json

        import skeleton_stage_map as ssm

        arch_path = _RO_ROOT / "configs" / "mrpc" / "static_skeletons_mrpc.json"
        archive = json.loads(arch_path.read_text(encoding="utf-8"))
        plans = ssm.build_stage_plans_from_archive(archive)
        for gk in ["block1_mrpc", "block2_mrpc", "block4", "block5_n0", "block5_n1", "block5_n2", "block5_n4"]:
            self.assertIn(gk, plans, f"{gk} missing from skeleton plans")
            active = set(plans[gk].active_rescale_rl_fields)
            self.assertTrue(active, f"{gk}: no active rescale RL fields — fusion map would have no rescale lever")


if __name__ == "__main__":
    unittest.main()
