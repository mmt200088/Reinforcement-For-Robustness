"""Reconstruct per-step fusion ``(option, K)`` from a flat best-action vector.

The pure matcher (:func:`fusion_fixed_action.match_option_id`) is torch-free and
is tested here against the committed mrpc fusion-count maps (``FusionCountMap.load``
is torch-free). It must:

* match the baseline option (option_id 0) and the BOOSTED fusion option
  (option_id 1, which carries ``explicit_field_values`` above the grid) by their
  ``action_indices`` while IGNORING the separately-decided K slot, and
* raise (never silently pick a wrong option) on no-match / ambiguity.

This is the foundation of the final-eval / GLUE boost fix: the persisted
``best_action_vec`` is a flat grid-index vector that cannot carry the boost, so
both standalone paths reconstruct ``option_by_step`` from it + the map and replay
the EXACT boosted config the RL search selected.

The schedule-walking helpers (``reconstruct_fusion_group`` /
``build_fusion_fixed_config``) need ``action_space.step_schedule`` which pulls
torch, so they are exercised in the torch-gated lane / on the server.
"""

from __future__ import annotations

import pathlib
import sys
import types
import unittest
from unittest import mock

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLB_DIR = _REPO_ROOT / "blb_stage2_rl"
_RO_ROOT = _REPO_ROOT / "Rescale_optimizer"
for _p in (str(_REPO_ROOT), str(_BLB_DIR), str(_RO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import fusion_count_map as fcm
import fusion_fixed_action as ffa


class MatchOptionIdTest(unittest.TestCase):
    def setUp(self):
        self.m = fcm.FusionCountMap.load("mrpc")
        # block2 is one of the boosted block-types (opt 1 carries the boost).
        self.graph = self.m.graphs["block2_mrpc"]

    def test_matches_baseline_option_zero(self):
        opt0 = next(o for o in self.graph.options if o.option_id == 0)
        slice0 = np.asarray(opt0.action_indices, dtype=int)
        self.assertEqual(
            ffa.match_option_id(action_slice=slice0, graph=self.graph, graph_key="block2_mrpc"),
            0,
        )

    def test_matches_boosted_option_ignoring_k_slot(self):
        opt1 = next(o for o in self.graph.options if o.option_id == 1)
        # Sanity: this option IS the boosted one (the whole reason the flat vec
        # alone loses information).
        self.assertTrue(opt1.boosted)
        self.assertTrue(opt1.explicit_field_values)
        slice1 = np.asarray(opt1.action_indices, dtype=int).copy()
        # The K slot is decided independently of the option; the matcher must
        # ignore it. Perturb it and still match option 1.
        k = int(self.graph.k_slot_index)
        slice1[k] = (int(slice1[k]) + 1) % 5
        self.assertEqual(
            ffa.match_option_id(action_slice=slice1, graph=self.graph, graph_key="block2_mrpc"),
            1,
        )

    def test_matches_legacy_all_max_baseline_with_slot_dims(self):
        # A legacy all-max full action vector encodes rescale slots as their max
        # grid index, while the fusion-count baseline option keeps fused-away /
        # absent rescales at idx0=None. When the slice is exactly all-max, it is
        # still an unambiguous baseline option.
        slot_dims = [15] * int(self.graph.block_num_slots)
        slot_dims[int(self.graph.k_slot_index)] = 5
        legacy_all_max = np.asarray([d - 1 for d in slot_dims], dtype=int)
        self.assertEqual(
            ffa.match_option_id(
                action_slice=legacy_all_max,
                graph=self.graph,
                graph_key="block2_mrpc",
                slot_dims=slot_dims,
            ),
            0,
        )

    def test_raises_on_no_match(self):
        bad = np.full(self.graph.block_num_slots, 99, dtype=int)
        with self.assertRaises(ValueError):
            ffa.match_option_id(action_slice=bad, graph=self.graph, graph_key="block2_mrpc")

    def test_every_committed_option_round_trips(self):
        # Every option's own action_indices must resolve back to itself across
        # all committed graphs (no accidental ambiguity introduced by boost).
        for gk, graph in self.m.graphs.items():
            for opt in graph.options:
                sl = np.asarray(opt.action_indices, dtype=int)
                self.assertEqual(
                    ffa.match_option_id(action_slice=sl, graph=graph, graph_key=gk),
                    int(opt.option_id),
                    f"{gk} option {opt.option_id} did not round-trip",
                )


class SelectFusionEvalMetadataTest(unittest.TestCase):
    """The final-eval metadata resolver wiring (torch-free branches).

    The reconstruct-from-vec branch needs ``step_schedule`` (torch) and runs in the
    torch lane; here we lock the branches that decide WHETHER to attach the persisted
    fusion group: only the trained best (== base_action) in a fusion run gets the
    boost-replay metadata; explicit configs win; per-slot runs and non-best
    candidates are left untouched.
    """

    BASE = [1, 2, 3, 4, 5]
    GROUP = {"option_by_step": {"0": 1}, "choices_by_step": []}

    def _call(self, **kw):
        params = dict(
            action_vec=self.BASE,
            base_action=self.BASE,
            existing_metadata={},
            fusion_group=self.GROUP,
            fusion_count_action=True,
            profile="mrpc",
            num_layers=12,
            gelu=[4] * 12,
            softmax=[6] * 12,
        )
        params.update(kw)
        return ffa.select_fusion_eval_metadata(**params)

    def test_attaches_persisted_group_for_best(self):
        md = self._call()
        self.assertEqual(md["schema_version"], "fusion_count_fixed_action_v1")
        self.assertEqual(md["group"], self.GROUP)

    def test_explicit_fusion_config_is_left_as_is(self):
        existing = {"schema_version": "fusion_count_fixed_action_v1", "group": {"option_by_step": {"3": 0}}}
        md = self._call(existing_metadata=existing)
        self.assertEqual(md["group"], {"option_by_step": {"3": 0}})  # not overwritten

    def test_per_slot_run_untouched(self):
        md = self._call(fusion_count_action=False, fusion_group=None)
        self.assertNotIn("schema_version", md)
        self.assertNotIn("group", md)

    def test_non_best_candidate_untouched(self):
        # A cost-matched random / range-mutated candidate differs from base_action
        # and must NOT get boost-replay (it is an arbitrary vec, not a map option).
        md = self._call(action_vec=[9, 9, 9, 9, 9])
        self.assertNotIn("schema_version", md)

    def test_no_base_action_untouched(self):
        md = self._call(base_action=None)
        self.assertNotIn("schema_version", md)


class ReconstructFusionGroupBaselineOwnedBlocksTest(unittest.TestCase):
    @staticmethod
    def _step(step_idx, block_idx, graph_key, offsets):
        return types.SimpleNamespace(
            step_idx=step_idx,
            layer_idx=0,
            block_idx=block_idx,
            graph_key_suffix=graph_key,
            full_vec_offsets=tuple(offsets),
            slot_dims=(15, 2),
        )

    @staticmethod
    def _graph():
        option = types.SimpleNamespace(
            option_id=1,
            action_indices=(7, 0),
            fusion_count=1,
            boosted=True,
        )
        return types.SimpleNamespace(k_slot_index=1, options=[option])

    def test_missing_block1_and_block3_maps_do_not_block_fusion_reconstruction(self):
        schedule = [
            self._step(0, 1, "block1_mrpc_large", (0, 1)),
            self._step(1, 2, "block2_mrpc_large", (2, 3)),
            self._step(2, 3, "block3_exp_n6", (4, 5)),
        ]
        fake_action_space = types.SimpleNamespace(
            K_LEVELS=(13, 11),
            step_schedule=lambda *_args, **_kwargs: schedule,
        )
        fusion_map = types.SimpleNamespace(
            graphs={"block2_mrpc_large": self._graph()},
        )

        with mock.patch.dict(sys.modules, {"action_space": fake_action_space}):
            result = ffa.reconstruct_fusion_group(
                [0, 0, 7, 1, 0, 0],
                fusion_map=fusion_map,
                num_layers=1,
                profile="mrpc_large",
                gelu=[1],
                softmax=[6],
            )

        self.assertEqual(result["option_by_step"], {"1": 1})
        self.assertEqual(result["summary"]["step_count"], 3)
        self.assertEqual(result["summary"]["total_fusion_count"], 1)
        self.assertEqual(result["summary"]["k_values"], [11])

    def test_missing_fusion_managed_block_map_still_fails_loudly(self):
        schedule = [self._step(0, 4, "block4", (0, 1))]
        fake_action_space = types.SimpleNamespace(
            K_LEVELS=(13, 11),
            step_schedule=lambda *_args, **_kwargs: schedule,
        )

        with mock.patch.dict(sys.modules, {"action_space": fake_action_space}):
            with self.assertRaisesRegex(KeyError, "block4"):
                ffa.reconstruct_fusion_group(
                    [7, 1],
                    fusion_map=types.SimpleNamespace(graphs={}),
                    num_layers=1,
                    profile="mrpc_large",
                    gelu=[1],
                    softmax=[6],
                )


if __name__ == "__main__":
    unittest.main()
