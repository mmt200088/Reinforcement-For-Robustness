"""Direct-replan fast enumeration path — torch-free unit + live-session tests.

The hot loop (``blb_stage2_rl/fusion_enum_fast.py``) must (a) iterate exactly
the same combo sequence as ``itertools.product`` for any contiguous rank
range, and (b) assemble installed-noise points / fusion / bits from the raw
``ReplanSession`` output with the same semantics the golden cfg path realizes.
(a) is locked here exactly; (b) is exercised against the REAL in-repo
ReplanSession on block2 with a hand-wired test template (production templates
are golden-derived on the server and gated by ``verify_template`` there).
"""
import itertools
import pathlib
import sys
from types import SimpleNamespace
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "Rescale_optimizer"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rfr.preparation.fusion import enumeration_fast as fef  # noqa: E402

try:
    from rescale_optimizer import ReplanSession
    _SESSION = ReplanSession.from_profile(profile="mrpc", root=str(_REPO / "Rescale_optimizer"))
except Exception:  # pragma: no cover
    _SESSION = None


class ComboRangeTest(unittest.TestCase):
    def test_iter_combo_range_matches_itertools_product(self):
        lens = [3, 1, 4, 2, 5]
        full = list(itertools.product(*[range(n) for n in lens]))
        total = len(full)
        for a, b in [(0, total), (7, 31), (0, 1), (total - 1, total), (17, 17)]:
            got = [tuple(c) for c in fef.iter_combo_range(lens, a, b)]
            self.assertEqual(got, full[a:b], (a, b))

    def test_contiguous_ranges_cover_exactly_once(self):
        lens = [4, 3, 5]
        total = 60
        edges = [0, 13, 14, 40, 60]
        seen = []
        for a, b in zip(edges, edges[1:]):
            seen.extend(tuple(c) for c in fef.iter_combo_range(lens, a, b))
        self.assertEqual(seen, list(itertools.product(*[range(n) for n in lens])))

    def test_no_enum_slots_single_empty_combo(self):
        self.assertEqual([tuple(c) for c in fef.iter_combo_range([], 0, 1)], [()])

    def test_derived_delta_uses_current_slot_sfs(self):
        class FakeSession:
            def __init__(self):
                self.last_deltas = None

            def replan(self, graph_key, *, t_new, delta_overrides, return_dict):
                raise AssertionError("fast enumeration requested full replan output")

            def replan_compact(self, graph_key, *, t_new, delta_overrides):
                self.last_deltas = dict(delta_overrides)
                return SimpleNamespace(
                    valid=True,
                    fusion_count=0,
                    total_bits=123,
                    compact_config={},
                )

        tpl = fef.FastEnumTemplate(
            graph_key="fake_block4",
            block_idx=4,
            n_block=16384,
            baseline_t_new=[30],
            baseline_deltas={"ctct_rot_softmax_mul_v": 52, "other": 17},
            skeleton_node_ids=[0],
            slot_t_targets=[[], []],
            slot_d_targets=[[], ["other"]],
            slot_choice_sfs=[[30, 24, 18], [22, 16, 12]],
            enum_positions=[1, 2],
            enum_choices=[[0, 1, 2], [0, 1, 2]],
            baseline_block_indices=[0, 0, 0],
            derived_deltas=[
                fef.DerivedDeltaSpec(
                    node="ctct_rot_softmax_mul_v",
                    terms=[fef.DeltaTermSpec(slot_idx=0), fef.DeltaTermSpec(slot_idx=1)],
                )
            ],
        )
        sess = FakeSession()
        res = fef.eval_combo_fast(tpl, sess, [1, 2])
        self.assertTrue(res["valid"])
        self.assertEqual(sess.last_deltas["ctct_rot_softmax_mul_v"], 24 + 12)
        self.assertEqual(sess.last_deltas["other"], 12)


@unittest.skipUnless(_SESSION is not None, "in-repo Rescale_optimizer unavailable")
class LiveSessionFastEvalTest(unittest.TestCase):
    """Hand-wired block2 template against the real ReplanSession.

    Wiring facts come from the literal mapper docstrings
    (``default_block2_cfg_to_delta``) and the block2 skeleton [0,2,4,6,8];
    this is TEST wiring only — production wiring is golden-derived.
    """

    def _template(self):
        rec = _SESSION.baselines["block2_mrpc"]
        t_base = list(rec.t_baseline)
        deltas = {
            "ctct_x_mean_over_std": "x2",
            "ctpt_gama1": 20,
            "ctpt_wq_wk": 22,
            "ctpt_rotKT_mask1": 15,
            "ctpt_rotKT_mask2": 15,
            "ctct_preprocess_qkt": "x2",
            "ctpt_mask": 15,
        }


        return fef.FastEnumTemplate(
            graph_key="block2_mrpc",
            block_idx=2,
            n_block=16384,
            baseline_t_new=t_base,
            baseline_deltas=deltas,
            skeleton_node_ids=[int(x) for x in rec.skeleton],
            slot_t_targets=[[0], []],
            slot_d_targets=[[], ["ctpt_gama1"]],
            slot_choice_sfs=[[28, 22, 16], [20, 16, 12]],
            enum_positions=[0, 1],
            enum_choices=[[14, 8, 2], [14, 10, 6]],
            baseline_block_indices=[14, 14],
            points=[
                fef.PointSpec(kind="source", distribution="fresh", N=16384, slot_idx=0, const_sf=28),
                fef.PointSpec(kind="rescale", distribution="rescale", N=16384, skel_pos=1, slot_idx=-1, const_sf=28),
                fef.PointSpec(kind="rescale", distribution="rescale", N=16384, skel_pos=4, slot_idx=-1, const_sf=28),
                fef.PointSpec(kind="encode", distribution="encoding", N=16384,
                              node="ctpt_gama1", slot_idx=1, const_sf=20),
                fef.PointSpec(kind="slot", distribution="encoding", N=16384, slot_idx=1),
                fef.PointSpec(kind="const", distribution="encoding", N=16384, const_sf=18),
            ],
        )

    def test_eval_combo_fast_baseline_and_fused_away(self):
        tpl = self._template()
        res = fef.eval_combo_fast(tpl, _SESSION, [0, 0])
        self.assertTrue(res["valid"])
        self.assertGreater(res["total_bits"], 0)
        self.assertGreaterEqual(res["fusion_count"], 0)
        self.assertGreater(res["total_variance"], 0.0)
        kinds = [p.distribution for p in res["points"]]
        self.assertIn("fresh", kinds)
        self.assertIn("encoding", kinds)


        res2 = fef.eval_combo_fast(tpl, _SESSION, [2, 2])
        self.assertIn("valid", res2)
        if res2.get("valid"):


            n_rescale_pts = sum(1 for p in res2["points"] if p.distribution == "rescale")
            self.assertLessEqual(n_rescale_pts, 2)

    def test_eval_combo_fast_is_deterministic(self):
        tpl = self._template()
        a = fef.eval_combo_fast(tpl, _SESSION, [1, 1])
        b = fef.eval_combo_fast(tpl, _SESSION, [1, 1])
        self.assertEqual(a.get("valid"), b.get("valid"))
        if a.get("valid"):
            self.assertEqual(a["fusion_count"], b["fusion_count"])
            self.assertEqual(a["total_bits"], b["total_bits"])
            self.assertEqual(
                [(p.scaling_factor, p.distribution, p.N) for p in a["points"]],
                [(p.scaling_factor, p.distribution, p.N) for p in b["points"]],
            )
            self.assertEqual(a["total_variance"], b["total_variance"])

    def test_enumerate_range_fast_reduces_like_shards(self):
        tpl = self._template()


        full_rows, full_valid = fef.enumerate_range_fast(tpl, _SESSION, start=0, stop=9)
        h1_rows, v1 = fef.enumerate_range_fast(tpl, _SESSION, start=0, stop=4)
        h2_rows, v2 = fef.enumerate_range_fast(tpl, _SESSION, start=4, stop=9)
        self.assertEqual(full_valid, v1 + v2)
        from rfr.preparation.fusion import enumeration as fusion_enum
        base = list(tpl.baseline_block_indices)
        opts_full = fusion_enum.group_min_noise_options(full_rows, base)
        opts_split = fusion_enum.group_min_noise_options(h1_rows + h2_rows, base)
        self.assertEqual(opts_full, opts_split)


if __name__ == "__main__":
    unittest.main(verbosity=2)
