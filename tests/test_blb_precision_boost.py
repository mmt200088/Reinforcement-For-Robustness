"""Tests for the fusion-option precision boost ("加大精度") — block2.

Two lanes:
  * CandidateGenTest — pure structural core (torch-free AND rescale_optimizer-free):
    construct a ReplanProbe by hand and assert short-prime detection + the exact
    4 candidate placements the design specifies for block2 fc=1.
  * BoostReplanTest — drives boost_option through the REAL ReplanSession (the cost
    source of truth): every candidate is replan-verified, exactly 4 reach all-q_max
    with fusion_count preserved, and the boosted option's chain is 60/60.
    Skipped where rescale_optimizer can't be imported.

See docs/superpowers/specs/2026-06-19-stage2-precision-boost-design.md.
"""

import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "blb_stage2_rl")):
    if p not in sys.path:
        sys.path.insert(0, p)

import precision_boost as pb  # noqa: E402

# block2 mrpc fc=1 option (the committed fusion map's option 1).
BASE2 = {
    "inv_std_fresh_sf": 21,
    "gamma_sf": 15,
    "wk_sf": 16,
    "kt_mask1_sf": 15,
    "kt_mask2_sf": 15,
    "qkt_merge_mask_sf": 15,
    "gamma_rescale_sf": 28,
    "kt_mask1_rescale_sf": 28,
    "qkt_matmul_rescale_sf": 28,
}


def _block2_fc1_probe() -> pb.ReplanProbe:
    # q_initial [29,31,58], fuse stage 1 into next -> q_final [60,58].
    return pb.ReplanProbe(
        valid=True, fusion_count=1,
        q_initial=(29, 31, 58), q_final=(60, 58),
        fusions=({"fused_position": 1, "fused_into": "next", "small_q": 29},),
    )


class CandidateGenTest(unittest.TestCase):
    def test_short_prime_detection(self):
        # the 58 is post-fusion stage 1 == pre-fusion rescale idx 2, deficit 2.
        self.assertEqual(pb.find_short_primes(_block2_fc1_probe(), 60), [(2, 2)])

    def test_no_short_prime_when_all_qmax(self):
        probe = pb.ReplanProbe(True, 1, (60, 60), (60, 60), ())
        self.assertEqual(pb.find_short_primes(probe, 60), [])

    def test_exactly_four_candidates(self):
        # block2 fc=1: deficit 2, c=1 → budget S=1 → 1 SF placed on one of the
        # four addable encodes (cascade-compensating every rescale strictly
        # upstream of that encode by +1).
        short = pb.find_short_primes(_block2_fc1_probe(), 60)
        cands = pb.generate_candidates(pb.BLOCK2_MRPC_TOPOLOGY, BASE2, short)
        got = sorted(tuple(sorted(c.edits.items())) for c in cands)
        expect = sorted([
            # (a) kt_mask2 — between R_pre (kt_mask1_rescale) and the qkt ×2: no
            #     rescale upstream of it → no compensation.
            (("kt_mask2_sf", 16),),
            # (b) gamma — cascade compensates BOTH rescales after it (gamma_rescale
            #     + kt_mask1_rescale) so every intermediate prime stays constant.
            (("gamma_rescale_sf", 29), ("gamma_sf", 16), ("kt_mask1_rescale_sf", 29)),
            # (c) wk — only kt_mask1_rescale is upstream-after it.
            (("kt_mask1_rescale_sf", 29), ("wk_sf", 17)),
            # (d) kt_mask1 — only kt_mask1_rescale is upstream-after it.
            (("kt_mask1_rescale_sf", 29), ("kt_mask1_sf", 16)),
        ])
        self.assertEqual(got, expect)

    def test_qkt_merge_is_not_a_candidate(self):
        # qkt_merge feeds the fixed q_tail (after the last rescale) — never fills a
        # short prime, so it must never appear in any candidate edit.
        cands = pb.generate_candidates(
            pb.BLOCK2_MRPC_TOPOLOGY, BASE2, pb.find_short_primes(_block2_fc1_probe(), 60),
        )
        self.assertTrue(all("qkt_merge_mask_sf" not in c.edits for c in cands))


@unittest.skipUnless(
    __import__("importlib").util.find_spec("rescale_optimizer") is not None,
    "rescale_optimizer required",
)
class BoostReplanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rescale_optimizer import ReplanSession  # noqa: E402
        cls.S = ReplanSession.from_profile(profile="mrpc", root=str(_REPO / "Rescale_optimizer"))

    def _replan_fn(self, slots):
        t_new = [
            slots["inv_std_fresh_sf"], slots["gamma_rescale_sf"],
            slots["kt_mask1_rescale_sf"], slots["qkt_matmul_rescale_sf"],
        ]
        deltas = {
            "ctct_x_mean_over_std": "x2",
            "ctpt_gama1": slots["gamma_sf"],
            "ctpt_wq_wk": slots["wk_sf"],
            "ctpt_rotKT_mask1": slots["kt_mask1_sf"],
            "ctpt_rotKT_mask2": slots["kt_mask2_sf"],
            "ctct_preprocess_qkt": "x2",
            "ctpt_mask": slots["qkt_merge_mask_sf"],
        }
        r = self.S.replan("block2_mrpc", t_new=t_new, delta_overrides=deltas, return_dict=True)["result"]
        return pb.ReplanProbe(
            valid=bool(r["valid"]), fusion_count=len(r["fusions"]),
            q_initial=tuple(int(x) for x in r["q_initial"]),
            q_final=tuple(int(x) for x in r["q_final"]),
            fusions=tuple(r["fusions"]),
        )

    @staticmethod
    def _noise_fn(slots, _probe):
        import noise_tables
        encs = ("gamma_sf", "wk_sf", "kt_mask1_sf", "kt_mask2_sf", "qkt_merge_mask_sf")
        v = noise_tables.variance(16384, slots["inv_std_fresh_sf"], "fresh")
        for k in encs:
            v += noise_tables.variance(16384, slots[k], "encoding")
        return v

    def test_base_chain_matches_design(self):
        probe = self._replan_fn(BASE2)
        self.assertEqual(probe.q_initial, (29, 31, 58))
        self.assertEqual(probe.q_final, (60, 58))
        self.assertEqual(probe.fusion_count, 1)

    def test_boost_reaches_all_qmax(self):
        res = pb.boost_option(
            topology=pb.BLOCK2_MRPC_TOPOLOGY, base_slots=BASE2,
            replan_fn=self._replan_fn, noise_fn=self._noise_fn, q_max=60,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.base_q_final, (60, 58))
        self.assertTrue(all(q == 60 for q in res.boosted_q_final))
        # all 4 cascade-compensated placements pass replan-verify.
        self.assertEqual(res.candidates_valid, 4)
        # the boosted option preserves fusion_count and raises >= 1 SF above base.
        self.assertEqual(self._replan_fn(res.boosted_slots).fusion_count, 1)
        self.assertTrue(any(res.boosted_slots[k] > BASE2[k] for k in BASE2))


# block4 mrpc fc=1 option (the committed block4.json option 1). v_mask is bound
# to softmax_out_mask (NOT a separate slot).
BASE4 = {
    "softmax_out_fresh_sf": 21,
    "v_fresh_sf": 17,
    "softmax_out_mask_sf": 10,
    "softmax_v_mask_sf": 11,
    "wo_sf": 11,
    "ln_mean_inv_d_sf": 11,
    "ln_var_inv_d_sf": 20,
    "softmax_v_matmul_rescale_sf": 31,
    "ln_mean_rescale_sf": 31,
    "ln_square_rescale_sf": 31,
}


class Block4CandidateGenTest(unittest.TestCase):
    """block4 fc=1: the short prime sits right after the ctct_square ×2, so it
    can only reach 59 (deficit 29 is odd → S=floor(29/2)=14, fill=28), and the
    14-SF budget is distributed across the addable upstream encodes with
    softmax_out_mask costing 2/SF (it is bound to v_mask)."""

    @staticmethod
    def _probe():
        # q_initial [27,33,31], fuse stage 1 into next -> q_final [60,31].
        return pb.ReplanProbe(
            valid=True, fusion_count=1,
            q_initial=(27, 33, 31), q_final=(60, 31),
            fusions=({"fused_position": 1, "fused_into": "next", "small_q": 27},),
        )

    def test_short_prime_and_fill(self):
        sp = pb.find_short_primes(self._probe(), 60)
        self.assertEqual(sp, [(2, 29)])
        # fill=28 → prime 31 -> 59 (NOT 60: all addable encodes are before the ×2
        # → even bit-weights only → an odd 29-bit fill is unreachable).
        self.assertEqual(pb.short_prime_fill(pb.BLOCK4_MRPC_TOPOLOGY, sp[0]), 28)

    def test_budget_double_counts_softmax_out_mask(self):
        cands = pb.generate_candidates(pb.BLOCK4_MRPC_TOPOLOGY, BASE4, [(2, 29)])
        # the all-on-softmax_out_mask candidate: x=7 (2*7=14) → som 10->17.
        all_soft = [c for c in cands if c.edits.get("softmax_out_mask_sf") == 17]
        self.assertTrue(all_soft, "expected an all-softmax_out_mask (x=7) candidate")
        c = all_soft[0]
        # only softmax_out_mask raised among encodes; cascade compensates the two
        # rescales upstream of the short prime by 14.
        self.assertEqual(c.edits.get("ln_mean_rescale_sf"), 45)
        for enc in ("softmax_v_mask_sf", "wo_sf", "ln_mean_inv_d_sf"):
            self.assertNotIn(enc, c.edits)

    def test_ln_var_is_not_a_candidate(self):
        # ln_var feeds the q_tail (after the short prime's rescale) — never fills.
        cands = pb.generate_candidates(pb.BLOCK4_MRPC_TOPOLOGY, BASE4, [(2, 29)])
        self.assertTrue(all("ln_var_inv_d_sf" not in c.edits for c in cands))


@unittest.skipUnless(
    __import__("importlib").util.find_spec("rescale_optimizer") is not None,
    "rescale_optimizer required",
)
class Block4BoostReplanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rescale_optimizer import ReplanSession  # noqa: E402
        cls.S = ReplanSession.from_profile(profile="mrpc", root=str(_REPO / "Rescale_optimizer"))

    def _replan_fn(self, slots):
        t_new = [
            slots["softmax_out_fresh_sf"], slots["softmax_v_matmul_rescale_sf"],
            slots["ln_mean_rescale_sf"], slots["ln_square_rescale_sf"],
        ]
        # ctct_rot_softmax_mul_v delta = v_fresh + v_mask, and v_mask is bound to
        # softmax_out_mask (the mask2 binding).
        deltas = {
            "ctpt_mask2": slots["softmax_out_mask_sf"],
            "ctct_rot_softmax_mul_v": slots["v_fresh_sf"] + slots["softmax_out_mask_sf"],
            "ctpt_mask": slots["softmax_v_mask_sf"],
            "ctpt_wo_attnout": slots["wo_sf"],
            "ctpt_inv_d_1": slots["ln_mean_inv_d_sf"],
            "ctct_square": "x2",
            "ctpt_inv_d_2": slots["ln_var_inv_d_sf"],
        }
        r = self.S.replan("block4", t_new=t_new, delta_overrides=deltas, return_dict=True)["result"]
        return pb.ReplanProbe(
            valid=bool(r["valid"]), fusion_count=len(r["fusions"]),
            q_initial=tuple(int(x) for x in r["q_initial"]),
            q_final=tuple(int(x) for x in r["q_final"]),
            fusions=tuple(r["fusions"]),
        )

    @staticmethod
    def _noise_fn(slots, _probe):
        import noise_tables
        v = (
            noise_tables.variance(16384, slots["softmax_out_fresh_sf"], "fresh")
            + noise_tables.variance(16384, slots["v_fresh_sf"], "fresh")
            # softmax_out_mask is installed twice (mask2 + the bound v_mask).
            + 2 * noise_tables.variance(16384, slots["softmax_out_mask_sf"], "encoding")
        )
        for k in ("softmax_v_mask_sf", "wo_sf", "ln_mean_inv_d_sf", "ln_var_inv_d_sf"):
            v += noise_tables.variance(16384, slots[k], "encoding")
        return v

    def test_base_chain_matches_design(self):
        probe = self._replan_fn(BASE4)
        self.assertEqual(probe.q_initial, (27, 33, 31))
        self.assertEqual(probe.q_final, (60, 31))
        self.assertEqual(probe.fusion_count, 1)

    def test_boost_reaches_max_fill_59(self):
        res = pb.boost_option(
            topology=pb.BLOCK4_MRPC_TOPOLOGY, base_slots=BASE4,
            replan_fn=self._replan_fn, noise_fn=self._noise_fn, q_max=60,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.base_q_final, (60, 31))
        # the short prime fills to 59 (its max), the fused modulus stays 60.
        self.assertEqual(res.boosted_q_final, (60, 59))
        self.assertEqual(self._replan_fn(res.boosted_slots).fusion_count, 1)
        # every distribution of the 14-SF budget is a valid all-q_max-fused chain.
        self.assertEqual(res.candidates_valid, res.candidates_tried)
        self.assertGreater(res.candidates_tried, 1)

    def test_all_softmax_distribution_is_valid(self):
        # the user's worked example: put the whole budget on softmax_out_mask.
        slots = dict(BASE4)
        slots.update({
            "softmax_out_mask_sf": 17, "softmax_v_matmul_rescale_sf": 45,
            "ln_mean_rescale_sf": 45,
        })
        probe = self._replan_fn(slots)
        self.assertEqual(probe.q_final, (60, 59))
        self.assertEqual(probe.fusion_count, 1)


# block5 n=2 mrpc fc=1 option (the committed block5_n2.json option 1).
BASE5 = {
    "x_centered_fresh_sf": 26,
    "gamma_sf": 19,
    "wffn1_sf": 20,
    "gelu_coeff_sf": 20,
    "normalize_rescale_sf": 31,
    "wffn1_rescale_sf": 31,
    "gelu_coeff_mul_rescale_sf_0": 31,
}


class Block5N2CandidateGenTest(unittest.TestCase):
    """block5_n2 fc=1: the short prime (51) has an addable encode (gelu_coeff)
    AFTER the ctct_gelu_x2 ×2 (bit-weight 1), so the odd deficit 9 is filled
    EXACTLY → 60 (block4 had no such encode and stopped at 59)."""

    @staticmethod
    def _probe():
        # q_initial [21,39,51], fuse stage 1 into next -> q_final [60,51].
        return pb.ReplanProbe(
            valid=True, fusion_count=1,
            q_initial=(21, 39, 51), q_final=(60, 51),
            fusions=({"fused_position": 1, "fused_into": "next", "small_q": 21},),
        )

    def test_short_prime_fills_to_60(self):
        sp = pb.find_short_primes(self._probe(), 60)
        self.assertEqual(sp, [(2, 9)])
        # max_fill == deficit (9) because the c=0 gelu_coeff encode (weight 1)
        # makes every integer fill reachable.
        self.assertEqual(pb.short_prime_fill(pb.BLOCK5_N2_MRPC_TOPOLOGY, sp[0]), 9)

    def test_candidate_count_and_mixed_channels(self):
        cands = pb.generate_candidates(pb.BLOCK5_N2_MRPC_TOPOLOGY, BASE5, [(2, 9)])
        # Σ (2·a_gamma + 2·a_wffn1 + 1·a_coeff) == 9 → a_coeff odd; 15 solutions.
        self.assertEqual(len(cands), 15)
        # the all-after candidate (gelu_coeff +9, no rescale comp).
        all_after = [c for c in cands if c.edits.get("gelu_coeff_sf") == 29]
        self.assertEqual(len(all_after), 1)
        self.assertEqual(set(all_after[0].edits), {"gelu_coeff_sf"})
        # an encode before the wffn1 ×2 cascade-compensates wffn1_rescale.
        comp = [c for c in cands if c.edits.get("gamma_sf", 19) > 19]
        self.assertTrue(comp and all("wffn1_rescale_sf" in c.edits for c in comp))


@unittest.skipUnless(
    __import__("importlib").util.find_spec("rescale_optimizer") is not None,
    "rescale_optimizer required",
)
class Block5N2BoostReplanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rescale_optimizer import ReplanSession  # noqa: E402
        cls.S = ReplanSession.from_profile(profile="mrpc", root=str(_REPO / "Rescale_optimizer"))

    def _replan_fn(self, slots):
        t_new = [
            slots["x_centered_fresh_sf"], slots["normalize_rescale_sf"],
            slots["wffn1_rescale_sf"], slots["gelu_coeff_mul_rescale_sf_0"],
        ]
        deltas = {
            "ctct_xmean_over_std": "x2",
            "ctpt_gamal": slots["gamma_sf"],
            "ctpt_wffn1": slots["wffn1_sf"],
            "ctct_gelu_x2": "x2",
            "ctpt_gelu_coeff": slots["gelu_coeff_sf"],
        }
        r = self.S.replan("block5_n2", t_new=t_new, delta_overrides=deltas, return_dict=True)["result"]
        return pb.ReplanProbe(
            valid=bool(r["valid"]), fusion_count=len(r["fusions"]),
            q_initial=tuple(int(x) for x in r["q_initial"]),
            q_final=tuple(int(x) for x in r["q_final"]),
            fusions=tuple(r["fusions"]),
        )

    @staticmethod
    def _noise_fn(slots, _probe):
        import noise_tables
        return sum(
            noise_tables.variance(16384, slots[k], "encoding")
            for k in ("gamma_sf", "wffn1_sf", "gelu_coeff_sf")
        )

    def test_base_chain_matches_design(self):
        probe = self._replan_fn(BASE5)
        self.assertEqual(probe.q_initial, (21, 39, 51))
        self.assertEqual(probe.q_final, (60, 51))

    def test_boost_reaches_60(self):
        res = pb.boost_option(
            topology=pb.BLOCK5_N2_MRPC_TOPOLOGY, base_slots=BASE5,
            replan_fn=self._replan_fn, noise_fn=self._noise_fn, q_max=60,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.base_q_final, (60, 51))
        self.assertEqual(res.boosted_q_final, (60, 60))  # reaches q_max exactly
        self.assertEqual(res.candidates_valid, res.candidates_tried)
        self.assertEqual(self._replan_fn(res.boosted_slots).fusion_count, 1)

    def test_user_worked_examples_valid(self):
        # case1 (all on gelu_coeff +9) and case3 (6+3) both reach 60.
        for slots in (
            {**BASE5, "gelu_coeff_sf": 29},
            {**BASE5, "gamma_sf": 21, "wffn1_sf": 21, "gelu_coeff_sf": 23, "wffn1_rescale_sf": 34},
        ):
            probe = self._replan_fn(slots)
            self.assertEqual(probe.q_final, (60, 60))
            self.assertEqual(probe.fusion_count, 1)


# block5 n=4 mrpc fc=1 option (the committed block5_n4.json option 1).
BASE5N4 = {
    "x_centered_fresh_sf": 26,
    "gamma_sf": 19,
    "wffn1_sf": 20,
    "gelu_coeff_sf": 20,
    "normalize_rescale_sf": 31,
    "wffn1_rescale_sf": 31,
    "gelu_power_rescale_sf_0": 31,
    "gelu_coeff_mul_rescale_sf_0": 31,
}


class Block5N4CandidateGenTest(unittest.TestCase):
    """block5_n4 fc=1: TWO ×2 between the before-encodes and the short prime
    (gamma/wffn1 are c=2 → 4 bits/SF), and TWO short primes after fusion (31 and
    51) — only the LAST (51) is boosted, the middle 31 is left."""

    @staticmethod
    def _probe():
        # q_initial [21,39,31,51], fuse 1&2 -> q_final [60,31,51].
        return pb.ReplanProbe(
            valid=True, fusion_count=1,
            q_initial=(21, 39, 31, 51), q_final=(60, 31, 51),
            fusions=({"fused_position": 1, "fused_into": "next", "small_q": 21},),
        )

    def test_two_short_primes_detected(self):
        # both the middle 31 (pre-idx 2) and the trailing 51 (pre-idx 3) are short.
        self.assertEqual(pb.find_short_primes(self._probe(), 60), [(2, 29), (3, 9)])

    def test_last_prime_fills_to_60(self):
        # the trailing 51 reaches 60 (the c=0 gelu_coeff gives the odd bit).
        self.assertEqual(pb.short_prime_fill(pb.BLOCK5_N4_MRPC_TOPOLOGY, (3, 9)), 9)

    def test_six_candidates_double_x2(self):
        cands = pb.generate_candidates(pb.BLOCK5_N4_MRPC_TOPOLOGY, BASE5N4, [(3, 9)])
        # 4·(a_gamma+a_wffn1) + a_coeff == 9 → a_g+a_w ∈ {0,1,2}: 1+2+3 = 6.
        self.assertEqual(len(cands), 6)
        # a before-encode (c=2) cascade-compensates BOTH wffn1_rescale and
        # gelu_power_rescale through the two doublings.
        comp = [c for c in cands if c.edits.get("gamma_sf", 19) > 19]
        self.assertTrue(comp)
        for c in comp:
            self.assertIn("wffn1_rescale_sf", c.edits)
            self.assertIn("gelu_power_rescale_sf_0", c.edits)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("rescale_optimizer") is not None,
    "rescale_optimizer required",
)
class Block5N4BoostReplanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rescale_optimizer import ReplanSession  # noqa: E402
        cls.S = ReplanSession.from_profile(profile="mrpc", root=str(_REPO / "Rescale_optimizer"))

    def _replan_fn(self, slots):
        t_new = [
            slots["x_centered_fresh_sf"], slots["normalize_rescale_sf"],
            slots["wffn1_rescale_sf"], slots["gelu_power_rescale_sf_0"],
            slots["gelu_coeff_mul_rescale_sf_0"],
        ]
        deltas = {
            "ctct_xmean_over_std": "x2",
            "ctpt_gamal": slots["gamma_sf"],
            "ctpt_wffn1": slots["wffn1_sf"],
            "ctct_gelu_x2": "x2",
            "ctct_gelu_x4": "x2",
            "ctpt_gelu_coeff": slots["gelu_coeff_sf"],
        }
        r = self.S.replan("block5_n4", t_new=t_new, delta_overrides=deltas, return_dict=True)["result"]
        return pb.ReplanProbe(
            valid=bool(r["valid"]), fusion_count=len(r["fusions"]),
            q_initial=tuple(int(x) for x in r["q_initial"]),
            q_final=tuple(int(x) for x in r["q_final"]),
            fusions=tuple(r["fusions"]),
        )

    @staticmethod
    def _noise_fn(slots, _probe):
        import noise_tables
        return sum(
            noise_tables.variance(16384, slots[k], "encoding")
            for k in ("gamma_sf", "wffn1_sf", "gelu_coeff_sf")
        )

    def test_base_chain_matches_design(self):
        probe = self._replan_fn(BASE5N4)
        self.assertEqual(probe.q_initial, (21, 39, 31, 51))
        self.assertEqual(probe.q_final, (60, 31, 51))

    def test_boost_last_prime_only(self):
        res = pb.boost_option(
            topology=pb.BLOCK5_N4_MRPC_TOPOLOGY, base_slots=BASE5N4,
            replan_fn=self._replan_fn, noise_fn=self._noise_fn, q_max=60,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res.base_q_final, (60, 31, 51))
        # the LAST prime (51) -> 60; the middle 31 is intentionally left.
        self.assertEqual(res.boosted_q_final, (60, 31, 60))
        self.assertEqual(res.candidates_valid, res.candidates_tried)
        self.assertEqual(self._replan_fn(res.boosted_slots).fusion_count, 1)

    def test_user_worked_examples_valid(self):
        # case1 (all-after +9), case3 8+1, case3 4+5 — all reach (60,31,60).
        for slots in (
            {**BASE5N4, "gelu_coeff_sf": 29},
            {**BASE5N4, "gamma_sf": 21, "gelu_coeff_sf": 21,
             "wffn1_rescale_sf": 33, "gelu_power_rescale_sf_0": 35},
            {**BASE5N4, "gamma_sf": 20, "gelu_coeff_sf": 25,
             "wffn1_rescale_sf": 32, "gelu_power_rescale_sf_0": 33},
        ):
            probe = self._replan_fn(slots)
            self.assertEqual(probe.q_final, (60, 31, 60))
            self.assertEqual(probe.fusion_count, 1)


def _cfg_sf_projection(cfg):
    """Stable (field, scaling_factor) projection of a block NoiseConfig — for
    comparing two cfgs built different ways without relying on dataclass __eq__."""
    out = {}
    for name in vars(cfg):
        val = getattr(cfg, name)
        for i, pt in enumerate(val if isinstance(val, tuple) else (val,)):
            sf = getattr(pt, "scaling_factor", None)
            if sf is not None:
                out[f"{name}[{i}]" if isinstance(val, tuple) else name] = int(sf)
    return out


@unittest.skipUnless(
    __import__("importlib").util.find_spec("torch") is not None
    and __import__("importlib").util.find_spec("rescale_optimizer") is not None,
    "torch + rescale_optimizer required",
)
class SFDirectEquivalenceTest(unittest.TestCase):
    """The SF-direct cfg builder MUST match the index path on an IN-GRID config
    (so boosting only differs where SFs go above baseline). Server-side gate for
    build_block_cfg_from_field_values — a drift here would silently corrupt every
    boosted option. torch + rescale_optimizer required (skipped locally)."""

    def _assert_sf_direct_matches(self, graph_key, block_idx, gelu_degree=4):
        from action_space import (
            _decode_block_field_values,
            action_vector_to_cfgs,
            build_block_cfg_from_field_values,
        )
        import fusion_enum
        import numpy as np
        ctx = fusion_enum.prepare_block_type_context(
            graph_key=graph_key, block_idx=block_idx, gelu_degree=gelu_degree, attn_degree=2,
            profile="mrpc", rescale_optimizer_root=str(_REPO / "Rescale_optimizer"),
            num_layers=12, ref_layer=1,
        )
        deg_g = int(ctx.gelu_per_layer[ctx.ref_layer])
        deg_a = int(ctx.attn_per_layer[ctx.ref_layer])
        base_indices = list(ctx.baseline_block_indices)
        full = ctx.baseline_full.copy()
        full[ctx.block_offset: ctx.block_offset + ctx.block_num_slots] = base_indices
        cfg_idx = action_vector_to_cfgs(
            full, ctx.max_sfs, num_layers=ctx.num_layers,
            gelu_degree=ctx.gelu_per_layer, attn_degree=ctx.attn_per_layer,
            only=(ctx.ref_layer, block_idx),
        ).cfgs_dict()[f"block{block_idx}"][ctx.ref_layer]
        fv = _decode_block_field_values(
            layer_idx=ctx.ref_layer, block_idx=block_idx, action_slice=np.asarray(base_indices, dtype=int),
            max_sfs=ctx.max_sfs, attn_degree=deg_a, gelu_degree=deg_g,
        )
        fv = {k: int(v) for k, v in fv.items() if v is not None}
        cfg_dir = build_block_cfg_from_field_values(
            block_idx, ctx.ref_layer, fv, N=ctx.N_block, gelu_degree=deg_g, attn_degree=deg_a,
        )
        self.assertEqual(_cfg_sf_projection(cfg_idx), _cfg_sf_projection(cfg_dir))

    def test_sf_direct_matches_index_path_block2(self):
        self._assert_sf_direct_matches("block2_mrpc", 2)

    def test_sf_direct_matches_index_path_block4(self):
        # also exercises the softmax_out_mask → v_mask binding through both paths.
        self._assert_sf_direct_matches("block4", 4)

    def test_sf_direct_matches_index_path_block5_n2(self):
        self._assert_sf_direct_matches("block5_n2", 5, gelu_degree=2)

    def test_sf_direct_matches_index_path_block5_n4(self):
        self._assert_sf_direct_matches("block5_n4", 5, gelu_degree=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
