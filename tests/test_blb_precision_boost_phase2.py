"""Tests for the fusion-option precision boost PHASE 2 ("二阶段加大精度").

Phase 1 (``precision_boost.boost_option``) raises the intermediate short modulus
primes; it leaves the final OUTPUT scale ("last node SF") fixed. Phase 2 raises
that output scale to its ceiling ``target = q_tail_bits - amplitude_budgets[-1] -
h_sf`` (read from the block's Rescale_optimizer config), distributing the gained
SF between the final encode and the last rescale's ``sf_post``. Most final encodes
may drop to a hardcoded floor of 15; protected fields such as block4's
``ln_var_inv_d_sf`` must stay at or above their base value. The last prime stays
as high as possible and every prime before it constant, at minimum installed
noise.

Lanes mirror the phase-1 test file:
  * pure structural / config lanes (torch-free AND rescale_optimizer-free);
  * real-replan lanes through ``ReplanSession`` (the cost source of truth),
    skipped where rescale_optimizer can't be imported.

See docs/adr (phase-2 precision boost).
"""

import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "blb_stage2_rl")):
    if p not in sys.path:
        sys.path.insert(0, p)

import noise_tables as nt  # noqa: E402  (torch-free: parses function_handler text)
from blb_stage2_rl import precision_boost as pb  # noqa: E402

_RO_ROOT = str(_REPO / "Rescale_optimizer")


class AboveTableNoiseSemanticsTest(unittest.TestCase):
    """SF above the noise-table max installs NO noise. var(46)~2.8e-25 and each +1
    SF is ~x0.25, so var(>46) is far below fp precision (var(49)~4e-27) — treating
    it as 0 is exact to machine precision. Lets the precision boost push an
    installed point past 46 (e.g. block4's ln_mean_rescale to 49) instead of being
    blocked. SF BELOW the table min stays an error (that regime is snapped upstream)."""

    def test_variance_above_table_max_is_zero(self):
        N = 16384
        tbl_max = max(nt.NOISE_VARIANCE_TABLE_BY_N[N])
        for sf in (tbl_max + 1, tbl_max + 3, 60):
            for dist in ("encoding", "fresh", "rescale"):
                self.assertEqual(nt.variance(N, sf, dist), 0.0, f"sf={sf} dist={dist}")

    def test_variance_within_table_unchanged(self):
        N = 16384
        self.assertGreater(nt.variance(N, max(nt.NOISE_VARIANCE_TABLE_BY_N[N]), "encoding"), 0.0)
        self.assertGreater(nt.variance(N, 20, "fresh"), 0.0)

    def test_variance_below_table_min_still_raises(self):
        N = 16384
        below = min(nt.NOISE_VARIANCE_TABLE_BY_N[N]) - 1
        with self.assertRaises(KeyError):
            nt.variance(N, below, "encoding")


@unittest.skipUnless(
    __import__("importlib").util.find_spec("torch") is not None,
    "torch required (function_handler imports torch)",
)
class FunctionHandlerAboveTableTest(unittest.TestCase):
    """The MODEL-FORWARD install path mirrors noise_tables: SF above the table max
    returns 0.0 (no noise) instead of raising; SF below the min still raises. Server
    only (function_handler imports torch)."""

    def test_get_input_noise_variance_by_N_above_max_is_zero(self):
        import function_handler as fh
        N = 16384
        tbl_max = max(fh.NOISE_VARIANCE_TABLE_BY_N[N])
        for sf in (tbl_max + 1, tbl_max + 3, 60):
            for dist in ("encoding", "fresh", "rescale"):
                self.assertEqual(fh.get_input_noise_variance_by_N(sf, dist, N), 0.0)

        self.assertGreater(fh.get_input_noise_variance_by_N(tbl_max, "encoding", N), 0.0)
        with self.assertRaises(ValueError):
            fh.get_input_noise_variance_by_N(min(fh.NOISE_VARIANCE_TABLE_BY_N[N]) - 1, "encoding", N)


class TargetOutputSFTest(unittest.TestCase):
    """target = q_tail_bits - amplitude_budgets[-1] - h_sf, read straight from
    the block's RO config (general; a changed JSON yields a changed target)."""

    def test_target_matches_formula_for_every_block(self):

        cases = {
            "block2_mrpc": 46,
            "block4": 53,
            "block5_n1": 48,
            "block5_n2": 43,
            "block5_n4": 43,
        }
        for gk, want in cases.items():
            got = pb.target_output_sf(gk, profile="mrpc", root=_RO_ROOT)
            self.assertEqual(got, want, f"{gk}: target {got} != {want}")


BASE2_P1 = {
    "inv_std_fresh_sf": 21, "gamma_sf": 15, "wk_sf": 16, "kt_mask1_sf": 16,
    "kt_mask2_sf": 15, "qkt_merge_mask_sf": 15, "gamma_rescale_sf": 28,
    "kt_mask1_rescale_sf": 29, "qkt_matmul_rescale_sf": 28,
}
BASE4_P1 = {
    "softmax_out_fresh_sf": 21, "v_fresh_sf": 17, "softmax_out_mask_sf": 14,
    "softmax_v_mask_sf": 13, "wo_sf": 13, "ln_mean_inv_d_sf": 13, "ln_var_inv_d_sf": 20,
    "softmax_v_matmul_rescale_sf": 39, "ln_mean_rescale_sf": 45, "ln_square_rescale_sf": 31,
}
BASE5_P1 = {
    "x_centered_fresh_sf": 26, "gamma_sf": 21, "wffn1_sf": 21, "gelu_coeff_sf": 23,
    "normalize_rescale_sf": 31, "wffn1_rescale_sf": 34, "gelu_coeff_mul_rescale_sf_0": 31,
}


class Phase2CandidateGenTest(unittest.TestCase):
    """Structural enumeration (torch-free, replan-free): the composition space is
    ``final_encode ∈ [floor, base+delta]``, ``sf_post = target − final_encode``,
    with the pre-scale rise supplied upstream. Asserts the user's named methods
    and the block4 ``ln_var_inv_d_sf`` base-value guard."""

    def _by_field(self, cands, **want):
        """Candidates whose edits match every (field == value) in want."""
        out = []
        for c in cands:
            if all(c.edits.get(f) == v for f, v in want.items()):
                out.append(c)
        return out

    def test_block2_three_methods(self):

        cands = pb.generate_phase2_candidates(
            pb.BLOCK2_MRPC_TOPOLOGY, BASE2_P1, target_output_sf=46, base_last_prime=60,
        )

        for c in cands:
            sp = c.edits["qkt_matmul_rescale_sf"]
            fe = c.edits["qkt_merge_mask_sf"]
            self.assertEqual(sp + fe, 46, f"output != 46 in {c.edits}")

        self.assertTrue(self._by_field(cands, qkt_merge_mask_sf=18, qkt_matmul_rescale_sf=28))

        self.assertTrue(self._by_field(cands, qkt_merge_mask_sf=16, qkt_matmul_rescale_sf=30))

        self.assertTrue(self._by_field(cands, qkt_merge_mask_sf=15, qkt_matmul_rescale_sf=31))

        self.assertTrue(all(c.edits["qkt_merge_mask_sf"] >= 15 for c in cands))

    def test_block4_ln_var_guard_blocks_decrease_to_floor_when_uncapped(self):


        cands = pb.generate_phase2_candidates(
            pb.BLOCK4_MRPC_TOPOLOGY, BASE4_P1, target_output_sf=53, base_last_prime=59,
            max_installed_sf=60,
        )
        for c in cands:
            self.assertEqual(c.edits["ln_square_rescale_sf"] + c.edits["ln_var_inv_d_sf"], 53)
        self.assertFalse(self._by_field(cands, ln_var_inv_d_sf=15, ln_square_rescale_sf=38))
        self.assertTrue(all(c.edits["ln_var_inv_d_sf"] >= BASE4_P1["ln_var_inv_d_sf"] for c in cands))

    def test_block4_ln_var_guard_applies_under_default_cap(self):


        cands = pb.generate_phase2_candidates(
            pb.BLOCK4_MRPC_TOPOLOGY, BASE4_P1, target_output_sf=53, base_last_prime=59,
        )
        self.assertTrue(cands)
        for c in cands:
            self.assertEqual(c.edits["ln_square_rescale_sf"] + c.edits["ln_var_inv_d_sf"], 53)
            self.assertLessEqual(max(c.edits.values()), 60, f"over q_max in {c.edits}")
            self.assertGreaterEqual(c.edits["ln_var_inv_d_sf"], BASE4_P1["ln_var_inv_d_sf"])
        self.assertFalse(self._by_field(cands, ln_var_inv_d_sf=15, ln_square_rescale_sf=38))
        self.assertTrue(self._by_field(cands, ln_var_inv_d_sf=20, ln_square_rescale_sf=33))

    def test_generality_composition_adapts_to_chain_change(self):


        def sigset(cands):
            return sorted(tuple(sorted(c.edits.items())) for c in cands)

        c60 = pb.generate_phase2_candidates(pb.BLOCK2_MRPC_TOPOLOGY, BASE2_P1, 46, base_last_prime=60)
        c59 = pb.generate_phase2_candidates(pb.BLOCK2_MRPC_TOPOLOGY, BASE2_P1, 46, base_last_prime=59)
        self.assertTrue(c60 and c59)
        self.assertNotEqual(sigset(c60), sigset(c59), "composition must depend on the last prime")

        c46 = pb.generate_phase2_candidates(pb.BLOCK2_MRPC_TOPOLOGY, BASE2_P1, 46, base_last_prime=60)
        c45 = pb.generate_phase2_candidates(pb.BLOCK2_MRPC_TOPOLOGY, BASE2_P1, 45, base_last_prime=60)
        self.assertNotEqual(sigset(c46), sigset(c45), "composition must depend on the target")
        for c in c45:
            self.assertEqual(c.edits["qkt_matmul_rescale_sf"] + c.edits["qkt_merge_mask_sf"], 45)

    def test_block5_no_final_encode_sf_post_only(self):

        cands = pb.generate_phase2_candidates(
            pb.BLOCK5_N2_MRPC_TOPOLOGY, BASE5_P1, target_output_sf=43, base_last_prime=60,
        )
        self.assertTrue(cands)
        for c in cands:

            self.assertEqual(c.edits["gelu_coeff_mul_rescale_sf_0"], 43)
            self.assertNotIn("qkt_merge_mask_sf", c.edits)
            self.assertNotIn("ln_var_inv_d_sf", c.edits)


BASE5N4_P1 = {
    "x_centered_fresh_sf": 26, "gamma_sf": 20, "wffn1_sf": 21, "gelu_coeff_sf": 21,
    "normalize_rescale_sf": 31, "wffn1_rescale_sf": 33, "gelu_power_rescale_sf_0": 35,
    "gelu_coeff_mul_rescale_sf_0": 31,
}


BASE5N1 = {
    "x_centered_fresh_sf": 22, "gamma_sf": 15, "wffn1_sf": 16, "gelu_coeff_sf": 16,
    "normalize_rescale_sf": 28, "gelu_coeff_mul_rescale_sf_0": 31,
}


class EffectiveTargetTest(unittest.TestCase):
    """Install points may now run up to q_max=60 (>46 = no noise), so the only clamp
    is the modulus limit, not 46. Every mrpc config ceiling (<=53, n1's 48) is <= the
    no-final-encode ceiling (60), so NONE are clamped — block5_n1 now reaches its full
    48 (was clamped to 46 under the old <=46 install cap)."""

    def test_effective_target_no_longer_clamped_below_qmax(self):
        cases = {
            "block2_mrpc": (pb.BLOCK2_MRPC_TOPOLOGY, 46, 46),
            "block4": (pb.BLOCK4_MRPC_TOPOLOGY, 53, 53),
            "block5_n1": (pb.BLOCK5_N1_MRPC_TOPOLOGY, 48, 48),
            "block5_n2": (pb.BLOCK5_N2_MRPC_TOPOLOGY, 43, 43),
            "block5_n4": (pb.BLOCK5_N4_MRPC_TOPOLOGY, 43, 43),
        }
        for gk, (topo, cfg, eff) in cases.items():
            self.assertEqual(pb.target_output_sf(gk, "mrpc", _RO_ROOT), cfg, f"{gk} config")
            self.assertEqual(pb.effective_output_target(topo, cfg), eff, f"{gk} effective")

    def test_effective_target_still_clamps_above_qmax(self):


        self.assertEqual(pb.effective_output_target(pb.BLOCK5_N1_MRPC_TOPOLOGY, 999), 60)
        self.assertEqual(pb.effective_output_target(pb.BLOCK4_MRPC_TOPOLOGY, 999), 120)


def _noise_fn_for(topology):
    """Deterministic installed-variance proxy over every named topology node
    (fresh/encode/rescale), mirroring the builder's SummedInstalledVariance so
    the driver's min-noise choice is well-defined for the test."""
    import noise_tables
    kd = {"fresh": "fresh", "encode": "encoding", "rescale": "rescale"}
    fields = [(n.cfg_field, kd[n.kind]) for n in topology.nodes if n.cfg_field and n.kind in kd]

    def nf(slots, _probe):
        return sum(noise_tables.variance(16384, int(slots[f]), d) for f, d in fields if f in slots)

    return nf


@unittest.skipUnless(
    __import__("importlib").util.find_spec("rescale_optimizer") is not None,
    "rescale_optimizer required",
)
class Phase2BoostReplanTest(unittest.TestCase):
    """Drive ``boost_option_phase2`` through the REAL ReplanSession for all four
    boostable block types: it must reach output==target with the fusion_count and
    every prior prime preserved, and return the minimum-noise composition (verified
    independently against the candidate set)."""

    @classmethod
    def setUpClass(cls):
        from rescale_optimizer import ReplanSession
        cls.S = ReplanSession.from_profile(profile="mrpc", root=_RO_ROOT)


    def _probe2(self, s):
        t = [s["inv_std_fresh_sf"], s["gamma_rescale_sf"], s["kt_mask1_rescale_sf"], s["qkt_matmul_rescale_sf"]]
        d = {"ctct_x_mean_over_std": "x2", "ctpt_gama1": s["gamma_sf"], "ctpt_wq_wk": s["wk_sf"],
             "ctpt_rotKT_mask1": s["kt_mask1_sf"], "ctpt_rotKT_mask2": s["kt_mask2_sf"],
             "ctct_preprocess_qkt": "x2", "ctpt_mask": s["qkt_merge_mask_sf"]}
        return self._mk(self.S.replan("block2_mrpc", t_new=t, delta_overrides=d, return_dict=True)["result"])

    def _probe4(self, s):
        t = [s["softmax_out_fresh_sf"], s["softmax_v_matmul_rescale_sf"], s["ln_mean_rescale_sf"], s["ln_square_rescale_sf"]]
        d = {"ctpt_mask2": s["softmax_out_mask_sf"], "ctct_rot_softmax_mul_v": s["v_fresh_sf"] + s["softmax_out_mask_sf"],
             "ctpt_mask": s["softmax_v_mask_sf"], "ctpt_wo_attnout": s["wo_sf"], "ctpt_inv_d_1": s["ln_mean_inv_d_sf"],
             "ctct_square": "x2", "ctpt_inv_d_2": s["ln_var_inv_d_sf"]}
        return self._mk(self.S.replan("block4", t_new=t, delta_overrides=d, return_dict=True)["result"])

    def _probe5n2(self, s):
        t = [s["x_centered_fresh_sf"], s["normalize_rescale_sf"], s["wffn1_rescale_sf"], s["gelu_coeff_mul_rescale_sf_0"]]
        d = {"ctct_xmean_over_std": "x2", "ctpt_gamal": s["gamma_sf"], "ctpt_wffn1": s["wffn1_sf"],
             "ctct_gelu_x2": "x2", "ctpt_gelu_coeff": s["gelu_coeff_sf"]}
        return self._mk(self.S.replan("block5_n2", t_new=t, delta_overrides=d, return_dict=True)["result"])

    def _probe5n4(self, s):
        t = [s["x_centered_fresh_sf"], s["normalize_rescale_sf"], s["wffn1_rescale_sf"],
             s["gelu_power_rescale_sf_0"], s["gelu_coeff_mul_rescale_sf_0"]]
        d = {"ctct_xmean_over_std": "x2", "ctpt_gamal": s["gamma_sf"], "ctpt_wffn1": s["wffn1_sf"],
             "ctct_gelu_x2": "x2", "ctct_gelu_x4": "x2", "ctpt_gelu_coeff": s["gelu_coeff_sf"]}
        return self._mk(self.S.replan("block5_n4", t_new=t, delta_overrides=d, return_dict=True)["result"])

    def _probe5n1(self, s):
        t = [s["x_centered_fresh_sf"], s["normalize_rescale_sf"], s["gelu_coeff_mul_rescale_sf_0"]]
        d = {"ctct_xmean_over_std": "x2", "ctpt_gamal": s["gamma_sf"], "ctpt_wffn1": s["wffn1_sf"],
             "ctpt_gelu_coeff": s["gelu_coeff_sf"]}
        return self._mk(self.S.replan("block5_n1", t_new=t, delta_overrides=d, return_dict=True)["result"])

    @staticmethod
    def _mk(r):
        return pb.ReplanProbe(
            valid=bool(r["valid"]), fusion_count=len(r["fusions"]),
            q_initial=tuple(int(x) for x in r["q_initial"]),
            q_final=tuple(int(x) for x in r["q_final"]),
            fusions=tuple(r["fusions"]),
            t_final=tuple(int(x) for x in r["t_final"]),
        )

    def _cases(self):

        return [
            ("block2_mrpc", pb.BLOCK2_MRPC_TOPOLOGY, BASE2_P1, 46, "qkt_merge_mask_sf", self._probe2),
            ("block4", pb.BLOCK4_MRPC_TOPOLOGY, BASE4_P1, 53, "ln_var_inv_d_sf", self._probe4),
            ("block5_n1", pb.BLOCK5_N1_MRPC_TOPOLOGY, BASE5N1, 48, None, self._probe5n1),
            ("block5_n2", pb.BLOCK5_N2_MRPC_TOPOLOGY, BASE5_P1, 43, None, self._probe5n2),
            ("block5_n4", pb.BLOCK5_N4_MRPC_TOPOLOGY, BASE5N4_P1, 43, None, self._probe5n4),
        ]

    def test_phase2_reaches_target_min_noise_all_blocks(self):
        for gk, topo, base, config_target, final_field, probe in self._cases():
            with self.subTest(block=gk):
                noise_fn = _noise_fn_for(topo)

                target = pb.effective_output_target(topo, config_target)
                base_probe = probe(dict(base))
                self.assertTrue(base_probe.valid)
                res = pb.boost_option_phase2(
                    topology=topo, base_slots=base, target_output_sf=config_target,
                    replan_fn=probe, noise_fn=noise_fn, q_max=60,
                )
                self.assertIsNotNone(res, f"{gk}: phase-2 returned None")
                self.assertEqual(res.output_sf, target)


                p = probe(res.boosted_slots)
                self.assertTrue(p.valid)
                self.assertEqual(p.fusion_count, base_probe.fusion_count)
                self.assertEqual(p.q_final[:-1], base_probe.q_final[:-1], f"{gk}: prior primes changed")
                fe = int(res.boosted_slots[final_field]) if final_field else 0
                self.assertEqual(int(p.t_final[-1]) + fe, target, f"{gk}: output != target")

                self.assertGreaterEqual(int(p.q_final[-1]), 59)


                best = None
                for c in pb.generate_phase2_candidates(topo, base, target, int(base_probe.q_final[-1]), q_max=60):
                    slots = dict(base)
                    slots.update(c.edits)
                    cp = probe(slots)
                    cfe = int(slots[final_field]) if final_field else 0
                    if (cp.valid and cp.fusion_count == base_probe.fusion_count
                            and cp.q_final[:-1] == base_probe.q_final[:-1]
                            and int(cp.t_final[-1]) + cfe == target):
                        v = noise_fn(slots, cp)
                        best = v if best is None else min(best, v)
                self.assertIsNotNone(best)
                self.assertLessEqual(res.total_variance, best + 1e-18, f"{gk}: not the min-noise composition")


                self.assertLessEqual(max(int(v) for v in res.boosted_slots.values()), 60)

    def test_phase2_target_is_not_hardcoded(self):


        noise_fn = _noise_fn_for(pb.BLOCK2_MRPC_TOPOLOGY)
        for tgt in (44, 45):
            res = pb.boost_option_phase2(
                topology=pb.BLOCK2_MRPC_TOPOLOGY, base_slots=BASE2_P1, target_output_sf=tgt,
                replan_fn=self._probe2, noise_fn=noise_fn, q_max=60,
            )
            self.assertIsNotNone(res)
            self.assertEqual(res.output_sf, tgt)
            p = self._probe2(res.boosted_slots)
            self.assertEqual(int(p.t_final[-1]) + int(res.boosted_slots["qkt_merge_mask_sf"]), tgt)

    def test_block4_user_special_case_now_installable(self):


        slots = dict(BASE4_P1)
        slots.update({"ln_var_inv_d_sf": 15, "ln_square_rescale_sf": 38,
                      "ln_mean_inv_d_sf": 17, "ln_mean_rescale_sf": 49})
        p = self._probe4(slots)
        self.assertTrue(p.valid)
        self.assertEqual(int(p.q_final[-1]), 60)
        self.assertEqual(int(p.t_final[-1]) + 15, 53)
        self.assertGreater(int(p.t_final[1]), 46)
        self.assertEqual(nt.variance(16384, int(p.t_final[1]), "rescale"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
