"""group_min_noise_options' KEPT options must be golden-CONSISTENT: the option's
claimed (fusion_count, total_bits) has to be reproduced by a real golden replan
of its DECODED action vector.

This broke the rte/sst2 block2 build. The fast direct-replan template
(``fusion_enum_fast``) is golden-DERIVED from per-slot probes; a slot interaction
it does not capture can make it feed ``replan`` a different ``(t_new,
delta_overrides)`` than golden for SOME combos. The block2 fc=1 option whose
three SF-irrelevant rescales (gamma/kt_mask1/qkt_matmul) decode to the lex-min
SF 15 is exactly such a combo: golden classifies it as **fusion 0** (those low
rescales stop the chain fusing — verified by real replan), but the fast path
stored it as **fusion 1**, so it became the kept fc=1 representative. The
precision boost then could not raise that non-fusing base to the output target
(the build emitted ``output_sf=43`` while the gate wants 46).

``verify_template``'s RANDOM probes missed the specific deterministic kept combo
(block2 passed 512 probes). The kept options are few and deterministic, so
golden-re-checking exactly them catches the escape; the builder falls back to a
full golden enumeration (the source of truth) on any mismatch — the real fc=1
options ALL boost to the target.

Torch-free: ``verify_kept_options_golden`` takes an injected ``eval_fn``.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "blb_stage2_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blb_stage2_rl import fusion_enum as fe


class KeptOptionGoldenVerifyTest(unittest.TestCase):
    def _opts(self):


        return [
            {"option_id": 0, "fusion_count": 0, "total_bits": 100, "action_indices": [14, 14, 14]},
            {"option_id": 1, "fusion_count": 1, "total_bits": 90, "action_indices": [7, 14, 1]},
        ]

    def _ev(self, table):
        def ev(_ctx, ai):
            return table[tuple(int(x) for x in ai)]
        return ev

    def test_all_consistent_no_problem(self):
        ev = self._ev({
            (14, 14, 14): {"valid": True, "fusion_count": 0, "total_bits": 100},
            (7, 14, 1): {"valid": True, "fusion_count": 1, "total_bits": 90},
        })
        self.assertEqual(fe.verify_kept_options_golden(None, self._opts(), eval_fn=ev), [])

    def test_fusion_count_mismatch_reported(self):


        ev = self._ev({
            (14, 14, 14): {"valid": True, "fusion_count": 0, "total_bits": 100},
            (7, 14, 1): {"valid": True, "fusion_count": 0, "total_bits": 90},
        })
        probs = fe.verify_kept_options_golden(None, self._opts(), eval_fn=ev)
        self.assertEqual([p[0] for p in probs], [1])
        self.assertIn("fusion_count", probs[0][1])

    def test_golden_invalid_reported(self):
        ev = self._ev({
            (14, 14, 14): {"valid": True, "fusion_count": 0, "total_bits": 100},
            (7, 14, 1): {"valid": False},
        })
        probs = fe.verify_kept_options_golden(None, self._opts(), eval_fn=ev)
        self.assertEqual([p[0] for p in probs], [1])
        self.assertIn("INVALID", probs[0][1].upper())

    def test_total_bits_mismatch_reported(self):
        ev = self._ev({
            (14, 14, 14): {"valid": True, "fusion_count": 0, "total_bits": 100},
            (7, 14, 1): {"valid": True, "fusion_count": 1, "total_bits": 77},
        })
        probs = fe.verify_kept_options_golden(None, self._opts(), eval_fn=ev)
        self.assertEqual([p[0] for p in probs], [1])
        self.assertIn("total_bits", probs[0][1])

    def test_baseline_consistent_option_zero_never_flagged_spuriously(self):

        opts = [
            {"option_id": 0, "fusion_count": 0, "total_bits": 100, "action_indices": [14, 14, 14]},
            {"option_id": 1, "fusion_count": 1, "total_bits": 90, "action_indices": [7, 14, 14]},
        ]
        ev = self._ev({
            (14, 14, 14): {"valid": True, "fusion_count": 0, "total_bits": 100},
            (7, 14, 14): {"valid": True, "fusion_count": 1, "total_bits": 90},
        })
        self.assertEqual(fe.verify_kept_options_golden(None, opts, eval_fn=ev), [])


if __name__ == "__main__":
    unittest.main()
