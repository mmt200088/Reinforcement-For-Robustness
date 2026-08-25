"""group_min_noise_options must treat an option 0 that installs the SAME noise as
the baseline as the baseline by RESULT-EQUIVALENCE, not raw action indices.

This broke the rte build: block2_rte option 0 had idx 1 at three RESCALE slots
(gamma_rescale / kt_mask1_rescale / qkt_matmul_rescale) while the all-max baseline
used idx 14, so the strict "option 0 == baseline" raw-index guard wrongly failed.

The actual cause is an SF-IRRELEVANT rescale: those rescales (calibrated anchor SF
28, enumerated levels SF 15..28 — NONE below the noise-table min 10) inject no
Gaussian noise in the fusion=0 baseline (their SF only affects modulus-chain
validity). Every SF level therefore installs the identical noise, so the min-noise
dedup kept the lex-min representative index (idx 1; idx 0 = None excluded) instead
of the baseline's idx 14. mrpc never hit it. A COLLAPSED low-baseline slot (levels
all snap to the table-min SF) is the other way this can happen; the fix handles both.

Fix: when option 0 is result-equivalent to the baseline (its installed_signature
matches the baseline's), rewrite option 0's indices to the canonical baseline so the
runtime baseline detection (make_all_max_action_vector) stays consistent; a genuine
installed-plan difference still raises (the guard keeps its protective value).

Torch-free: group_min_noise_options + EvaluatedConfig are pure.
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


def _ec(action_indices, fusion_count, total_bits, total_variance, sig):
    return fe.EvaluatedConfig(
        action_indices=tuple(action_indices),
        fusion_count=int(fusion_count),
        total_bits=int(total_bits),
        total_variance=float(total_variance),
        installed_signature=sig,
        slots={},
    )


class GroupMinNoiseBaselineEquivalenceTest(unittest.TestCase):
    def test_result_equivalent_option0_rewritten_to_baseline(self):


        ec0 = _ec((14, 14, 1), 0, 100, 1.0, "S0")
        ec1 = _ec((14, 10, 1), 1, 90, 2.0, "S1")
        opts = fe.group_min_noise_options(
            [ec0, ec1], (14, 14, 14), baseline_installed_signature="S0",
        )
        self.assertEqual(opts[0]["fusion_count"], 0)
        self.assertEqual(opts[0]["action_indices"], [14, 14, 14])
        self.assertEqual(opts[1]["fusion_count"], 1)

    def test_exact_match_unchanged(self):

        ec0 = _ec((14, 14, 14), 0, 100, 1.0, "S0")
        opts = fe.group_min_noise_options(
            [ec0], (14, 14, 14), baseline_installed_signature="S0",
        )
        self.assertEqual(opts[0]["action_indices"], [14, 14, 14])

    def test_genuinely_different_option0_still_raises(self):

        ec0 = _ec((14, 14, 1), 0, 100, 1.0, "DIFFERENT")
        with self.assertRaises(ValueError):
            fe.group_min_noise_options(
                [ec0], (14, 14, 14), baseline_installed_signature="S0",
            )

    def test_no_signature_keeps_strict_index_guard(self):

        ec0 = _ec((14, 14, 1), 0, 100, 1.0, "S0")
        with self.assertRaises(ValueError):
            fe.group_min_noise_options([ec0], (14, 14, 14))


if __name__ == "__main__":
    unittest.main()
