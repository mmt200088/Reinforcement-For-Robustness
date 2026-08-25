"""The precision boost must align its base chain with the RUNTIME chain by
resetting a NOISE-IRRELEVANT topology rescale to its baseline SF.

Server ground truth (block2 rte/sst2 fc=1 option): the three rescale slots
(gamma/kt_mask1/qkt_matmul) are NOT noise-install points at runtime — their cfg
fields are None, so the bridge's t_new uses the baseline sf_post (t_new =
[21,28,28,28]) and the runtime is INSENSITIVE to their action SF. But the boost
decodes them via _decode_block_field_values to their raw action SF (15, below
baseline 28), so it replans a lower-precision chain than the runtime installs and
stalls the output at 43 (target 46). Canonicalising them to baseline (28) makes
the boost chain match the runtime (t_new=[..,28,28,28]) and reach 46 — server-
confirmed fc + installed-noise signature preserved.

`canonicalize_noise_irrelevant_rescales` resets each topology rescale to its
baseline SF ONLY when that preserves (valid, fusion_count, installed signature),
so a genuinely noise-relevant rescale (its SF moves an installed point) is left
untouched. Pure: probe_fn / sig_fn injected (torch-free).
"""

from __future__ import annotations

import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "blb_stage2_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blb_stage2_rl import precision_boost as pb


def _topo():

    return pb.ChainTopology(
        graph_key="mock",
        nodes=(
            pb.ChainNode("fresh", "f"),
            pb.ChainNode("rescale", "R1"),
            pb.ChainNode("rescale", "R2"),
            pb.ChainNode("rescale", "R3"),
        ),
    )


class _Probe:
    def __init__(self, valid, fusion_count):
        self.valid = valid
        self.fusion_count = fusion_count


class CanonicalizeRescaleTest(unittest.TestCase):
    BASELINE = {"R1": 28, "R2": 30, "R3": 28}

    def _run(self, base_fv, *, sig_depends_on=("R2",), fc=1, valid=True):

        def probe_fn(slots):
            return _Probe(valid, fc)

        def sig_fn_for(slots):
            return tuple((k, int(slots[k])) for k in sig_depends_on if k in slots)


        last = {}

        def probe(slots):
            last["slots"] = dict(slots)
            return probe_fn(slots)

        def sig(_probe):
            return sig_fn_for(last["slots"])

        return pb.canonicalize_noise_irrelevant_rescales(
            base_fv, _topo(), self.BASELINE, probe_fn=probe, sig_fn=sig,
        )

    def test_noise_irrelevant_rescale_reset_to_baseline(self):

        out = self._run({"R1": 15, "R2": 30, "R3": 28}, sig_depends_on=("R2",))
        self.assertEqual(out["R1"], 28)

    def test_noise_relevant_rescale_left_untouched(self):

        out = self._run({"R1": 28, "R2": 20, "R3": 28}, sig_depends_on=("R2",))
        self.assertEqual(out["R2"], 20)

    def test_mixed_only_irrelevant_canonicalised(self):
        out = self._run({"R1": 15, "R2": 20, "R3": 28}, sig_depends_on=("R2",))
        self.assertEqual(out["R1"], 28)
        self.assertEqual(out["R2"], 20)

    def test_already_at_or_above_baseline_skipped(self):
        out = self._run({"R1": 28, "R2": 30, "R3": 28}, sig_depends_on=())
        self.assertEqual(out, {"R1": 28, "R2": 30, "R3": 28})

    def test_fusion_count_change_blocks_canonicalisation(self):

        def probe(slots):
            return _Probe(True, 1 if int(slots["R1"]) == 15 else 0)

        def sig(_p):
            return ("const",)

        out = pb.canonicalize_noise_irrelevant_rescales(
            {"R1": 15, "R2": 30, "R3": 28}, _topo(), self.BASELINE,
            probe_fn=probe, sig_fn=sig,
        )
        self.assertEqual(out["R1"], 15)

    def test_invalid_base_returns_unchanged(self):
        out = self._run({"R1": 15, "R2": 30, "R3": 28}, valid=False)
        self.assertEqual(out["R1"], 15)


if __name__ == "__main__":
    unittest.main()
