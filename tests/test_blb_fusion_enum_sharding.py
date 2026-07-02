import itertools
import pathlib
import sys
import unittest
from unittest import mock


_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "blb_stage2_rl"), str(_REPO / "Rescale_optimizer"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fusion_enum  # noqa: E402


class FusionEnumShardingTests(unittest.TestCase):
    def test_product_shard_matches_modulo_product_order(self):
        choices = [[10, 11, 12], [20], [30, 31, 32, 33]]
        full = list(itertools.product(*choices))
        for num_shards in (1, 2, 5, 17):
            for shard_idx in range(num_shards):
                expected = [combo for i, combo in enumerate(full) if i % num_shards == shard_idx]
                got = list(fusion_enum._iter_product_shard(choices, shard_idx, num_shards))
                self.assertEqual(got, expected, (shard_idx, num_shards))

    def test_enumerate_shard_does_not_skip_spin_full_product(self):
        class Ctx:
            baseline_block_indices = (9, 9, 9)
            enum_choices = [[1, 2], [3, 4, 5], [6, 7]]
            enum_positions = [0, 1, 2]

        full = list(itertools.product(*Ctx.enum_choices))
        expected = [combo for i, combo in enumerate(full) if i % 3 == 1]
        seen = []

        def fake_eval(_ctx, block):
            seen.append(tuple(int(x) for x in block))
            return {
                "valid": True,
                "fusion_count": len(seen) % 2,
                "total_bits": 100 - len(seen),
                "points": [],
            }

        class NoiseOrder:
            def total_variance(self, _points):
                return 0.0

        with (
            mock.patch.object(fusion_enum, "_iter_product_shard", wraps=fusion_enum._iter_product_shard) as shard_iter,
            mock.patch.object(fusion_enum, "_eval_block", side_effect=fake_eval),
        ):
            _rows, num_valid = fusion_enum.enumerate_shard(
                Ctx(), shard_idx=1, num_shards=3, noise_order=NoiseOrder()
            )

        shard_iter.assert_called_once_with(Ctx.enum_choices, 1, 3)
        self.assertEqual(num_valid, len(expected))
        self.assertEqual(seen, expected)

    def test_degeneracy_probe_reuses_corner_evaluation(self):
        class Ctx:
            baseline_block_indices = (9, 9)
            enum_choices = [[1, 2], [3, 4]]
            enum_positions = [0, 1]

        seen = []

        def fake_eval(_ctx, block):
            seen.append(tuple(int(x) for x in block))
            return {"valid": True, "fusion_count": 0}

        with mock.patch.object(fusion_enum, "_eval_block", side_effect=fake_eval):
            result = fusion_enum.degeneracy_probe(Ctx(), num_random=0)

        self.assertTrue(result["degenerate"])
        self.assertEqual(seen, [(9, 9), (1, 3)])
        self.assertEqual(result["samples_checked"], 1)


if __name__ == "__main__":
    unittest.main()
