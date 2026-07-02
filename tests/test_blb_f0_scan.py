import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class BLBF0ScanTests(unittest.TestCase):
    def _records(self):
        # NOTE: ``wffn2_rescale_sf`` was removed from the RL action space on
        # 2026-05-14 because mrpc baseline never places a rescale at that
        # node. The test now exercises ``mean_rescale_sf`` (still active).
        return [
            {
                "global_index": 0,
                "layer": 0,
                "block": "block1",
                "field": "mean_rescale_sf",
                "kind": "R",
                "distribution": "scaling",
                "operation": "ctct_mean_rescale",
                "effective": True,
                "N": 8192,
                "action_values": [16, 18],
            },
            {
                "global_index": 1,
                "layer": 0,
                "block": "block1",
                "field": "output_truncation_k",
                "kind": "K",
                "distribution": "truncation",
                "operation": "output_truncation_k",
                "effective": True,
                "N": 8192,
                "action_values": [8, 9, 11, 13, 10, 12],
            },
            {
                "global_index": 2,
                "layer": 0,
                "block": "block1",
                "field": "gelu_out_sf",
                "kind": "F",
                "distribution": "fresh",
                "operation": "ctpt_gelu_out",
                "effective": False,
                "N": 8192,
                "action_values": [22, 24],
            },
        ]

    def test_smallest_cost_rows_does_not_full_sort(self):
        from scripts.blb_f0_scan_feasible_domain import _smallest_cost_rows

        rows = [
            {"total_bits_sum": 30, "fusion_count": 1, "action_hash": "c"},
            {"total_bits_sum": 10, "fusion_count": 2, "action_hash": "b"},
            {"total_bits_sum": 10, "fusion_count": 1, "action_hash": "a"},
            {"total_bits_sum": 20, "fusion_count": 0, "action_hash": "d"},
        ]

        with mock.patch("builtins.sorted", side_effect=AssertionError("full sort")):
            best = _smallest_cost_rows(rows, 3)

        self.assertEqual([row["action_hash"] for row in best], ["a", "b", "d"])

    def test_per_slot_summary_scans_rows_once(self):
        from scripts.blb_f0_scan_feasible_domain import _build_per_slot_summary_rows

        class SinglePassRows:
            def __init__(self, rows):
                self.rows = list(rows)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise AssertionError("per-slot summary should not rescan all rows per slot")
                return iter(self.rows)

        rows = SinglePassRows([
            {
                "slot_global_index": 0,
                "optimizer_valid": True,
                "delta_total_bits": -2,
                "delta_fusion_count": 0,
            },
            {
                "slot_global_index": 0,
                "optimizer_valid": False,
                "delta_total_bits": 0,
                "delta_fusion_count": 0,
            },
            {
                "slot_global_index": 1,
                "optimizer_valid": True,
                "delta_total_bits": 3,
                "delta_fusion_count": -1,
            },
        ])

        summary = _build_per_slot_summary_rows(
            baseline_action=[1, 2],
            records=self._records(),
            rows=rows,
        )

        self.assertEqual(rows.iterations, 1)
        self.assertEqual(summary[0]["candidate_count"], 2)
        self.assertEqual(summary[0]["valid_count"], 1)
        self.assertEqual(summary[0]["improving_valid_count"], 1)
        self.assertEqual(summary[0]["best_delta_total_bits"], -2)
        self.assertEqual(summary[1]["candidate_count"], 1)
        self.assertEqual(summary[1]["best_delta_fusion_count"], -1)

    def test_mask_reuses_sorted_allowed_indices_for_values(self):
        from scripts.blb_f0_scan_feasible_domain import _build_mask

        real_sorted = sorted
        sort_calls = 0

        def counting_sorted(*args, **kwargs):
            nonlocal sort_calls
            sort_calls += 1
            return real_sorted(*args, **kwargs)

        with mock.patch("builtins.sorted", side_effect=counting_sorted):
            mask = _build_mask(
                baseline_action=[1, 3, 1],
                action_dims=[2, 6, 2],
                records=self._records(),
                per_slot_rows=[
                    {
                        "slot_global_index": 0,
                        "optimizer_valid": True,
                        "candidate_index": 0,
                    }
                ],
                source="test",
            )

        self.assertEqual(sort_calls, len(mask["slots"]))
        self.assertEqual(mask["slots"][0]["allowed_indices"], [0, 1])
        self.assertEqual(mask["slots"][0]["allowed_values"], [16, 18])

    def test_candidate_indices_below_baseline_are_streamed(self):
        from scripts.blb_f0_scan_feasible_domain import _candidate_indices_below_baseline

        candidates = _candidate_indices_below_baseline(5)

        self.assertIsInstance(candidates, range)
        self.assertEqual(list(candidates), [0, 1, 2, 3, 4])

    def test_scan_stops_when_baseline_is_invalid(self):
        from scripts.blb_f0_scan_feasible_domain import run_scan_core

        def evaluate(_action, _source):
            return {
                "optimizer_valid": False,
                "total_bits_sum": 0,
                "fusion_count": 0,
                "avg_k": 13.0,
                "invalid_chain": {"reason": "baseline invalid"},
            }

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "baseline"):
                run_scan_core(
                    baseline_action=[1, 3],
                    action_dims=[2, 6],
                    records=self._records(),
                    evaluate_action=evaluate,
                    output_dir=td,
                    metadata={"profile": "mrpc"},
                    beam_size=4,
                    beam_depths=[1],
                    random_samples=4,
                    random_seed=1,
                )
            self.assertTrue((Path(td) / "baseline_f0.json").exists())

    def test_scan_writes_per_slot_mask_and_rank_fields(self):
        from scripts.blb_f0_scan_feasible_domain import run_scan_core

        def evaluate(action, _source):
            # Baseline [1, 3] is Trust-0-like. Lowering slot 0 is valid and
            # cheaper; lowering K to 2 is valid but cost-irrelevant.
            return {
                "optimizer_valid": True,
                "total_bits_sum": 100 - (1 if action[0] == 0 else 0),
                "fusion_count": 0,
                "avg_k": 13.0 if action[1] == 3 else 11.0,
                "invalid_chain": None,
                "q_bits": [60, 50],
                "candidate_key": f"candidate-{action[0]}-{action[1]}-{action[2]}",
            }

        with tempfile.TemporaryDirectory() as td:
            result = run_scan_core(
                baseline_action=[1, 3, 1],
                action_dims=[2, 6, 2],
                records=self._records(),
                evaluate_action=evaluate,
                output_dir=td,
                metadata={"profile": "mrpc", "num_layers": 1},
                beam_size=4,
                beam_depths=[1],
                random_samples=8,
                random_seed=2,
                multi_random_samples=8,
                multi_mutation_counts=[2],
            )
            out = Path(td)
            self.assertTrue((out / "manifest.json").exists())
            self.assertTrue((out / "per_slot_scan.jsonl").exists())
            self.assertTrue((out / "suggested_action_mask.json").exists())
            self.assertEqual(result["baseline"]["rescale_cost"]["rank_key"], [100, 0])
            self.assertEqual(result["baseline"]["f0_sort_key"], [0, 100, 0])

            rows = [
                json.loads(line)
                for line in (out / "per_slot_scan.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(rows)
            required = {
                "slot_global_index",
                "layer",
                "block",
                "field",
                "kind",
                "baseline_index",
                "candidate_index",
                "optimizer_valid",
                "total_bits_sum",
                "fusion_count",
                "delta_total_bits",
                "delta_fusion_count",
                "candidate_key",
            }
            self.assertTrue(required.issubset(rows[0].keys()))
            k_rows = [row for row in rows if row["kind"] == "K"]
            self.assertTrue(k_rows)
            self.assertTrue(all(row["optimizer_cost_irrelevant"] for row in k_rows))

            mask = json.loads((out / "suggested_action_mask.json").read_text(encoding="utf-8"))
            self.assertEqual(mask["action_width"], 3)
            self.assertEqual(len(mask["slots"]), 3)
            for slot in mask["slots"]:
                self.assertIn(slot["baseline_index"], slot["allowed_indices"])
                self.assertTrue(slot["allowed_indices"])
            inactive_slot = next(slot for slot in mask["slots"] if slot["global_index"] == 2)
            self.assertEqual(inactive_slot["reason"], "ineffective_compat_slot_baseline_only")
            self.assertEqual(inactive_slot["allowed_indices"], [1])

            random_report = json.loads((out / "masked_random_validity.json").read_text(encoding="utf-8"))
            self.assertIn("by_mutation_count", random_report)
            self.assertGreaterEqual(random_report["by_mutation_count"]["1"]["valid_rate"], 0.95)
            self.assertIn("total_bits_min", random_report["by_mutation_count"]["1"])
            self.assertIn("fusion_count_mean", random_report["by_mutation_count"]["1"])
            self.assertIn("best_action_hashes", random_report["by_mutation_count"]["1"])
            self.assertTrue((out / "multi_random_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
