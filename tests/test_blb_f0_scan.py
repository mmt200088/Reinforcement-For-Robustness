import json
import tempfile
import unittest
from pathlib import Path


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
