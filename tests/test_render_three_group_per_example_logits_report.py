import copy
import importlib
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

from tests.test_render_three_group_fusion_stability_report import make_run_payloads


GROUPS = (
    "all_fusion0",
    "block2_block5_all_layers_fusionmax",
    "block2_block4_block5_all_layers_fusion1",
)


def _load_report_module():
    return importlib.import_module(
        "scripts.render_three_group_per_example_logits_report"
    )


def _group(run, name):
    return next(group for group in run["group_results"] if group["name"] == name)


def _cross_entropy(logits, label):
    peak = max(logits)
    return peak + math.log(sum(math.exp(value - peak) for value in logits)) - logits[label]


def _set_raw_metrics(run_payloads, *, loss, accuracy, weighted_f1):
    for run in run_payloads:
        for group in run["group_results"]:
            values = {
                "loss": [loss] * 5,
                "metric1": [accuracy] * 5,
                "metric2": [weighted_f1] * 5,
            }
            group["trial_metrics"] = values
            group["metrics"] = {
                "loss_mean": loss,
                "loss_std": 0.0,
                "loss_max": loss,
                "metric1_mean": accuracy,
                "metric1_std": 0.0,
                "metric1_min": accuracy,
                "metric2_mean": weighted_f1,
                "metric2_std": 0.0,
                "metric2_min": weighted_f1,
            }


def make_fixture(expected_examples=2):
    runs = make_run_payloads()
    rows = []
    dataset_indices = list(range(10, 10 + expected_examples))
    for run in runs:
        for group_name in GROUPS:
            trial_seeds = _group(run, group_name)["terminal_probe"][
                "per_worker_trial_seeds"
            ][0]
            for trial_index, trial_seed in enumerate(trial_seeds):
                for probe_position, dataset_idx in enumerate(dataset_indices):
                    gold = probe_position % 2
                    logits = [3.0, -2.0] if gold == 0 else [-2.0, 3.0]
                    rows.append(
                        {
                            "schema_version": "fusion-count-per-example-v1",
                            "run_seed": run["seed"],
                            "group": group_name,
                            "trial_index": trial_index,
                            "trial_seed": trial_seed,
                            "probe_position": probe_position,
                            "dataset_idx": dataset_idx,
                            "input_ids": [101, dataset_idx, 102, 0],
                            "attention_mask": [1, 1, 1, 0],
                            "token_type_ids": [0, 0, 1, 0],
                            "gold_label": gold,
                            "predicted_label": gold,
                            "correct": True,
                            "logits": logits,
                        }
                    )
    loss = statistics.fmean(
        _cross_entropy(row["logits"], row["gold_label"])
        for row in rows[:expected_examples]
    )
    _set_raw_metrics(runs, loss=loss, accuracy=1.0, weighted_f1=1.0)
    return runs, rows, copy.deepcopy(runs)


def failure_codes(summary, gate_name=None):
    return {
        failure["code"]
        for gate in summary["gates"]
        if gate_name is None or gate["name"] == gate_name
        for failure in gate["failures"]
    }


class PredictionSummaryTests(unittest.TestCase):
    def setUp(self):
        self.report = _load_report_module()

    def _summary(self, runs, rows, prior_runs, **kwargs):
        return self.report.build_prediction_summary(
            run_payloads=runs,
            prediction_rows=rows,
            prior_run_payloads=prior_runs,
            source_commit="abc123",
            expected_examples=2,
            **kwargs,
        )

    def test_build_prediction_summary_recomputes_metrics_and_input_aggregates(self):
        runs, rows, prior_runs = make_fixture()

        summary = self._summary(runs, rows, prior_runs)

        self.assertTrue(summary["all_gates_pass"])
        self.assertEqual(summary["row_count"], 5 * 3 * 5 * 2)
        self.assertEqual(
            summary["groups"]["all_fusion0"]["inputs"]["10"]["correct_count"],
            25,
        )
        aggregate = summary["groups"]["all_fusion0"]["inputs"]["10"]
        self.assertEqual(aggregate["trial_count"], 25)
        self.assertEqual(aggregate["correct_rate"], 1.0)
        self.assertEqual(aggregate["mean_logits"], [3.0, -2.0])
        self.assertEqual(aggregate["std_logits"], [0.0, 0.0])
        self.assertEqual(aggregate["input_ids"], [101, 10, 102, 0])
        self.assertEqual(summary["trial_results"][0]["incorrect_dataset_indices"], [])
        self.assertEqual(summary["trial_results"][0]["correct_dataset_indices"], [10, 11])
        self.assertEqual(summary["changed_examples"]["count"], 0)
        self.assertEqual(
            [gate["name"] for gate in summary["gates"]],
            [
                "base_three_group",
                "prediction_completeness",
                "input_identity",
                "logits_prediction",
                "recomputed_metrics",
                "shared_trial_seeds",
                "prior_equivalence",
            ],
        )

    def test_base_three_group_gate_reuses_existing_report_gates(self):
        runs, rows, prior_runs = make_fixture()
        _group(runs[0], "all_fusion0")["step_records"][0][
            "model_uses_replan_config"
        ] = False

        summary = self._summary(runs, rows, prior_runs)

        self.assertFalse(summary["all_gates_pass"])
        base_gate = next(
            gate for gate in summary["gates"] if gate["name"] == "base_three_group"
        )
        self.assertFalse(base_gate["passed"])
        self.assertIn("base_gate_failed", failure_codes(summary, "base_three_group"))
        self.assertIn("steps_install", json.dumps(base_gate["failures"]))

    def test_prediction_gate_rejects_wrong_argmax_duplicate_idx_and_nonfinite_logits(self):
        mutations = {
            "prediction_argmax": lambda rows: rows[0].update(
                {"predicted_label": 1, "correct": False}
            ),
            "duplicate_dataset_idx": lambda rows: rows[1].update(
                {
                    "dataset_idx": rows[0]["dataset_idx"],
                    "input_ids": rows[0]["input_ids"],
                    "gold_label": rows[0]["gold_label"],
                    "predicted_label": rows[0]["predicted_label"],
                    "logits": rows[0]["logits"],
                }
            ),
            "non_finite_logits": lambda rows: rows[0].update(
                {"logits": [float("inf"), 0.0]}
            ),
        }

        for expected_code, mutate in mutations.items():
            with self.subTest(expected_code=expected_code):
                runs, rows, prior_runs = make_fixture()
                mutate(rows)
                summary = self._summary(runs, rows, prior_runs)
                self.assertFalse(summary["all_gates_pass"])
                self.assertIn(expected_code, failure_codes(summary))

    def test_input_identity_rejects_unstable_input_and_gold_label(self):
        for field, replacement in (
            ("input_ids", [101, 999, 102, 0]),
            ("attention_mask", [1, 1, 0, 0]),
            ("token_type_ids", [0, 1, 1, 0]),
            ("gold_label", 0),
        ):
            with self.subTest(field=field):
                runs, rows, prior_runs = make_fixture()
                rows[-1][field] = replacement
                summary = self._summary(runs, rows, prior_runs)
                self.assertIn("unstable_input_identity", failure_codes(summary))

    def test_logits_gate_rejects_wrong_shape_correctness_and_malformed_rows_without_crashing(self):
        mutations = (
            ("logit_count", lambda row: row.update({"logits": [1.0]})),
            ("prediction_correctness", lambda row: row.update({"correct": False})),
            ("row_mapping", lambda rows: rows.__setitem__(0, "not-a-row")),
        )
        for expected_code, mutate in mutations:
            with self.subTest(expected_code=expected_code):
                runs, rows, prior_runs = make_fixture()
                if expected_code == "row_mapping":
                    mutate(rows)
                else:
                    mutate(rows[0])
                summary = self._summary(runs, rows, prior_runs)
                self.assertFalse(summary["all_gates_pass"])
                self.assertIn(expected_code, failure_codes(summary))
                json.dumps(summary, allow_nan=False)

    def test_recomputed_metric_gate_rejects_accuracy_f1_and_loss_mismatches(self):
        for metric, delta in (("metric1", 0.1), ("metric2", 0.1), ("loss", 1e-4)):
            with self.subTest(metric=metric):
                runs, rows, prior_runs = make_fixture()
                _group(runs[0], GROUPS[0])["trial_metrics"][metric][0] += delta
                prior_runs = copy.deepcopy(runs)
                summary = self._summary(runs, rows, prior_runs)
                self.assertIn("recomputed_metric_mismatch", failure_codes(summary))

    def test_prior_metric_gate_rejects_changed_trial_metric(self):
        runs, rows, prior_runs = make_fixture()
        _group(runs[0], GROUPS[0])["trial_metrics"]["metric1"][0] += 0.001

        summary = self._summary(runs, rows, prior_runs)

        self.assertFalse(summary["all_gates_pass"])
        self.assertIn("prior_trial_metric_mismatch", failure_codes(summary))

    def test_shared_trial_seed_gate_rejects_group_seed_difference(self):
        runs, rows, prior_runs = make_fixture()
        group = _group(runs[1], GROUPS[1])
        group["terminal_probe"]["per_worker_trial_seeds"][0][-1] += 1

        summary = self._summary(runs, rows, prior_runs)

        self.assertIn("shared_trial_seed_mismatch", failure_codes(summary))

    def test_prediction_completeness_requires_exact_file_count_and_hierarchy(self):
        runs, rows, prior_runs = make_fixture()
        missing_trial_rows = [
            row
            for row in rows
            if not (
                row["run_seed"] == runs[0]["seed"]
                and row["group"] == GROUPS[0]
                and row["trial_index"] == 0
            )
        ]

        wrong_files = self._summary(
            runs, rows, prior_runs, prediction_file_count=4
        )
        missing_trial = self._summary(runs, missing_trial_rows, prior_runs)

        self.assertIn("prediction_file_count", failure_codes(wrong_files))
        self.assertIn("trial_row_count", failure_codes(missing_trial))

    def test_cross_entropy_is_numerically_stable_for_extreme_logits(self):
        self.assertEqual(self.report.stable_cross_entropy([1000.0, -1000.0], 0), 0.0)
        self.assertAlmostEqual(
            self.report.stable_cross_entropy([-1000.0, 1000.0], 0),
            2000.0,
        )


class HtmlReportTests(unittest.TestCase):
    def setUp(self):
        self.report = _load_report_module()

    def test_html_embeds_all_rows_filter_controls_and_static_summaries(self):
        runs, rows, prior_runs = make_fixture()
        summary = self.report.build_prediction_summary(
            run_payloads=runs,
            prediction_rows=rows,
            prior_run_payloads=prior_runs,
            source_commit="abc123",
            expected_examples=2,
        )

        html_text = self.report.render_html(summary, rows)

        for control_id in (
            "seed-filter",
            "group-filter",
            "trial-filter",
            "correct-filter",
            "dataset-idx-filter",
            "prediction-data",
        ):
            self.assertIn(f'id="{control_id}"', html_text)
        self.assertIn("abc123", html_text)
        self.assertIn("Correct dataset IDs", html_text)
        self.assertIn("Incorrect dataset IDs", html_text)
        self.assertIn("Per-input aggregate", html_text)
        self.assertIn("input_ids", html_text)
        self.assertIn("attention_mask", html_text)
        self.assertIn("token_type_ids", html_text)
        self.assertIn("logits", html_text)
        embedded = html_text.split('id="prediction-data">', 1)[1].split(
            "</script>", 1
        )[0]
        self.assertEqual(json.loads(embedded), rows)

    def test_html_escapes_script_termination_and_paginates_at_100_rows(self):
        runs, rows, prior_runs = make_fixture()
        rows[0]["input_ids"] = ["</script><script>alert(1)</script>"]
        summary = self.report.build_prediction_summary(
            run_payloads=runs,
            prediction_rows=rows,
            prior_run_payloads=prior_runs,
            source_commit="abc123",
            expected_examples=2,
        )

        html_text = self.report.render_html(summary, rows)

        embedded = html_text.split('id="prediction-data">', 1)[1].split(
            "</script>", 1
        )[0]
        self.assertNotIn("</script>", embedded.lower())
        self.assertIn(r"<\/script>", embedded.lower())
        self.assertIn("const PAGE_SIZE = 100", html_text)
        self.assertIn("slice(pageStart, pageStart + PAGE_SIZE)", html_text)
        self.assertNotIn("predictionRows.map", html_text)


class ProductionShapeAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _load_report_module()
        cls.runs, cls.rows, cls.prior_runs = make_fixture(expected_examples=408)

    def test_generated_production_shape_contains_exactly_30600_rows(self):
        summary = self.report.build_prediction_summary(
            run_payloads=self.runs,
            prediction_rows=self.rows,
            prior_run_payloads=self.prior_runs,
            source_commit="production-shape",
        )

        self.assertTrue(summary["all_gates_pass"])
        self.assertEqual(summary["row_count"], 30_600)
        self.assertEqual(summary["expected_row_count"], 30_600)

    def test_cli_writes_strict_outputs_and_returns_nonzero_when_a_gate_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = []
            rows_by_seed = {
                seed: [row for row in self.rows if row["run_seed"] == seed]
                for seed in (run["seed"] for run in self.runs)
            }
            for index, (run, prior) in enumerate(zip(self.runs, self.prior_runs)):
                run_path = root / f"run-{index}.json"
                prior_path = root / f"prior-{index}.json"
                prediction_path = root / f"prediction-{index}.jsonl"
                run_path.write_text(json.dumps(run), encoding="utf-8")
                prior_path.write_text(json.dumps(prior), encoding="utf-8")
                selected_rows = copy.deepcopy(rows_by_seed[run["seed"]])
                if index == 0:
                    selected_rows[0]["predicted_label"] = 1
                prediction_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in selected_rows),
                    encoding="utf-8",
                )
                args.extend(["--run-json", str(run_path)])
                args.extend(["--prediction-jsonl", str(prediction_path)])
                args.extend(["--prior-run-json", str(prior_path)])
            output_json = root / "summary.json"
            output_html = root / "report.html"
            args.extend(
                [
                    "--source-commit",
                    "deadbeef",
                    "--output-json",
                    str(output_json),
                    "--output-html",
                    str(output_html),
                ]
            )

            return_code = self.report.main(args)

            self.assertEqual(return_code, 1)
            summary_text = output_json.read_text(encoding="utf-8")
            summary = json.loads(summary_text)
            self.assertFalse(summary["all_gates_pass"])
            self.assertIn("prediction_argmax", failure_codes(summary))
            self.assertNotIn("NaN", summary_text)
            self.assertNotIn("Infinity", summary_text)
            self.assertIn('id="prediction-data"', output_html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
