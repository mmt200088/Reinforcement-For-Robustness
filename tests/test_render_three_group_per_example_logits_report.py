import copy
import importlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


def _weighted_f1(gold, predicted):
    weighted = 0.0
    for label in (0, 1):
        support = sum(value == label for value in gold)
        if not support:
            continue
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(gold, predicted)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(gold, predicted)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(gold, predicted)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        weighted += support * (
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return weighted / len(gold)


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


def _synchronize_metrics(run_payloads, rows):
    for run in run_payloads:
        for group_name in GROUPS:
            group = _group(run, group_name)
            values = {"loss": [], "metric1": [], "metric2": []}
            for trial_index in range(5):
                trial_rows = [
                    row
                    for row in rows
                    if row["run_seed"] == run["seed"]
                    and row["group"] == group_name
                    and row["trial_index"] == trial_index
                ]
                gold = [row["gold_label"] for row in trial_rows]
                predicted = [row["predicted_label"] for row in trial_rows]
                values["loss"].append(
                    statistics.fmean(
                        _cross_entropy(row["logits"], row["gold_label"])
                        for row in trial_rows
                    )
                )
                values["metric1"].append(
                    sum(a == b for a, b in zip(gold, predicted)) / len(gold)
                )
                values["metric2"].append(_weighted_f1(gold, predicted))
            group["trial_metrics"] = values
            group["metrics"] = {
                "loss_mean": statistics.fmean(values["loss"]),
                "loss_std": statistics.pstdev(values["loss"]),
                "loss_max": max(values["loss"]),
                "metric1_mean": statistics.fmean(values["metric1"]),
                "metric1_std": statistics.pstdev(values["metric1"]),
                "metric1_min": min(values["metric1"]),
                "metric2_mean": statistics.fmean(values["metric2"]),
                "metric2_std": statistics.pstdev(values["metric2"]),
                "metric2_min": min(values["metric2"]),
            }


def _make_incorrect(row):
    row["logits"] = [-2.0, 3.0] if row["gold_label"] == 0 else [3.0, -2.0]
    row["predicted_label"] = 1 - row["gold_label"]
    row["correct"] = False


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

    def test_required_row_fields_are_enforced_and_token_type_ids_is_optional(self):
        required_fields = (
            "schema_version",
            "run_seed",
            "group",
            "trial_index",
            "trial_seed",
            "probe_position",
            "dataset_idx",
            "input_ids",
            "attention_mask",
            "gold_label",
            "predicted_label",
            "correct",
            "logits",
        )
        for field in required_fields:
            with self.subTest(field=field):
                runs, rows, prior_runs = make_fixture()
                rows[0].pop(field)
                summary = self._summary(runs, rows, prior_runs)
                self.assertFalse(summary["all_gates_pass"])
                self.assertIn(
                    "missing_required_field",
                    failure_codes(summary, "prediction_completeness"),
                )
                first_trial = summary["trial_results"][0]
                self.assertIsNone(first_trial["recomputed_loss"])

        for token_type_value in (None, "missing"):
            with self.subTest(token_type_value=token_type_value):
                runs, rows, prior_runs = make_fixture()
                for row in rows:
                    if token_type_value == "missing":
                        row.pop("token_type_ids")
                    else:
                        row["token_type_ids"] = None
                summary = self._summary(runs, rows, prior_runs)
                self.assertTrue(summary["all_gates_pass"])

    def test_model_input_tensors_require_flat_nonempty_integer_lists(self):
        mutations = {
            "input_tensor_empty": lambda row: row.update({"input_ids": []}),
            "input_tensor_element": lambda row: row.update(
                {"input_ids": [101, "10", 102, 0]}
            ),
            "input_tensor_nested": lambda row: row.update(
                {"input_ids": [101, [10], 102, 0]}
            ),
            "input_tensor_bool": lambda row: row.update(
                {"token_type_ids": [0, False, 1, 0]}
            ),
            "input_length_mismatch": lambda row: row.update(
                {"attention_mask": [1, 1, 1]}
            ),
            "attention_mask_value": lambda row: row.update(
                {"attention_mask": [1, 2, 1, 0]}
            ),
        }

        for expected_code, mutate in mutations.items():
            with self.subTest(expected_code=expected_code):
                runs, rows, prior_runs = make_fixture()
                mutate(rows[0])
                summary = self._summary(runs, rows, prior_runs)
                self.assertIn(
                    expected_code,
                    failure_codes(summary, "input_identity"),
                )
                self.assertIsNone(summary["trial_results"][0]["recomputed_loss"])
                aggregate = summary["groups"][GROUPS[0]]["inputs"]["10"]
                self.assertEqual(aggregate["trial_count"], 24)

    def test_probe_positions_are_complete_unique_and_order_independent(self):
        runs, rows, prior_runs = make_fixture()
        rows[0], rows[1] = rows[1], rows[0]
        reordered = self._summary(runs, rows, prior_runs)
        self.assertTrue(reordered["all_gates_pass"])

        cases = {}
        runs, rows, prior_runs = make_fixture()
        rows[1]["probe_position"] = 0
        cases["duplicate_probe_position"] = (runs, rows, prior_runs)
        runs, rows, prior_runs = make_fixture()
        rows.pop(1)
        cases["probe_position_set"] = (runs, rows, prior_runs)
        runs, rows, prior_runs = make_fixture()
        rows[0]["probe_position"] = True
        cases["probe_position_type"] = (runs, rows, prior_runs)

        for expected_code, payloads in cases.items():
            with self.subTest(expected_code=expected_code):
                summary = self._summary(*payloads)
                self.assertIn(
                    expected_code,
                    failure_codes(summary, "prediction_completeness"),
                )
                self.assertIsNone(summary["trial_results"][0]["recomputed_loss"])

    def test_position_to_dataset_mapping_must_be_stable_across_trials(self):
        runs, rows, prior_runs = make_fixture()
        target = [
            row
            for row in rows
            if row["run_seed"] == runs[0]["seed"]
            and row["group"] == GROUPS[0]
            and row["trial_index"] == 1
        ]
        target[0]["probe_position"], target[1]["probe_position"] = (
            target[1]["probe_position"],
            target[0]["probe_position"],
        )

        summary = self._summary(runs, rows, prior_runs)

        self.assertIn(
            "unstable_position_mapping",
            failure_codes(summary, "input_identity"),
        )
        affected = next(
            trial
            for trial in summary["trial_results"]
            if trial["seed"] == runs[0]["seed"]
            and trial["group"] == GROUPS[0]
            and trial["trial_index"] == 1
        )
        self.assertIsNone(affected["recomputed_loss"])

    def test_trial_seed_requires_integer_nonbool_and_exact_terminal_value(self):
        for replacement, expected_code in (
            (True, "trial_seed_type"),
            ([123], "trial_seed_type"),
            (123, "trial_seed_mismatch"),
        ):
            with self.subTest(replacement=replacement):
                runs, rows, prior_runs = make_fixture()
                rows[0]["trial_seed"] = replacement
                summary = self._summary(runs, rows, prior_runs)
                self.assertIn(
                    expected_code,
                    failure_codes(summary, "prediction_completeness"),
                )
                self.assertIsNone(summary["trial_results"][0]["recomputed_loss"])

    def test_malformed_json_types_produce_structured_failures_without_type_error(self):
        mutations = {
            "current_run_seed_type": lambda runs, rows, priors: runs[0].update(
                {"seed": []}
            ),
            "prior_run_seed_type": lambda runs, rows, priors: priors[0].update(
                {"seed": {"bad": "seed"}}
            ),
            "row_run_seed_type": lambda runs, rows, priors: rows[0].update(
                {"run_seed": []}
            ),
            "row_group_type": lambda runs, rows, priors: rows[0].update(
                {"group": {"bad": "group"}}
            ),
        }

        for expected_code, mutate in mutations.items():
            with self.subTest(expected_code=expected_code):
                runs, rows, prior_runs = make_fixture()
                mutate(runs, rows, prior_runs)
                summary = self._summary(runs, rows, prior_runs)
                self.assertFalse(summary["all_gates_pass"])
                self.assertIn(expected_code, failure_codes(summary))
                json.dumps(summary, allow_nan=False)

    def test_bool_and_nonmapping_run_seeds_fail_structurally(self):
        runs, rows, prior_runs = make_fixture()
        runs[0]["seed"] = True
        prior_runs[1] = ["not", "a", "mapping"]

        summary = self._summary(runs, rows, prior_runs)

        self.assertFalse(summary["all_gates_pass"])
        self.assertIn("current_run_seed_type", failure_codes(summary))
        self.assertIn("prior_run_mapping", failure_codes(summary))

    def test_changed_examples_detects_equal_rates_with_different_aligned_patterns(self):
        runs, rows, _ = make_fixture()
        seed = runs[0]["seed"]
        for group_name, trial_index in zip(GROUPS, (0, 1, 2)):
            row = next(
                row
                for row in rows
                if row["run_seed"] == seed
                and row["group"] == group_name
                and row["trial_index"] == trial_index
                and row["dataset_idx"] == 10
            )
            _make_incorrect(row)
        _synchronize_metrics(runs, rows)
        prior_runs = copy.deepcopy(runs)

        summary = self._summary(runs, rows, prior_runs)

        self.assertTrue(summary["all_gates_pass"])
        changed = next(
            item
            for item in summary["changed_examples"]["examples"]
            if item["dataset_idx"] == 10
        )
        self.assertTrue(changed["reasons"]["cross_group_pattern_difference"])
        self.assertTrue(changed["reasons"]["within_group_noise_variation"])
        self.assertEqual(set(changed["group_rates"].values()), {24 / 25})
        self.assertEqual(
            {len(pattern) for pattern in changed["group_patterns"].values()},
            {25},
        )
        self.assertNotEqual(
            changed["group_patterns"][GROUPS[0]],
            changed["group_patterns"][GROUPS[1]],
        )

    def test_changed_examples_detects_same_pattern_with_noise_variation(self):
        runs, rows, _ = make_fixture()
        seed = runs[0]["seed"]
        for group_name in GROUPS:
            row = next(
                row
                for row in rows
                if row["run_seed"] == seed
                and row["group"] == group_name
                and row["trial_index"] == 0
                and row["dataset_idx"] == 10
            )
            _make_incorrect(row)
        _synchronize_metrics(runs, rows)
        prior_runs = copy.deepcopy(runs)

        summary = self._summary(runs, rows, prior_runs)

        self.assertTrue(summary["all_gates_pass"])
        changed = next(
            item
            for item in summary["changed_examples"]["examples"]
            if item["dataset_idx"] == 10
        )
        self.assertFalse(changed["reasons"]["cross_group_pattern_difference"])
        self.assertTrue(changed["reasons"]["within_group_noise_variation"])
        self.assertEqual(
            len({tuple(pattern) for pattern in changed["group_patterns"].values()}),
            1,
        )

    def test_cross_entropy_is_numerically_stable_for_extreme_logits(self):
        self.assertEqual(self.report.stable_cross_entropy([1000.0, -1000.0], 0), 0.0)
        self.assertAlmostEqual(
            self.report.stable_cross_entropy([-1000.0, 1000.0], 0),
            2000.0,
        )

    def test_metric_helpers_match_hard_coded_asymmetric_goldens(self):
        self.assertAlmostEqual(
            self.report._weighted_f1(
                [0, 0, 0, 1, 1],
                [0, 1, 1, 1, 1],
            ),
            0.5666666666666667,
            places=15,
        )
        self.assertAlmostEqual(
            self.report.stable_cross_entropy([1.2, -0.7], 1),
            2.0393867582829603,
            places=15,
        )

    def test_trial_loss_is_sample_weighted_across_unequal_batch_grouping(self):
        runs, rows, _ = make_fixture(expected_examples=3)
        logits_by_position = {
            0: [0.2, -0.1],
            1: [-0.3, 0.8],
            2: [1.1, 0.4],
        }
        for row in rows:
            row["logits"] = logits_by_position[row["probe_position"]]
            row["predicted_label"] = row["gold_label"]
            row["correct"] = True
        _set_raw_metrics(
            runs,
            loss=0.41495887282313854,
            accuracy=1.0,
            weighted_f1=1.0,
        )
        prior_runs = copy.deepcopy(runs)

        summary = self.report.build_prediction_summary(
            run_payloads=runs,
            prediction_rows=rows,
            prior_run_payloads=prior_runs,
            source_commit="golden-loss",
            expected_examples=3,
        )

        self.assertTrue(summary["all_gates_pass"])
        self.assertAlmostEqual(
            summary["trial_results"][0]["recomputed_loss"],
            0.41495887282313854,
            places=15,
        )

    def test_prior_runs_require_exact_unique_groups_but_allow_shuffled_order(self):
        mutations = {
            "prior_duplicate_groups": lambda priors: priors[0][
                "group_results"
            ].append(copy.deepcopy(priors[0]["group_results"][0])),
            "prior_required_groups": lambda priors: priors[0][
                "group_results"
            ].pop(),
        }
        for expected_code, mutate in mutations.items():
            with self.subTest(expected_code=expected_code):
                runs, rows, prior_runs = make_fixture()
                mutate(prior_runs)
                summary = self._summary(runs, rows, prior_runs)
                self.assertIn(
                    expected_code,
                    failure_codes(summary, "prior_equivalence"),
                )

        runs, rows, prior_runs = make_fixture()
        prior_runs.reverse()
        for prior in prior_runs:
            prior["group_results"].reverse()
        summary = self._summary(runs, rows, prior_runs)
        self.assertTrue(summary["all_gates_pass"])


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


class PythonCompatibilityTests(unittest.TestCase):
    def test_renderer_has_no_pep604_union_syntax(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "render_three_group_per_example_logits_report.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(" | ", source)

    def test_renderer_imports_with_real_python39_when_available(self):
        candidates = [
            os.environ.get("PYTHON39"),
            shutil.which("python3.9"),
            "/usr/bin/python3",
        ]
        interpreter = None
        for candidate in candidates:
            if not candidate or not Path(candidate).is_file():
                continue
            version = subprocess.run(
                [
                    candidate,
                    "-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if version.returncode == 0 and version.stdout.strip() == "3.9":
                interpreter = candidate
                break
        if interpreter is None:
            self.skipTest("Python 3.9 interpreter is not available")

        completed = subprocess.run(
            [
                interpreter,
                "-c",
                (
                    "import scripts.render_three_group_per_example_logits_report "
                    "as report; assert report.SCHEMA_VERSION == "
                    "'three-group-per-example-logits-v1'"
                ),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


class OutputTransactionTests(unittest.TestCase):
    def setUp(self):
        self.report = _load_report_module()

    @staticmethod
    def _args(output_json, output_html):
        return [
            "--run-json",
            "run.json",
            "--prediction-jsonl",
            "predictions.jsonl",
            "--prior-run-json",
            "prior.json",
            "--source-commit",
            "transaction-test",
            "--output-json",
            str(output_json),
            "--output-html",
            str(output_html),
        ]

    def test_cli_rejects_outputs_resolving_to_same_path(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "report.out"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "distinct"):
                self.report.main(self._args(output, output))
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")
            self.assertEqual([path.name for path in Path(td).iterdir()], ["report.out"])

    def test_render_failure_preserves_existing_outputs_without_temp_leaks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_json = root / "summary.json"
            output_html = root / "report.html"
            output_json.write_text("old-json", encoding="utf-8")
            output_html.write_text("old-html", encoding="utf-8")
            with (
                mock.patch.object(self.report, "_load_json", return_value={}),
                mock.patch.object(self.report, "_load_jsonl", return_value=[]),
                mock.patch.object(
                    self.report,
                    "build_prediction_summary",
                    return_value={"all_gates_pass": True},
                ),
                mock.patch.object(
                    self.report,
                    "render_html",
                    side_effect=RuntimeError("render failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    self.report.main(self._args(output_json, output_html))

            self.assertEqual(output_json.read_text(encoding="utf-8"), "old-json")
            self.assertEqual(output_html.read_text(encoding="utf-8"), "old-html")
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"summary.json", "report.html"},
            )

    def test_partial_temp_write_failure_preserves_outputs_and_cleans_temps(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_json = root / "summary.json"
            output_html = root / "report.html"
            output_json.write_text("old-json", encoding="utf-8")
            output_html.write_text("old-html", encoding="utf-8")
            real_write = self.report._write_temp_file
            call_count = 0

            def fail_second_write(path, text):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("second write failed")
                return real_write(path, text)

            with mock.patch.object(
                self.report, "_write_temp_file", side_effect=fail_second_write
            ):
                with self.assertRaisesRegex(OSError, "second write failed"):
                    self.report._write_outputs_transactionally(
                        output_json,
                        output_html,
                        '{"new":true}\n',
                        "<html>new</html>",
                    )

            self.assertEqual(output_json.read_text(encoding="utf-8"), "old-json")
            self.assertEqual(output_html.read_text(encoding="utf-8"), "old-html")
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"summary.json", "report.html"},
            )

    def test_second_replace_failure_rolls_back_both_outputs_and_cleans_temps(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_json = root / "summary.json"
            output_html = root / "report.html"
            output_json.write_text("old-json", encoding="utf-8")
            output_html.write_text("old-html", encoding="utf-8")
            real_replace = os.replace
            failed = False

            def fail_html_promotion(source, destination):
                nonlocal failed
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    not failed
                    and destination_path == output_html
                    and source_path.name.startswith(f".{output_html.name}.tmp-")
                ):
                    failed = True
                    raise OSError("html replace failed")
                return real_replace(source, destination)

            with mock.patch.object(
                self.report.os, "replace", side_effect=fail_html_promotion
            ):
                with self.assertRaisesRegex(OSError, "html replace failed"):
                    self.report._write_outputs_transactionally(
                        output_json,
                        output_html,
                        '{"new":true}\n',
                        "<html>new</html>",
                    )

            self.assertTrue(failed)
            self.assertEqual(output_json.read_text(encoding="utf-8"), "old-json")
            self.assertEqual(output_html.read_text(encoding="utf-8"), "old-html")
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"summary.json", "report.html"},
            )


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

    def test_cli_malformed_json_types_write_strict_diagnostics_and_return_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = []
            rows_by_seed = {
                seed: [row for row in self.rows if row["run_seed"] == seed]
                for seed in (run["seed"] for run in self.runs)
            }
            for index, (original_run, original_prior) in enumerate(
                zip(self.runs, self.prior_runs)
            ):
                run = copy.deepcopy(original_run)
                prior = copy.deepcopy(original_prior)
                selected_rows = copy.deepcopy(rows_by_seed[original_run["seed"]])
                if index == 0:
                    run["seed"] = []
                    selected_rows[0]["run_seed"] = []
                    selected_rows[1]["group"] = {"bad": "group"}
                if index == 1:
                    prior["seed"] = {"bad": "seed"}
                run_path = root / f"run-{index}.json"
                prior_path = root / f"prior-{index}.json"
                prediction_path = root / f"prediction-{index}.jsonl"
                run_path.write_text(json.dumps(run), encoding="utf-8")
                prior_path.write_text(json.dumps(prior), encoding="utf-8")
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
                    "malformed-json",
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
            self.assertTrue(
                {
                    "current_run_seed_type",
                    "prior_run_seed_type",
                    "row_run_seed_type",
                    "row_group_type",
                }.issubset(failure_codes(summary))
            )
            self.assertNotIn("NaN", summary_text)
            self.assertNotIn("Infinity", summary_text)
            self.assertIn('id="prediction-data"', output_html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
