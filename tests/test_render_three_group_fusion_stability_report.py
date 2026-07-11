import importlib
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path


SEEDS = [20260721, 20261721, 20262721, 20263721, 20264721]
INSTALL_PATH = (
    "BLBStage2SequentialEnv.evaluate_step -> commit_step -> "
    "BLBStage2Env.step(boosted_overrides)"
)
GROUP_PATTERNS = {
    "all_fusion0": {"block2": 0, "block4": 0, "block5_n4": 0},
    "block2_block5_all_layers_fusionmax": {
        "block2": 1,
        "block4": 0,
        "block5_n4": 1,
    },
    "block2_block4_block5_all_layers_fusion1": {
        "block2": 1,
        "block4": 1,
        "block5_n4": 1,
    },
}
EXPECTED_TOTALS = {
    "all_fusion0": 0,
    "block2_block5_all_layers_fusionmax": 24,
    "block2_block4_block5_all_layers_fusion1": 36,
}


def _load_report_module():
    return importlib.import_module("scripts.render_three_group_fusion_stability_report")


def _step_schedule(pattern):
    steps = []
    for layer in range(12):
        blocks = (
            ((2, "block2_mrpc"), (4, "block4"), (5, "block5_n4"))
            if layer == 0
            else (
                (1, "block1_mrpc"),
                (2, "block2_mrpc"),
                (4, "block4"),
                (5, "block5_n4"),
            )
        )
        for block_idx, graph_key in blocks:
            pattern_key = "block5_n4" if block_idx == 5 else f"block{block_idx}"
            fusion_count = pattern.get(pattern_key, 0)
            steps.append(
                {
                    "step_idx": len(steps),
                    "layer_idx": layer,
                    "block_idx": block_idx,
                    "graph_key": graph_key,
                    "option_id": fusion_count,
                    "map_option_id": fusion_count,
                    "k_index": 3,
                    "k_value": 13,
                    "valid": True,
                    "fusion_count_replan": fusion_count,
                    "boosted": bool(fusion_count),
                    "replan_application": {
                        "applied_before_forward": True,
                        "model_uses_replan_config": True,
                    },
                    "model_uses_replan_config": True,
                }
            )
    assert len(steps) == 47
    return steps


def _trial_values(run_index, group_index):
    losses = [
        0.310 + group_index * 0.004 + run_index * 0.0007 + trial * 0.00011
        for trial in range(5)
    ]
    metric1 = [
        0.880 + group_index * 0.003 - run_index * 0.0002 + trial * 0.00009
        for trial in range(5)
    ]
    metric2 = [
        0.870 + group_index * 0.002 - run_index * 0.0001 + trial * 0.00007
        for trial in range(5)
    ]
    return {"loss": losses, "metric1": metric1, "metric2": metric2}


def _group_payload(name, run_index, group_index):
    pattern = GROUP_PATTERNS[name]
    trial_values = _trial_values(run_index, group_index)
    metrics = {
        "loss_mean": statistics.fmean(trial_values["loss"]),
        "loss_std": statistics.pstdev(trial_values["loss"]),
        "metric1_mean": statistics.fmean(trial_values["metric1"]),
        "metric1_std": statistics.pstdev(trial_values["metric1"]),
        "metric2_mean": statistics.fmean(trial_values["metric2"]),
        "metric2_std": statistics.pstdev(trial_values["metric2"]),
        "loss_max": max(trial_values["loss"]),
        "metric1_min": min(trial_values["metric1"]),
        "metric2_min": min(trial_values["metric2"]),
    }
    return {
        "name": name,
        "metrics": metrics,
        "trial_metrics": trial_values,
        "fusion_total": EXPECTED_TOTALS[name],
        "fusion_by_block": {
            "1": 0,
            "2": pattern["block2"] * 12,
            "4": pattern["block4"] * 12,
            "5": pattern["block5_n4"] * 12,
        },
        "k_distribution": {"13": 47},
        "block5_graph_counts": {"block5_n4": 12},
        "step_records": _step_schedule(pattern),
        "terminal_probe": {
            "k": 5,
            "deterministic_probe_seed": SEEDS[run_index],
            "per_worker_trial_indices": [list(range(5))],
            "per_worker_trial_seeds": [
                [SEEDS[run_index] ^ (trial * 2654435761) for trial in range(5)]
            ],
        },
    }


def make_run_payload(run_index):
    group_results = [
        _group_payload(name, run_index, group_index)
        for group_index, name in enumerate(GROUP_PATTERNS)
    ]
    return {
        "schema_version": "fusion_count_action_eval_rlpath_compare_v1",
        "seed": SEEDS[run_index],
        "shared_group_seed": True,
        "repeat": 5,
        "probe_size": 408,
        "stage1_gelu": [4] * 12,
        "stage1_softmax": [6] * 12,
        "install_path": INSTALL_PATH,
        "group_results": group_results,
    }


def make_run_payloads():
    return [make_run_payload(run_index) for run_index in range(5)]


def _group(payload, group_name):
    return next(
        group
        for group in payload["group_results"]
        if group["name"] == group_name
    )


def _pooled_values(payloads, group_name, metric):
    return [
        value
        for payload in payloads
        for value in _group(payload, group_name)["trial_metrics"][metric]
    ]


def _gate(summary, name):
    return next(gate for gate in summary["gates"] if gate["name"] == name)


class BuildSummaryTests(unittest.TestCase):
    def test_pools_raw_trials_and_builds_all_paired_comparisons(self):
        report = _load_report_module()
        payloads = make_run_payloads()

        summary = report.build_summary(
            run_payloads=payloads,
            source_commit="abc123",
        )

        self.assertTrue(summary["all_gates_pass"])
        self.assertEqual(summary["total_evaluations"], 75)
        self.assertIsInstance(report.render_html(summary), str)
        self.assertEqual(
            set(summary["comparisons"]),
            {
                "b2b5_minus_control",
                "b2b4b5_minus_control",
                "b2b4b5_minus_b2b5",
            },
        )

        for group_name in GROUP_PATTERNS:
            for metric in ("loss", "metric1", "metric2"):
                expected = _pooled_values(payloads, group_name, metric)
                actual = summary["groups"][group_name]["pooled_metrics"][metric]
                self.assertEqual(actual["count"], 25)
                self.assertTrue(math.isfinite(actual["mean"]))
                self.assertTrue(math.isfinite(actual["std"]))
                self.assertAlmostEqual(actual["mean"], statistics.fmean(expected))
                self.assertAlmostEqual(actual["std"], statistics.pstdev(expected))

        treatment = _pooled_values(
            payloads, "block2_block4_block5_all_layers_fusion1", "metric1"
        )
        control = _pooled_values(payloads, "all_fusion0", "metric1")
        paired = [new - old for new, old in zip(treatment, control)]
        actual_delta = summary["comparisons"]["b2b4b5_minus_control"][
            "paired_deltas"
        ]["metric1"]
        self.assertEqual(actual_delta["count"], 25)
        self.assertAlmostEqual(actual_delta["mean"], statistics.fmean(paired))
        self.assertAlmostEqual(actual_delta["std"], statistics.pstdev(paired))

    def test_missing_or_wrong_evaluator_schema_version_fails_completeness_gate(self):
        report = _load_report_module()
        mutations = {
            "missing": lambda payload: payload.pop("schema_version"),
            "wrong": lambda payload: payload.update(
                {"schema_version": "fusion_count_action_eval_rlpath_compare_v0"}
            ),
        }

        for case, mutate in mutations.items():
            with self.subTest(case=case):
                payloads = make_run_payloads()
                mutate(payloads[2])
                summary = report.build_summary(
                    run_payloads=payloads,
                    source_commit="abc123",
                )
                self.assertFalse(summary["all_gates_pass"])
                completeness = _gate(summary, "completeness")
                self.assertFalse(completeness["passed"])
                self.assertIn(
                    "schema_version",
                    json.dumps(completeness["failures"]).lower(),
                )

    def test_non_deterministic_trial_seed_list_fails_trial_metadata_gate(self):
        report = _load_report_module()
        payloads = make_run_payloads()
        probe = _group(
            payloads[1], "block2_block5_all_layers_fusionmax"
        )["terminal_probe"]
        probe["per_worker_trial_seeds"] = [[101, 102, 103, 104, 105]]

        summary = report.build_summary(
            run_payloads=payloads,
            source_commit="abc123",
        )

        self.assertFalse(summary["all_gates_pass"])
        metadata_gate = _gate(summary, "trial_metadata")
        self.assertFalse(metadata_gate["passed"])
        self.assertIn("deterministic", json.dumps(metadata_gate["failures"]).lower())

    def test_reported_metrics_must_match_raw_trial_statistics(self):
        report = _load_report_module()
        payloads = make_run_payloads()
        group = _group(
            payloads[3], "block2_block4_block5_all_layers_fusion1"
        )
        group["metrics"]["loss_mean"] += 0.1

        summary = report.build_summary(
            run_payloads=payloads,
            source_commit="abc123",
        )

        self.assertFalse(summary["all_gates_pass"])
        consistency_gate = _gate(summary, "metric_consistency")
        self.assertFalse(consistency_gate["passed"])
        failures = json.dumps(consistency_gate["failures"]).lower()
        self.assertIn("loss_mean", failures)
        self.assertIn("mismatch", failures)

    def test_wrong_block4_replan_is_a_structured_pattern_gate_failure(self):
        report = _load_report_module()
        payloads = make_run_payloads()
        treatment = _group(
            payloads[2], "block2_block4_block5_all_layers_fusion1"
        )
        block4_step = next(
            step for step in treatment["step_records"] if step["block_idx"] == 4
        )
        block4_step["fusion_count_replan"] = 0

        summary = report.build_summary(
            run_payloads=payloads,
            source_commit="abc123",
        )

        self.assertFalse(summary["all_gates_pass"])
        pattern_gate = _gate(summary, "fusion_pattern")
        self.assertFalse(pattern_gate["passed"])
        self.assertTrue(pattern_gate["failures"])
        self.assertIn("block4", json.dumps(pattern_gate["failures"]).lower())

    def test_step_records_must_follow_exact_schedule_order_and_index(self):
        report = _load_report_module()

        def swap_first_two(payloads):
            records = _group(payloads[0], "all_fusion0")["step_records"]
            records[0], records[1] = records[1], records[0]

        def duplicate_step_index(payloads):
            records = _group(payloads[0], "all_fusion0")["step_records"]
            records[1]["step_idx"] = records[0]["step_idx"]

        for case, mutate in {
            "record order": swap_first_two,
            "step index": duplicate_step_index,
        }.items():
            with self.subTest(case=case):
                payloads = make_run_payloads()
                mutate(payloads)
                summary = report.build_summary(
                    run_payloads=payloads,
                    source_commit="abc123",
                )
                self.assertFalse(summary["all_gates_pass"])
                pattern_gate = _gate(summary, "fusion_pattern")
                self.assertFalse(pattern_gate["passed"])
                failures = json.dumps(pattern_gate["failures"]).lower()
                self.assertRegex(failures, r"order|index|step_idx")

    def test_missing_required_group_or_trial_fails_completeness_gate(self):
        report = _load_report_module()
        mutations = {
            "missing group": lambda payloads: payloads[0]["group_results"].pop(0),
            "missing trial": lambda payloads: _group(payloads[4], "all_fusion0")[
                "trial_metrics"
            ]["loss"].pop(),
        }

        for case, mutate in mutations.items():
            with self.subTest(case=case):
                payloads = make_run_payloads()
                mutate(payloads)
                summary = report.build_summary(
                    run_payloads=payloads,
                    source_commit="abc123",
                )
                self.assertFalse(summary["all_gates_pass"])
                self.assertFalse(_gate(summary, "completeness")["passed"])


class CliTests(unittest.TestCase):
    def _write_payloads(self, directory, payloads):
        paths = []
        for index, payload in enumerate(payloads):
            path = directory / f"run_{index}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(path)
        return paths

    def _argv(self, run_paths, output_json, output_html):
        argv = []
        for path in run_paths:
            argv.extend(["--run-json", str(path)])
        argv.extend(
            [
                "--source-commit",
                "abc123",
                "--output-json",
                str(output_json),
                "--output-html",
                str(output_html),
            ]
        )
        return argv

    def test_cli_writes_strict_json_and_visible_complete_html(self):
        report = _load_report_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_paths = self._write_payloads(directory, make_run_payloads())
            output_json = directory / "summary.json"
            output_html = directory / "report.html"

            rc = report.main(self._argv(run_paths, output_json, output_html))

            self.assertEqual(rc, 0)
            summary = json.loads(
                output_json.read_text(encoding="utf-8"),
                parse_constant=lambda value: self.fail(
                    f"non-standard JSON constant emitted: {value}"
                ),
            )
            self.assertTrue(summary["all_gates_pass"])
            self.assertEqual(summary["total_evaluations"], 75)
            self.assertEqual(summary["expected_evaluations"], 75)
            self.assertEqual(summary["observed_evaluations"], 75)
            self.assertEqual(summary["included_evaluations"], 75)
            html = output_html.read_text(encoding="utf-8")
            for visible_text in (
                "B2=1",
                "B4=1",
                "B5=1",
                "K=13",
                "mean +/- std",
                "75",
                "b2b5_minus_control",
                "b2b4b5_minus_control",
                "b2b4b5_minus_b2b5",
                ", ".join(str(seed) for seed in SEEDS),
                "Stage-1 GELU",
                "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]",
                "Stage-1 Softmax",
                "[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]",
                "INCLUDED",
            ):
                self.assertIn(visible_text, html)

    def test_bad_payload_returns_one_and_preserves_diagnostic_artifacts(self):
        report = _load_report_module()
        payloads = make_run_payloads()
        payloads[1]["group_results"] = [
            group
            for group in payloads[1]["group_results"]
            if group["name"] != "block2_block5_all_layers_fusionmax"
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_paths = self._write_payloads(directory, payloads)
            output_json = directory / "diagnostic.json"
            output_html = directory / "diagnostic.html"

            rc = report.main(self._argv(run_paths, output_json, output_html))

            self.assertEqual(rc, 1)
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_html.is_file())
            summary = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertFalse(summary["all_gates_pass"])
            self.assertFalse(_gate(summary, "completeness")["passed"])
            self.assertEqual(summary["expected_evaluations"], 75)
            self.assertEqual(summary["observed_evaluations"], 70)
            self.assertEqual(summary["included_evaluations"], 60)
            self.assertEqual(summary["total_evaluations"], 60)
            self.assertEqual(summary["expected_seeds"], SEEDS)
            self.assertEqual(summary["observed_seeds"], SEEDS)
            self.assertEqual(
                summary["included_seeds"],
                [seed for seed in SEEDS if seed != SEEDS[1]],
            )
            html = output_html.read_text(encoding="utf-8")
            self.assertIn("completeness", html.lower())
            for visible_text in (
                "Expected evaluations",
                "Observed evaluations",
                "Included evaluations",
                "EXCLUDED",
            ):
                self.assertIn(visible_text, html)

    def test_non_finite_payload_still_writes_strict_diagnostic_artifacts(self):
        report = _load_report_module()
        payloads = make_run_payloads()
        payloads[0]["seed"] = float("nan")
        malformed_group = _group(payloads[0], "all_fusion0")
        malformed_group["fusion_by_block"]["2"] = float("inf")
        malformed_group["metrics"]["loss_mean"] = float("-inf")

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_paths = self._write_payloads(directory, payloads)
            output_json = directory / "non_finite_diagnostic.json"
            output_html = directory / "non_finite_diagnostic.html"

            rc = report.main(self._argv(run_paths, output_json, output_html))

            self.assertEqual(rc, 1)
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_html.is_file())
            summary = json.loads(
                output_json.read_text(encoding="utf-8"),
                parse_constant=lambda value: self.fail(
                    f"non-standard JSON constant emitted: {value}"
                ),
            )
            self.assertFalse(summary["all_gates_pass"])
            html = output_html.read_text(encoding="utf-8").lower()
            self.assertIn("fail", html)
            self.assertIn("completeness", html)


if __name__ == "__main__":
    unittest.main()
