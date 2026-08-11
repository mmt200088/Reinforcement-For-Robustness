from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]


class Stage2PersistentLauncherTest(unittest.TestCase):
    @staticmethod
    def _install_fake_flock(fakebin):
        fake_flock = fakebin / "flock"
        fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        fake_flock.chmod(0o755)

    def test_layerwise_stage2_defaults_to_elastic_reward_devices(self):
        source = (
            REPO_ROOT / "llama_7B_LayerImportance.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("--elastic-gpu-mode", source)
        self.assertIn("scripts/elastic_gpu_supervisor.py", source)
        self.assertIn(
            '[ "$RUN_MODE" = "stage2-only" ] && '
            '[ "$S_BLB_V3_REWARD_DEVICES" = "false" ]',
            source,
        )
        self.assertIn('BLB_V3_REWARD_DEVICES="auto"', source)
        self.assertIn("--blb_v3_reward_devices", source)

    def test_elastic_off_retains_direct_python_launch(self):
        argv = self._capture_stage2_launcher_argv(
            ["--elastic-gpu-mode", "off"]
        )

        self.assertEqual(argv[0], "rl_tune.py")
        self.assertNotIn("scripts/elastic_gpu_supervisor.py", argv)

    def test_python_public_decision_fields_reach_evaluator_constructor(self):
        tune_tree = ast.parse((REPO_ROOT / "rl_tune.py").read_text(encoding="utf-8"))
        train_fn = next(
            node for node in tune_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        train_args = {arg.arg: ast.unparse(arg.annotation) for arg in train_fn.args.args}
        self.assertEqual(train_args["blb_v3_decision_granularity"], "str")
        self.assertEqual(train_args["blb_v3_reward_design"], "str")
        self.assertEqual(train_args["blb_v3_policy_network_variant"], "str")
        evaluator_call = next(
            node for node in ast.walk(train_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "LayerImportanceEvaluator"
        )
        forwarded = {keyword.arg for keyword in evaluator_call.keywords}
        self.assertIn("blb_v3_decision_granularity", forwarded)
        self.assertIn("blb_v3_reward_design", forwarded)
        self.assertIn("blb_v3_policy_network_variant", forwarded)

        evaluator_tree = ast.parse(
            (REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        )
        evaluator_class = next(
            node for node in evaluator_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LayerImportanceEvaluator"
        )
        init_fn = next(
            node for node in evaluator_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        init_args = {arg.arg for arg in init_fn.args.args}
        self.assertIn("blb_v3_decision_granularity", init_args)
        self.assertIn("blb_v3_reward_design", init_args)
        self.assertIn("blb_v3_policy_network_variant", init_args)
        assigned_attrs = {
            target.attr
            for node in ast.walk(init_fn)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        }
        self.assertIn("blb_v3_decision_granularity", assigned_attrs)
        self.assertIn("blb_v3_reward_design", assigned_attrs)
        self.assertIn("blb_v3_policy_network_variant", assigned_attrs)

    def test_python_robust_constraint_fields_reach_evaluator_constructor(self):
        expected = {
            "stage2_stability_multiplier": "float",
            "blb_v3_baseline_groups": "int",
            "blb_v3_baseline_trials_per_group": "int",
            "blb_v3_constraint_bootstrap_samples": "int",
            "blb_v3_online_constraint_probability": "float",
            "blb_v3_promotion_constraint_probability": "float",
            "blb_v3_final_constraint_probability": "float",
            "blb_v3_min_convergence_episodes": "int",
            "blb_v3_convergence_patience_updates": "int",
        }
        tune_tree = ast.parse((REPO_ROOT / "rl_tune.py").read_text(encoding="utf-8"))
        train_fn = next(
            node for node in tune_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        train_args = {arg.arg: ast.unparse(arg.annotation) for arg in train_fn.args.args}
        for name, annotation in expected.items():
            self.assertEqual(train_args[name], annotation)
        evaluator_call = next(
            node for node in ast.walk(train_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "LayerImportanceEvaluator"
        )
        forwarded = {keyword.arg for keyword in evaluator_call.keywords}
        self.assertTrue(set(expected).issubset(forwarded))

    def _capture_stage2_launcher_argv(self, extra_args):
        with tempfile.TemporaryDirectory(prefix="stage2_public_config_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = ""
            result = subprocess.run(
                [
                    "bash",
                    "llama_7B_LayerImportance.sh",
                    "run",
                    "rl",
                    "--preset",
                    "mrpc-blb-stage2-rl",
                    "--mode",
                    "stage2-only",
                    "--persistent-root",
                    str(tmp / "persistent"),
                    "--stage2-search-episodes",
                    "170",
                    "--fresh",
                    *extra_args,
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0, msg=result.stdout + "\n" + result.stderr,
            )
            for _ in range(50):
                if capture.is_file():
                    break
                import time

                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke python")
            return [
                part.decode("utf-8")
                for part in capture.read_bytes().split(b"\0")[:-1]
            ]

    def _capture_comparator_persistent_path(
            self,
            alias,
            extra_args=(),
            *,
            include_preset=True,
            return_argv=False,
    ):
        with tempfile.TemporaryDirectory(prefix=f"{alias}_persistent_route_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            persistent_root = tmp / "persistent"
            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = ""
            command = [
                "bash",
                "llama_7B_LayerImportance.sh",
                "run",
                alias,
            ]
            if include_preset:
                command.extend(("--preset", "mrpc-blb-stage2-rl"))
            command.extend((
                "--persistent-root",
                str(persistent_root),
                "--fresh",
                *extra_args,
            ))
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0, msg=result.stdout + "\n" + result.stderr,
            )
            for _ in range(50):
                if capture.is_file():
                    break
                import time

                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke python")
            argv = [
                part.decode("utf-8")
                for part in capture.read_bytes().split(b"\0")[:-1]
            ]
            self.assertNotIn("--formal_dataset_protocol", argv)
            self.assertNotIn("--formal_mrpc_fixture_path", argv)
            self.assertEqual(
                Path(
                    argv[
                        argv.index(
                            "--mrpc_reproducibility_fixture_path"
                        )
                        + 1
                    ]
                ),
                REPO_ROOT
                / "fixtures"
                / "reproducibility"
                / "mrpc_validation_v1.json",
            )
            if return_argv:
                return argv
            output_dir = Path(argv[argv.index("--output_dir") + 1])
            return output_dir.relative_to(persistent_root)

    def _capture_persistent_slug_and_metadata(self, extra_args):
        with tempfile.TemporaryDirectory(prefix="stage2_constraint_identity_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = ""
            result = subprocess.run(
                [
                    "bash",
                    "llama_7B_LayerImportance.sh",
                    "run",
                    "rl",
                    "--preset",
                    "mrpc-blb-stage2-rl",
                    "--mode",
                    "stage2-only",
                    "--persistent-root",
                    str(tmp / "persistent"),
                    "--stage2-search-episodes",
                    "170",
                    "--fresh",
                    *extra_args,
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0, msg=result.stdout + "\n" + result.stderr,
            )
            for _ in range(50):
                if capture.is_file():
                    break
                import time

                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke python")
            argv = [
                part.decode("utf-8")
                for part in capture.read_bytes().split(b"\0")[:-1]
            ]
            output_dir = Path(argv[argv.index("--output_dir") + 1])
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            return output_dir.name, metadata

    def test_comparator_aliases_use_disjoint_backend_persistent_directories(self):
        expected_prefixes = {
            "bo_rf": ("bo_rf", "bert-base", "mrpc"),
            "greedy": ("greedy", "bert-base", "mrpc"),
            "coinn_ga": ("coinn_ga", "bert-base", "mrpc"),
        }

        for alias, expected_prefix in expected_prefixes.items():
            with self.subTest(alias=alias):
                relative_path = self._capture_comparator_persistent_path(alias)
                self.assertEqual(relative_path.parts[:3], expected_prefix)
                self.assertEqual(
                    relative_path.parts[3],
                    "s1t0.001_s2t0.001_s2st2.0",
                )

    def test_comparator_aliases_fix_canonical_batch_without_preset(self):
        for alias in ("bo_rf", "greedy", "coinn_ga"):
            with self.subTest(alias=alias):
                argv = self._capture_comparator_persistent_path(
                    alias,
                    include_preset=False,
                    return_argv=True,
                )
                self.assertEqual(
                    argv[argv.index("--batch_size") + 1],
                    "64",
                )
                self.assertEqual(
                    argv[argv.index("--micro_batch_size") + 1],
                    "64",
                )

    def test_comparator_aliases_enable_three_single_gpu_probe_workers(self):
        for alias in ("bo_rf", "greedy", "coinn_ga"):
            with self.subTest(alias=alias):
                argv = self._capture_comparator_persistent_path(
                    alias,
                    return_argv=True,
                )
                self.assertEqual(
                    argv[argv.index("--blb_v3_reward_devices") + 1],
                    "0",
                )
                self.assertEqual(
                    argv[argv.index("--stage2_workers_per_device") + 1],
                    "3",
                )

    def test_comparator_aliases_pin_final_eval_and_stage2_seeds(self):
        for alias in ("bo_rf", "greedy", "coinn_ga"):
            for include_preset in (False, True):
                with self.subTest(
                        alias=alias,
                        include_preset=include_preset,
                ):
                    argv = self._capture_comparator_persistent_path(
                        alias,
                        include_preset=include_preset,
                        return_argv=True,
                    )
                    self.assertEqual(
                        argv[argv.index("--final_eval_random_seed") + 1],
                        "42",
                    )
                    self.assertEqual(
                        argv[argv.index("--blb_v3_seed") + 1],
                        "42",
                    )
                    if alias == "bo_rf":
                        for flag, expected in (
                            ("--blb_v3_search_initial_design_size", "64"),
                            ("--blb_v3_search_candidate_pool_size", "2048"),
                            ("--blb_v3_search_patience_generations", "100"),
                            ("--blb_v3_search_rf_n_estimators", "128"),
                            ("--blb_v3_search_rf_min_samples_leaf", "2"),
                        ):
                            with self.subTest(
                                    alias=alias,
                                    include_preset=include_preset,
                                    flag=flag,
                            ):
                                self.assertEqual(
                                    argv[argv.index(flag) + 1],
                                    expected,
                                )
                    elif alias == "coinn_ga":
                        self.assertEqual(
                            argv[
                                argv.index(
                                    "--blb_v3_search_patience_generations"
                                ) + 1
                            ],
                            "5",
                        )

    def test_comparator_smoke_aliases_apply_smoke_settings(self):
        for alias in ("bo_rf", "greedy", "coinn_ga"):
            with self.subTest(alias=alias):
                argv = self._capture_comparator_persistent_path(
                    alias,
                    extra_args=("--comparator-smoke",),
                    return_argv=True,
                )
                for flag, expected in (
                    ("--comparator_smoke", "true"),
                    ("--blb_v3_search_evaluation_budget", "1"),
                    ("--blb_v3_search_full_validation", "false"),
                    ("--stage2_k_trials", "3"),
                    ("--final_eval_random_seed", "42"),
                    ("--blb_v3_seed", "42"),
                    ("--skip_final_eval", "true"),
                ):
                    with self.subTest(alias=alias, flag=flag):
                        self.assertEqual(
                            argv[argv.index(flag) + 1],
                            expected,
                        )

    def test_comparator_aliases_apply_full_run_settings_without_smoke(self):
        expected_budgets = {
            "bo_rf": "50000",
            "greedy": "2176782336",
            "coinn_ga": "45664",
        }
        for alias, expected_budget in expected_budgets.items():
            with self.subTest(alias=alias):
                argv = self._capture_comparator_persistent_path(
                    alias,
                    return_argv=True,
                )
                for flag, expected in (
                    ("--comparator_smoke", "false"),
                    ("--blb_v3_search_evaluation_budget", expected_budget),
                    ("--blb_v3_search_full_validation", "true"),
                    ("--skip_final_eval", "false"),
                ):
                    with self.subTest(alias=alias, flag=flag):
                        self.assertEqual(
                            argv[argv.index(flag) + 1],
                            expected,
                        )

    def test_comparator_aliases_reject_fixed_setting_overrides(self):
        cases = (
            ("bo_rf", ("--algorithm", "ga")),
            ("bo_rf", ("--algorithm=ga",)),
            ("greedy", ("--mode", "stage2-only")),
            ("coinn_ga", ("--blb-v3-search-backend", "greedy")),
            ("bo_rf", ("--blb-v3-search-evaluation-budget", "1")),
            ("bo_rf", ("--blb-v3-search-initial-design-size", "1")),
            ("bo_rf", ("--blb-v3-search-initial-design-size=1",)),
            ("bo_rf", ("--blb-v3-search-candidate-pool-size", "1")),
            ("bo_rf", ("--blb-v3-search-candidate-pool-size=1",)),
            ("bo_rf", ("--blb-v3-search-patience-generations", "1")),
            ("bo_rf", ("--blb-v3-search-patience-generations=1",)),
            ("coinn_ga", ("--blb-v3-search-population-size", "63")),
            ("coinn_ga", ("--blb-v3-search-population-size=63",)),
            ("greedy", ("--stage2-k-trials", "5")),
            ("bo_rf", ("--random-seed", "7")),
            ("greedy", ("--random-seed=7",)),
            ("coinn_ga", ("--blb-v3-seed", "8")),
            ("bo_rf", ("--blb-v3-seed=8",)),
            ("bo_rf", ("--blb-v3-search-rf-n-estimators", "1")),
            ("bo_rf", ("--blb-v3-search-rf-n-estimators=1",)),
            ("bo_rf", ("--blb-v3-search-rf-min-samples-leaf", "99")),
            ("bo_rf", ("--blb-v3-search-rf-min-samples-leaf=99",)),
            ("bo_rf", ("--batch-size", "8")),
            ("greedy", ("--batch-size=8",)),
            (
                "greedy",
                ("--mrpc-reproducibility-fixture-path", "other.json"),
            ),
            (
                "coinn_ga",
                ("--mrpc-reproducibility-fixture-path=other.json",),
            ),
            ("bo_rf", ("--stage2-rl-variant", "legacy_v2")),
            ("bo_rf", ("--stage2-rl-variant=legacy_v2",)),
            ("greedy", ("--stage2-probe-size", "1")),
            ("greedy", ("--stage2-probe-size=1",)),
            ("bo_rf", ("--stage2-communication-importance-ratio", "0.5")),
            ("greedy", ("--stage2-communication-importance-ratio=0.5",)),
            ("coinn_ga", ("--blb-v3-truncation-backend", "decimal")),
            ("bo_rf", ("--blb-v3-truncation-backend=decimal",)),
            ("greedy", ("--blb-v3-truncation-ring-bits", "44")),
            ("coinn_ga", ("--blb-v3-truncation-ring-bits=44",)),
            ("bo_rf", ("--blb-v3-truncation-source-fractional-bits", "23")),
            ("greedy", ("--blb-v3-truncation-source-fractional-bits=23",)),
        )
        for alias, extra_args in cases:
            with self.subTest(alias=alias, extra_args=extra_args):
                result = subprocess.run(
                    [
                        "bash",
                        "llama_7B_LayerImportance.sh",
                        "run",
                        alias,
                        *extra_args,
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "comparator 子命令不允许覆盖固定的算法、预算或验证参数",
                    result.stdout + result.stderr,
                )

    def test_stage2_public_decision_and_reward_defaults_and_overrides_reach_python(self):
        default_argv = self._capture_stage2_launcher_argv([])
        self.assertEqual(
            default_argv[default_argv.index("--blb_v3_decision_granularity") + 1],
            "layer",
        )
        self.assertEqual(
            default_argv[default_argv.index("--blb_v3_reward_design") + 1],
            "robust_constrained",
        )
        self.assertEqual(
            default_argv[
                default_argv.index("--blb_v3_policy_network_variant") + 1
            ],
            "shared_gtrxl_small_v1",
        )

        explicit_argv = self._capture_stage2_launcher_argv([
            "--blb-v3-decision-granularity",
            "layer",
            "--blb-v3-reward-design",
            "robust_constrained",
            "--blb-v3-policy-network-variant",
            "separate_critic_gtrxl_v1",
        ])
        self.assertEqual(
            explicit_argv[explicit_argv.index("--blb_v3_decision_granularity") + 1],
            "layer",
        )
        self.assertEqual(
            explicit_argv[explicit_argv.index("--blb_v3_reward_design") + 1],
            "robust_constrained",
        )
        self.assertEqual(
            explicit_argv[
                explicit_argv.index("--blb_v3_policy_network_variant") + 1
            ],
            "separate_critic_gtrxl_v1",
        )

    def test_active_layerwise_robust_defaults_reach_python(self):
        argv = self._capture_stage2_launcher_argv([])
        expected = {
            "--stage2_limit_tolerance": "0.001",
            "--stage2_stability_multiplier": "2.0",
            "--stage2_k_trials": "3",
            "--blb_v3_baseline_groups": "5",
            "--blb_v3_baseline_trials_per_group": "3",
            "--blb_v3_constraint_bootstrap_samples": "4096",
            "--blb_v3_online_constraint_probability": "0.50",
            "--blb_v3_promotion_constraint_probability": "0.80",
            "--blb_v3_final_constraint_probability": "0.95",
            "--blb_v3_promotion_validation_trials": "15",
            "--blb_v3_final_selection_validation_trials": "15",
            "--blb_v3_min_convergence_episodes": "90000",
            "--blb_v3_convergence_patience_updates": "100",
            "--blb_v3_rollout_size": "120",
            "--stage2_rl_lr": "5e-5",
        }
        for option, value in expected.items():
            with self.subTest(option=option):
                self.assertIn(option, argv)
                self.assertEqual(argv[argv.index(option) + 1], value)
        self.assertEqual(argv[argv.index("--blb_v3_substage_mode") + 1], "false")
        self.assertNotIn("--blb_v3_warmstart_neighbor_sampling", argv)
        self.assertNotIn("--blb_v3_fusion_probe_interval", argv)
        self.assertNotIn("--blb_v3_fusion_exploration_epsilon", argv)

    def test_constraint_probability_gates_must_be_monotonic(self):
        with tempfile.TemporaryDirectory(prefix="stage2_probability_order_") as td:
            tmp = Path(td)
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            fake_python = fakebin / "python"
            fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            result = subprocess.run(
                [
                    "bash",
                    "llama_7B_LayerImportance.sh",
                    "run",
                    "rl",
                    "--preset",
                    "mrpc-blb-stage2-rl",
                    "--persistent-root",
                    str(tmp / "persistent"),
                    "--fresh",
                    "--blb-v3-online-constraint-probability",
                    "0.90",
                    "--blb-v3-promotion-constraint-probability",
                    "0.80",
                    "--blb-v3-final-constraint-probability",
                    "0.95",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("online <= promotion <= final", result.stdout + result.stderr)

    def test_layerwise_robust_persistence_uses_stability_multiplier(self):
        slug, metadata = self._capture_persistent_slug_and_metadata([
            "--stage2-stability-tolerance",
            "3.0",
            "--stage2-stability-multiplier",
            "2.25",
        ])

        self.assertEqual(slug, "s1t0.001_s2t0.001_s2st2.25")
        self.assertEqual(metadata["stage2_stability_multiplier"], 2.25)
        self.assertNotIn("stage2_stability_tolerance", metadata)
        self.assertEqual(metadata["blb_v3_decision_granularity"], "layer")
        self.assertEqual(metadata["blb_v3_reward_design"], "robust_constrained")
        self.assertEqual(
            metadata["blb_v3_policy_network_variant"], "shared_gtrxl_small_v1"
        )

    def test_block_stage1_aligned_rollback_persistence_uses_legacy_tolerance(self):
        slug, metadata = self._capture_persistent_slug_and_metadata([
            "--blb-v3-decision-granularity",
            "block",
            "--blb-v3-reward-design",
            "stage1_aligned",
            "--stage2-stability-tolerance",
            "3.0",
            "--stage2-stability-multiplier",
            "2.25",
        ])

        self.assertEqual(slug, "s1t0.001_s2t0.001_s2st3.0")
        self.assertEqual(metadata["stage2_stability_tolerance"], 3.0)
        self.assertNotIn("stage2_stability_multiplier", metadata)
        self.assertEqual(metadata["blb_v3_decision_granularity"], "block")
        self.assertEqual(metadata["blb_v3_reward_design"], "stage1_aligned")

    def test_layerwise_resume_rejects_legacy_metadata_without_multiplier(self):
        with tempfile.TemporaryDirectory(prefix="stage2_legacy_metadata_") as td:
            tmp = Path(td)
            root = tmp / "persistent"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            run_dir = (
                root / "rl" / "bert-base" / "mrpc"
                / "s1t0.001_s2t0.001_s2st2.0"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                json.dumps({
                    "stage1_accuracy_tolerance": 0.001,
                    "stage2_limit_tolerance": 0.001,
                    "stage2_stability_tolerance": 2.0,
                }),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "bash",
                    "llama_7B_LayerImportance.sh",
                    "run",
                    "rl",
                    "--preset",
                    "mrpc-blb-stage2-rl",
                    "--mode",
                    "stage2-only",
                    "--persistent-root",
                    str(root),
                    "--stage2-search-episodes",
                    "170",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CONSTRAINT_MISMATCH", result.stdout + result.stderr)
        self.assertIn("stage2_stability_multiplier", result.stdout + result.stderr)

    def test_layerwise_resume_rejects_different_policy_network_arm(self):
        with tempfile.TemporaryDirectory(prefix="stage2_network_mismatch_") as td:
            tmp = Path(td)
            root = tmp / "persistent"
            run_dir = (
                root / "rl" / "bert-base" / "mrpc"
                / "s1t0.001_s2t0.001_s2st2.0"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(
                json.dumps({
                    "stage1_accuracy_tolerance": 0.001,
                    "stage2_limit_tolerance": 0.001,
                    "stage2_stability_multiplier": 2.0,
                    "blb_v3_policy_network_variant": "shared_gtrxl_v1",
                }),
                encoding="utf-8",
            )
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            result = subprocess.run(
                [
                    "bash",
                    "llama_7B_LayerImportance.sh",
                    "run",
                    "rl",
                    "--preset",
                    "mrpc-blb-stage2-rl",
                    "--mode",
                    "stage2-only",
                    "--persistent-root",
                    str(root),
                    "--stage2-search-episodes",
                    "170",
                    "--blb-v3-policy-network-variant",
                    "separate_critic_gtrxl_v1",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        output = result.stdout + result.stderr
        self.assertIn("CONSTRAINT_MISMATCH", output)
        self.assertIn("blb_v3_policy_network_variant", output)

    def test_stage2_public_decision_and_reward_options_reject_unknown_values(self):
        for option, value in (
            ("--blb-v3-decision-granularity", "token"),
            ("--blb-v3-reward-design", "legacy_unknown"),
            ("--blb-v3-policy-network-variant", "larger_maybe"),
        ):
            with self.subTest(option=option):
                result = subprocess.run(
                    [
                        "bash",
                        "llama_7B_LayerImportance.sh",
                        "run",
                        "rl",
                        "--preset",
                        "mrpc-blb-stage2-rl",
                        option,
                        value,
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)

    def test_stage2_launcher_defaults_fixed_config_to_all4(self):
        with tempfile.TemporaryDirectory(prefix="stage2_all4_launcher_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = ""

            result = subprocess.run(
                [
                    "bash",
                    "llama_7B_LayerImportance.sh",
                    "run",
                    "rl",
                    "--preset",
                    "mrpc-blb-stage2-rl",
                    "--mode",
                    "stage2-only",
                    "--persistent-root",
                    str(tmp / "persistent"),
                    "--stage2-search-episodes",
                    "170",
                    "--fresh",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            for _ in range(50):
                if capture.is_file():
                    break
                import time

                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke python")
            raw_argv = capture.read_bytes().split(b"\0")
            self.assertEqual(raw_argv[-1], b"")
            argv = [part.decode("utf-8") for part in raw_argv[:-1]]

        self.assertEqual(
            argv[argv.index("--stage2_fixed_config_source") + 1],
            "all4",
        )
        self.assertEqual(
            argv[argv.index("--stage2_fixed_config_path") + 1],
            "",
        )

    def test_stage2_launcher_auto_forwards_visible_gpus(self):
        with tempfile.TemporaryDirectory(prefix="stage2_gpu_audit_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            python3_capture = tmp / "python3_invocations.txt"
            fake_python3 = fakebin / "python3"
            fake_python3.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\n' "$*" >> {str(python3_capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python3.chmod(0o755)
            fake_nvidia_smi = fakebin / "nvidia-smi"
            fake_nvidia_smi.write_text(
                "#!/usr/bin/env bash\nexit 0\n",
                encoding="utf-8",
            )
            fake_nvidia_smi.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = "0,1"

            cmd = [
                "bash",
                "llama_7B_LayerImportance.sh",
                "run",
                "rl",
                "--preset",
                "mrpc-blb-stage2-rl",
                "--mode",
                "stage2-only",
                "--persistent-root",
                str(tmp / "persistent"),
                "--stage2-search-episodes",
                "170",
                "--stage2-fixed-config-source",
                "json",
                "--stage2-fixed-config",
                "glue_final_configs_best_ppo.json",
                "--fresh",
            ]
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + "\n" + result.stderr,
            )
            for _ in range(50):
                if (
                    python3_capture.is_file()
                    and "scripts/elastic_gpu_supervisor.py"
                    in python3_capture.read_text(encoding="utf-8")
                ):
                    break
                import time

                time.sleep(0.1)
            self.assertTrue(
                python3_capture.is_file()
                and "scripts/elastic_gpu_supervisor.py"
                in python3_capture.read_text(encoding="utf-8"),
                msg="launcher did not invoke the intercepted elastic supervisor",
            )
            combined = result.stdout + "\n" + result.stderr
            self.assertNotIn("[gpu-audit][WARN]", combined)
            self.assertIn("scripts/elastic_gpu_supervisor.py", combined)
            self.assertIn("--blb_v3_reward_devices auto", combined)

    def test_stage2_rl_launches_inside_constraint_persistent_dir(self):
        with tempfile.TemporaryDirectory(prefix="stage2_persist_launcher_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            persistent_root = tmp / "persistent"
            expected_output_dir = (
                persistent_root
                / "rl"
                / "bert-base"
                / "mrpc"
                / "s1t0.001_s2t0.001_s2st2.0"
            )

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["CAPTURE"] = str(capture)

            cmd = [
                "bash",
                "llama_7B_LayerImportance.sh",
                "run",
                "rl",
                "--preset",
                "mrpc-blb-stage2-rl",
                "--mode",
                "stage2-only",
                "--persistent-root",
                str(persistent_root),
                "--stage1-accuracy-tolerance",
                "0.001",
                "--stage2-limit-tolerance",
                "0.001",
                "--stage2-stability-tolerance",
                "3.0",
                "--stage2-search-episodes",
                "170",
                "--stage2-fixed-config-source",
                "json",
                "--stage2-fixed-config",
                "glue_final_configs_best_ppo.json",
                "--fresh",
            ]
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + "\n" + result.stderr,
            )
            for _ in range(50):
                if capture.is_file():
                    break
                import time
                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke python")
            argv = [
                part.decode("utf-8")
                for part in capture.read_bytes().split(b"\0")
                if part
            ]
            latest_run_dir = expected_output_dir.parent / "LATEST_RUN_DIR"
            self.assertTrue(latest_run_dir.is_file())
            self.assertEqual(latest_run_dir.read_text(encoding="utf-8").strip(), str(expected_output_dir))

        self.assertIn("--output_dir", argv)
        self.assertEqual(argv[argv.index("--output_dir") + 1], str(expected_output_dir))
        self.assertIn("--decoupled_layout", argv)
        self.assertEqual(argv[argv.index("--decoupled_layout") + 1], "false")

    def test_stage2_run_tag_creates_separate_persistent_slug(self):
        with tempfile.TemporaryDirectory(prefix="stage2_persist_tag_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            persistent_root = tmp / "persistent"
            expected_output_dir = (
                persistent_root
                / "rl"
                / "bert-base"
                / "mrpc"
                / "s1t0.001_s2t0.001_s2st2.0__gate_gN_20260625"
            )

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["CAPTURE"] = str(capture)

            cmd = [
                "bash",
                "llama_7B_LayerImportance.sh",
                "run",
                "rl",
                "--preset",
                "mrpc-blb-stage2-rl",
                "--mode",
                "stage2-only",
                "--persistent-root",
                str(persistent_root),
                "--run-tag",
                "gate_gN_20260625",
                "--stage1-accuracy-tolerance",
                "0.001",
                "--stage2-limit-tolerance",
                "0.001",
                "--stage2-stability-tolerance",
                "3.0",
                "--stage2-search-episodes",
                "170",
                "--stage2-fixed-config-source",
                "json",
                "--stage2-fixed-config",
                "glue_final_configs_best_ppo.json",
                "--fresh",
            ]
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + "\n" + result.stderr,
            )
            for _ in range(50):
                if capture.is_file():
                    break
                import time
                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke python")
            argv = [
                part.decode("utf-8")
                for part in capture.read_bytes().split(b"\0")
                if part
            ]

        self.assertIn("--output_dir", argv)
        self.assertEqual(argv[argv.index("--output_dir") + 1], str(expected_output_dir))

    def test_stage2_preset_defaults_to_current_formal_constraints(self):
        with tempfile.TemporaryDirectory(prefix="stage2_persist_preset_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            self._install_fake_flock(fakebin)
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_flock = fakebin / "flock"
            fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_flock.chmod(0o755)

            persistent_root = tmp / "persistent"
            expected_output_dir = (
                persistent_root
                / "rl"
                / "bert-base"
                / "mrpc"
                / "s1t0.001_s2t0.001_s2st2.0"
            )

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = ""

            cmd = [
                "bash",
                "llama_7B_LayerImportance.sh",
                "run",
                "rl",
                "--preset",
                "mrpc-blb-stage2-rl",
                "--persistent-root",
                str(persistent_root),
                "--stage2-search-episodes",
                "0",
                "--stage2-fixed-config-source",
                "json",
                "--stage2-fixed-config",
                "glue_final_configs_best_ppo.json",
                "--fresh",
            ]
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + "\n" + result.stderr,
            )
            for _ in range(50):
                if capture.is_file():
                    break
                import time
                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke python")
            argv = [
                part.decode("utf-8")
                for part in capture.read_bytes().split(b"\0")
                if part
            ]

        self.assertEqual(argv[argv.index("--output_dir") + 1], str(expected_output_dir))
        self.assertEqual(argv[argv.index("--stage1_accuracy_tolerance") + 1], "0.001")
        self.assertEqual(argv[argv.index("--stage2_limit_tolerance") + 1], "0.001")
        self.assertEqual(argv[argv.index("--stage2_stability_tolerance") + 1], "1.2")
        self.assertEqual(argv[argv.index("--stage2_stability_multiplier") + 1], "2.0")
        self.assertEqual(argv[argv.index("--stage2_rl_episodes") + 1], "0")

        preset = (REPO_ROOT / "presets" / "mrpc-blb-stage2-rl.conf").read_text(
            encoding="utf-8",
        )
        self.assertIn("--stage2-search-episodes 150000", preset)
        self.assertIn("--blb-v3-min-convergence-episodes 90000", preset)
        self.assertIn("--blb-v3-convergence-patience-updates 100", preset)

    def test_server_command_short_rl_runs_use_tagged_persistent_dirs(self):
        source = (REPO_ROOT / "SERVER_COMMAND.md").read_text(encoding="utf-8")
        self.assertIn("--run-tag \"ab_${tag}_${TS}\"", source)
        self.assertIn("--run-tag \"gate_${tag}_${TS}\"", source)
        self.assertIn("${tag}_persistent_verify.txt", source)
        self.assertIn("--min-episodes \"$EPISODES_AB\"", source)

    def test_stage2_training_loop_flushes_status_on_ppo_update(self):
        source = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text(
            encoding="utf-8",
        )
        callback_start = source.index("def _ppo_update_end_callback(")
        callback_end = source.index("    t_start = time.time()", callback_start)
        callback_source = source[callback_start:callback_end]
        self.assertIn("status.update_after_ppo_update", callback_source)

    def test_stage2_training_loop_refreshes_live_curves_after_ppo_update(self):
        source = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text(
            encoding="utf-8",
        )
        callback_start = source.index("def _ppo_update_end_callback(")
        callback_end = source.index("    t_start = time.time()", callback_start)
        callback_source = source[callback_start:callback_end]
        self.assertIn("_write_live_training_curves", callback_source)
        self.assertIn("live_curve_refresh", source)
        self.assertIn("write_training_curves(", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
