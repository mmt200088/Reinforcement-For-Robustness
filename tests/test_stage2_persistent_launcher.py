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
    def test_python_public_decision_fields_reach_evaluator_constructor(self):
        tune_tree = ast.parse((REPO_ROOT / "rl_tune.py").read_text(encoding="utf-8"))
        train_fn = next(
            node for node in tune_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        train_args = {arg.arg: ast.unparse(arg.annotation) for arg in train_fn.args.args}
        self.assertEqual(train_args["blb_v3_decision_granularity"], "str")
        self.assertEqual(train_args["blb_v3_reward_design"], "str")
        evaluator_call = next(
            node for node in ast.walk(train_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "LayerImportanceEvaluator"
        )
        forwarded = {keyword.arg for keyword in evaluator_call.keywords}
        self.assertIn("blb_v3_decision_granularity", forwarded)
        self.assertIn("blb_v3_reward_design", forwarded)

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

    def _capture_persistent_slug_and_metadata(self, extra_args):
        with tempfile.TemporaryDirectory(prefix="stage2_constraint_identity_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
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

        explicit_argv = self._capture_stage2_launcher_argv([
            "--blb-v3-decision-granularity",
            "layer",
            "--blb-v3-reward-design",
            "robust_constrained",
        ])
        self.assertEqual(
            explicit_argv[explicit_argv.index("--blb_v3_decision_granularity") + 1],
            "layer",
        )
        self.assertEqual(
            explicit_argv[explicit_argv.index("--blb_v3_reward_design") + 1],
            "robust_constrained",
        )

    def test_active_layerwise_robust_defaults_reach_python(self):
        argv = self._capture_stage2_launcher_argv([])
        expected = {
            "--stage2_limit_tolerance": "0.001",
            "--stage2_stability_multiplier": "2.0",
            "--stage2_k_trials": "5",
            "--blb_v3_baseline_groups": "5",
            "--blb_v3_baseline_trials_per_group": "5",
            "--blb_v3_constraint_bootstrap_samples": "4096",
            "--blb_v3_online_constraint_probability": "0.50",
            "--blb_v3_promotion_constraint_probability": "0.80",
            "--blb_v3_final_constraint_probability": "0.95",
            "--blb_v3_promotion_validation_trials": "25",
            "--blb_v3_final_selection_validation_trials": "25",
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
            root = Path(td) / "persistent"
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
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CONSTRAINT_MISMATCH", result.stdout + result.stderr)
        self.assertIn("stage2_stability_multiplier", result.stdout + result.stderr)

    def test_stage2_public_decision_and_reward_options_reject_unknown_values(self):
        for option, value in (
            ("--blb-v3-decision-granularity", "token"),
            ("--blb-v3-reward-design", "legacy_unknown"),
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

    def test_stage2_launcher_warns_when_visible_gpus_are_not_forwarded_to_gpu_flags(self):
        with tempfile.TemporaryDirectory(prefix="stage2_gpu_audit_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
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
            combined = result.stdout + "\n" + result.stderr
            self.assertIn("[gpu-audit][WARN]", combined)
            self.assertIn("--blb-v3-reward-devices 0,1", combined)

    def test_stage2_rl_launches_inside_constraint_persistent_dir(self):
        with tempfile.TemporaryDirectory(prefix="stage2_persist_launcher_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
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
