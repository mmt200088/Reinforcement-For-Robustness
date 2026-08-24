from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "llama_7B_LayerImportance.sh"


class Stage2PersistentLauncherTest(unittest.TestCase):
    @staticmethod
    def _fake_bin(root: Path, capture: Path) -> Path:
        fake_bin = root / "fakebin"
        fake_bin.mkdir()
        (fake_bin / "flock").write_text(
            "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
        )
        (fake_bin / "flock").chmod(0o755)
        (fake_bin / "python").write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                tmp={str(capture)!r}.tmp.$$
                printf '%s\\0' "$@" > "$tmp"
                mv "$tmp" {str(capture)!r}
                exit 0
                """
            ),
            encoding="utf-8",
        )
        (fake_bin / "python").chmod(0o755)
        return fake_bin

    def _capture(
        self,
        algorithm: str,
        *extra: str,
        mode: str | None = None,
    ) -> tuple[list[str], Path, dict]:
        with tempfile.TemporaryDirectory(prefix="production_launcher_") as td:
            root = Path(td)
            capture = root / "argv.nul"
            fake_bin = self._fake_bin(root, capture)
            persistent_root = root / "persistent"
            command = [
                "bash",
                str(LAUNCHER),
                "run",
                algorithm,
                "--persistent-root",
                str(persistent_root),
                "--fresh",
                "--elastic-gpu-mode",
                "off",
            ]
            if mode is not None:
                command.extend(("--mode", mode))
            command.extend(extra)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
            result = subprocess.run(
                command,
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
                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke Python")
            argv = [
                item.decode("utf-8")
                for item in capture.read_bytes().split(b"\0")[:-1]
            ]
            output_dir = Path(argv[argv.index("--output_dir") + 1])
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            return argv, output_dir, metadata

    @staticmethod
    def _value(argv: list[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    def test_launcher_uses_one_training_entrypoint(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        for removed in (
            "rl_tune_general.py",
            "rl_tune_genetic.py",
            "rl_ga_compare_runner.py",
            "--stage2-rl-variant",
            "legacy_v2",
        ):
            self.assertNotIn(removed, source)
        self.assertIn("python rl_tune.py", source)
        for backend in ("bo_rf", "greedy", "coinn_ga"):
            self.assertIn(f'SEARCH_BACKEND="{backend}"', source)

    def test_stage1_and_stage2_use_the_expected_production_routes(self):
        stage1_argv, stage1_path, stage1_meta = self._capture(
            "rl", "--run-tag", "stage1-test", mode="stage1-only"
        )
        self.assertEqual(stage1_argv[0], "rl_tune.py")
        self.assertEqual(self._value(stage1_argv, "--skip_stage1_rl"), "false")
        self.assertEqual(self._value(stage1_argv, "--skip_noise_rl"), "true")
        self.assertEqual(self._value(stage1_argv, "--decoupled_layout"), "true")
        self.assertEqual(stage1_path.parts[-2:], ("stage1", "bert base mrpc"))
        self.assertEqual(stage1_meta["policy_network_variant"], "shared_gtrxl_small_v1")

        stage2_argv, stage2_path, stage2_meta = self._capture(
            "rl", "--run-tag", "stage2-test", mode="stage2-only"
        )
        self.assertEqual(self._value(stage2_argv, "--skip_stage1_rl"), "true")
        self.assertEqual(self._value(stage2_argv, "--skip_noise_rl"), "false")
        self.assertEqual(
            self._value(stage2_argv, "--stage2_fixed_config_source"), "all4"
        )
        self.assertEqual(stage2_path.parts[-4:-1], ("rl", "bert-base", "mrpc"))
        self.assertEqual(stage2_meta["stage2_stability_multiplier"], 2.0)

    def test_stage2_auto_mode_uses_elastic_supervisor_and_reward_auto(self):
        with tempfile.TemporaryDirectory(prefix="elastic_launcher_") as td:
            result = subprocess.run(
                [
                    "bash",
                    str(LAUNCHER),
                    "run",
                    "rl",
                    "--mode",
                    "stage2-only",
                    "--persistent-root",
                    str(Path(td) / "persistent"),
                    "--fresh",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("scripts/elastic_gpu_supervisor.py", result.stdout)
            self.assertIn("--blb_v3_reward_devices auto", result.stdout)

    def test_comparator_contracts_are_fixed_and_disjoint(self):
        expected = {
            "bo_rf": ("50000", "2000"),
            "greedy": ("2176782336", "100"),
            "coinn_ga": ("11464", "5"),
        }
        for algorithm, (budget, patience) in expected.items():
            with self.subTest(algorithm=algorithm):
                argv, path, metadata = self._capture(algorithm)
                self.assertEqual(
                    path.parts[-4:-1], (algorithm, "bert-base", "mrpc")
                )
                self.assertEqual(metadata["algorithm"], algorithm)
                self.assertEqual(self._value(argv, "--batch_size"), "16")
                self.assertEqual(
                    self._value(argv, "--stage2_inference_batch_size"), "64"
                )
                self.assertEqual(
                    self._value(argv, "--blb_v3_search_backend"), algorithm
                )
                self.assertEqual(
                    self._value(argv, "--blb_v3_search_evaluation_budget"),
                    budget,
                )
                self.assertEqual(
                    self._value(argv, "--blb_v3_search_patience_generations"),
                    patience,
                )
                self.assertEqual(
                    self._value(argv, "--stage2_fixed_config_source"),
                    "stage1_result",
                )
                self.assertEqual(
                    self._value(argv, "--blb_v3_final_selection_top_n"), "5"
                )

    def test_comparator_stage1_only_preserves_formal_budgets(self):
        expected = {
            "bo_rf": ("10000", "1000"),
            "greedy": ("2176782336", "100"),
            "coinn_ga": ("11464", "5"),
        }
        for algorithm, (budget, patience) in expected.items():
            with self.subTest(algorithm=algorithm):
                argv, _, _ = self._capture(
                    algorithm, "--comparator-stage1-only"
                )
                self.assertEqual(self._value(argv, "--skip_noise_rl"), "true")
                self.assertEqual(self._value(argv, "--skip_final_eval"), "true")
                self.assertEqual(
                    self._value(argv, "--blb_v3_search_evaluation_budget"),
                    budget,
                )
                self.assertEqual(
                    self._value(argv, "--blb_v3_search_patience_generations"),
                    patience,
                )

    def test_comparator_smoke_keeps_real_trials_but_skips_strict_eval(self):
        argv, _, _ = self._capture("bo_rf", "--comparator-smoke")
        self.assertEqual(
            self._value(argv, "--blb_v3_search_evaluation_budget"), "1"
        )
        self.assertEqual(
            self._value(argv, "--blb_v3_search_full_validation"), "false"
        )
        self.assertEqual(self._value(argv, "--stage2_k_trials"), "3")
        self.assertEqual(self._value(argv, "--skip_final_eval"), "true")

    def test_comparator_rejects_contract_overrides(self):
        for override in (
            ("--batch-size", "8"),
            ("--random-seed", "7"),
            ("--stage2-k-trials", "5"),
            ("--blb-v3-search-evaluation-budget", "1"),
        ):
            with self.subTest(override=override):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "run", "bo_rf", *override],
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("do not allow overriding", result.stderr)

    def test_removed_commands_and_options_are_rejected(self):
        cases = (
            ("general", "train"),
            ("compare",),
            ("run", "ga"),
            ("run", "rl", "--stage2-rl-variant", "blb_v3"),
            ("run", "rl", "--rl-algo", "ppo"),
        )
        for case in cases:
            with self.subTest(case=case):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), *case],
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
