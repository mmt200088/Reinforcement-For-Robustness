"""Source-text wiring checks for the PPO->GRPO swap (2026-05-31).

Torch-free: these read the source files as text and assert that the GRPO path is
threaded everywhere (update fns, dispatch, config fields, CLI flags, output-dir
swap) WITHOUT removing the PPO path. They run on a torch-free dev box; the GRPO
*math* is covered separately by tests/test_grpo_common.py, and behavioral RL
correctness is verified server-side.
"""
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


class GrpoHelperPresent(unittest.TestCase):
    def test_grpo_common_exports(self):
        text = _read("grpo_common.py")
        self.assertIn("def grpo_group_normalize(", text)
        self.assertIn("def grpo_per_step_advantages(", text)


class GrpoUpdateFnsParallelToPPO(unittest.TestCase):
    def test_stage2_has_both_updates(self):
        text = _read("blb_stage2_rl/sequential_policy.py")
        # PPO update preserved, GRPO update added alongside it.
        self.assertIn("def sequential_ppo_update(", text)
        self.assertIn("def sequential_grpo_update(", text)
        # GRPO uses the group-relative advantage (no GAE) + drops value loss.
        self.assertIn("def grpo_advantages(", text)

    def test_stage1_has_both_updates(self):
        text = _read("layer_importance_evaluator.py")
        self.assertIn("def ppo_update_gtrxl(", text)
        self.assertIn("def grpo_update_gtrxl(", text)


class DispatchOnRlAlgo(unittest.TestCase):
    def test_stage2_runner_dispatches(self):
        text = _read("blb_stage2_rl/sequential_runner.py")
        self.assertIn("sequential_grpo_update(", text)
        self.assertIn('rl_algo", "ppo")).lower() == "grpo"', text)
        # frozen reference snapshot + checkpoint round-trip
        self.assertIn("reference_policy", text)
        self.assertIn("grpo_reference_policy", text)

    def test_stage1_evaluator_dispatches(self):
        text = _read("layer_importance_evaluator.py")
        self.assertIn('self.rl_algo == "grpo"', text)
        self.assertIn("grpo_update_gtrxl(", text)
        self.assertIn("stage1_grpo_reference.pt", text)


class ConfigFieldsThreaded(unittest.TestCase):
    def test_config_fields(self):
        self.assertIn("rl_algo: str = \"ppo\"", _read("blb_stage2_rl/runner.py"))
        self.assertIn("grpo_kl_beta: float = 0.04", _read("blb_stage2_rl/runner.py"))
        self.assertIn("rl_algo: str = \"ppo\"", _read("blb_stage2_rl/sequential_runner.py"))

    def test_rl_tune_forwards_to_evaluator(self):
        text = _read("rl_tune.py")
        self.assertIn("rl_algo: str = \"ppo\"", text)
        self.assertIn("rl_algo=rl_algo", text)
        self.assertIn("grpo_kl_beta=grpo_kl_beta", text)


class LauncherFlagAndOutputDir(unittest.TestCase):
    def test_launcher_flag_and_chapter_swap(self):
        text = _read("llama_7B_LayerImportance.sh")
        self.assertIn("--rl-algo)", text)
        self.assertIn("--grpo-kl-beta)", text)
        # output tree swaps to GRPO Chapter (same structure) only for grpo
        self.assertIn('PERSISTENT_ROOT="GRPO Chapter/persistent"', text)
        self.assertIn('RUNS_ROOT="GRPO Chapter/runs"', text)
        # CMD forwards the flags to rl_tune
        self.assertIn("--rl_algo", text)
        self.assertIn("--grpo_kl_beta", text)

    def test_ppo_remains_default(self):
        text = _read("llama_7B_LayerImportance.sh")
        self.assertIn('RL_ALGO="ppo"', text)


if __name__ == "__main__":
    unittest.main()
