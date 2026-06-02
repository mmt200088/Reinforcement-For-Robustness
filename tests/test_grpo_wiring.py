"""Torch-free checks that GRPO cannot be selected from project entrypoints.

The historical GRPO math/helpers may remain for auditability, but normal CLI,
evaluator, and runner configuration must reject GRPO after the MRPC
generalization study showed it is not suitable for this project.
"""
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


class GrpoEntrypointsDisabled(unittest.TestCase):
    def test_launcher_rejects_grpo_and_no_longer_swaps_chapter(self):
        text = _read("llama_7B_LayerImportance.sh")
        self.assertIn("--rl-algo)", text)
        self.assertIn("--grpo-kl-beta)", text)
        self.assertIn("GRPO 已在本项目中永久禁用", text)
        self.assertIn('case "$RL_ALGO" in ppo)', text)
        self.assertNotIn('PERSISTENT_ROOT="GRPO Chapter/persistent"', text)
        self.assertNotIn('RUNS_ROOT="GRPO Chapter/runs"', text)
        self.assertIn("--rl_algo", text)
        self.assertIn('RL_ALGO="ppo"', text)
        self.assertNotIn("--grpo_kl_beta", text)

    def test_python_entrypoints_reject_non_ppo(self):
        for rel in (
            "rl_tune.py",
            "layer_importance_evaluator.py",
            "blb_stage2_rl/runner.py",
            "blb_stage2_rl/sequential_runner.py",
        ):
            text = _read(rel)
            self.assertIn("GRPO has been disabled for this project", text, rel)
        self.assertIn('if self.rl_algo != "ppo"', _read("layer_importance_evaluator.py"))
        self.assertIn("def __post_init__(self)", _read("blb_stage2_rl/runner.py"))
        self.assertIn("def __post_init__(self)", _read("blb_stage2_rl/sequential_runner.py"))

    def test_runtime_dispatch_is_ppo_only(self):
        evaluator = _read("layer_importance_evaluator.py")
        sequential = _read("blb_stage2_rl/sequential_runner.py")
        sequential_policy = _read("blb_stage2_rl/sequential_policy.py")
        self.assertNotIn('self.rl_algo == "grpo"', evaluator)
        self.assertNotIn("stage1_grpo_reference.pt", evaluator)
        self.assertNotIn("sequential_grpo_update(", sequential)
        self.assertIn("self.ppo_update_gtrxl(", evaluator)
        self.assertIn("sequential_ppo_update(", sequential)
        self.assertIn("Use ppo_update_gtrxl instead", evaluator)
        self.assertIn("Use sequential_ppo_update instead", sequential_policy)


if __name__ == "__main__":
    unittest.main()
