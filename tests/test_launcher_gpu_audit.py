from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts" / "launcher_gpu_audit.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("launcher_gpu_audit", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LauncherGpuAuditTest(unittest.TestCase):
    def test_stage1_warns_when_multiple_gpus_visible_without_stage1_devices(self):
        audit = _load_audit_module()

        warnings = audit.audit_launch(
            search_algorithm="rl",
            run_mode="stage1-only",
            stage2_rl_variant="blb_v3",
            cuda_visible_devices="0,1,2,3",
            stage1_rl_devices="",
            stage2_rl_devices="",
            blb_v3_reward_devices="",
            stage2_k_trials=4,
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn("--stage1-rl-devices 0,1,2,3", warnings[0])

    def test_stage1_is_quiet_when_stage1_devices_are_set(self):
        audit = _load_audit_module()

        warnings = audit.audit_launch(
            search_algorithm="rl",
            run_mode="stage1-only",
            stage2_rl_variant="blb_v3",
            cuda_visible_devices="0,1,2,3",
            stage1_rl_devices="0,1,2,3",
            stage2_rl_devices="",
            blb_v3_reward_devices="",
            stage2_k_trials=4,
        )

        self.assertEqual(warnings, [])

    def test_stage2_warns_when_multiple_gpus_visible_without_stage2_parallel_flags(self):
        audit = _load_audit_module()

        warnings = audit.audit_launch(
            search_algorithm="rl",
            run_mode="stage2-only",
            stage2_rl_variant="blb_v3",
            cuda_visible_devices="0,1,2,3",
            stage1_rl_devices="",
            stage2_rl_devices="",
            blb_v3_reward_devices="",
            stage2_k_trials=4,
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn("--stage2-rl-devices 0,1,2,3", warnings[0])
        self.assertIn("--blb-v3-reward-devices 0,1,2,3", warnings[0])

    def test_reward_devices_warn_when_k_trials_under_uses_visible_devices(self):
        audit = _load_audit_module()

        warnings = audit.audit_launch(
            search_algorithm="rl",
            run_mode="stage2-only",
            stage2_rl_variant="blb_v3",
            cuda_visible_devices="0,1,2,3",
            stage1_rl_devices="",
            stage2_rl_devices="",
            blb_v3_reward_devices="0,1,2,3",
            stage2_k_trials=2,
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn("--stage2-k-trials 2", warnings[0])
        self.assertIn("4 reward devices", warnings[0])

    def test_strict_cli_returns_nonzero_when_warning_exists(self):
        audit = _load_audit_module()

        rc = audit.main([
            "--search-algorithm", "rl",
            "--run-mode", "stage1-only",
            "--stage2-rl-variant", "blb_v3",
            "--cuda-visible-devices", "0,1",
            "--stage1-rl-devices", "",
            "--stage2-rl-devices", "",
            "--blb-v3-reward-devices", "",
            "--stage2-k-trials", "4",
            "--strict",
        ])

        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
