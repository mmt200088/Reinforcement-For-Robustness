import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class OptimizerCostImportTests(unittest.TestCase):
    def test_package_context_uses_package_action_space_with_legacy_path_present(self):
        code = """
import rfr.search.common.action_space as package_action_space
import rfr.preparation.rescale.optimizer_cost as optimizer_cost

print(int(optimizer_cost.ActionDecodeResult is package_action_space.ActionDecodeResult))
"""
        env = dict(os.environ)
        extra_paths = [
            str(REPO_ROOT),
            str(REPO_ROOT / "blb_stage2_rl"),
            str(REPO_ROOT / "configs/preparation/rescale"),
        ]
        if env.get("PYTHONPATH"):
            extra_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(extra_paths)

        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip().splitlines()[-1], "1")


if __name__ == "__main__":
    unittest.main()
