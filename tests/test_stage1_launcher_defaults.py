from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_search.sh"


class Stage1LauncherDefaultTest(unittest.TestCase):
    def test_stage1_only_rl_defaults_to_elastic_auto_devices(self):
        source = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn('ELASTIC_GPU_MODE="auto"', source)
        self.assertIn("--elastic-gpu-mode", source)
        self.assertIn("rfr.search.runtime.supervisor", source)
        self.assertIn(
            '[ "$MODE" = "stage1-only" ] && [ -z "$STAGE1_RL_DEVICES" ]',
            source,
        )
        self.assertIn('STAGE1_RL_DEVICES="auto"', source)

    def test_stage1_only_rl_uses_high_throughput_batch_default_when_unspecified(self):
        source = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn('BATCH_SIZE="128"', source)
        for preset in (ROOT / "configs/presets").glob("*-stage1-rl.conf"):
            self.assertIn("--batch-size 512", preset.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
