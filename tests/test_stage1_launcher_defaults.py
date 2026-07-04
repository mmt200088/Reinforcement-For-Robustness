from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "llama_7B_LayerImportance.sh"


class Stage1LauncherDefaultTest(unittest.TestCase):
    def test_stage1_only_rl_uses_high_throughput_batch_default_when_unspecified(self):
        source = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn('STAGE1_RL_DEFAULT_BATCH_SIZE="128"', source)
        self.assertIn(
            '[ "$SEARCH_ALGORITHM" = "rl" ] && [ "$RUN_MODE" = "stage1-only" ] && [ "$S_BATCH_SIZE" = "false" ]',
            source,
        )
        self.assertIn('BATCH_SIZE="$STAGE1_RL_DEFAULT_BATCH_SIZE"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
