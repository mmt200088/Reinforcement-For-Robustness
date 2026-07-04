import unittest

from tests.source_inspection_utils import source_text


class BLBDiagnosticsStaticTest(unittest.TestCase):
    def test_ppo_warning_means_use_shared_streaming_helper(self):
        text = source_text("blb_stage2_rl/diagnostics.py")
        self.assertIn("from stats_utils import mean_or_default", text)
        self.assertIn("mean_or_default(u.entropy for u in self._ppo_history[-3:])", text)
        self.assertIn(
            "mean_or_default(u.clip_fraction for u in self._ppo_history[-3:])",
            text,
        )
        self.assertNotIn(
            "np.mean([u.entropy for u in self._ppo_history[-3:]])",
            text,
        )
        self.assertNotIn(
            "np.mean([u.clip_fraction for u in self._ppo_history[-3:]])",
            text,
        )


if __name__ == "__main__":
    unittest.main()
