import tempfile
import unittest
from pathlib import Path

from rfr.search.common import local_optimum as rlo


class RLLocalOptimumStreamingInputTest(unittest.TestCase):
    def test_attribute_collapse_uses_optional_series_once(self):
        priority = [3] * 80 + [1] * 80
        fusion_count = (1.0 if i < 80 else 8.0 for i in range(160))
        margin = (0.2 if i < 80 else -0.3 for i in range(160))

        lines = rlo.attribute_collapse(
            priority=priority,
            fusion_count=fusion_count,
            worst_signed_margin=margin,
        )
        text = "\n".join(lines)

        self.assertIn("fusion 均值", text)
        self.assertIn("margin(mu) 均值", text)
        self.assertIn("末段 8.00", text)
        self.assertIn("末段 -0.3000", text)
        self.assertNotIn("nan", text.lower())

    def test_write_report_reuses_materialized_episode_returns_for_detection(self):
        returns = (1.0 for _ in range(120))
        entropies = (0.01 for _ in range(120))
        best_scores = (1.0 for _ in range(120))

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "local_optimum.txt"
            result = rlo.write_local_optimum_report(
                str(path),
                episode_returns=returns,
                episode_entropies=entropies,
                best_score_history=best_scores,
                title="Stage-X",
            )

            self.assertEqual(result, str(path))
            text = path.read_text(encoding="utf-8")

        self.assertIn("完成回合数: 120", text)
        self.assertIn("recent_reward_mean: 1.0", text)
        self.assertNotIn("样本不足", text)


if __name__ == "__main__":
    unittest.main()
