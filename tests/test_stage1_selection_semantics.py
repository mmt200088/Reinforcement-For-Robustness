from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "layer_importance_evaluator.py"


def _source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def _method_region(name: str) -> str:
    text = _source_text()
    start = text.index(f"    def {name}")
    next_start = text.find("\n    def ", start + 1)
    if next_start == -1:
        next_start = len(text)
    return text[start:next_start]


class Stage1SelectionSemanticsTest(unittest.TestCase):
    def test_stage1_rl_masks_degree_zero_relu_action(self):
        text = _source_text()
        self.assertIn(
            "STAGE1_GELU_ACTION_MASK = np.array([True, True, True, False], dtype=bool)",
            text,
        )

        region = _method_region("get_gelu_action_mask")
        self.assertIn("return STAGE1_GELU_ACTION_MASK.copy()", region)
        self.assertNotIn("return np.array([True, True, True, True]", region)

    def test_stage1_final_selection_keeps_raw_reward_best_config(self):
        text = _source_text()
        marker = "# ---- 局部最优检测：写入 pruning_search_log.txt ----"
        start = text.index(marker)
        end = text.index("# ---------------------------------------------------------\n        # Plot:", start)
        final_selection = text[start:end]
        self.assertNotIn("global_best_config is not None", final_selection)
        self.assertNotIn("search_best_config is not None", final_selection)
        self.assertIn("_select_stage1_reward_best_config", final_selection)

    def test_confirmed_candidate_priority_is_metric_then_loss_then_cost(self):
        region = _method_region("_is_better_confirmed_candidate")
        metric_pos = region.index("cand_metric_sum")
        loss_pos = region.index("cand_loss")
        cost_pos = region.index("cand_cost")
        self.assertLess(metric_pos, loss_pos)
        self.assertLess(loss_pos, cost_pos)

    def test_stage1_training_loop_does_not_reconfirm_window_candidates(self):
        text = _source_text()
        self.assertNotIn("confirm_stage1_candidate(", text)
        self.assertNotIn("候选确认", text)


if __name__ == "__main__":
    unittest.main()
