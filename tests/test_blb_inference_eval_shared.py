from __future__ import annotations

import importlib.util
import pathlib
import types
import unittest

import numpy as np


def _function_region_from_source(source: str, name: str) -> str:
    marker = f"def {name}("
    start = source.index(marker)
    next_def = source.find("\ndef ", start + len(marker))
    next_class = source.find("\nclass ", start + len(marker))
    candidates = [pos for pos in (next_def, next_class) if pos != -1]
    end = min(candidates) if candidates else len(source)
    return source[start:end]


class InstalledInferenceEvalSourceTest(unittest.TestCase):
    def test_full_eval_defers_per_batch_tensor_cpu_syncs(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        source = (repo / "blb_stage2_rl" / "inference_eval.py").read_text(
            encoding="utf-8"
        )
        region = _function_region_from_source(
            source, "run_installed_model_on_dataloader"
        )
        loop_region = region[
            region.index("with torch.inference_mode():"):region.index("avg_loss =")
        ]

        self.assertIn("loss_tensors.append(loss_t.detach()", loop_region)
        self.assertIn("normalize_logits_tensor_for_metrics", loop_region)
        self.assertNotIn("loss_t.detach().item()", loop_region)
        self.assertNotIn("normalize_logits_for_metrics(\n                logits_t", loop_region)

    def test_probe_trial_defers_per_batch_tensor_cpu_syncs(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        source = (repo / "blb_stage2_rl" / "inference_eval.py").read_text(
            encoding="utf-8"
        )
        region = _function_region_from_source(source, "run_installed_probe_trial")
        loop_region = region[
            region.index("with torch.inference_mode():"):region.index("finally:")
        ]

        self.assertIn("loss_tensors.append(loss_t.detach()", loop_region)
        self.assertIn("trial_pred_tensors.append", loop_region)
        self.assertNotIn("compute_probe_batch_metrics(", loop_region)
        self.assertNotIn(".cpu().numpy()", loop_region)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch unavailable")
class SharedInstalledInferenceEvalTest(unittest.TestCase):
    def test_probe_trial_and_full_eval_share_metric_semantics(self):
        import torch
        import torch.nn.functional as F

        from blb_stage2_rl.eval_metrics import metric_pair_for_dataset, sample_weighted_mean
        from blb_stage2_rl.inference_eval import (
            run_installed_model_on_dataloader,
            run_installed_probe_trial,
        )

        class LogitEchoModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None):
                del attention_mask, token_type_ids
                logits = input_ids.float()
                loss = F.cross_entropy(logits, labels.long()) if labels is not None else None
                return types.SimpleNamespace(loss=loss, logits=logits)

        class ProbeBatch:
            def __init__(self, logits, labels):
                self.input_ids = torch.tensor(logits, dtype=torch.float32)
                self.attention_mask = torch.ones(len(labels), 2, dtype=torch.long)
                self.labels = torch.tensor(labels, dtype=torch.long)
                self.token_type_ids = None

        batches = [
            ProbeBatch([[3.0, 0.0], [0.0, 3.0]], [0, 1]),
            ProbeBatch([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]], [1, 1, 0]),
        ]
        dataloader = [
            {
                "input_ids": b.input_ids,
                "attention_mask": b.attention_mask,
                "labels": b.labels,
            }
            for b in batches
        ]
        model = LogitEchoModel()
        expected_losses = [
            float(F.cross_entropy(b.input_ids, b.labels).item()) for b in batches
        ]
        expected_counts = [int(b.labels.numel()) for b in batches]
        all_logits = np.concatenate([b.input_ids.numpy() for b in batches], axis=0)
        all_labels = np.concatenate([b.labels.numpy() for b in batches], axis=0)
        expected_m1, expected_m2 = metric_pair_for_dataset("mrpc", all_labels, all_logits)
        expected_loss = sample_weighted_mean(expected_losses, expected_counts)

        probe_loss, probe_m1, probe_m2 = run_installed_probe_trial(
            model,
            batches,
            is_regression=False,
            metric_profile="mrpc",
        )
        full = run_installed_model_on_dataloader(
            model,
            dataloader,
            device=torch.device("cpu"),
            metric_profile="mrpc",
            use_train=False,
            split_name="validation_full",
        )

        self.assertAlmostEqual(probe_loss, expected_loss, places=7)
        self.assertAlmostEqual(full.loss, expected_loss, places=7)
        self.assertAlmostEqual(probe_m1, expected_m1, places=7)
        self.assertAlmostEqual(full.metric1, expected_m1, places=7)
        self.assertAlmostEqual(probe_m2, expected_m2, places=7)
        self.assertAlmostEqual(full.metric2, expected_m2, places=7)

    def test_full_eval_preserves_single_logit_binary_threshold_semantics(self):
        import torch
        import torch.nn.functional as F

        from blb_stage2_rl.inference_eval import run_installed_model_on_dataloader

        class SingleLogitModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, labels=None):
                del attention_mask
                logits = input_ids.float().reshape(-1)
                two_class_logits = torch.stack([1.0 - logits, logits], dim=1)
                loss = F.cross_entropy(two_class_logits, labels.long())
                return types.SimpleNamespace(loss=loss, logits=logits)

        dataloader = [
            {
                "input_ids": torch.tensor([0.2, 0.7], dtype=torch.float32),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "labels": torch.tensor([0, 1], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([0.8, 0.1], dtype=torch.float32),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "labels": torch.tensor([0, 0], dtype=torch.long),
            },
        ]
        result = run_installed_model_on_dataloader(
            SingleLogitModel(),
            dataloader,
            device=torch.device("cpu"),
            metric_profile="sst2",
            use_train=False,
            split_name="validation_full",
        )

        # Threshold predictions are [0, 1, 1, 0], not argmax over shape (B,1).
        self.assertAlmostEqual(result.metric1, 0.75)
        self.assertAlmostEqual(result.metric2, 0.75)


if __name__ == "__main__":
    unittest.main()
