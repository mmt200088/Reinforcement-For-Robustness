import unittest
from types import SimpleNamespace

import torch


class BLBActionMaskTests(unittest.TestCase):
    def test_baseline_only_mask_keeps_action_length_and_forces_baseline(self):
        from blb_stage2_rl.action_mask import build_action_mask
        from blb_stage2_rl.action_space import make_all_max_action_vector

        baseline = make_all_max_action_vector(num_layers=2)
        mask = build_action_mask(num_layers=2, mode="baseline_only")

        self.assertEqual(len(mask), len(baseline))
        for slot_mask, baseline_idx in zip(mask, baseline.tolist()):
            self.assertTrue(slot_mask[int(baseline_idx)])
            self.assertEqual(int(slot_mask.sum()), 1)

    def test_policy_sample_and_evaluate_share_masked_logprob(self):
        from blb_stage2_rl.action_mask import build_action_mask
        from blb_stage2_rl.action_space import layer_dims, make_all_max_action_vector
        from blb_stage2_rl.policy import BLBStage2Policy

        baseline = make_all_max_action_vector(num_layers=1)
        mask = build_action_mask(num_layers=1, mode="baseline_only")
        policy = BLBStage2Policy(
            state_dim=3,
            num_layers=1,
            per_layer_dims=layer_dims(),
            first_input_levels=5,
            d_hidden=8,
            d_layer_emb=4,
        )

        state = torch.zeros(1, 3)
        action, sampled_logprob, _value = policy.sample_action(
            state,
            deterministic=False,
            action_mask=mask,
        )
        eval_logprob, _entropy, _value = policy.evaluate_action(
            state,
            action,
            action_mask=mask,
        )

        self.assertEqual(action.squeeze(0).tolist(), baseline.tolist())
        self.assertTrue(torch.allclose(sampled_logprob, eval_logprob))

    def test_near_baseline_keeps_inactive_slots_baseline_only(self):
        from blb_stage2_rl.action_mask import build_action_mask
        from blb_stage2_rl.action_space import (
            describe_action_vector,
            load_max_sfs,
            make_all_max_action_vector,
        )

        baseline = make_all_max_action_vector(num_layers=1)
        mask = build_action_mask(
            num_layers=1,
            mode="near_baseline",
            gelu_degree=[1],
            attn_degree=[2],
            profile="mrpc",
        )
        desc = describe_action_vector(
            baseline,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=1,
            gelu_degree=[1],
            attn_degree=[2],
            profile="mrpc",
        )

        inactive = [r for r in desc["records"] if not r.get("effective", True)]
        self.assertTrue(inactive)
        for record in inactive:
            slot_mask = mask[int(record["global_index"])]
            self.assertEqual(int(slot_mask.sum()), 1)
            self.assertTrue(slot_mask[int(baseline[int(record["global_index"])])])

    def test_action_bias_raises_baseline_sampling_probability(self):
        from blb_stage2_rl.action_space import layer_dims, make_all_max_action_vector
        from blb_stage2_rl.policy import BLBStage2Policy

        torch.manual_seed(1234)
        baseline = make_all_max_action_vector(num_layers=1)
        dims = layer_dims() + [5]
        policy = BLBStage2Policy(
            state_dim=3,
            num_layers=1,
            per_layer_dims=layer_dims(),
            first_input_levels=5,
            d_hidden=8,
            d_layer_emb=4,
        )
        state = torch.zeros(200, 3)
        bias = []
        for slot_idx, dim in enumerate(dims):
            row = [0.0] * int(dim)
            row[int(baseline[slot_idx])] = 7.0
            bias.append(row)

        unbiased, _lp, _value = policy.sample_action(state, action_bias=None)
        biased, _lp, _value = policy.sample_action(state, action_bias=bias)

        baseline_t = torch.as_tensor(baseline)
        unbiased_rate = (unbiased == baseline_t).float().mean().item()
        biased_rate = (biased == baseline_t).float().mean().item()
        self.assertGreater(biased_rate, unbiased_rate + 0.35)

    def test_evaluate_action_accepts_mask_and_bias_for_legal_action(self):
        from blb_stage2_rl.action_mask import build_action_mask
        from blb_stage2_rl.action_space import layer_dims, make_all_max_action_vector
        from blb_stage2_rl.policy import BLBStage2Policy

        baseline = make_all_max_action_vector(num_layers=1)
        mask = build_action_mask(num_layers=1, mode="baseline_only")
        dims = layer_dims() + [5]
        bias = []
        for slot_idx, dim in enumerate(dims):
            row = [0.0] * int(dim)
            row[int(baseline[slot_idx])] = 5.5
            bias.append(row)
        policy = BLBStage2Policy(
            state_dim=3,
            num_layers=1,
            per_layer_dims=layer_dims(),
            first_input_levels=5,
            d_hidden=8,
            d_layer_emb=4,
        )

        state = torch.zeros(1, 3)
        action = torch.as_tensor(baseline, dtype=torch.long).unsqueeze(0)
        log_prob, entropy, value = policy.evaluate_action(
            state,
            action,
            action_mask=mask,
            action_bias=bias,
        )

        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(torch.isfinite(value).all())

    def test_per_dim_entropy_can_report_masked_entropy(self):
        from blb_stage2_rl.action_mask import build_action_mask
        from blb_stage2_rl.action_space import layer_dims
        from blb_stage2_rl.policy import BLBStage2Policy

        mask = build_action_mask(num_layers=1, mode="baseline_only")
        policy = BLBStage2Policy(
            state_dim=3,
            num_layers=1,
            per_layer_dims=layer_dims(),
            first_input_levels=5,
            d_hidden=8,
            d_layer_emb=4,
        )
        state = torch.zeros(2, 3)

        raw_entropy = policy.per_dim_entropy(state)
        masked_entropy = policy.per_dim_entropy(state, action_mask=mask)

        self.assertEqual(tuple(raw_entropy.shape), tuple(masked_entropy.shape))
        self.assertGreater(float(raw_entropy.mean().item()), 0.1)
        self.assertLess(float(masked_entropy.abs().max().item()), 1e-6)

    def test_load_mask_rejects_inactive_slot_opened_to_nonbaseline(self):
        import json
        import tempfile
        from pathlib import Path

        from blb_stage2_rl.action_mask import load_action_mask_file
        from blb_stage2_rl.action_space import (
            action_dims_for_config,
            describe_action_vector,
            load_max_sfs,
            make_all_max_action_vector,
        )

        baseline = make_all_max_action_vector(num_layers=1)
        dims = action_dims_for_config(num_layers=1)
        desc = describe_action_vector(
            baseline,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=1,
            gelu_degree=[4],
            attn_degree=[4],
            profile="mrpc",
        )
        records = sorted(desc["records"], key=lambda r: int(r["global_index"]))
        inactive = next(r for r in records if not r.get("effective", True))
        slots = []
        for idx, dim in enumerate(dims):
            baseline_idx = int(baseline[idx])
            allowed = [baseline_idx]
            if idx == int(inactive["global_index"]):
                extra = 0 if baseline_idx != 0 else min(1, int(dim) - 1)
                allowed = sorted(set([baseline_idx, extra]))
            slots.append({
                "global_index": idx,
                "baseline_index": baseline_idx,
                "allowed_indices": allowed,
            })

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mask.json"
            path.write_text(json.dumps({
                "schema": "blb_action_mask_v1",
                "action_width": len(dims),
                "baseline_action": baseline.tolist(),
                "slots": slots,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ineffective"):
                load_action_mask_file(
                    path,
                    expected_width=len(dims),
                    baseline_action=baseline.tolist(),
                    action_dims=dims,
                    slot_records=records,
                )

    def test_ppo_update_passes_mask_and_bias_to_evaluate_action(self):
        from blb_stage2_rl.action_mask import build_action_mask
        from blb_stage2_rl.action_space import layer_dims, make_all_max_action_vector
        from blb_stage2_rl.policy import BLBStage2Policy, PPOConfig, RolloutBuffer, ppo_update

        class RecordingPolicy(BLBStage2Policy):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.seen_mask = None
                self.seen_bias = None

            def evaluate_action(self, state, action, action_mask=None, action_bias=None):
                self.seen_mask = action_mask
                self.seen_bias = action_bias
                return super().evaluate_action(
                    state,
                    action,
                    action_mask=action_mask,
                    action_bias=action_bias,
                )

        baseline = make_all_max_action_vector(num_layers=1)
        mask = build_action_mask(num_layers=1, mode="baseline_only")
        bias = [[0.0] * int(dim) for dim in layer_dims() + [5]]
        for slot_idx, baseline_idx in enumerate(baseline.tolist()):
            bias[slot_idx][int(baseline_idx)] = 5.5

        policy = RecordingPolicy(
            state_dim=3,
            num_layers=1,
            per_layer_dims=layer_dims(),
            first_input_levels=5,
            d_hidden=8,
            d_layer_emb=4,
        )
        buffer = RolloutBuffer()
        buffer.add(
            state=torch.zeros(3).numpy(),
            action=baseline,
            log_prob=0.0,
            reward=1.0,
            value=0.0,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        ppo_update(
            policy,
            optimizer,
            buffer,
            PPOConfig(n_epochs=1, minibatch_size=1),
            torch.device("cpu"),
            action_mask=mask,
            action_bias=bias,
        )

        self.assertIs(policy.seen_mask, mask)
        self.assertIs(policy.seen_bias, bias)

    def test_runner_train_config_reads_action_mask_fields(self):
        from blb_stage2_rl.runner import BLBStage2RLRunner

        ev = SimpleNamespace(
            stage2_rl_episodes=240,
            stage2_ppo_lr_initial=1e-4,
            dataset_key="mrpc",
            stage2_k_trials=1,
            blb_v3_action_mask_enabled="true",
            blb_v3_action_mask_mode="from_file",
            blb_v3_action_mask_file="reports/mask.json",
            blb_v3_action_mask_baseline_logit_bonus="2.5",
            blb_v3_action_mask_source="phase1_f0_scan",
        )

        cfg = BLBStage2RLRunner(ev)._build_train_config_from_evaluator(ev)

        self.assertTrue(cfg.action_mask_enabled)
        self.assertEqual(cfg.action_mask_mode, "from_file")
        self.assertEqual(cfg.action_mask_file, "reports/mask.json")
        self.assertEqual(cfg.action_mask_baseline_logit_bonus, 2.5)
        self.assertEqual(cfg.action_mask_source, "phase1_f0_scan")
