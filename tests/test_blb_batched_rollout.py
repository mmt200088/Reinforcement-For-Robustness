"""Tests for the batched lockstep rollout sampler (2026-06-21).

The batched rollout (``collect_fusion_episodes_batched``) advances a worker's
episodes in lockstep with ONE GTrXL forward per step, then samples per row from
the (action-independent) logits with a batch-/device-invariant seeded sampler
(``BLBStage2SequentialPolicy.sample_from_logits``). The correctness guarantees:

  * the seeded sampler is a CORRECT categorical sampler (empirical freqs match
    softmax(logits)), and masked levels are never drawn;
  * it is BATCH-INVARIANT: a row's action/log_prob depends only on (its logits,
    its seed), so it is identical whether sampled in a batch of B or alone — this
    is what makes ``collect_fusion_episode`` (B=1) and the batched driver
    float-equivalent;
  * ``forward_and_mask`` batched over B rows matches the per-row forward within
    float (the residual ~1e-6 batched-GEMM difference the design accepts in
    place of bit-exact 1==N).

torch-required (the sampler is torch-based) -> these run on the GPU/CPU-torch
server. The end-to-end B=1-vs-B=W episode equivalence (needs the real model +
Rescale_optimizer) is the SERVER_COMMAND self-test, not a unit test.
"""

import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "blb_stage2_rl")):
    if p not in sys.path:
        sys.path.insert(0, p)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("torch") is not None,
    "torch required",
)
class BatchedRolloutSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        from blb_stage2_rl.sequential_policy import (
            BLBStage2SequentialPolicy,
            SequentialPolicyConfig,
        )
        cls.torch = torch
        H, S = 12, 2
        state_dim = 4 + H + 5 + 1 + H * S + H * 3
        cfg = SequentialPolicyConfig(
            state_dim=int(state_dim),
            max_step_dim=int(S),
            max_num_levels=6,
            horizon=int(H),
            block_count=5,
            num_layers=12,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            dropout=0.0,
            step_embed_dim=16,
            layer_embed_dim=16,
            block_embed_dim=8,
            prev_action_embed_dim=4,
            cont_proj_dim=32,
            actor_dim=32,
            critic_dim=32,
            default_prior_scale=0.0,
        )
        torch.manual_seed(0)
        cls.cfg = cfg
        cls.policy = BLBStage2SequentialPolicy(cfg).eval()
        cls.S = S
        cls.device = torch.device("cpu")

    def _random_logits(self, B):
        torch = self.torch
        S, L = self.S, int(self.cfg.max_num_levels)
        g = torch.Generator().manual_seed(123 + B)
        logits = torch.randn(B, S, L, generator=g)
        safe = logits.clone()
        slot_mask = torch.ones(B, S, dtype=torch.bool)
        return logits, safe, slot_mask

    def test_sampler_is_batch_invariant(self):
        # A row's draw depends ONLY on (its logits, its seed): sampling row e
        # inside a batch of B must equal sampling it alone with the same logits
        # slice + seed. This is the invariance the batched/serial equivalence
        # rests on (here the logits are IDENTICAL slices, so it must be EXACT).
        torch = self.torch
        B = 5
        logits, safe, slot_mask = self._random_logits(B)
        seeds = [1000 + i * 7 for i in range(B)]
        actions_B, logp_B = self.policy.sample_from_logits(logits, safe, slot_mask, seeds)
        for e in range(B):
            a1, lp1 = self.policy.sample_from_logits(
                logits[e:e + 1], safe[e:e + 1], slot_mask[e:e + 1], [seeds[e]],
            )
            self.assertTrue(torch.equal(actions_B[e:e + 1], a1),
                            f"row {e} action differs batched vs alone")
            self.assertTrue(torch.allclose(logp_B[e:e + 1], lp1, atol=0),
                            f"row {e} log_prob differs batched vs alone")

    def test_sampler_seed_determinism_and_variation(self):
        # Same seed -> same action; different seeds -> (generally) explore.
        logits, safe, slot_mask = self._random_logits(1)
        a_a, _ = self.policy.sample_from_logits(logits, safe, slot_mask, [42])
        a_b, _ = self.policy.sample_from_logits(logits, safe, slot_mask, [42])
        self.assertTrue(self.torch.equal(a_a, a_b), "same seed must reproduce the action")
        # actions is [1, S]; take row 0 -> a flat (a0, a1, ...) hashable tuple.
        seen = {
            tuple(self.policy.sample_from_logits(logits, safe, slot_mask, [s])[0][0].tolist())
            for s in range(200)
        }
        self.assertGreater(len(seen), 1, "different seeds should explore >1 action")

    def test_sampler_matches_categorical_distribution(self):
        # Empirical action frequencies over many seeds approximate softmax(logits).
        torch = self.torch
        L = int(self.cfg.max_num_levels)
        logits = torch.randn(1, 1, L)
        safe = logits.clone()
        slot_mask = torch.ones(1, 1, dtype=torch.bool)
        probs = torch.softmax(logits[0, 0], dim=-1)
        N = 8000
        counts = torch.zeros(L)
        for s in range(N):
            a, _ = self.policy.sample_from_logits(logits, safe, slot_mask, [s])
            counts[int(a[0, 0].item())] += 1
        emp = counts / N
        self.assertTrue(
            torch.allclose(emp, probs, atol=0.03),
            f"empirical {emp.tolist()} vs softmax {probs.tolist()}",
        )

    def test_masked_levels_never_sampled(self):
        # action_level_mask leaving only levels {1,3} legal -> only those drawn.
        torch = self.torch
        L = int(self.cfg.max_num_levels)
        logits = torch.zeros(1, 1, L)
        allowed = torch.zeros(1, 1, L, dtype=torch.bool)
        allowed[0, 0, 1] = True
        allowed[0, 0, 3] = True
        slot_mask = torch.ones(1, 1, dtype=torch.bool)
        levels = torch.full((1, 1), L, dtype=torch.long)
        masked = logits + self.policy._build_logit_mask(
            slot_mask, levels, L, action_level_mask=allowed,
        )
        safe = torch.where(torch.isfinite(masked).any(-1, keepdim=True), masked,
                           torch.zeros_like(masked))
        drawn = {
            int(self.policy.sample_from_logits(masked, safe, slot_mask, [s])[0][0, 0].item())
            for s in range(500)
        }
        self.assertTrue(drawn.issubset({1, 3}), f"sampled a masked level: {drawn}")
        self.assertEqual(drawn, {1, 3}, "both allowed levels should be reachable")

    def test_forward_and_mask_batched_vs_per_row(self):
        # The batched forward matches the per-row forward within float (the
        # ~1e-6 batched-GEMM residual the design accepts). Built states are valid
        # per-step observations (current_step one-hot at index 0).
        torch = self.torch
        import numpy as np
        B = 4
        S = self.S
        H = int(self.cfg.horizon)
        sd = int(self.cfg.state_dim)
        states = []
        rng = np.random.default_rng(7)
        for _ in range(B):
            st = np.zeros(sd, dtype=np.float32)
            st[0:4] = rng.random(4)
            st[4] = 1.0  # current_step = 0
            states.append(torch.from_numpy(st))
        obs_B = torch.stack(states, dim=0)
        slot_mask = torch.ones(B, S, dtype=torch.bool)
        levels = torch.full((B, S), int(self.cfg.max_num_levels), dtype=torch.long)
        lg_B, _, val_B = self.policy.forward_and_mask(
            obs_B, slot_mask, levels, truncate_to_current=False,
        )
        for e in range(B):
            lg_1, _, val_1 = self.policy.forward_and_mask(
                obs_B[e:e + 1], slot_mask[e:e + 1], levels[e:e + 1],
                truncate_to_current=False,
            )
            self.assertTrue(torch.allclose(lg_B[e:e + 1], lg_1, atol=1e-4),
                            f"row {e} logits drift > 1e-4 batched vs per-row")
            self.assertTrue(torch.allclose(val_B[e:e + 1], val_1, atol=1e-4),
                            f"row {e} value drift > 1e-4 batched vs per-row")


if __name__ == "__main__":
    unittest.main(verbosity=2)
