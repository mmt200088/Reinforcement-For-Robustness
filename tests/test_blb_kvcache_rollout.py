"""Correctness gate for the ADR/2026-06-19 KV-cache rollout speedup.

User dropped the strict 1==N byte-identical requirement to unlock real speedups,
but still requires NO QUALITY LOSS. Since the KV-cached incremental rollout forward
(`SequentialGTrXLBlock.forward_incremental`) REIMPLEMENTS `nn.MultiheadAttention`'s
math (to actually cache K/V — nn.MHA recomputes in_proj on key/value every call),
the safety net is THIS self-test: the per-token incremental forward must match the
original full-horizon `forward` within float (~1e-5). A wrong reimplementation
FAILS here (on the server, where torch is available) and never reaches a real run,
so quality cannot silently regress. Also covers cache lifecycle (fresh cache per
sequence => no cross-sequence contamination).

torch-required: skipped where torch isn't installed (e.g. the local torch-free CI),
run on the GPU server before the rollout-loop integration + the before/after
reward-curve quality check.
"""

import unittest

try:
    import numpy as np
    import torch  # noqa: F401

    from blb_stage2_rl.sequential_policy import (
        BLBStage2SequentialPolicy,
        SequentialGTrXLBlock,
        SequentialPolicyConfig,
    )
    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised on the torch-free path
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch + sequential_policy required")
class KVCacheBlockEquivalenceTest(unittest.TestCase):
    def _run_incremental(self, block, x):
        k_cache = v_cache = None
        outs = []
        for t in range(int(x.shape[1])):
            out_t, k_cache, v_cache = block.forward_incremental(
                x[:, t:t + 1, :], k_cache, v_cache,
            )
            outs.append(out_t)
        return torch.cat(outs, dim=1)

    def test_incremental_matches_full_forward(self):
        # The whole point: KV-cached incremental == full causal forward (within float).
        torch.manual_seed(0)
        d_model, n_heads, d_ff, H = 32, 4, 64, 7
        block = SequentialGTrXLBlock(d_model, n_heads, d_ff, dropout=0.0).eval()
        x = torch.randn(1, H, d_model)
        causal = torch.triu(torch.ones(H, H, dtype=torch.bool), diagonal=1)
        with torch.no_grad():
            out_full = block(x, attn_mask=causal)
            out_inc = self._run_incremental(block, x)
        self.assertEqual(out_full.shape, out_inc.shape)
        max_diff = (out_full - out_inc).abs().max().item()
        self.assertLess(
            max_diff, 1e-5,
            f"KV-cache incremental rollout must match the full forward (no quality "
            f"loss); max_diff={max_diff}",
        )

    def test_matches_across_head_dims(self):
        # Guard the in_proj split / head reshape / scaling at a different config.
        torch.manual_seed(3)
        block = SequentialGTrXLBlock(48, 6, 96, dropout=0.0).eval()
        x = torch.randn(1, 5, 48)
        causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        with torch.no_grad():
            self.assertLess(
                (block(x, attn_mask=causal) - self._run_incremental(block, x)).abs().max().item(),
                1e-5,
            )

    def test_cache_lifecycle_no_contamination(self):
        # Fresh cache per sequence must be independent (rollout resets per episode).
        torch.manual_seed(1)
        block = SequentialGTrXLBlock(16, 2, 32, dropout=0.0).eval()
        x1 = torch.randn(1, 4, 16)
        x2 = torch.randn(1, 4, 16)
        with torch.no_grad():
            a = self._run_incremental(block, x1)
            b = self._run_incremental(block, x1)  # same input, fresh cache -> identical
            self.assertLess((a - b).abs().max().item(), 1e-6)
            # a second, different sequence still matches its own full forward
            causal = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
            self.assertLess(
                (self._run_incremental(block, x2) - block(x2, attn_mask=causal)).abs().max().item(),
                1e-5,
            )


@unittest.skipUnless(_HAS_TORCH, "torch + sequential_policy required")
class PolicyIncrementalRolloutEquivalenceTest(unittest.TestCase):
    """End-to-end gate: the KV-cached rollout path must reproduce the validated
    full forward (logits + value) at every step, within float. This is the real
    no-quality-loss guarantee — if it passes on the server, the rollout's
    sampled actions / log-probs / values match the full path, so the search and
    reward curve are statistically identical to the non-cached run.
    """

    def _make_cfg(self) -> "SequentialPolicyConfig":
        H, S = 6, 2
        state_dim = 4 + H + 5 + 1 + H * S + H * 3
        return SequentialPolicyConfig(
            state_dim=int(state_dim),
            max_step_dim=int(S),
            max_num_levels=6,
            horizon=int(H),
            block_count=5,
            num_layers=2,
            d_model=32,
            n_heads=4,
            n_layers=3,
            d_ff=64,
            dropout=0.0,
            step_embed_dim=8,
            layer_embed_dim=8,
            block_embed_dim=4,
            prev_action_embed_dim=4,
            cont_proj_dim=16,
            actor_dim=16,
            critic_dim=16,
            default_prior_scale=0.0,
        )

    def _make_state(self, cfg, t, committed_actions, committed_signals):
        # Mimic BLBStage2SequentialEnv._build_obs exactly: a committed prefix
        # (steps 0..t-1) that NEVER changes across t, current-step one-hot at t,
        # and zeros for steps >= t. The skipped block-one-hot/layer fields don't
        # affect the token (the policy reads block/layer from buffers).
        H, S = int(cfg.horizon), int(cfg.max_step_dim)
        state = np.zeros(int(cfg.state_dim), dtype=np.float32)
        state[0:4] = np.array([0.5, 0.4, 0.3, 0.06], dtype=np.float32)
        state[4 + int(t)] = 1.0
        cursor = 4 + H + 5 + 1
        pa = np.zeros((H, S), dtype=np.float32)
        for i in range(int(t)):
            pa[i] = np.asarray(committed_actions[i], dtype=np.float32) / 8.0
        state[cursor:cursor + H * S] = pa.reshape(-1)
        cursor += H * S
        ps = np.zeros((H, 3), dtype=np.float32)
        for i in range(int(t)):
            ps[i] = np.asarray(committed_signals[i], dtype=np.float32)
        state[cursor:cursor + H * 3] = ps.reshape(-1)
        return state

    def test_build_token_at_matches_build_tokens(self):
        torch.manual_seed(7)
        cfg = self._make_cfg()
        policy = BLBStage2SequentialPolicy(cfg).eval()
        rng = np.random.default_rng(7)
        H, S = int(cfg.horizon), int(cfg.max_step_dim)
        committed_actions = rng.integers(0, cfg.max_num_levels, size=(H, S))
        committed_signals = rng.random((H, 3)).astype(np.float32)
        state_np = self._make_state(cfg, H - 1, committed_actions, committed_signals)
        state = torch.from_numpy(state_np).float().unsqueeze(0)
        with torch.no_grad():
            full_tokens, _cur = policy._build_tokens(state, truncate_to_current=False)
            for p in range(H):
                one = policy._build_token_at(state, p)[:, 0, :]
                diff = (one - full_tokens[:, p, :]).abs().max().item()
                self.assertLess(diff, 1e-5, f"_build_token_at mismatch at p={p}: {diff}")

    def test_incremental_rollout_matches_full_forward(self):
        torch.manual_seed(11)
        cfg = self._make_cfg()
        policy = BLBStage2SequentialPolicy(cfg).eval()
        # Exercise the warmstart prior path too (both paths add the same prior,
        # so it must cancel exactly).
        policy.apply_preferred_per_step_bias([1, 0], gain=1.3)
        rng = np.random.default_rng(11)
        H, S = int(cfg.horizon), int(cfg.max_step_dim)
        committed_actions = rng.integers(0, cfg.max_num_levels, size=(H, S))
        committed_signals = rng.random((H, 3)).astype(np.float32)
        prior_scale = 0.7

        cache = policy.new_rollout_cache()
        with torch.no_grad():
            for t in range(H):
                obs_t = torch.from_numpy(
                    self._make_state(cfg, t, committed_actions, committed_signals)
                ).float().unsqueeze(0)
                lg_inc, v_inc = policy.forward(
                    obs_t, prior_scale, truncate_to_current=True, kv_cache=cache,
                )
                lg_full, v_full = policy.forward(
                    obs_t, prior_scale, truncate_to_current=True,
                )
                self.assertLess(
                    (lg_inc - lg_full).abs().max().item(), 1e-5,
                    f"logits mismatch at step {t}",
                )
                self.assertLess(
                    (v_inc - v_full).abs().max().item(), 1e-5,
                    f"value mismatch at step {t}",
                )
                if t < H - 1:
                    obs_next = torch.from_numpy(
                        self._make_state(cfg, t + 1, committed_actions, committed_signals)
                    ).float().unsqueeze(0)
                    policy.commit_kv_cache(obs_next, t, cache)
            self.assertEqual(cache.length, H - 1)

    def test_sample_action_value_matches_full(self):
        # The rollout calls sample_action/evaluate_action with kv_cache; their
        # forward must match the non-cached forward (the value used for GAE and
        # the logits used for the categorical are identical within float).
        torch.manual_seed(5)
        cfg = self._make_cfg()
        policy = BLBStage2SequentialPolicy(cfg).eval()
        rng = np.random.default_rng(5)
        H, S = int(cfg.horizon), int(cfg.max_step_dim)
        committed_actions = rng.integers(0, cfg.max_num_levels, size=(H, S))
        committed_signals = rng.random((H, 3)).astype(np.float32)
        cache = policy.new_rollout_cache()
        with torch.no_grad():
            for t in range(H):
                obs_t = torch.from_numpy(
                    self._make_state(cfg, t, committed_actions, committed_signals)
                ).float().unsqueeze(0)
                slot_mask = torch.ones(1, S, dtype=torch.bool)
                levels = torch.full((1, S), cfg.max_num_levels, dtype=torch.long)
                _a, _lp, v_cached = policy.sample_action(
                    obs_t, slot_mask, levels, deterministic=True,
                    truncate_to_current=True, kv_cache=cache,
                )
                _a2, _lp2, v_full = policy.sample_action(
                    obs_t, slot_mask, levels, deterministic=True,
                    truncate_to_current=True,
                )
                self.assertLess((v_cached - v_full).abs().max().item(), 1e-5)
                if t < H - 1:
                    obs_next = torch.from_numpy(
                        self._make_state(cfg, t + 1, committed_actions, committed_signals)
                    ).float().unsqueeze(0)
                    policy.commit_kv_cache(obs_next, t, cache)


if __name__ == "__main__":
    unittest.main(verbosity=2)
