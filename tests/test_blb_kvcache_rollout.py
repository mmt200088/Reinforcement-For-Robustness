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
    import torch  # noqa: F401
    from blb_stage2_rl.sequential_policy import SequentialGTrXLBlock
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
