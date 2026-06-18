"""Production-scale equivalence + speed benchmark for the KV-cache rollout.

The 2026-06-19 rollout speedup (`--blb-v3-kv-cache-rollout`) is NOT byte-identical
(the user dropped 1==N for it), so the guarantee is "no quality loss within float".
The speedup lives ENTIRELY in the policy's per-step GTrXL forward — env / replan /
probe are unchanged — so this script measures exactly that, WITHOUT touching any
persistent run dir (a full training A/B would `--fresh`-wipe the canonical
Stage-2 best). It:

  1. Builds the REAL fusion-mode policy at production scale (horizon=59,
     d_model=256, n_heads=8, n_layers=4, d_ff=512, max_step_dim=2).
  2. Replays a synthetic-but-structurally-faithful rollout (committed prefix that
     never changes across steps; current-step one-hot; prev_action placeholder->
     committed at the i->i+1 boundary — exactly the env's obs invariants) and
     asserts at EVERY step that the KV-cached forward matches the full forward in
     logits, value, and the deterministic argmax action (the search-relevant
     outputs) within tolerance. This is the no-quality-loss proof at the scale a
     small unit test can't reach.
  3. Times the OFF (full O(H^2) per-step rebuild) vs ON (O(H) incremental + fixup)
     rollouts over many episodes and reports the per-episode rollout-forward
     speedup.

torch-required; run on the GPU server. Exit code 1 on any equivalence failure
(so SERVER_COMMAND can gate on it — a wrong impl never reaches a real run).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _build_state(cfg, t, committed_actions, committed_signals):
    H = int(cfg.horizon)
    S = int(cfg.max_step_dim)
    state = np.zeros(int(cfg.state_dim), dtype=np.float32)
    state[0:4] = np.array([0.5, 0.4, 0.3, 0.59], dtype=np.float32)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200, help="rollouts to time")
    ap.add_argument("--horizon", type=int, default=59)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    import torch

    from blb_stage2_rl.sequential_policy import (
        BLBStage2SequentialPolicy,
        SequentialPolicyConfig,
    )

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    H, S = int(args.horizon), 2
    state_dim = 4 + H + 5 + 1 + H * S + H * 3
    cfg = SequentialPolicyConfig(
        state_dim=int(state_dim),
        max_step_dim=int(S),
        max_num_levels=6,
        horizon=int(H),
        block_count=5,
        num_layers=12,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.0,
        step_embed_dim=16,
        layer_embed_dim=16,
        block_embed_dim=8,
        prev_action_embed_dim=4,
        cont_proj_dim=64,
        actor_dim=64,
        critic_dim=64,
        default_prior_scale=0.0,
    )
    torch.manual_seed(0)
    policy = BLBStage2SequentialPolicy(cfg).to(device).eval()
    # fusion mode puts a prior on the K slot only (ADR-011); exercise that path.
    policy.apply_preferred_per_step_bias([-1, 1], gain=2.5)

    slot_mask = torch.ones(1, S, dtype=torch.bool, device=device)
    levels = torch.full((1, S), cfg.max_num_levels, dtype=torch.long, device=device)
    prior_scale = 1.0
    rng = np.random.default_rng(123)

    def make_episode(seed):
        r = np.random.default_rng(seed)
        ca = r.integers(0, cfg.max_num_levels, size=(H, S))
        cs = r.random((H, 3)).astype(np.float32)
        states = [
            torch.from_numpy(_build_state(cfg, t, ca, cs)).float().to(device).unsqueeze(0)
            for t in range(H + 1)  # need obs_{t+1} for the fixup of step t
        ]
        return states

    # ---- equivalence (the no-quality-loss gate, production scale) ----
    print(f"[kvcache-bench] device={device} horizon={H} d_model={cfg.d_model} "
          f"n_layers={cfg.n_layers} tol={args.tol}")
    max_logit_diff = 0.0
    max_value_diff = 0.0
    action_mismatches = 0
    with torch.no_grad():
        for ep in range(8):
            states = make_episode(1000 + ep)
            cache = policy.new_rollout_cache()
            for t in range(H):
                lg_on, v_on = policy.forward(
                    states[t], prior_scale, truncate_to_current=True, kv_cache=cache,
                )
                lg_off, v_off = policy.forward(
                    states[t], prior_scale, truncate_to_current=True,
                )
                max_logit_diff = max(max_logit_diff, (lg_on - lg_off).abs().max().item())
                max_value_diff = max(max_value_diff, (v_on - v_off).abs().max().item())
                # search-relevant: the argmax action over the masked logits
                a_on = torch.argmax(lg_on, dim=-1)
                a_off = torch.argmax(lg_off, dim=-1)
                action_mismatches += int((a_on != a_off).sum().item())
                if t < H - 1:
                    policy.commit_kv_cache(states[t + 1], t, cache)
    print(f"[kvcache-bench] equivalence: max|logit_diff|={max_logit_diff:.2e} "
          f"max|value_diff|={max_value_diff:.2e} argmax_mismatches={action_mismatches}")
    ok = (max_logit_diff < args.tol and max_value_diff < args.tol
          and action_mismatches == 0)
    if not ok:
        print("[kvcache-bench][FATAL] KV-cache rollout DIVERGES from full forward "
              "at production scale — quality would regress; do NOT enable.")
        return 1
    print("[kvcache-bench] equivalence PASS — KV-cache is quality-neutral within float.")

    # ---- speed ----
    episodes = [make_episode(2000 + i) for i in range(int(args.episodes))]

    def run_off(states):
        for t in range(H):
            policy.forward(states[t], prior_scale, truncate_to_current=True)

    def run_on(states):
        cache = policy.new_rollout_cache()
        for t in range(H):
            policy.forward(states[t], prior_scale, truncate_to_current=True, kv_cache=cache)
            if t < H - 1:
                policy.commit_kv_cache(states[t + 1], t, cache)

    def timeit(fn):
        with torch.no_grad():
            for i in range(5):  # warmup
                fn(episodes[i])
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for st in episodes:
                fn(st)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / len(episodes)

    off_s = timeit(run_off)
    on_s = timeit(run_on)
    speedup = off_s / on_s if on_s > 0 else float("nan")
    print(f"[kvcache-bench] rollout-forward per episode (H={H} steps): "
          f"OFF={off_s * 1e3:.2f} ms  ON={on_s * 1e3:.2f} ms  speedup={speedup:.2f}x")
    print("[kvcache-bench] DONE (this measures ONLY the policy GTrXL forward; "
          "env/replan/probe are unchanged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
