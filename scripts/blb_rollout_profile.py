#!/usr/bin/env python3
"""Isolated profiler for the Stage-2 fusion-mode rollout GTrXL forward.

WHY THIS EXISTS (2026-06-21): both the KV-cache rollout and the first batched
rollout shipped to a 60k / A-B WITHOUT first proving, in isolation and under
SYMMETRIC synchronization, that the per-step GTrXL forward is (a) the dominant
rollout cost and (b) actually cheaper when batched across episodes. KV-cache was
the wrong lever (launch-bound, not FLOP-bound); the first batched A/B was
confounded (OFF measured async / ON synced, and the batched path disabled the
causal-prefix truncation). This script removes all of that noise: it builds the
REAL production fusion policy (no model / no Rescale_optimizer / no GLUE needed),
feeds it realistic per-step observations, and times the forward three ways with
the SAME cuda.synchronize discipline:

  * serial  per-episode = H forwards on [1, state_dim]   (truncate_to_current=True)
  * batched per-episode = H forwards on [B, state_dim]/B (truncate_to_current=True)
  * batched(no-trunc)   = the OLD buggy path (truncate_to_current=False, full H)

It prints, for the requested B values: the per-episode forward wall and the
serial/batched speedup, plus the truncate-vs-no-truncate delta and a per-stage
breakdown (obs host-build / forward / sample) so the forward's share of the
policy-side rollout step is explicit.

DECISION RULE printed at the end:
  * real operating point is B ~= rollout_size / (NGPU * workers_per_device).
    Pass the same rollout_size / GPU count / workers-per-device as the real A/B.
    Read the speedup AT THAT B, not the largest-B best case.
  * if batched(trunc) gives a clear (>~1.5x) per-episode forward speedup at the
    real B, batching is the right lever and a corrected A/B should confirm it
    end-to-end; if it is ~1.0x, the forward is not batch-amortizable in this
    regime and the feature should be abandoned like KV-cache.

torch-required -> runs on the GPU/CPU-torch server (no torch locally).
Usage:
  python3 scripts/blb_rollout_profile.py --rollout-size 60 --gpus 4 --workers-per-device 2
"""

import argparse
import statistics as st
import sys
import time


def _have_torch() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="timed repeats per (B, mode); median reported")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batches", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--rollout-size", type=int, default=32,
                    help="PPO window (BLBStage2TrainConfig.rollout_size) for the "
                         "real-B note")
    ap.add_argument("--gpus", type=int, default=None,
                    help="GPU count used by the real A/B; defaults to visible CUDA devices")
    ap.add_argument("--workers-per-device", type=int, default=2,
                    help="workers per GPU used by the real A/B")
    args = ap.parse_args()

    if not _have_torch():
        print("[FATAL] torch not importable - this profiler must run on the "
              "torch server", file=sys.stderr)
        return 2

    import torch

    sys.path.insert(0, ".")
    sys.path.insert(0, "blb_stage2_rl")
    from blb_stage2_rl.sequential_policy import (
        BLBStage2SequentialPolicy,
        SequentialPolicyConfig,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    gpu_count = (
        int(args.gpus)
        if args.gpus is not None
        else (int(torch.cuda.device_count()) if is_cuda else 1)
    )
    gpu_count = max(1, gpu_count)
    workers_per_device = max(1, int(args.workers_per_device))

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    # --- production fusion-mode policy config (mirrors sequential_runner) ---
    H = 59          # horizon (L=12 per-block schedule)
    S = 2           # fusion mode: slot0 = fusion option, slot1 = K
    L = 6           # max_num_levels = max(6, max_options=2) = 6
    state_dim = 4 + H + 5 + 1 + H * S + H * 3
    cfg = SequentialPolicyConfig(
        state_dim=int(state_dim),
        max_step_dim=int(S),
        max_num_levels=int(L),
        horizon=int(H),
        num_layers=12,
        block_count=5,
        # all trunk dims are dataclass defaults = the real run:
        # d_model=256 n_heads=8 n_layers=4 d_ff=512 dropout=0.1 ...
    )
    torch.manual_seed(0)
    policy = BLBStage2SequentialPolicy(cfg).to(device).eval()

    print("=" * 78)
    print("Stage-2 fusion rollout forward profiler")
    print(f"  device={device}  torch={torch.__version__}")
    print(f"  policy: d_model={cfg.d_model} heads={cfg.n_heads} layers={cfg.n_layers} "
          f"d_ff={cfg.d_ff} | H={H} S={S} L={L} state_dim={state_dim}")
    nparams = sum(p.numel() for p in policy.parameters())
    print(f"  params={nparams/1e6:.2f}M")
    print("=" * 78)

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    maxB = max(batches)

    # Pre-build per-step inputs for the largest batch; slice [:B] per run. The
    # forward cost is shape-driven, so random contents are representative.
    g = torch.Generator(device="cpu").manual_seed(7)
    obs_steps = []  # obs_steps[t] : [maxB, state_dim]
    for t in range(H):
        st_ = torch.rand(maxB, state_dim, generator=g) * 0.1
        st_[:, 4: 4 + H] = 0.0
        st_[:, 4 + t] = 1.0  # current_step one-hot at t
        obs_steps.append(st_.to(device))
    slot_mask_full = torch.ones(maxB, S, dtype=torch.bool, device=device)
    levels_full = torch.full((maxB, S), L, dtype=torch.long, device=device)
    alm_full = torch.ones(maxB, S, L, dtype=torch.bool, device=device)

    def run_forward(B, t, truncate):
        return policy.forward_and_mask(
            obs_steps[t][:B], slot_mask_full[:B], levels_full[:B],
            action_level_mask=alm_full[:B], truncate_to_current=bool(truncate),
        )

    # --- warmup ---
    for _ in range(int(args.warmup)):
        for t in range(H):
            run_forward(maxB, t, True)
            run_forward(maxB, t, False)
    sync()

    def time_episode_forward(B, truncate):
        """Median per-episode forward wall = (H forwards on [B,...]) / B."""
        samples = []
        for _ in range(int(args.repeat)):
            sync()
            t0 = time.perf_counter()
            for t in range(H):
                run_forward(B, t, truncate)
            sync()
            samples.append((time.perf_counter() - t0) / float(B))
        return st.median(samples)

    # --- per-stage breakdown of ONE policy-side rollout step (B=1) ---
    import numpy as np
    np_obs = np.asarray(obs_steps[H // 2][:1].cpu().numpy(), dtype=np.float32)
    sm_np = np.ones((1, S), dtype=bool)

    def time_stage(fn):
        xs = []
        for _ in range(max(20, int(args.repeat) * 4)):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            xs.append(time.perf_counter() - t0)
        return st.median(xs)

    obs_build = time_stage(lambda: (
        torch.from_numpy(np_obs).float().to(device),
        torch.from_numpy(sm_np).to(device),
    ))
    fwd_one = time_stage(lambda: run_forward(1, H // 2, True))
    lg, sf, _v = run_forward(1, H // 2, True)
    sample_one = time_stage(
        lambda: policy.sample_from_logits(lg, sf, slot_mask_full[:1], [12345])
    )
    print("\n[per-step policy-side breakdown, B=1, median ms]")
    print(f"  obs host-build (from_numpy+.to) : {obs_build*1e3:8.3f} ms")
    print(f"  forward_and_mask (truncate=True): {fwd_one*1e3:8.3f} ms  <-- batched")
    print(f"  sample_from_logits             : {sample_one*1e3:8.3f} ms")
    pol_step = obs_build + fwd_one + sample_one
    if pol_step > 0:
        print(f"  forward share of policy-side step: {100*fwd_one/pol_step:5.1f}%")

    # --- the decisive table ---
    serial_per_ep = time_episode_forward(1, True)  # B-independent
    print("\n[per-episode FORWARD wall, median, symmetric cuda.sync]")
    print(f"  serial (B=1, truncate=True) per-episode = {serial_per_ep*1e3:.3f} ms "
          f"(this is the OFF baseline; B-independent)")
    print("")
    header = (f"  {'B':>3} | {'batched(trunc)':>16} | {'speedup':>8} | "
              f"{'batched(noTrunc)':>16} | {'speedup':>8} | {'trunc gain':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = []
    for B in batches:
        bt = time_episode_forward(B, True)
        bnt = time_episode_forward(B, False)
        sp_t = serial_per_ep / bt if bt > 0 else float("nan")
        sp_nt = serial_per_ep / bnt if bnt > 0 else float("nan")
        tgain = bnt / bt if bt > 0 else float("nan")
        rows.append((B, bt, sp_t, bnt, sp_nt, tgain))
        print(f"  {B:>3} | {bt*1e3:>13.3f}ms | {sp_t:>7.2f}x | "
              f"{bnt*1e3:>13.3f}ms | {sp_nt:>7.2f}x | {tgain:>9.2f}x")

    # --- verdict at the real operating point ---
    print("\n" + "=" * 78)
    real_b = max(1, round(args.rollout_size / (gpu_count * workers_per_device)))
    print("real operating B = rollout_size / (NGPU * workers_per_device):")
    print(
        f"  rollout_size={args.rollout_size} NGPU={gpu_count} "
        f"workers_per_device={workers_per_device} -> B~={real_b}"
    )

    def speedup_at(target):
        best = min(rows, key=lambda r: abs(r[0] - target))
        return best[0], best[2]

    b_real, sp_real = speedup_at(real_b)
    print(f"  batched(trunc) forward speedup at nearest measured B={b_real}: {sp_real:.2f}x")
    if sp_real >= 1.5:
        print("  VERDICT: forward IS batch-amortizable at the real B -> batching is the "
              "right lever; run the corrected end-to-end A/B to confirm episode-level gain.")
    elif sp_real > 1.1:
        print("  VERDICT: MARGINAL forward gain at the real B -> end-to-end episode gain "
              "likely small (forward is only part of the episode); decide via the A/B.")
    else:
        print("  VERDICT: forward is NOT batch-amortizable at the real B (~1x) -> abandon "
              "like KV-cache; keep the multi-GPU/workers-per-device path.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
