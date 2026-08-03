# BLB Stage-2 RL Live Summary

- Run: `s1t0.001_s2t0.001_s2st2.0__bertlarge_sst2_stage2_k3_4gpu_1b34e949_20260801`
- Profile: `sst2_large`
- Phase: 已停止：checkpoint-boundary graceful stop
- Updated at: 2026-08-02T23:45:00.662434
- Elapsed seconds: 95639.386000
- Episode: 35280 / 50000 (70.56%)
- PPO updates: 294

## Reward

- Last reward: +1.531871
- Recent reward mean: +0.851670
- Best reward: +1.698342
- Best episode: n/a
- Last priority: 3
- Last invalid: False

## Last Terminal Metrics


## Last PPO Update

- `policy_loss`: -0.0038826579693704844
- `value_loss`: 0.2919137477874756
- `entropy`: 1.006639003753662
- `clip_fraction`: 0.05130208283662796
- `window_mean_return`: 0.9604265449951305
- `window_mean_invalid`: 0.0
- `approx_kl`: 0.002471446292474866
- `lr`: 5e-05
- `ent_coef`: 0.0

## Key Artifacts

- Status JSON: `blb_stage2_status.json`
- Live summary: `blb_stage2_live_summary.md`
- Episodes: `diagnostics/episodes.jsonl`
- PPO updates: `diagnostics/ppo_updates.jsonl`
- Diagnostics summary: `diagnostics/diagnostics_summary.md`
- Details batches: `details/`
- Training curve: `blb_stage2_training_curve.png`
- Entropy curve: `blb_stage2_entropy_curve.png`
- Final report: `blb_stage2_report.md`
