# BLB Stage-2 RL Live Summary

- Run: `s1t0.001_s2t0.001_s2st2.0__bertlarge_rte_stage2_k3_4gpu_a9559610_20260731`
- Profile: `rte_large`
- Phase: 已停止：checkpoint-boundary graceful stop
- Updated at: 2026-08-01T20:13:02.087986
- Elapsed seconds: 84906.758000
- Episode: 33720 / 50000 (67.44%)
- PPO updates: 281

## Reward

- Last reward: -3.040408
- Recent reward mean: +0.725969
- Best reward: +1.406860
- Best episode: n/a
- Last priority: 1
- Last invalid: False

## Last Terminal Metrics


## Last PPO Update

- `policy_loss`: -0.002084304578602314
- `value_loss`: 0.22306004166603088
- `entropy`: 0.6276459097862244
- `clip_fraction`: 0.0240885429084301
- `window_mean_return`: 0.7448450551912056
- `window_mean_invalid`: 0.0
- `approx_kl`: 0.0017824929673224688
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
