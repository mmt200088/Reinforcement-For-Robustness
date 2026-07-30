# BLB Stage-2 RL Live Summary

- Run: `s1t0.001_s2t0.001_s2st2.0__bertlarge_mrpc_stage2_k3_4gpu_0764e710_20260730`
- Profile: `mrpc_large`
- Phase: PPO training (24-step layerwise robust)
- Updated at: 2026-07-31T00:44:56.002496
- Elapsed seconds: 72406.589000
- Episode: 24600 / 150000 (16.40%)
- PPO updates: 205

## Reward

- Last reward: +1.459018
- Recent reward mean: +0.866050
- Best reward: +1.531940
- Best episode: n/a
- Last priority: 3
- Last invalid: False

## Last Terminal Metrics


## Last PPO Update

- `policy_loss`: -0.0023111305199563503
- `value_loss`: 0.15776289999485016
- `entropy`: 0.777920663356781
- `clip_fraction`: 0.02938368171453476
- `window_mean_return`: 0.9543613770723702
- `window_mean_invalid`: 0.0
- `approx_kl`: 0.002301697852090001
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
