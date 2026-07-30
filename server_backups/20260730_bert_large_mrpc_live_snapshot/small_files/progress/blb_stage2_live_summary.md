# BLB Stage-2 RL Live Summary

- Run: `s1t0.001_s2t0.001_s2st2.0__bertlarge_mrpc_stage2_k3_4gpu_0764e710_20260730`
- Profile: `mrpc_large`
- Phase: PPO training (24-step layerwise robust)
- Updated at: 2026-07-30T17:12:27.110957
- Elapsed seconds: 45257.698000
- Episode: 12000 / 150000 (8.00%)
- PPO updates: 100

## Reward

- Last reward: +1.354591
- Recent reward mean: +0.898886
- Best reward: +1.500693
- Best episode: n/a
- Last priority: 3
- Last invalid: False

## Last Terminal Metrics


## Last PPO Update

- `policy_loss`: -0.0023497152142226696
- `value_loss`: 0.12880684435367584
- `entropy`: 0.8576430678367615
- `clip_fraction`: 0.03676215186715126
- `window_mean_return`: 0.978766568287686
- `window_mean_invalid`: 0.0
- `approx_kl`: 0.0016445053042843938
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
