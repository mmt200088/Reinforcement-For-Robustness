# BLB Stage-2 RL Live Summary

- Run: `s1t0.001_s2t0.001_s2st2.0__bertbase_sst2_stage2_k3_4gpu_stage1best20260625_1b34e949_20260803`
- Profile: `sst2`
- Phase: max_episodes_reached
- Updated at: 2026-08-04T01:50:56.115456
- Elapsed seconds: 41923.856000
- Episode: 50000 / 50000 (100.00%)
- PPO updates: 417

## Reward

- Last reward: -3.353952
- Recent reward mean: +0.057927
- Best reward: +1.458915
- Best episode: n/a
- Last priority: 1
- Last invalid: False

## Last Terminal Metrics


## Last PPO Update

- `policy_loss`: -0.0018212709110230207
- `value_loss`: 0.43613576889038086
- `entropy`: 0.2693555951118469
- `clip_fraction`: 0.00937500037252903
- `window_mean_return`: 0.08404778218722671
- `window_mean_invalid`: 0.0
- `approx_kl`: 0.0017703680787235498
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
