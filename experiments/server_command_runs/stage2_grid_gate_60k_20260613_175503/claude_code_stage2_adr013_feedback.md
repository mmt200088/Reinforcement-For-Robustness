# Stage-2 ADR-013 60k server result feedback

Run: `stage2_grid_gate_60k_20260613_175503`
Source: `4e3aec0589b8940e8894dfbaaa3cfdfe9433eb1a`

## Status

Not a successful 60k completion. The watchdog stopped the long run at `40320` episodes / last episode `40319` because P3 stayed below 2% for 12 consecutive windows.

## Gate

```text
==== [gate] 判读 ====
[gate][PASS] rollout_sig 逐窗逐字相同（1卡 == 5卡）
[gate] episodes.jsonl 数值逐项对比: PASS（完全一致）  n=300/300
[gate] forced_fusion_probe episodes in gate run: [(80, 'forced_fusion_probe_b2', 12), (280, 'forced_fusion_probe_b5', 25)]
[gate] probe presence/rotation/fc check: OK
1-GPU : 857 ep/h (1260s)
5-GPU : 2842 ep/h (380s)  speedup=3.32x (ideal 5x)
```

## Best found before collapse

- episode: `23645`
- total_reward: `40.727`
- terminal_reward: `42.814`
- priority: `3`
- fusion_count: `27` (`b2=10`, `b4=6`, `b5=11`)
- loss_mean: `0.3410`
- metric1/metric2: `0.8617` / `0.8617`
- effective changed slots vs baseline: `188`

## Collapse signature

First rolling600 with P3=0: `episode=24599`, reward_mean=`-6.915`, fusion_mean=`28.037`, metric1_mean=`0.7687`.

Final health window showed all P1 and fusion around 35, with b2/b4/b5 all around 11.7. Entropy did not go to zero, so this looks like over-fusion reward/constraint shaping rather than dead exploration.

Local artifacts:

- `experiments/server_command_runs/stage2_grid_gate_60k_20260613_175503/stage2_adr013_compact_summary.json`
- `experiments/server_command_runs/stage2_grid_gate_60k_20260613_175503/stage2_adr013_collapse_report.html`
- `reports/html_reports/20260614_stage2_adr013_60k_collapse_report.html`
