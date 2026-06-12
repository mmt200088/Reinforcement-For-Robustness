# Stage-2 ADR-011 60k result for Claude Code

Source/run context:
- Server runroot: `/hy-tmp/server_command_stage2_adr011_a45f651_20260611_225936`
- Local artifact root: `experiments/server_command_runs/stage2_grid_gate_60k_20260612_004130`
- Source sync marker used for the server snapshot: `e28a610d85b4`
- HTML report: `reports/html_reports/20260612_stage2_adr011_60k_final.html`

Execution outcome:
- Completed: `60,000/60,000` episodes.
- Wall time: `13.85 h` (49847 s), long-run throughput `4333 ep/h`.
- Gate passed: 1-GPU == 5-GPU rollout signatures and episode JSON exactly; 1-GPU 1148 ep/h, 5-GPU 4154 ep/h, speedup 3.62x.
- Health: invalid steps `0`, loss-cap/collapse sentinels `0`.
- Priority counts: P1=1327, P2=9, P3=58664.

Search outcome:
- Raw max reward: ep `80`, mode `forced_fusion_probe_b2`, fusion_count `12`, reward `39.556036`, priority `P3`.
- Diagnostics rank-best: ep `39998`, mode `radius1`, fusion_count `0`, reward `39.544891`, priority `P3`.
- Tail fusion averages: tail100=0.000, tail1000=0.060, tail5000=0.060, tail10000=0.060.
- Final PPO update: update `1000`, entropy `0.000000`, clip_fraction `0.000000`.

Interpretation:
- ADR-011 is mechanically verified: deterministic gate passed, forced fusion probes appeared, and the full run did not collapse.
- The highest raw reward came from forced block2 fusion, so profitable fusion evidence exists.
- The PPO policy still converged to no-fusion/truncation-only behavior: tail fusion is near zero and the rank-best candidate has fusion_count=0.
- Recommended next debug target: policy/rank-selection retention of profitable block2/block5 fusion evidence, not map validity or replan application.
