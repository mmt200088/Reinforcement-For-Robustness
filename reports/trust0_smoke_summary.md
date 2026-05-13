# Trust-0 Smoke Summary
## Conclusion
- Current code runs BLB Stage-2 RL short smoke on server llm_ist + RTX 4090.
- Baseline preflight now derives the metric limit from all-max BLB baseline: raw=0.9375, all_max_blb=0.8750, drop=0.00494, Acc/F1 limit ~= 0.87006.
- 240-episode F1 diagnostic completed with no error_summary.txt; best_reward=10.7333 at episode 204.
- Non-anchor invalid_rate is 90% (36/40), and the final rollout is 20/20 invalid. Stop per taskbook; do not increase episodes yet.

## F0 Baseline
- valid=True, total_bits_sum=14989, fusion_count=0, rank_key=[14989, 0]
- candidate_key=`7a54dd548e4faab7223cd651be35d95431911aae67880e0bdacd2b298bd7fe1b`, avg_k=13.0, q_bits diagnostic count=60

## F1 Rollout
- episodes=240, rollouts=12, total_samples=240, overall invalid_rate=0.150
- P0 invalid=36, P1=0, P2=0, P3=204
- non-anchor episodes 201-240: invalid_rate=0.900, warning_count=2

## Failures And Fixes
- s1t0.00491: command conflict between explicit Stage-1 episodes and skip_stage1_rl=True.
- s1t0.00492: exposed threshold semantics bug; fixed by deriving threshold after all-max BLB preflight and added regression coverage.
- Initial F0 baseline lacked Stage-1 degree context; added degree CLI to blb_eval_action.py and reran F0.

## Recommendation
Next phase should do action mask curriculum / F0 feasible-domain scan before any long training.

## Evidence Files
- trace_csv: `reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__progress__blb_stage2_episode_trace.csv`
- status_json: `reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__progress__blb_stage2_status.json`
- report_md: `reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__progress__blb_stage2_report.md`
- warning_txt: `reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__warning.txt`
- details_txt: `reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/stage2_noise__details__noise_ppo_step_info_1-360.txt`
- log: `reports/trust0_smoke_artifacts/s1t0.00494_s2t0.00494_s2st0.00494/logs__trust0_smoke_nonanchor.log`
