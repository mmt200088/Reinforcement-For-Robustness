# Stage2 RL First-10k Curve Optimization Design

## Goal

Extend the Stage-2 sequential RL stabilization work from a 600-episode collapse
smoke to a monitored first-10,000-episode research run. The target is a healthy
learning curve, not merely process survival: reward windows should remain
positive and make search progress, terminal metrics must avoid collapse
sentinels such as `loss_mean=100`, priority must not enter sustained P1(acc),
and PPO/exploration/GPU metrics must not show unexplained pathological jumps.
Per user correction during the first 10k attempt, occasional negative reward
spikes or isolated P1(acc) episodes are acceptable if rolling reward averages
remain healthy and such events do not become frequent or sustained.

## Diagnosis From 600 Episodes

The previous 600-episode run fixed the episode-120/121 collapse mode. The
server monitor reported 600 episodes, 480 post-anchor episodes, P1(acc)=0,
post-anchor `loss_mean` in `0.3350..0.3442`, and both GPUs active.

The remaining first-10k risk is curve quality. PPO entropy dropped from about
`5.88` at episode 120 to `0.0035` at episode 180, then recovered only to about
`0.025` by episode 600. Clip fraction similarly dropped to about `0.002..0.022`.
Reward and best reward were still improving, so this is not failure evidence,
but it is a falsifiable risk: the policy may become too narrow for a useful
10k search unless entropy and safe-neighbor exploration can be tuned.

## Approach

Keep the safe-neighbor sequential algorithm unchanged for the first 10k attempt.
Expose the exploration controls through the launcher so future experiments can
change only `SERVER_COMMAND.md`:

- safe-neighbor ramp length
- safe-neighbor max mutations
- safe-neighbor max radius
- safe-neighbor on/off
- warmstart bias gain
- steady entropy coefficient
- anchor entropy coefficient
- entropy ramp length

Also make `episodes.jsonl` carry terminal loss/priority/metric and
safe-neighbor fields directly. The 10k watchdog should read structured JSONL
instead of relying on text-log parsing for core health checks.

## First 10k Candidate Settings

Use the same safe baseline as the 600-episode run, with conservative
safe-neighbor expansion. The first candidate used `ramp=3000`,
`max_mutations=16`, `max_radius=3`; server evidence showed reward improvement
but P1 clustering when radius reached 2. The current safer candidate therefore
keeps radius at 1 and expands mutation count more slowly:

- `--stage2-search-episodes 10000`
- `--skip-final-eval`
- `--stage2-rollout-size 60`
- `--blb-v3-warmstart-anchor-episodes 120`
- `--blb-v3-ent-coef 0.04`
- `--blb-v3-ent-coef-ramp-episodes 1200`
- `--blb-v3-warmstart-neighbor-ramp-episodes 6000`
- `--blb-v3-warmstart-neighbor-max-mutations 8`
- `--blb-v3-warmstart-neighbor-max-radius 1`
- `--blb-v3-reward-devices 0,1`

Hypothesis: higher and slower entropy plus radius-1 safe-neighbor support
should keep the first-10k curve from narrowing too early while avoiding the
observed radius-2 accuracy-collapse region.

Failed wider-neighbor evidence: artifact
`experiments/server_command_runs/stage2_rl_first10k_curve_20260520_141129`
stopped at 1784 episodes. Overall reward did not broadly collapse, but P1 was
clustered in episodes 1699-1757; all P1 cases had `safe_neighbor_radius=2`,
and 7/64 episodes at `(mutations=9, radius=2)` were P1, including two
`loss_mean=100` sentinels.

## Success Criteria

- Local tests pass for parameter plumbing and structured episode diagnostics.
- Server tests pass before training.
- A fresh 10k dual-GPU run completes or, if stopped by watchdog, leaves a clear
  failure report.
- Hard failures are absent: no loss cap, no NaN/inf losses, no sustained P1,
  no invalid-step resurgence, no stale episode progress, and both GPUs used.
- Isolated negative-reward or P1 episodes are not by themselves failure
  evidence; evaluate their frequency plus rolling 60/300/1000 reward windows.
- Curve-quality checks are acceptable: rolling reward windows do not collapse,
  best reward continues to improve beyond the 600-episode baseline, entropy and
  clip fraction do not indicate a dead policy without search progress, and cost
  signals improve without accuracy/stability regressions.
