# ADR-007: v2-style clipped-shaping + tier-bonus reward (supersedes ADR-002 impl)

- **Status**: Accepted (supersedes implementation of [ADR-002](ADR-002-hard-priority-reward.md))
- **Date**: 2026-05-18
- **Tags**: rl-design, reward, evaluation

## Context

ADR-002 specified a hard-priority reward:

```
if acc < acc_threshold:    reward = -P1_PEN - acc_violation_scaled       # ≈ -150
elif loss_std > stab_thr:  reward = -P2_PEN - stab_violation_scaled      # ≈ -150
else:                       reward = w_bits·bits_drop + w_fusion·fusion_drop + w_k·k_drop
```

In two consecutive training runs (commit `42cfbe4` and commit `d0ab4bc`, see
diagnostics at `Parting Chapter/persistent/.../s1t0.005_*` and
`reports/stage2_rl/failed_runs/2026-05-18_dynamic_stab_calibration_fallback/`)
the implementation collapsed:

- baseline noisy `loss_std = 0.0048` (very stable because all-max SF baseline
  is the least-noisy schedule)
- derived `stab_threshold = 0.05` (floor kicked in)
- typical RL candidate `loss_std ≈ 1.0` (3 trials over high-noise BLB install
  produces high cross-entropy variance even when the model still works)
- → every episode trips P2, reward formula `-50 + (0.05 - 1.0)·100 ≈ -145`
- → the diagnostic label `priority=P3(cost)` was hardcoded `1 if invalid_steps>0 else 3`
  and therefore lied — masking the real source of the failure
- → PPO learned "minimize loss_std" = "retreat toward baseline" instead of cost
- 1000 episodes of training produced **zero** cost improvement; reward never
  crossed 0 (stayed in `-213, -130`)

Attempted fix in `d0ab4bc` (dynamic stab calibration via P90 of 5 random valid
samples) failed because **577-dim uniform-random actions are almost surely
invalid_chain**: 25 attempts produced 0 valid samples, calibration aborted,
threshold stayed at the bad 0.05.

The hard-priority structure has another inherent risk: the penalties (-50,
-100, -200) are large relative to PPO advantage estimation. Once every action
lands in the same priority tier, the relative differences are tiny and PPO
loses gradient signal entirely.

`noise_rl_module_v2.py` (the legacy single-N stage-2 RL that ran on the same
research codebase and "worked very well" per the user) uses a different
design:

```python
shaping_reward = perf_score + cost_score + stability_penalty - violation_penalty
clipped_shaping = clip(shaping, -5, +5)        # bounded
tier_bonus = 0
if metric_ok:    tier_bonus += 20              # NOISE_STAGE_METRIC_TIER_BONUS
  if stab_ok:    tier_bonus += 20              # NOISE_STAGE_STABILITY_TIER_BONUS
terminal_reward = clipped_shaping + tier_bonus  # ∈ [-5, +45]
```

The crucial properties:

1. The shaping is **bounded** to a small range so PPO advantage estimation
   stays well-conditioned.
2. The "priority" semantics is preserved via **large additive tier jumps**
   (+20 / +40) — a metric-ok candidate is always at least 15 points above any
   metric-failing one.
3. Stability is a **continuous soft penalty** (`-lambda × excess`), not a hard
   gate. Even when `loss_std` is well above `stab_threshold`, the penalty just
   saturates against the clip; cost signal remains visible inside the
   clipped-shaping.
4. Stability and cost are **silenced when metric fails** (per v2 line 1647),
   so PPO focuses on "satisfy acc first" rather than being distracted by
   stability noise in the infeasible region.

## Decision

Switch BLB Stage-2 RL `compute_reward` to the v2-style formula:

```python
margin_acc       = (acc - acc_threshold) / max(|baseline_acc - acc_threshold|, 0.01)
cost_score       = cost_weight × (bits_score + fusion_score + k_score) / 3
                   # each *_score = drop / typical_drop
stab_excess      = max(0, loss_std - stab_threshold)
stability_penalty = -lambda_stab × stab_excess

metric_ok        = acc_violation == 0 AND not invalid_chain
stab_ok          = stab_excess == 0

shaping_raw      = margin_acc
                 + (-invalid_penalty if invalid else 0)
                 + (cost_score if metric_ok else 0)
                 + (stability_penalty if metric_ok else 0)
shaping_clipped  = clip(shaping_raw, -5, +5)

tier_bonus       = (0 if not metric_ok
                    else tier_metric_bonus + (tier_stability_bonus if stab_ok else 0))

total_reward     = shaping_clipped + tier_bonus
```

Default weights (all from v2 verified values, in `RewardWeights`):

| Field | Default | Note |
|---|---|---|
| `cost_weight` | 1.0 | cost normalization multiplier |
| `lambda_stab` | 5.0 | stability soft penalty weight |
| `invalid_penalty` | 5.0 | enough to saturate the clip floor |
| `reward_clip_min/max` | -5 / +5 | bounded shaping |
| `tier_metric_bonus` | +20 | metric-ok = acc + no-invalid |
| `tier_stability_bonus` | +20 | additional bonus when stab_ok |
| `margin_denom_floor` | 0.01 | denom safety for tight tolerance |

Persistent-dir slug is bumped with `_rdv2` suffix
(`s1t0.005_s2t0.005_s2st0.005_rdv2`) so old checkpoints don't silently mix
with the new design. Any future reward redesign should bump this tag.

**Hard-priority intent is preserved**: by construction, every metric-failing
candidate scores `≤ +5` (clip max) and every metric-passing one scores `≥ +15`
(clip min + tier_metric_bonus). Cost can never compensate for an
accuracy/invalid violation because it's silenced (`effective_cost_score = 0`)
when metric fails. The user's design intent from ADR-002 ("cost reward must
never offset an accuracy / stability violation") is satisfied.

**Why this fixes the stuck-reward problem**:

- baseline action: `shaping ≈ 0 + 0 - 0` → clip = 0, tier = +40, **total = +40**
- best candidate (cost-down, still metric-ok, stab-fail): `shaping ≈ 0.3 - 5` (clipped) = -5 (clipped), tier = +20, **total = +15**
- worst candidate (metric-fail or invalid): `shaping ≈ -5` (clipped), tier = 0, **total = -5**

PPO sees a spread of ~45 points with smooth differential gradients in each
tier. The previous -213 → -130 "training" was an artifact of all candidates
being stuck in a constant-ish P2; now there's a genuine 50-point spread per
episode.

## Alternatives considered

| Option | Why rejected |
|--------|--------------|
| Keep ADR-002 hard-priority, just relax `stab_threshold` to 1.5 | Doesn't address the underlying issue — large -150 penalties still dominate PPO advantages, only the bucket shifts |
| Keep hard-priority but bound penalties to [-10, +10] | Still no continuous signal — every P1 action gets the same -10; PPO can't distinguish "barely failed acc" from "totally failed acc" |
| Dynamic threshold from random-action P90 | Already tried (commit `d0ab4bc`); 577-dim random is almost always invalid; calibration aborts. Could fix by sampling from warmstart-biased policy, but that's brittle compared to "always bounded reward" |
| Lagrangian / CPO | Engineering complexity high; existing PPO is mature; not needed for this scale of action space |
| Per-block per-step soft rewards only (drop terminal entirely) | Loses the per-episode-level signal; per-step shaping is already in place additively and is too local for cost trade-offs |

## Consequences

**Positive**:

- PPO advantage estimation is well-conditioned (bounded reward).
- Cost signal is visible to PPO whenever metric is OK — drives actual cost
  optimization, not just retreat-to-baseline.
- Reward priority semantics preserved via tier bonus jumps; ranking is
  unambiguous.
- The `priority` field is still meaningful for reporting and ranking
  (tier 1 / 2 / 3 in `RewardBreakdown.priority`).
- No more reliance on a calibration loop that depends on random-action
  validity — uses the v2 formula `baseline_std × (1 + tol)` directly.

**Negative / trade-offs**:

- Departs from ADR-002's literal formula (large negative penalties), though
  preserves its intent. Any reader looking only at ADR-002 will be confused
  until they find this ADR.
- The reward magnitude is no longer interpretable as "violation amount" —
  it's a clipped-and-tiered score in [-5, +45].
- `lambda_stab = 5` and the tier bonuses (+20 / +40) are hyperparameters; if
  acc + stab both consistently pass for every candidate, the tier bonus
  saturates and PPO must rely on the ±0.5 cost differential within the
  shaping. Watch for "best reward plateaus at +40.X" in diagnostics — that's
  the symptom of saturation.

**Things to watch / future re-evaluation triggers**:

- If diagnostics show >80% of episodes with `metric_ok=True, stab_ok=True`
  (saturating at +40), bump `cost_weight` or widen the shaping clip to
  recover within-tier discrimination.
- If diagnostics show `metric_ok=True` but `stab_ok=False` for ALL episodes
  (best stuck at +15 to +25), `stab_threshold` is too tight given the
  intrinsic per-trial noise — loosen `stability_tol` in the preset.
- If priority-1 candidates dominate forever, `acc_threshold` is unreachable
  given Stage-1 degrees — re-examine Stage-1 config.

## References

- Linked code:
  - `blb_stage2_rl/reward.py` — new `compute_reward`
  - `blb_stage2_rl/sequential_runner.py:880-` — calibration + log
  - `noise_rl_module_v2.py:1620-1670` — v2 reference implementation
  - `llama_7B_LayerImportance.sh:1127` — slug `_rdv2` tag
- Linked failure reports:
  - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/diagnostics/diagnostics_summary.md`
    (last run under ADR-002 implementation)
  - `reports/stage2_rl/failed_runs/2026-05-18_dynamic_stab_calibration_fallback/report.html`
    (random-action calibration aborted)
- Supersedes: [ADR-002](ADR-002-hard-priority-reward.md) (the *implementation*
  details only; the *design intent* of hard-priority is preserved).
