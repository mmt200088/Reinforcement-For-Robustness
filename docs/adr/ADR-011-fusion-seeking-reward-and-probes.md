# ADR-011: Fusion-seeking reward (split P3 cost budget) + scheduled fusion probes

Date: 2026-06-11
Status: Accepted
Amends: ADR-008's fusion-count reward (the per-block weighted saving stays;
its normalization/budgeting changes) and the warmstart-prior / curriculum
contract from the 2026-06-05 fusion curriculum decision (probes are an
additive standing re-exploration mechanism).

## Context

The first full 60k fusion-count run (artifacts
`experiments/server_command_runs/stage2_grid_gate_60k_20260611_031751`,
determinism gate PASS, 3.62× on 5 GPUs) converged to **fusion_count = 0** —
the best action found was the zero-fusion baseline with deep K. Offline group
evaluation (Codex, `fusion_count_newenum_eval_20260611_010156`) proves this is
a search failure, not a property of the problem:

- `one_hot_block2` (fusion=12): loss 0.3837 ≤ baseline 0.3846 — free.
- `one_hot_block5_n1` (fusion=10): loss 0.3835 — free.
- `block2+block5 all fusionmax` (fusion=24): loss 0.3845 — free.
- `one_hot_block4` (fusion=12): loss 0.624 — genuinely harmful at 12 layers
  (but ≤4 block4 layers measured harmless: 0.3845/0.3810/0.3730).

So ≥24 fusions were free to harvest; RL took none. Episode forensics
(`episodes.jsonl`, 60000 rows): fusion>0 was attempted 436 times total, 435
of them before episode ~2000; after the curriculum ramp dissolved the policy
never sampled fusion again. Best P3 reward with fusion==0 was 39.41 (late)
vs 38.18 with fusion>0 (early) — **the reward itself preferred the
zero-fusion baseline**.

Four compounding causes:

1. **K-pot dilution.** The P3 cost was `cost_norm × p3_cost_budget(4.5)` with
   one shared normalization (`MAX_ACTUAL ≈ 6190` for mrpc-47, of which 2350
   is truncation weight). Marginal reward of one block2 fusion = +0.109, one
   block5 fusion = +0.029 — invisible against episode reward noise. Worse,
   K alone could reach 2350/6190×4.5 = **1.71**, ≈ the 24-fusion saving
   (1.66): fusion was *redundant* for reward maximization.
2. **P3 metric-margin pull toward zero noise.** The margin term (≤0.5)
   systematically favors the lowest-noise config — the fusion-0 baseline.
3. **Permanent warmstart prior on a 2-way choice.** The fusion option slot's
   preferred=0 prior decays to a 0.15 floor but never to zero (gain 2.5);
   it kept pulling the binary fusion choice back to option 0 forever.
4. **Failure tax + no re-exploration.** Under the 0.5% stability tolerance,
   15.2% of fusion attempts landed P1/P2 (fusion episodes' loss_std ≈ 2.2×
   baseline), giving the fusion logit a large negative advantage during the
   only window it was ever sampled; once collapsed, nothing ever resampled it.

## Decision

Four coordinated changes (all in this repo revision):

1. **Split the P3 cost budget** (`reward.FUSION_COST_BUDGET_FRACTION = 2/3`):
   `ext_score = fusion_norm × (budget × 2/3) + trunc_norm × (budget × 1/3)`,
   each component normalized over its OWN maximum
   (`fusion_cost.FusionCostResult.fusion_norm / trunc_norm`). With budget 4.5
   and mrpc-47 (fusion max 3840): one block2 flip = +0.117, block4 = +0.102,
   block5 = +0.031; block2+block5 full fusion = +1.78 ≫ margin cap 0.5; K
   alone caps at 1.5. Fusion is now strictly required to top P3. The user's
   80:150:130:40:50 value ratios are preserved within the fusion component.
2. **No warmstart prior on the fusion option slot** (`preferred = [-1, K]`;
   `apply_preferred_per_step_bias` accepts -1 = "no prior on this slot").
   The 60-episode forced anchor alone grounds cold start; K keeps its prior.
3. **Scheduled forced-fusion probes**
   (`fusion_curriculum.fusion_probe_target_block`, default interval 200,
   `--blb-v3-fusion-probe-interval`, 0 disables): every 200 post-anchor
   episodes, one episode forces fusion option 1 on ONE block type (rotating
   block2 → block5 → block4) at baseline K, scored normally (forced-branch
   `evaluate_action` under the open mask — PPO-sound, same mechanism as the
   anchor). Pure function of the absolute episode index → deterministic and
   identical across episode-parallel workers (1==N preserved). 300 probes in
   60k ≈ 0.5% overhead. The block4 probe is *expected* to land P1 — truthful
   negative evidence the policy needs in order to learn selective block4 use.
4. **Tolerances for the rerun** (user spec, run flags not code defaults):
   `--stage2-stability-tolerance 5.0` (500%) and `--stage2-limit-tolerance
   0.005` (0.5%). Plus a real bug fix: `RewardWeights.stab_tolerance` (used
   by the v3 m1/m2/loss std channels inside `compute_reward`) silently stayed
   at its dataclass default 0.5 regardless of the CLI flag; the sequential
   runner now writes the CLI tolerance into the weights, so relaxing
   stability relaxes ALL stability channels.

## Alternatives considered

- **Raise `p3_cost_budget` instead of splitting**: inflates K and fusion
  equally — does not fix K's ability to consume the whole cost reward.
- **Inflate block5's weight so its flip is "visible"**: silently overrides the
  user's stated cost ratios; rejected. Visibility is restored by removing the
  failure tax (tolerances), the prior, and by probes feeding evidence.
- **Force fusion floor / mask out option 0**: violates mental-model item 2's
  spirit (RL must be able to choose baseline); rejected.
- **Probe forcing ALL fusable blocks at once**: includes 12× block4 → P1
  every probe → feeds "fusion is bad" evidence; rejected in favor of
  per-block-type rotation.

## Consequences

- Reward scale within P3 changes (fusion-heavy configs now out-rank
  fusion-0+deep-K). Policy/critic head shapes unchanged → checkpoint variant
  string stays `…_fusioncount_v1`, but reward values are not comparable with
  pre-ADR-011 runs; cross-run reward comparisons must split at this ADR.
- `terminal_cost_rank_score` (unbounded P3 rank) is unchanged — candidate
  ranking already weighted fusion correctly; only the PPO scalar was broken.
- The gate run adds a probe-presence verdict (ep60=block2, ep260=block5,
  fusion_count ≥ 10) so a regressed probe path fails fast on the server.
- If the rerun still parks at fusion=0 WITH these fixes, the next suspects
  are (in order): per-episode advantage noise vs +0.031 block5 signal,
  curriculum ramp length, and the value-head's ability to separate per-block
  contributions (potential per-block reward attribution work).
