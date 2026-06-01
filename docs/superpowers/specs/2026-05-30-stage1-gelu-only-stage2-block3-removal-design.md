# Stage-1 gelu-only (softmax fixed deg-6, gelu+ReLU), Stage-2 block3 removal

- **Date:** 2026-05-30
- **Status:** Approved; implementing. **B, A, C = DONE & verified (py_compile +
  torch-free tests; behavioral verification is server-side, no torch locally).
  D = TODO** — the ReLU→Stage-2-block5 guardrail; continue in a fresh session.
- **Method change — invalidates existing Stage-1 AND Stage-2 checkpoints (fresh
  retrain required).**

## Progress (2026-05-30)

- **[DONE] B — gelu degree-0 = ReLU.** Verified by py_compile + grep:
  - `layer_importance_evaluator.py:72` → `GELU_MAP = {0:4, 1:2, 2:1, 3:0}`
    (+ comment updated).
  - `get_gelu_action_mask` (~line 2008) → `np.array([True,True,True,True])`
    (4th action enabled; docstring index line updated to `3=degree0(ReLU)`).
  - `function_handler.py:2678` `replace_layer_gelu` → `new_act = nn.ReLU() if
    int(degree) == 0 else PolynomialGELU(degree=degree)` (`nn` already imported
    at function_handler.py:4; reuse cache machinery unchanged). No other repo
    site constructs `PolynomialGELU(degree=0)`.
- **[DONE] A** — Stage-1 gelu-only. Active `GTrXLStrategyNetwork` (+ legacy
  `PolicyNetwork`) lost `head_s` / `embed_prev_s` / `prev_s` token; forward /
  get_action_and_logprob / evaluate_actions are gelu-only. `RecurrentRolloutBuffer`
  + `EpisodeRollout` dropped the `prev_s_actions`/`actions_s` streams.
  `TransformerOptEnv.step(gelu_action)` fixes softmax at `FIXED_SOFTMAX_DEGREE=6`;
  `_get_state` dropped `softmax_norm`+`softmax_history` (44→31) and now exposes the
  6 policy continuous features BY NAME via `get_policy_cont_features()`, so both
  rollout loops + the worker stop magic-indexing the flat state (the silent-bug
  risk under no-torch). PPO update / both rollout loops / worker /
  `stage1_rl/parallel_runner.py` carry no softmax stream; `softmax=[6]*L` flows via
  `env.softmax_config`. `_resolve_stage2_fixed_stage1_config` forces the Stage-2
  handoff softmax to `[6]*L` (block3_exp_n6). Resume guard raises clearly on a
  legacy softmax-bearing checkpoint (`head_s`/`embed_prev_s` keys → tell user
  `--fresh`). Dead `LSTMStrategyNetwork` + `SOS_TOKEN_SOFTMAX`/`SOFTMAX_MAP`
  constants intentionally kept. Verified: py_compile + tests/
  test_stage1_selection_semantics.py + exhaustive straggler sweep.
- **[DONE] C** — Stage-2 block 3 removed from the *decided* schedule.
  `_LAYER0_BLOCK_ORDER=(2,4,5)`, `_LAYER_GE_1_BLOCK_ORDER=(1,2,4,5)`,
  `horizon_for_num_layers = 3 + (L-1)*4` (47 for L=12). `_BLOCK3_FIELDS` /
  `build_block3_cfg_from_action` kept; the legacy 577-dim full vec is UNCHANGED
  (block 3 slots preserved). The sequential env now seeds `_pending_full_vec` with
  the all-max baseline (`make_all_max_action_vector`) instead of all-min, so block
  3 — never written by a step — stays frozen at the static_skeletons baseline.
  Bridge never installs block 3 noise (`replace_layer_block3_noise` kept, uncalled,
  applied uniformly to baseline+candidates). **Cost exclusion:** block 3 stays in
  the optimizer requests (modulus chain unchanged) but, because the cost reward is
  baseline-relative (`bits_gain`/`fusion_gain`/`k_gain` = baseline − candidate) and
  block 3 is frozen at the SAME value in both the candidate and the archive-derived
  baseline, it cancels EXACTLY → literal zero-effect exclusion with no edits to the
  reward/aggregate math (editing only the candidate side would mismatch the archive
  baseline that includes block 3). Substage path verified compatible (filters
  step_schedule to active block ∈ {1,2,4,5}; `frozen_base` carries block 3 baseline).
  Diagnostics `/59` → horizon-aware. New torch-free
  tests/test_blb_block3_removed_schedule.py. Verified: py_compile + that test +
  full torch-free BLB suite (no new failures vs the 25 pre-existing torch-absent
  errors).
- **[TODO] D** — Stage-2 baseline_bootstrap / block5 raises a clear error on a
  degree-0 layer (ReLU→block5 is a future task).

## Locked decisions

| # | Decision |
|---|----------|
| A | Stage-1 RL decides **gelu only**. Softmax is no longer an action; every layer is fixed to **degree 6**. The policy's softmax head + prev-softmax embedding are **fully removed**. |
| B | gelu gains a **degree-0 = ReLU** action by reusing the currently-masked 4th gelu slot. `GELU_MAP = {0:4, 1:2, 2:1, 3:0}`; degree 0 installs `nn.ReLU()` (not the degree-0 polynomial). |
| C | Stage-2 **block3 fully removed from RL**: not decided, no noise installed, and its cost excluded from every RL cost signal. Noise interface (`replace_layer_block3_noise` etc.) is **kept but unused** (may be re-enabled later). |
| D | gelu degree-0 = ReLU only affects **Stage-1** this round. If Stage-2 baseline bootstrap / block5 hits a degree-0 layer, it **raises a clear error** ("degree-0/ReLU Stage-2 block5 not implemented") — never silently uses a polynomial. |

## A. Stage-1 softmax removal (gelu-only)

- `softmax_degrees` is always `[6]*L`. `SOFTMAX_MAP` stays defined but is not an
  action source.
- `GTrXLStrategyNetwork` + `PolicyNetwork`: remove `head_s`, `embed_prev_s`,
  `prev_s_actions` token input, `SOS_TOKEN_SOFTMAX`; `forward` /
  `get_action_and_logprob` / `evaluate_actions` produce/consume gelu only.
- `TransformerOptEnv.step(gelu_action)` (softmax arg dropped); remove softmax
  dims from the state vector; reward optimizes gelu cost only (softmax cost is a
  constant).
- `_stage1_collect_episode_in_worker` + `stage1_rl/parallel_runner.py`: drop the
  softmax action stream.
- All `stage1_evaluate` / `evaluate_model` call sites pass `softmax=[6]*L`.
- Resume: if a loaded checkpoint has the old (softmax-bearing) shape, error
  clearly and tell the user to `--fresh`.

## B. gelu degree-0 = ReLU

- `GELU_MAP = {0:4, 1:2, 2:1, 3:0}` + comment update (4th action is now
  degree-0=ReLU, not a masked filler).
- `get_gelu_action_mask` → `[True, True, True, True]` (4th action enabled).
- `replace_layer_gelu`: `degree == 0` installs a cached `nn.ReLU()` (reuses the
  approx-module cache machinery). `GELU_COST[0] = -1.0` already exists.
- `PolynomialGELU` degree-0 polynomial path is left intact (unused by Stage-1 now)
  but is no longer what action degree-0 maps to.

## C. Stage-2 block3 removal

- `blb_stage2_rl/action_space.py`: drop block3's 8 slots from the decided action
  space; `step_schedule` skips block3 (horizon 59→~47). Keep `_BLOCK3_FIELDS` /
  `build_block3_cfg_from_action` defined but unused by the schedule.
- `blb_rl_bridge.py`: `apply` never installs block3 noise (`block3_cfgs` always
  empty); `replace_layer_block3_noise` kept, not called.
- `baseline_bootstrap`: no block3 action/cfg generated.
- **Cost exclusion:** verify `ReplanSession` — if block3 can be dropped from
  optimizer requests without breaking the modulus chain, drop it; otherwise the
  optimizer still processes block3 internally to keep the chain valid, but its
  bits/fusion/k are excluded from every RL cost signal / reward / ranking
  (block3 is fixed → its cost is a constant → zero effect on RL search).
- Stage-2 policy horizon/action dims change → old Stage-2 checkpoints invalid.

## D. ReLU → Stage-2 block5 guardrail

- Stage-2 `baseline_bootstrap` (and any block5 cfg builder) raises on a degree-0
  layer with an explicit message. Stage-1 is free to choose degree-0; turning a
  degree-0 Stage-1 into a Stage-2 ciphertext config is a separate future task.

## Testing

- Torch-free: `GELU_MAP` degree-0 mapping + mask all-True; `step_schedule` after
  block3 removal (horizon, dims, no block_idx==3 step); Stage-2 record / action
  decode has no block3 cfgs.
- Torch (server): a tiny Stage-1 forward installs ReLU for a degree-0 layer and
  degree-{4,2,1} polynomials elsewhere; a tiny Stage-2 build errors cleanly on a
  degree-0 layer.

## Sequencing / risks

- Implement B → A → C → D, verifying each.
- Independent of the (not-yet-implemented) persistence/final-eval specs, but
  touches the same `action_space`/policy/bridge; coordinate with the local
  uncommitted substage/OSR work to avoid conflicts.
