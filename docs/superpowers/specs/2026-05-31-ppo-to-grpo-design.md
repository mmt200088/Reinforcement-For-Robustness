# PPO → GRPO (selectable; minimal swap of the advantage estimator)

- **Date:** 2026-05-31
- **Status:** Implemented; locally verified (py_compile + torch-free tests + the
  GRPO math/wiring tests + preset validator + `bash -n`). Behavioral RL
  correctness is **server-side** (no torch locally).
- **Principle:** *replace the RL algorithm, NOT the RL design.* PPO is kept
  intact; GRPO is selectable; rollout / reward / action-space / warmstart /
  curriculum / multi-GPU / persistence layout are all untouched.

## Grilled decisions (locked)

1. **Group = the existing per-update episode window.** Every episode starts from
   the same frozen model + static_skeletons baseline, so one update window is one
   GRPO group. No explicit group sampling, no new group-size knob.
2. **Advantage = group-relative episode return**, outcome-supervised:
   `A_i = (R_i − mean)/(std+ε)` over the window, broadcast to every step of
   episode i. `R_i` = undiscounted sum of that episode's rewards. **Drops** GAE +
   the critic value loss + return-normalization + the old MAD adv-norm.
3. **Critic head kept but untrained** (no value loss → no gradient) so network /
   checkpoint shapes are unchanged.
4. **+ Reference-KL** (faithful GRPO): a frozen reference = policy snapshot at
   GRPO start; `+ β·KL(π_θ‖π_ref)` via the unbiased **k3** estimator
   `exp(Δ)−Δ−1`, Δ = ref_logp − new_logp. `--grpo-kl-beta` (default 0.04).
   Reference is frozen for the whole run and restored on resume (Stage-2: in the
   checkpoint as `grpo_reference_policy`; Stage-1: sidecar
   `stage1_grpo_reference.pt`).
5. **Kept** (critic-independent stability): clipped-ratio surrogate, entropy
   bonus, per-slot entropy recovery, KL-adaptive LR + early-stop, warmstart-prior
   replay, non-finite-minibatch backoff.
6. **Single `--rl-algo {ppo,grpo}`** (default `ppo`), both stages together.
7. **Output → `GRPO Chapter/…`** (identical internal structure to
   `Parting Chapter`) when `grpo`. Separate trees ⇒ PPO/GRPO checkpoints never
   collide; resume auto-selects the right tree.

## Implementation map

- `grpo_common.py` (new, torch-free): `grpo_group_normalize`,
  `segment_episode_returns`, `grpo_per_step_advantages`. Tested by
  `tests/test_grpo_common.py`.
- **Stage-2** `blb_stage2_rl/sequential_policy.py`: `sequential_grpo_update`
  (mirror of `sequential_ppo_update`); `SequentialRolloutBuffer.grpo_advantages`.
  `sequential_runner.py`: `SequentialTrainConfig.rl_algo`/`grpo_kl_beta`;
  `train_sequential(reference_policy=…)` + dispatch at the update site;
  reference snapshot/restore in `run_sequential_via_runner`; checkpoint key
  `grpo_reference_policy`; algo-aware phase label.
- **Stage-1** `layer_importance_evaluator.py`: `grpo_update_gtrxl` (mirror of
  `ppo_update_gtrxl`); evaluator `rl_algo`/`grpo_kl_beta`; frozen reference
  snapshot (sidecar) after net setup+resume; dispatch at the
  `(episode+1)%PPO_UPDATE_INTERVAL` update. Works for single- and multi-GPU
  rollout (both route through that update; reference forward is on cuda:0).
- **Config/CLI** `blb_stage2_rl/runner.py`: `BLBStage2TrainConfig.rl_algo`/
  `grpo_kl_beta` + `_build_train_config_from_evaluator` copies them from the
  evaluator. `rl_tune.py`: `--rl_algo`/`--grpo_kl_beta` → evaluator.
  `llama_7B_LayerImportance.sh`: `--rl-algo`/`--grpo-kl-beta`, validation,
  `GRPO Chapter` root swap (only when `--persistent-root` not overridden),
  forwards both flags to `rl_tune`.

## Verification

- Local ceiling (no torch): `py_compile` (7 files), `bash -n`, the torch-free
  `tests/test_grpo_common.py` (group-norm math, episode segmentation, broadcast,
  clip) + `tests/test_grpo_wiring.py` (source-text: both update fns present,
  dispatch, config fields, CLI flags, dir swap), preset validator (15/15),
  ruff-clean new files. The torch-free BLB gate is unchanged (25 pre-existing
  no-torch errors, zero new).
- **Server (todo):** a short `--rl-algo grpo` run per stage; confirm outputs land
  under `GRPO Chapter/`, GRPO metrics include `kl_ref`/`kl_beta`, reward curve is
  a normal RL curve, and PPO mode is byte-for-byte unchanged.

## Compatibility notes

- Built on top of the uncommitted gelu-only (A) + block3-removal (C) working
  tree: Stage-1 GRPO mirrors the *current* gelu-only `evaluate_actions`
  signature; Stage-2 GRPO is advantage-only so the block3-removed schedule is
  irrelevant to it. Nothing committed yet.
