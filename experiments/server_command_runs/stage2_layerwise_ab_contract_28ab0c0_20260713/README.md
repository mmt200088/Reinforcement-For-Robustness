# Stage-2 Layerwise A/B Command Contract Evidence

This bundle verifies the no-GPU command contract for the pending production
1GPU-versus-5GPU Stage-2 gate at source
`28ab0c082f2a77ef90bb3347e72bcf989412e676`.

## Problem

The production source `24e919cea64f22b1b869b9bf22d57bbb776bac5e`
dispatches `decision_granularity=layer` and `reward_design=robust_constrained`.
That path explicitly rejects the old gate's `--stage2-rl-devices` episode
parallelism and requires `--blb-v3-reward-devices` K-trial splitting instead.
The old gate also overrode the production batch, rollout, tolerance,
warmstart, curriculum, and forced-fusion settings.

## Result

- The single command builder now drives both the recorded preflight and the
  real launch, so the evidence command cannot drift from the executed array.
- Physical `CUDA_VISIBLE_DEVICES` values are remapped to process-local reward
  device ids: the one-GPU case uses `0`; the five-GPU case uses `0,1,2,3,4`.
- Both cases use batch 64, rollout 120, all-4 Stage-1 config, PPO, layerwise
  decisions, robust constraints, and the same baseline/probability/trial
  settings as the formal run.
- Legacy neighbor curriculum, warmstart anchor, forced-fusion probe,
  exploration epsilon, batch 512, relaxed tolerances, and
  `--stage2-rl-devices` are absent.
- `PRINT_EFFECTIVE_COMMANDS=1` writes both complete commands and exits before
  querying `nvidia-smi`, allowing a non-contaminating readiness gate.

## Verification

- RED source `119511c330d4cce98edc99aef7b680bebfc1f4b2` failed because the old
  implementation invoked the fake `nvidia-smi` and had no effective-command
  preflight.
- GREEN passed `bash -n`, all 18 targeted launcher/comparator tests, and a real
  five-device command preflight with CUDA hidden.
- The server worktree was clean before and after verification. GitHub fetch
  timed out, so the GREEN source was fetched only after the Git mirror returned
  the exact expected branch SHA.

This is command-contract evidence, not the final GPU result. The strict
episode/PPO equality and throughput gate remains pending until the active
60,000-episode process releases all five GPUs and its layerwise source is
integrated.

## Files

- `red/`: focused failing test and exact RED source/status evidence.
- `green_final/`: syntax result, 18-test result, exact Git sync evidence, and
  the generated one-GPU/five-GPU commands.
