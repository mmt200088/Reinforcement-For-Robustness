# Fixed Block2/Block5 Stability Report Design

## Goal

Make the existing MRPC fixed-fusion comparison report visibly and verifiably
include the five-repeat stability evaluation that is already present in the raw
server results. This is a reporting-only change: it must not rerun inference or
introduce a second evaluation path.

## Source Data

The report continues to consume the two existing
`run_fusion_count_action_eval_rlpath.py` result files. Each required control and
treatment group must provide:

- top-level `repeat == 5`;
- terminal-probe `k == 5`;
- exactly five trial indices, `0..4`;
- exactly five recorded deterministic trial seeds;
- finite, non-negative `loss_std`, `metric1_std`, and `metric2_std` values;
- the existing metric means and conservative extrema (`loss_max`,
  `metric1_min`, and `metric2_min`).

The evaluator remains the merged canonical chain:

`SequentialEnv.evaluate_step -> commit_step -> BLBStage2Env.step`

No report code may recreate model configuration, replan, noise installation, or
metric evaluation.

## Report Changes

The comparison HTML will:

1. display Loss, Accuracy, and Weighted F1 as `mean +/- std` for control and
   treatment;
2. retain treatment-minus-control mean deltas;
3. add a stability table showing standard deviations and conservative extrema
   for every pair/group;
4. show trial count, trial indices, and deterministic seeds;
5. include a stability gate in the protocol-gate table.

The JSON comparison summary will retain separate numeric mean/std fields and
add normalized trial metadata. It will not replace numbers with formatted
strings.

## Failure Policy

Report generation fails with a non-zero exit status when either GELU pair lacks
exactly five trials, required standard deviations, or complete trial metadata.
Action/install gates remain unchanged. `all_gates_pass` requires both the
existing action/install gates and the new stability gate.

## Verification

Use test-first development:

- a synthetic five-trial payload must render `mean +/- std`, extrema, and all
  five seeds;
- a four-trial payload must fail the stability gate and return non-zero from the
  CLI;
- focused tests run on the server from an exact pushed commit;
- regenerate the two existing HTML copies from the already committed raw JSON,
  then commit and push the updated reports and gate evidence locally.

## Non-Goals

- No GPU inference rerun.
- No changes to noise generation, reward, fusion maps, replan, model install, or
  Stage-2 RL.
- No new evaluation implementation outside the existing merged module.
