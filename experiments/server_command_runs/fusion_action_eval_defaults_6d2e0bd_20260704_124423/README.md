# Fusion Action Eval Default JSON Evidence

Source commit: `6d2e0bd`

Optimization: `scripts/run_fusion_count_action_eval.py` now computes fixed
Stage-1 parser default JSON strings and the fixed manual Stage-2 noise JSON
once at module load, then reuses those strings in `main()` and `_run_one()`.
This avoids repeated `json.dumps()` work when the fusion-count action eval CLI
builds its parser and when it schedules multiple Paean final-eval action runs.

Server evidence:

- RED: `/hy-tmp/fusion_action_eval_stage1_defaults_red_bc3870b_20260704_124423`
  ran the new focused test against the previous source and failed with
  `red.rc=1` because the old implementation still dumped the fixed defaults
  inline.
- GREEN: `/hy-tmp/fusion_action_eval_stage1_defaults_green_20260704_124423`
  ran `python3 -m py_compile scripts/run_fusion_count_action_eval.py` and
  `python3 -m unittest tests.test_run_fusion_count_action_eval -v`.
  `py_compile.rc=0`, `green.rc=0`, 7 tests passed.
