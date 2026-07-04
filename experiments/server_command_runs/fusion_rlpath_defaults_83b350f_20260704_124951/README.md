# Fusion RL-Path Default JSON Evidence

Source commit: `83b350f`

Optimization: `scripts/run_fusion_count_action_eval_rlpath.py` now computes the
fixed Stage-1 GELU and Softmax default JSON strings once at module load and
reuses those strings for argparse defaults. This avoids repeating the same
`json.dumps()` calls whenever the RL-path fusion-count comparison CLI builds
its parser.

Server evidence:

- RED: `/hy-tmp/fusion_rlpath_stage1_defaults_red_6a033d3_20260704_124951`
  ran the new focused test against the previous source and failed with
  `red.rc=1` because the old implementation still dumped the fixed defaults
  inline inside `main()`.
- GREEN: `/hy-tmp/fusion_rlpath_stage1_defaults_green_20260704_124951`
  ran `python3 -m py_compile scripts/run_fusion_count_action_eval_rlpath.py`
  and `python3 -m unittest tests.test_run_fusion_count_action_eval_rlpath -v`.
  `py_compile.rc=0`, `green.rc=0`, 14 tests passed.
