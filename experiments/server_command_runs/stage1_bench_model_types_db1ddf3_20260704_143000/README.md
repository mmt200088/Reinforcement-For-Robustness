# Stage-1 Benchmark Model Choices Evidence

Source commit: `db1ddf3`

Optimization: `scripts/stage1_approx_reuse_benchmark.py` now builds
`_MODEL_TYPES = tuple(_DIMS)` once and reuses that tuple for argparse
`--model-type` choices instead of rebuilding `list(_DIMS)` whenever the
benchmark parser is created.

Server evidence:

- RED: `/hy-tmp/stage1_bench_model_types_red_9c3952c_20260704_143000` ran the
  new focused parser-choice test against the previous source and failed with
  `red.rc=1` because the old implementation still used `choices=list(_DIMS)`.
- GREEN: `/hy-tmp/stage1_bench_model_types_green_20260704_143000` ran
  `python3 -m py_compile scripts/stage1_approx_reuse_benchmark.py
  tests/test_stage1_approx_reuse_benchmark.py stats_utils.py` and the complete
  `python3 -m unittest tests.test_stage1_approx_reuse_benchmark -v` suite.
  `py_compile.rc=0`, `green.rc=0`, 2 tests passed.
