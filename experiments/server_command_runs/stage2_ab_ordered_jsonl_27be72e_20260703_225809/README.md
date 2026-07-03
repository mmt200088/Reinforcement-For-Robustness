# Stage-2 A/B Ordered JSONL Load Evidence

Source commit: `27be72e`
Red-test commit: `77a78e0`

## Optimization

`scripts/stage2_ngpu_ab_compare.py` now detects whether an input JSONL stream is
already ordered by `episode` or `update`. Normal append-only Stage-2 artifacts
return directly without sorting; out-of-order artifacts still fall back to
sorting so comparison semantics are preserved.

This reduces unnecessary `O(n log n)` sorting work in the 1GPU-vs-NGPU
throughput/equality gate for the common ordered-log case.

## Server Verification

Red run:

- Directory: `rfr_stage2_ab_ordered_red_77a78e0_20260703_225605/`
- Command: `PYTHONPATH="$PWD" python -m unittest tests.test_stage2_ngpu_ab_compare.Stage2NgpuCompareTests.test_load_jsonl_does_not_unconditionally_sort_ordered_logs -v`
- Result: `red_rc=1`
- Expected failure: `_load_jsonl()` contained an unconditional `.sort(...)`.

Green run:

- Directory: `rfr_stage2_ab_ordered_green_27be72e_20260703_225809/`
- Commands:
  - `PYTHONPATH="$PWD" python -m py_compile scripts/stage2_ngpu_ab_compare.py tests/test_stage2_ngpu_ab_compare.py`
  - `PYTHONPATH="$PWD" python -m unittest tests.test_stage2_ngpu_ab_compare -v`
- Results:
  - `green_py_compile_rc=0`
  - `green_unittest_rc=0`
  - `Ran 10 tests ... OK`
