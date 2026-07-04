# Stage-2 A/B Excluded-Key Reuse Verification

Source commit: `623cd5d`

Server temp root: `/hy-tmp/rfr_ngpu_excluded_keys_20260704_101330`

Purpose:

- Avoid rebuilding the canonical comparison excluded-key set for every row and
  for both sides of every Stage-2 1GPU-vs-NGPU A/B comparison.
- `compare_rows()` now materializes the excluded keys once per comparison and
  reuses that set in the per-row canonicalization hot loop.
- The comparison semantics stay unchanged for normal lists/sets and improve
  correctness for single-pass iterables passed as `excluded_keys`.

Server verification:

- RED package: old `scripts/stage2_ngpu_ab_compare.py` plus the new regression
  test.
  - Command: `python3 -m unittest
    tests.test_stage2_ngpu_ab_compare.Stage2NgpuCompareTests.test_compare_rows_materializes_excluded_keys_once
    -v`
  - Result: expected failure at the old repeated `set(excluded_keys)` path.
  - Log: `red.log`
- GREEN package: source commit `623cd5d` changes plus tests.
  - Command: same single regression test.
  - Result: OK.
  - Log: `green.log`
- Wider GREEN:
  - Command: `python3 -m py_compile scripts/stage2_ngpu_ab_compare.py
    tests/test_stage2_ngpu_ab_compare.py jsonl_utils.py json_utils.py &&
    python3 -m unittest tests.test_stage2_ngpu_ab_compare -v`
  - Result: OK, 11 tests.
  - Log: `green_full.log`
