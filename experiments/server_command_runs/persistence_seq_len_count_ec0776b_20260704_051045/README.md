# Persistence `_seq_len()` Streaming Count Evidence

Source commit: `ec0776b` (`Count unsized curve lengths without lists`)

This run verifies a report/artifact-path optimization in
`blb_stage2_rl/persistence.py`: unsized iterable length fallback now counts
items directly with `sum(1 for _ in values)` instead of materializing
`list(values)`.

## Server Verification

- Red package: `/hy-tmp/rfr_seq_len_count_red_ced7730_20260704_051007`
- Green package: `/hy-tmp/rfr_seq_len_count_green_20260704_051045`
- Red test:
  `python3 -m unittest tests.test_blb_stage2_outputs.UpgradedCurvesTest.test_seq_len_counts_iterable_without_list_materialization -v`
- Green gate:
  `python3 -m py_compile blb_stage2_rl/persistence.py tests/test_blb_stage2_outputs.py`
- Green gate:
  `python3 -m unittest tests.test_blb_stage2_outputs -v`
- Source guards:
  `grep -F -n 'return sum(1 for _ in values)' blb_stage2_rl/persistence.py`
  and no `return len(list(values))`.

## Result

- Red: failed because old `_seq_len()` called `list(values)`.
- Green: passed `py_compile`, all 31 `tests.test_blb_stage2_outputs` tests,
  and both source guards.

