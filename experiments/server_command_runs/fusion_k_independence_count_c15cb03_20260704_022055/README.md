# Fusion K-independence streaming-count evidence

Source commit: `c15cb03`

Purpose: verify that `check_k_independence()` counts sample configs during the
existing scan instead of materializing `sample_configs` again with
`len(list(sample_configs))`.

Server runs:

- Red: `red/red_status.txt` records `red_rc=1`; the regression test failed on
  the old implementation because a streamed generator was already consumed and
  `samples_checked` was reported as `0` instead of `2`.
- Green: `green/green_status.txt` records `py_compile_rc=0`,
  `unittest_rc=0`, and `source_guard_rc=0`.

Green command coverage:

- `python -m py_compile blb_stage2_rl/fusion_enum.py`
- `python -m unittest tests.test_blb_fusion_count_map.CheckKIndependenceTest tests.test_blb_fusion_count_map.GroupMinNoiseOptionsTest -v`
- Source guard confirming `samples_checked += 1` is present and
  `len(list(sample_configs))` is absent from `check_k_independence()`.
