# Paper Figures Payload Reuse Verification

Source commit: `b6dda66`

Server temp root: `/hy-tmp/rfr_paper_payload_reuse_20260704_094735`

Purpose:

- Avoid copying JSON-native `list` and `dict` payloads while loading paper
  figure run metadata.
- `best_action_vec`, `best_slots`, `baseline_slots`,
  `diff_vs_baseline`, and `first_invalid_counts` now reuse payloads returned
  by `read_json_file()` when they are already native JSON containers.
- Non-native truthy containers keep the old compatibility path through
  `list(...)` / `dict(...)`.

Server verification:

- RED package: old `tools/paper_figures.py` plus the new regression test.
  - Command: `python3 -m unittest
    tests.test_paper_figures.PaperFiguresTest.test_load_run_reuses_json_native_action_payloads_without_copy
    -v`
  - Result: expected failure at `best_action_vec=list(...)`.
  - Log: `red.log`
- GREEN package: source commit `b6dda66` changes plus tests.
  - Command: same single regression test.
  - Result: OK.
  - Log: `green.log`
- Wider GREEN:
  - Command: `python3 -m py_compile tools/paper_figures.py
    tests/test_paper_figures.py && python3 -m unittest
    tests.test_paper_figures -v`
  - Result: OK, 5 tests.
  - Log: `green_full.log`
