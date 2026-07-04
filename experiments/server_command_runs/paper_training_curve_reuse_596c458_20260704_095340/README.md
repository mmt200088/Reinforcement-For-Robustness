# Paper Training Curve Reward Series Reuse Verification

Source commit: `596c458`

Server temp root: `/hy-tmp/rfr_paper_training_curve_reuse_20260704_095340`

Purpose:

- Avoid copying already-loaded paper-figure episode reward lists before plotting
  training curves.
- `fig_training_curves()` now passes native list-backed `RunData.episodes`
  directly into plot calls.
- Non-list iterable reward series keep the old compatibility path through
  `[float(value) for value in values]`.

Server verification:

- RED package: old `tools/paper_figures.py` plus the new regression test.
  - Command: `python3 -m unittest
    tests.test_paper_figures.PaperFiguresTest.test_training_curve_reuses_native_episode_rewards_without_copy
    -v`
  - Result: expected failure at the old `[float(value) for value in r.episodes]`
    copy.
  - Log: `red.log`
- GREEN package: source commit `596c458` changes plus tests.
  - Command: same single regression test.
  - Result: OK.
  - Log: `green.log`
- Wider GREEN:
  - Command: `python3 -m py_compile tools/paper_figures.py
    tests/test_paper_figures.py && python3 -m unittest
    tests.test_paper_figures -v`
  - Result: OK, 6 tests.
  - Log: `green_full.log`
