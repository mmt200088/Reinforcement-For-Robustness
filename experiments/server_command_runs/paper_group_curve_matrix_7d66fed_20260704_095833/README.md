# Paper Group Training Curve Matrix Verification

Source commit: `7d66fed`

Server temp root: `/hy-tmp/rfr_paper_group_curve_matrix_20260704_095833`

Purpose:

- Avoid per-seed sliced-list copies in `fig_training_curves(..., group_label=...)`.
- The grouped training-curve path now fills the reward matrix by streaming the
  first `min_len` values from each seed through `itertools.islice()` and
  `np.fromiter()`.
- This keeps the required numpy matrix for mean/std bands while avoiding
  intermediate `s[:min_len]` lists for every seed.

Server verification:

- RED package: old `tools/paper_figures.py` plus the new regression test.
  - Command: `python3 -m unittest
    tests.test_paper_figures.PaperFiguresTest.test_group_training_curve_avoids_seed_slice_copies
    -v`
  - Result: expected failure at the old `s[:min_len]` copy.
  - Log: `red.log`
- GREEN package: source commit `7d66fed` changes plus tests.
  - Command: same single regression test.
  - Result: OK.
  - Log: `green.log`
- Wider GREEN:
  - Command: `python3 -m py_compile tools/paper_figures.py
    tests/test_paper_figures.py && python3 -m unittest
    tests.test_paper_figures -v`
  - Result: OK, 7 tests.
  - Log: `green_full.log`
