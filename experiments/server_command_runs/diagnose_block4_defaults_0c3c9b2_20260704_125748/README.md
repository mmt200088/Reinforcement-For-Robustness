# Block4 Diagnosis Default JSON Evidence

Source commit: `0c3c9b2`

Optimization: `scripts/diagnose_block4_fusion_install.py` now reuses the
Stage-1 GELU and Softmax default JSON strings exported by the RL-path fusion
diagnostic helper instead of calling `json.dumps()` again when building its
argument parser.

Server evidence:

- RED: `/hy-tmp/diagnose_block4_defaults_red_a7fb68e_20260704_125748` ran the
  new focused static test against the previous source and failed with
  `red.rc=1` because the old implementation still dumped the defaults inline.
- GREEN: `/hy-tmp/diagnose_block4_defaults_green_20260704_125748` ran
  `python3 -m py_compile scripts/diagnose_block4_fusion_install.py` and
  `python3 -m unittest tests.test_diagnose_block4_fusion_install_static -v`.
  `py_compile.rc=0`, `green.rc=0`, 2 tests passed.
