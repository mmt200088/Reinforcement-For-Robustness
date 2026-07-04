# Paper Figures CLI Figure Names Evidence

Source commit: `9356c77`

Optimization: `tools/paper_figures.py` now builds `ALL_FIG_NAMES` once from
`ALL_FIGS` and reuses that tuple for the `--figs` argparse default and help
text. This avoids rebuilding `list(ALL_FIGS.keys())` every time the parser is
created.

Server evidence:

- RED: `/hy-tmp/paper_figs_cli_names_red_f778692_20260704_132500` ran the new
  focused parser-default test against the previous source and failed with
  `red.rc=1` because the old implementation still used `list(ALL_FIGS.keys())`.
- GREEN: `/hy-tmp/paper_figs_cli_names_green_20260704_132500` ran
  `python3 -m py_compile tools/paper_figures.py tests/test_paper_figures.py
  jsonl_utils.py json_utils.py` and the complete
  `python3 -m unittest tests.test_paper_figures -v` suite.
  `py_compile.rc=0`, `green.rc=0`, 8 tests passed.
