# Paper Figure Episode Column Streaming Evidence

Source commit: `dcfea75`
Red-test commit: `86a60a3`

## Optimization

`tools/paper_figures.load_run()` now reads `episodes.jsonl` as a direct
`total_reward` float column through shared `jsonl_utils.read_jsonl_float_field()`.
The training-curve and LaTeX-summary paths no longer build one
`{"total_reward": ...}` dictionary per episode when only reward values are used.

This keeps figure outputs semantically equivalent while reducing per-row object
allocation for long Stage-2 episode logs.

## Server Verification

Red run:

- Directory: `rfr_paper_episode_column_red_86a60a3_20260703_224715/`
- Command: `PYTHONPATH="$PWD" python -m unittest tests.test_paper_figures.PaperFiguresTest.test_load_run_projects_large_jsonl_rows_to_needed_fields tests.test_paper_figures.PaperFiguresTest.test_load_run_streams_episode_rewards_as_float_column -v`
- Result: `red_rc=1`
- Expected failures:
  - `load_run()` returned `{"total_reward": ...}` dictionaries.
  - `load_run()` still called `read_jsonl_fields()` for `episodes.jsonl`.

Green run:

- Directory: `rfr_paper_episode_column_green_dcfea75_20260703_225013/`
- Commands:
  - `PYTHONPATH="$PWD" python -m py_compile jsonl_utils.py tools/paper_figures.py tests/test_paper_figures.py tests/test_jsonl_utils.py`
  - `PYTHONPATH="$PWD" python -m unittest tests.test_paper_figures tests.test_jsonl_utils -v`
- Results:
  - `green_py_compile_rc=0`
  - `green_unittest_rc=0`
  - `Ran 20 tests ... OK`
