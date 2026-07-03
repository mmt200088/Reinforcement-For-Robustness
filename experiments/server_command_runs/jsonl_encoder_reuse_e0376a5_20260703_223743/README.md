# JSONL Encoder Reuse Evidence

Source commit: `e0376a5`
Red-test commit: `6d2798a`

## Optimization

`jsonl_utils.write_jsonl_rows()` now creates one `JSONEncoder` per finite JSONL
write and reuses `iterencode()` for every row, instead of calling `json.dump()`
inside the row loop.

This preserves the existing line-delimited output while reducing repeated
encoder construction overhead for bounded report and diagnostic JSONL artifacts
such as feasible-domain scans.

## Server Verification

Red run:

- Directory: `rfr_jsonl_encoder_red_6d2798a_20260703_223530/`
- Command: `PYTHONPATH="$PWD" python -m unittest tests.test_jsonl_utils.JsonlUtilsTest.test_write_jsonl_rows_reuses_encoder_without_json_dump_calls -v`
- Result: `red_rc=1`
- Expected failure: current implementation called `json.dump()` in the row loop.

Green run:

- Directory: `rfr_jsonl_encoder_green_e0376a5_20260703_223743/`
- Commands:
  - `PYTHONPATH="$PWD" python -m py_compile jsonl_utils.py tests/test_jsonl_utils.py`
  - `PYTHONPATH="$PWD" python -m unittest tests.test_jsonl_utils -v`
- Results:
  - `green_py_compile_rc=0`
  - `green_unittest_rc=0`
  - `Ran 15 tests ... OK`
