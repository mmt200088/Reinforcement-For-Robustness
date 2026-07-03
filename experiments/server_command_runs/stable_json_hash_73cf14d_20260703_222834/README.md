# Stable JSON Hash Streaming Evidence

Source commit: `73cf14d`
Red-test commit: `f5a1d0e`

## Optimization

`json_utils.stable_json_hash()` now streams canonical JSON chunks from the
shared stable `JSONEncoder` directly into `sha256`, instead of materializing the
full `stable_json_key()` string and then encoding it.

This preserves the stable hash contract while lowering peak string allocation
for large action registries, masks, candidate payloads, and report/artifact
identity keys.

## Server Verification

Red run:

- Directory: `rfr_stable_json_hash_red_f5a1d0e_20260703_222618/`
- Command: `PYTHONPATH="$PWD" python -m unittest tests.test_rl_data_points.RLDataPointWriterTest.test_stable_json_hash_streams_without_materializing_key -v`
- Result: `red_rc=1`
- Expected failure: current implementation called `stable_json_key()`.

Green run:

- Directory: `rfr_stable_json_hash_green_73cf14d_20260703_222834/`
- Commands:
  - `PYTHONPATH="$PWD" python -m py_compile json_utils.py tests/test_rl_data_points.py`
  - `PYTHONPATH="$PWD" python -m unittest tests.test_rl_data_points -v`
- Results:
  - `green_py_compile_rc=0`
  - `green_unittest_rc=0`
  - `Ran 24 tests ... OK`
