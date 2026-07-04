# OSR Baseline Action Hash Streaming

Source commit: `22b2fc8`

This run verifies that `blb_stage2_rl/osr.py` computes the
`baseline_action_vec_hash` for OSR fingerprints by streaming the compact JSON
array representation directly into sha256. This preserves the old
`json.dumps(..., separators=(",", ":"))` hash bytes while avoiding a full
Python list copy plus a full JSON string allocation.

Server temporary sources:

- RED: `/hy-tmp/osr_hash_stream_red_e44fddd_20260704_125000`
- GREEN: `/hy-tmp/osr_hash_stream_green_20260704_125000`

Verification:

- `red.rc`: `1`, expected failure on old source because the fingerprint path
  still used `reshape(-1).tolist()` and `json.dumps(bvec, ...)`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: full `tests.test_blb_osr` passed with 14 tests.
