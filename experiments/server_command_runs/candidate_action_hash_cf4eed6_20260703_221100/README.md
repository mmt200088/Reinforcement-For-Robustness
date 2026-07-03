# Candidate Action Hash Verification

Purpose: verify the `blb_stage2_rl/candidate_store.py` action-hash encoding
optimization.

- Red commit: `e4f4bea`
- Green/source commit: `cf4eed6`
- Server run roots:
  - `/hy-tmp/rfr_candidate_action_hash_e4f4bea_20260703_221100`
  - `/hy-tmp/rfr_candidate_action_hash_cf4eed6_20260703_221400`

Checks:

- Red target unittest:
  `tests.test_blb_stage2_rl_regressions.BLBPlaybookArtifactRegressionTests.test_candidate_action_hash_avoids_json_dumps_for_integer_vectors`
  failed because `_action_hash_from_tuple()` still called `json.dumps`.
- Green `py_compile` for `blb_stage2_rl/candidate_store.py` and
  `tests/test_blb_stage2_rl_regressions.py` returned `0`.
- Green target unittest returned `OK`.

Scope:

- The hash output stays byte-compatible with the old compact JSON integer array
  payload, e.g. `[4,3,2,-1]`.
- The implementation now streams the payload directly into `sha256`, avoiding
  the temporary list, JSON string, and UTF-8 encoded copy for normalized integer
  action tuples.
- This affects candidate/artifact identity generation only; it does not change
  reward, action sampling, optimizer semantics, or RL scheduling.
