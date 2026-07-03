# Candidate-store ndarray normalization evidence

Source commit: `0aa212a`

Purpose: verify that `normalize_action_indices()` handles ndarray-backed
candidate action vectors through a direct flattening iterator instead of
copying through `.tolist()`.

Server runs:

- Red: `red/red_status.txt` records `red_rc=1`; `red/logs/red_unittest.log`
  shows the old path called the ndarray sentinel `.tolist()` method.
- Green: `green/green_status.txt` records `py_compile_rc=0`,
  `unittest_rc=0`, and `source_guard_rc=0`.

Green command coverage:

- `python -m py_compile blb_stage2_rl/candidate_store.py`
- `python -m unittest tests.test_blb_candidate_store_identity.BLBCandidateStoreIdentityTests -v`
- source guard confirming `action_indices.reshape(-1)` appears before the
  legacy `action_indices.tolist()` compatibility branch
