# Candidate Store JSONL Streaming Verification

- Runtime source commit: `f6a51e8e3fec689e2d9d64c56f83bf87a512c6bd`
- Test-stabilization commit verified as head: `53b3cb8e75fba0e326f0ed371553a5087597e395`
- Parent red-check commit: `b26552a8b8f371399706f65d76ae241b67281b0f`
- Server run directory: `/hy-tmp/rfr_candidate_store_jsonl_f6a51e8_20260703_1917`
- Server Python: `Python 3.11.12`

The runtime source commit streams append-only candidate-store JSONL rows with a
reused `JSONEncoder.iterencode()` instead of allocating a complete
`json.dumps(...)+ "\n"` string for every candidate. The follow-up commit only
stabilized an existing path-mocking test so the focused suite runs against the
current shared JSONL reader.

## Checks

- Red check: parent source plus an inline verification script that requires
  `CandidateStore.append()` to write with `writelines()`.
  Result: `red_rc=0`.
- Green check: `python3 -m unittest tests.test_blb_candidate_store_identity -v`
  from the `53b3cb8` source package.
  Result: `green_rc=0`, `9` tests OK.
- Server `git ls-remote` succeeded for `refs/heads/jk_standard_rl`.
  Result: `server_ls_remote.rc=0`.

## Evidence Files

- `summary.txt`: server return codes and log paths.
- `red.log`: expected parent-implementation failure.
- `green.log`: passing focused test output from the verified head.
- `archive_sha256_server.txt`: hashes of the exact archives unpacked on server.
- `source_identity.txt`: server timestamp, expected commits, host, and Python.
- `server_ls_remote.log` / `server_ls_remote.rc`: server GitHub remote check.
