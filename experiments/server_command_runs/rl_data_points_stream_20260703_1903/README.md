# RL Data Point Streaming Verification

- Source commit: `12cf8c4561a1e64f687d004466b1ef577dfd6907`
- Parent red-check commit: `e38aa12681910a1fb9047525511bec41ab3cb040`
- Local remote proof before server run: `origin/jk_standard_rl = 12cf8c4561a1e64f687d004466b1ef577dfd6907`
- Server run directory: `/hy-tmp/rfr_rl_data_points_stream_12cf8c4_20260703_1903`
- Server Python: `Python 3.11.12`

The server-side `git ls-remote` attempt timed out after 45 seconds
(`server_ls_remote.rc = 124`), so this verification used minimal `git archive`
packages exported locally from the already-pushed commits and copied to the
server. The server did not edit project source; it only unpacked the archives
and ran the checks below.

## Checks

- Red check: parent source plus stdin verification script, expected to catch
  old whole-file JSON writes and per-row `json.dumps()` JSONL writes.
  Result: `red_rc=0`.
- Green check: `python3 -m unittest tests.test_rl_data_points -v` from the
  `12cf8c4` source package.
  Result: `green_rc=0`, `22` tests OK.

## Evidence Files

- `summary.txt`: server return codes and log paths.
- `red.log`: expected failures from the parent implementation.
- `green.log`: passing focused unit test output from the source commit.
- `archive_sha256_server.txt`: hashes of the exact archives unpacked on server.
- `source_identity.txt`: server timestamp, expected commits, host, and Python.
- `server_ls_remote.log` / `server_ls_remote.rc`: server GitHub fetch attempt.
