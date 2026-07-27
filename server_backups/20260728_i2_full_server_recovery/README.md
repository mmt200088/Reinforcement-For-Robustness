# i-2 Full Server Recovery Snapshot

This directory is the recovery snapshot for the GPUShare server state captured
at `2026-07-27T18:09:40Z` (`2026-07-28 02:09:40` Asia/Shanghai), immediately
after the Stage-2 H/M/L precision-preset CUDA smoke completed.

## Coverage

- Seven Git worktrees are represented by exact commit IDs and tracked-tree
  hashes in `git_worktrees.tsv`.
- All server-only untracked files and relevant ignored result files from those
  worktrees are in the payload.
- All 149 non-worktree roots under `/hy-tmp` are in the payload, including
  experiment outputs, logs, Git bundles, 18/21/27-group results, exact-probe
  results, RL structured data, and the H/M/L real-chain smoke.
- The payload contains 1,829 files or symlinks and 3,281,366,297 uncompressed
  bytes. Its seven Git-compatible parts total 576,596,530 bytes.
- Capture found no live project process under `/hy-tmp`; the payload therefore
  has a stable write boundary.

Only disposable Python/test caches and the reproducible Hugging Face cache
payloads are omitted. The excluded Hugging Face cache is 1,799,158,734 bytes;
`hf_cache_sources.json` records exact repository revisions and
`hf_cache_manifest.tsv` records all 90 original cache paths, sizes, and hashes.

## Integrity Evidence

- `PART_SHA256SUMS.tsv` authenticates each archive part.
- `METADATA_SHA256SUMS.tsv` authenticates the capture and recovery metadata.
- `payload_manifest.tsv` authenticates every restored server-only path.
- A server-side extraction rehearsal restored and rehashed all 1,829 payload
  entries successfully: `PAYLOAD_RESTORE_OK files=1829 bytes=3281366297`.
- Every worktree commit was confirmed reachable from the Git remote at capture
  time. Six were exact remote tips; the remaining commit was an ancestor of
  `jk_standard_rl`.

## Restore

Start with a normal source checkout that has all remote branch objects:

```bash
git clone <repository-url> RFR-source
git -C RFR-source fetch origin '+refs/heads/*:refs/remotes/origin/*'
mkdir /path/to/empty-recovery-root
./server_backups/20260728_i2_full_server_recovery/restore.sh \
  /path/to/RFR-source \
  /path/to/empty-recovery-root
```

The script verifies metadata and archive hashes, reconstructs each worktree at
its exact captured commit, overlays the server-only payload, and verifies every
payload path against `payload_manifest.tsv`. It deliberately refuses a
non-empty recovery root.

After verification, move or synchronize
`/path/to/empty-recovery-root/hy-tmp/` to the replacement server's
`/hy-tmp/`. Recreate the Hugging Face cache from the revisions in
`hf_cache_sources.json`; use `hf_cache_manifest.tsv` if byte-for-byte cache
verification is required.

## Important Files

- `capture_summary.json`: machine-readable scope and byte counts.
- `hy_tmp_top_level_inventory.tsv`: every captured top-level server path.
- `git_worktrees.tsv` and `git_refs.tsv`: exact Git reconstruction data.
- `payload_manifest.tsv`: path-level SHA-256 manifest.
- `ignored_cache_summary.tsv`: included results versus excluded disposable
  cache counts.
- `project_processes.json`, `system_snapshot.txt`, `gpu_inventory.txt`:
  capture-time environment evidence.
- `capture_server_state.py`: reproducible inventory/capture tool.
- `restore.sh`: guarded recovery and verification tool.
