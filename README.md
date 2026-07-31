# RFR server-data cloud archive (2026-07-31)

This branch is a self-contained, content-addressed overlay for the server data
that was present in the local RFR working tree on 2026-07-31. It is paired with
the immutable base branch:

- Base branch: `codex/local-primary-head-backup-20260731`
- Base commit: `c78aa8a80abd8718d0e123589e2c54ac13485ad2`
- Overlay branch: `codex/server-data-cloud-archive-20260731`

The base commit preserves every tracked file. The overlay preserves every
current untracked or modified experiment, report, Stage-1/Stage-2 result, and
supporting data file. A second overlay preserves research logs and generated
documents hidden by `.gitignore`. Content is deduplicated by SHA-256 and split
into chunks below GitHub's 100 MiB per-file limit.

## Exact server restore

Run these commands on the server, using an empty destination:

```bash
git clone --filter=blob:none --single-branch \
  --branch codex/local-primary-head-backup-20260731 \
  git@github.com:mmt200088/Reinforcement-For-Robustness.git \
  /hy-tmp/RFR-restored-20260731

git clone --filter=blob:none --single-branch \
  --branch codex/server-data-cloud-archive-20260731 \
  git@github.com:mmt200088/Reinforcement-For-Robustness.git \
  /hy-tmp/RFR-data-overlay-20260731

python3 /hy-tmp/RFR-data-overlay-20260731/restore_to_server.py \
  --target /hy-tmp/RFR-restored-20260731 \
  --verify-base-only

python3 /hy-tmp/RFR-data-overlay-20260731/restore_to_server.py \
  --target /hy-tmp/RFR-restored-20260731
```

Success requires both `BASE_VERIFY_OK` and `RESTORE_OK`. A later audit can run:

```bash
python3 /hy-tmp/RFR-data-overlay-20260731/restore_to_server.py \
  --target /hy-tmp/RFR-restored-20260731 \
  --verify-overlay-only
```

To combine the archived data with the latest source instead of reproducing the
historical working tree exactly, clone the desired source commit first and
apply the same overlay command to that checkout.

## Inventory

- `archive_metadata.json`: immutable provenance and aggregate counts.
- `base_tracked_manifest.jsonl`: Git blob identity and size of every tracked
  data file, including historical chapters, experiment outputs, submissions,
  reports, checkpoints, and analysis outputs.
- `overlay_manifest.jsonl`: path, size, mode, time, Git blob, and SHA-256 for
  every overlay file.
- `local_worktree_inventory.json`: every local worktree, branch/commit, size,
  dirty state, and the remote refs that preserve its commit.
- `stage1best_chain_consumed_marker.patch`: the sole inactive-worktree change,
  preserving the intentional deletion of an already-consumed `finish.md`.
- `server_restore_evidence.json`: GitHub-to-server download hashes and the
  successful base, overlay, and combined restore verification markers.
- `chunks.json`: ordered payload chunks and their hashes.
- `ignored_overlay_manifest.jsonl` and `ignored_chunks.json`: equivalent
  inventory and chunk metadata for ignored logs and generated documents.
- `SHA256SUMS`: control-file and payload-chunk hashes.
- `overlay_objects.tar.gz.part-*`: deduplicated content-addressed payload.
- `ignored_overlay_objects.tar.gz.part-*`: deduplicated ignored-file payload.
- `restore_to_server.py`: restore and verification tool.

Finder metadata (`.DS_Store`) and the empty accidental root file named `=` are
not research data and are deliberately excluded.
