# Stage-2 K-convergence server cleanup recovery backup

This directory is the content-complete recovery record for three temporary
server worktrees removed from `/hy-tmp` on 2026-07-18.

The original directories totalled 20.53 GB of file content. Most of that data
was tracked content already represented by Git commits, so the backup stores a
reconstructable snapshot instead of duplicating the same repository data:

1. `base_commit.txt` identifies the Git tree containing every tracked file.
2. `tracked_changes.patch` restores every staged or unstaged tracked change.
3. `extra_files.tar.gz.part-*` restores every untracked or ignored file.
4. The TSV inventories and SHA-256 manifests verify every restored file.

## Snapshot index

| Snapshot | Original server path | Base commit | Original bytes | Tracked files | Status rows | Extra files | Extra archive bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| `stage2_k_convergence_5f223f0` | `/hy-tmp/stage2_k_convergence_5f223f0` | `e154f54484f017555264d515df87967c9bf24dab` | 6,844,619,439 | 7,392 | 30 | 79 | 1,516,346 |
| `stage2_k_convergence_tdd_red_20260715` | `/hy-tmp/stage2_k_convergence_tdd_red_20260715` | `14187ee1c1778f4dee598cb755017effc8332869` | 6,848,047,317 | 7,350 | 43 | 109 | 2,810,101 |
| `stage2_k_convergence_validation_20260715` | `/hy-tmp/stage2_k_convergence_validation_20260715` | `14187ee1c1778f4dee598cb755017effc8332869` | 6,840,453,309 | 7,350 | 31 | 97 | 2,314,045 |

## Files in each snapshot directory

| File | Meaning |
|---|---|
| `README.md` | Snapshot-specific origin, purpose, counts, and recovery notes. |
| `base_commit.txt` | Exact Git commit used to reconstruct all tracked files. |
| `tracked_tree.txt` | Exhaustive tracked-file list with Git blob IDs and sizes. |
| `working_tree_status.txt` | Exact pre-deletion Git status, including every untracked path. |
| `tracked_changes.patch` | Full-index binary patch for all modified tracked files. |
| `tracked_changes_stat.txt` | Human-readable summary of the tracked patch. |
| `modified_tracked_files.tsv` | Per-file SHA-256, size, state, path, and description for changed tracked files. |
| `extra_files.tsv` | Per-file SHA-256, size, type, path, and purpose for every untracked or ignored file. |
| `extra_files.tar.gz.part-*` | Split gzip archive containing all paths listed in `extra_files.tsv`. |
| `extra_archive.sha256` | SHA-256 of the reconstructed unsplit gzip archive. |
| `archive_parts.sha256` | SHA-256 of each stored archive part. |
| `git_remotes.txt` | Original worktree Git remote configuration. |
| `git_submodules.txt` | Original submodule commit/status record. |
| `source_metadata.env` | Machine-readable source path, size, commit, and inventory counts. |
| `metadata_files.sha256` | Integrity manifest for all snapshot metadata files. |

`extra_files.tsv` is the exhaustive answer to what each non-Git file was: each
row records the path and classifies it as structured RL evidence, a historical
source/experiment archive, a design document, provenance metadata, generated
cache metadata, or another exact-restoration file.

## Verification and restoration

From any checkout containing this backup commit:

```bash
server_backups/20260718_stage2_k_convergence_cleanup_recovery/restore_snapshot.sh verify-all
```

To recreate one deleted server directory in a new Git worktree:

```bash
server_backups/20260718_stage2_k_convergence_cleanup_recovery/restore_snapshot.sh \
  restore stage2_k_convergence_5f223f0 /desired/restore/path
```

The restore command checks the base commit, applies the tracked binary patch,
extracts every extra file, and verifies all recorded content hashes.

## Deliberately excluded directories

This cleanup does not delete or package the active source tree, the resumable
10,200-episode checkpoint, the previous long natural-convergence run, or the
older layerwise-robust final run. Those remain on the server.
