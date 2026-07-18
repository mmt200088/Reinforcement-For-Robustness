# Server deletion receipt

## Remote backup gate

- Backup commit pushed before deletion: `918db8b1073a41bae06e06152f57e7e4ebd82b0c`
- Verified remote branch: `origin/jk_standard_rl`
- Verified recovery branch: `origin/codex/stage2-k-convergence`
- Both remote refs resolved to the exact backup commit before deletion.
- The backup commit contains 50 recovery files under this dedicated directory.

## Restore rehearsal

Before deleting the remaining snapshots, the
`stage2_k_convergence_5f223f0` directory was deleted and reconstructed from its
backup representation:

1. Checked out all 7,392 tracked files from base commit
   `e154f54484f017555264d515df87967c9bf24dab`.
2. Applied `tracked_changes.patch`.
3. Extracted all 79 archived untracked or ignored paths.
4. Verified every extra and modified tracked file against its SHA-256 record.
5. Verified that the reconstructed `git status --porcelain -uall` was byte-for-byte
   identical to `working_tree_status.txt`.

Result: `FULL_RESTORE_TEST_OK`.

## Deleted server directories

Deletion completed at `2026-07-18T12:41:42+08:00`.

| Deleted path | Original content bytes | Recovery snapshot |
|---|---:|---|
| `/hy-tmp/stage2_k_convergence_5f223f0` | 6,844,619,439 | `stage2_k_convergence_5f223f0/` |
| `/hy-tmp/stage2_k_convergence_tdd_red_20260715` | 6,848,047,317 | `stage2_k_convergence_tdd_red_20260715/` |
| `/hy-tmp/stage2_k_convergence_validation_20260715` | 6,840,453,309 | `stage2_k_convergence_validation_20260715/` |

All three paths were verified absent after deletion.

## Disk result

- Before cleanup: 47 GB used, 3.4 GB available, 94% utilization.
- After cleanup: 28 GB used, 23 GB available, 55% utilization.

## Preserved active evidence

The following were checked after deletion and were not modified:

- Active source tree: `/hy-tmp/rfr_runtime_optimization`
- Diagnostic episode rows: `10,320`
- Valid resumable checkpoint episode: `10,200`
- Valid resumable PPO update: `85`
- Algorithm revision: `dual_resource_maxmin_shapley_multifidelity_convergence_v9`
- Checkpoint SHA-256:
  `22350c2523f40e174214fbae3742053889f5209fbcf5335d8ddfb50ac3810da3`
- Checkpoint load result: `CHECKPOINT_LOAD_OK`

The previous long natural-convergence run and the older layerwise-robust final
run were also explicitly verified present and were not deleted.
