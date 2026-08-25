# Local Workspace Recovery Index

This branch preserves files and Git states that existed only in the retired
local workspace before its cleanup on 2026-08-25. None of these branches are
canonical or deployment inputs.

## High-value results

| Content | Remote branch | Commit |
| --- | --- | --- |
| Formal Stage-1 BO-RF, Greedy, and COINN-GA raw results | `codex/local-worktree-backups-20260825/comparator-stage1-formal-results` | `c1548bf595c90c0c44eaff2a160c19bc37170640` |
| Paper reasoning-trace report and figures | `codex/local-worktree-backups-20260825/dirty-paper-reasoning-traces` | `db9e1d14b791b5048798e861639f9e6dd4d2f052` |
| Standalone Stage-1 GA result bundle | this branch, under `local_recovery/RFR-Results/` | recorded by this branch tip |

## Design and audit history

| Content | Remote branch | Commit |
| --- | --- | --- |
| Comparator single-GPU design commit | `codex/local-worktree-backups-20260825/comparator-single-gpu-design` | `981cc901191bcb172608c998d51364a3daaaf5b6` |
| Comparator design plus uncommitted plan/spec edits | `codex/local-worktree-backups-20260825/dirty-comparator-single-gpu-docs` | `97076019fa0f28a5fbee68dd7ae713c2619d4cbd` |
| Fusion-map Softmax parity handoff | `codex/local-worktree-backups-20260825/fusion-map-six-profile-handoff` | `cdc835c629cc5aa5e4ab89340d51f6210e4c6fa2` |

## Superseded historical states

| Content | Remote branch | Commit | Classification |
| --- | --- | --- | --- |
| Stage-2 runtime-gate RED test | `codex/local-worktree-backups-20260825/stage2-runtime-gate-red` | `1bbbf92a3d36b8766ca80a42570e7b0dcbd5ce77` | Obsolete failing-test history |
| JSONL helper consolidation stash | `codex/local-worktree-backups-20260825/jsonl-helper-stash` | `ea502a443a707510bc5d1ade5437bdf307df0214` | Obsolete code for removed utilities |
| Stage-1 GA 200-generation source | `codex/local-worktree-backups-20260825/stage1-ga-200-source` | `45fc16316b7235ab1f2c4d400b1883667e6badc2` | Patch-equivalent to canonical history |
| Stage-1 GA aggregate manifest v2 | `codex/local-worktree-backups-20260825/stage1-ga-200-manifest-v2` | `95509b1413a75a857e62b9fa42546e1761567d89` | Superseded aggregate metadata |
| Stage-1 GA full-run manifest and handoff merge | `codex/local-worktree-backups-20260825/stage1-ga-full-run-manifest` | `1f5524161439c3ec831179cf979e05cd95630784` | Superseded aggregate metadata |
| Unfinished Stage-1 GA aggregate manifest | `codex/local-worktree-backups-20260825/dirty-stage1-ga-200-aggregate` | `a5eaf8fff03b8996a8c980e1282d22b9e53ec0b2` | Incomplete historical draft |

The two `.DS_Store` files found in old worktrees were intentionally not backed
up because they contain only Finder metadata.

## Recovery

Fetch a backup branch and inspect it in a separate worktree:

```bash
git fetch origin <branch>
git worktree add ../recovered-worktree origin/<branch>
```

`SHA256SUMS` verifies every standalone file stored under `local_recovery/`.
