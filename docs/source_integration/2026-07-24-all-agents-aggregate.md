# All-agent source aggregate, 2026-07-24

This document records the source selection used for the server deployment.
Remote heads were refreshed with an all-heads fetch before integration. The
deployed source must be a clean commit on
`codex/all-agents-integrated-20260724`; a dirty worktree or an arbitrary agent
branch is not a deployable aggregate.

## Aggregate base

- `origin/jk_standard_rl` at
  `a8c026a78c81ad985f47c1f2e88b02aad241e215` is an ancestor.
- `origin/codex/stage2-runtime-efficiency-integrated-3d89` at
  `2356a58fabf20ad1744be0f7d998acfcad9232a2` is the integration base.
- The base already contains the completed Block3 wiring, candidate-promotion
  index, green whole-project runtime changes, server artifact evacuation, small
  GTrXL implementation, terminal-probe CUDA optimizations, shared probe pool,
  bounded diagnostics, and compact indexed candidate evidence.

## Added on top

- `origin/codex/five-profile-fusion-audit` at
  `c49fed05f0bac59b2471143a8d3e41c65c2b93d1`: canonical installed-SF audit
  code, multi-profile runner, tests, and the final six-profile report.
- `origin/codex/stage2-glue-ep114240` at
  `13e923da134a960b5026281b6df3ed3fb4003fc6`: independent-noise RNG replay
  fix, regression tests, and two-seed GLUE artifacts.
- `origin/codex/stage2-network-ablation-v10` at
  `bc9de82e22e5602155766f7117086f766ff7588a`: small-GTrXL default and
  ablation-strategy documentation. Its implementation commits are already in
  the aggregate base.

These changes were replayed onto the newest aggregate base instead of merging
their older branch bases. This preserves the latest production code while
retaining each completed result.

## Superseded branches

- `origin/codex/block-sf-audit` at
  `aff4d1ccc7ae1d2e8dabef97d9466a28e0e5948b` is superseded by the six-profile
  audit, which includes its BERT-base MRPC case plus five additional profiles.
- `origin/codex/stage2-rl-runtime-opt-48b03e8` at
  `6268ee02fdbf070065ea04f72966cffc0461300f` is superseded by
  `2356a58f`. Its accepted production optimizations were ported and reverified;
  its larger probe-batch variants were explicitly rejected under exact-output
  parity and are not active defaults.
- The local `jk_standard_rl` worktree at
  `c78aa8a80abd8718d0e123589e2c54ac13485ad2` is a dirty, older local line
  behind the refreshed remote history. It is left untouched and is not a valid
  server source snapshot.
- `origin/main`, `origin/bert-large`, and old Stage-1 report branches are
  historical or unrelated roots, not current agent integration lines.

## Deployment invariant

Before deployment, refresh all remote heads again and stop if a current agent
branch advanced after this inventory. Push the aggregate commit, transfer that
exact commit through Git, and verify both commit ID and source tree ID on the
server before running tests or training.
