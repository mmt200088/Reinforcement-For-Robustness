# Aggregate Review

- Remote refresh: `git fetch origin '+refs/heads/*:refs/remotes/origin/*' --prune`
- Selected completed aggregate base: `117f9459`
- Aggregate base refs: `origin/jk_standard_rl` and
  `origin/codex/stage2-k-presets-network-budget-20260728`
- Optimization branch at review: `63a80e2c`
- `117f9459` is an ancestor of the optimization branch.
- The only remote head newer than `117f9459` is this optimization branch.
- Older unmerged remote heads are historical, isolated, or superseded work.
  None has a newer completed handoff that replaces `117f9459`.
- The diverged `54dfeaa0` branch contains the older K6/K7 server-gate evidence;
  its production action-domain work is already represented in the later
  aggregate lineage, while `117f9459` carries the newer precision-preset
  production chain.

See `remote_heads_snapshot.tsv` and `remote_heads_not_merged.txt` for the exact
ref snapshot used for this decision.
