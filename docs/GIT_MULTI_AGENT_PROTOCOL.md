# Multi-Agent Git Protocol

## Roles

An **ordinary task agent** owns one isolated task branch. The **aggregator** is
the single agent explicitly authorized by the user to integrate completed work,
advance `jk_standard_rl`, and deploy the canonical checkout.

## Task Agent Workflow

1. Fetch remote refs and run the task-start boundary check.
2. Create an isolated `codex/task-<task-id>` branch and worktree from the
   current canonical commit.
3. Modify source locally. Do not edit source on a server.
4. Commit and push every coherent source change.
5. Run focused and complete verification from the exact pushed commit.
6. Publish `agent_handoffs/tasks/<task-id>.json` with the source commit, source
   tree, changed paths, test evidence, and completion status.
7. Run the task-finish boundary check. Do not update `jk_standard_rl`.

An ordinary handoff is `deployment_eligible=false`. It may be
`aggregate_eligible=true` only when its source and evidence are complete.

## Aggregator Workflow

1. Fetch every remote head immediately before integration.
2. Run aggregate preflight and review all completed, non-superseded handoffs.
3. Record every remote head as integrated, superseded, rejected, or unrelated.
4. Create a clean `codex/aggregate-*` branch from current canonical.
5. Integrate source commits, not result branches or unfinished worktree state.
6. Resolve conflicts by reviewing behavior; never merge mechanically.
7. Run local static checks, then push the aggregate candidate.
8. Synchronize a clean server checkout through Git to that exact commit.
9. Run required server tests and experiments. Return server-generated evidence
   through a result branch.
10. Run aggregate-finalize, then fast-forward `jk_standard_rl` only if every
    mandatory gate passes.
11. Verify local, remote, and server full commit IDs and tree IDs are identical
    and all canonical checkouts are tracked-clean.

## Safety Rules

- Never use `--no-verify`.
- Never deploy a task branch, result branch, dirty tree, or partial aggregate.
- Never reset, clean, or overwrite unrelated user or agent changes.
- Never store credentials, private host details, or personal paths in Git.
- Server source changes occur only through Git fetch/checkout or a verified
  bundle. Runtime artifacts may be created on the server.
- If a mandatory gate fails, canonical does not move.

The boundary commands are implemented by `scripts/repo_sync_guard.py`; use its
`--help` output for the current command syntax.
