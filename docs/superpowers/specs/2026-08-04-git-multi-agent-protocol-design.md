# Multi-Agent Git and Server Synchronization Protocol Design

## Objective

Keep the canonical local checkout, GitHub aggregate branch, and server source
checkout on one explicit source commit and tree while multiple agents work in
parallel. Prevent dirty shared worktrees, partial-agent deployments, stale
source edits, and accidental deployment of archive, recovery, experiment, or
unfinished branches.

The protocol must be enforceable by repository tooling rather than relying
only on prose. It must preserve the existing rule that source is edited
locally, Git is the only source transport, and the server only runs code and
returns generated artifacts.

## Current State and Failure Mode

The canonical branch is `jk_standard_rl`. Agent work is normally placed on
`codex/*` branches and may use linked worktrees. Historical branches also use
`codex/*` for aggregates, archives, recovery snapshots, experiment evidence,
and result capture.

The previous failure combined four independent states:

- the primary checkout remained on an old local branch tip;
- other worktrees continued to advance `origin/jk_standard_rl`;
- generated reports and documentation were written into the stale primary
  checkout;
- the repository fetchspec exposed only the canonical branch, hiding some
  remote agent heads from normal fetches.

Git cannot infer whether an agent branch is complete, superseded, rejected, or
an archive. A clean GitHub branch alone therefore cannot prove that the source
is the latest valid aggregate.

## Non-Negotiable Invariants

- `jk_standard_rl` is the only canonical aggregate branch.
- The primary local checkout of `jk_standard_rl` remains tracked-clean and is
  not an agent development workspace or an experiment output directory.
- Ordinary agents never commit directly to or push `jk_standard_rl`.
- Every source task starts from the current canonical commit in an isolated
  worktree on a `codex/task-*` branch.
- Only completed, non-superseded work with a valid handoff may enter an
  aggregate.
- Archive, recovery, result, rejected, and in-progress branches are never
  deployed as source.
- Server source changes arrive only through Git at an exact aggregate commit.
- A server checkout with tracked changes is never updated or executed as the
  canonical source checkout.
- Local, remote, and server commit and tree IDs must match before a run starts.
- Git operations must not destroy unexpected local changes. Dirty state is
  quarantined to a recovery branch before canonical synchronization.

## Branch Roles

| Pattern | Role | May enter source aggregate | May deploy |
| --- | --- | --- | --- |
| `jk_standard_rl` | Canonical aggregate | N/A | Yes |
| `codex/task-<task>-<date>` | Agent source work | With completed handoff | No |
| `codex/aggregate-<date>` | Temporary integration branch | Becomes canonical after gates | No |
| `codex/result-<run>-<date>` | Server-generated compact results | Artifacts only after review | No |
| `codex/archive-<name>` | Immutable historical archive | No | No |
| `codex/recovery-<name>` | Dirty-state or incident recovery | No direct merge | No |
| `codex/experiment-<name>` | Rejected or exploratory work | No unless reclassified | No |

Existing branches are not renamed mechanically. Their role is recorded in an
aggregate manifest when they are considered for integration.

## Roles and Authority

### Ordinary Agent

An ordinary agent may create a task worktree, edit locally, test, commit, push
its task branch, and publish a handoff. It may not update the canonical branch
or server canonical checkout.

### Aggregator

Exactly one active aggregator owns a given aggregate cycle. It refreshes all
remote heads, validates handoffs, records dispositions, integrates selected
work, verifies the aggregate, and advances the canonical branch. Aggregator
authority is explicit for the command invocation; it is not inferred from an
author name or branch name.

### Server Runner

The server runner may fetch, create or update a clean deployment checkout at
an exact canonical commit, run approved commands, and publish result branches.
It may not edit, format, patch, or commit source files.

## Tracked Protocol Files

The implementation introduces:

- `docs/GIT_MULTI_AGENT_PROTOCOL.md`: concise human workflow and recovery
  procedures.
- `agent_handoffs/schema.json`: JSON schema for task handoffs and aggregate
  manifests.
- `agent_handoffs/README.md`: status definitions and examples.
- `agent_handoffs/tasks/`: completed task handoffs that remain relevant to
  aggregate provenance.
- `agent_handoffs/aggregates/`: immutable aggregate manifests.
- `scripts/repo_sync_guard.py`: local, aggregate, and server validation CLI.
- `scripts/install_git_protocol_hooks.sh`: installs the repository-owned hook
  path in the shared local Git configuration.
- `.githooks/pre-push`: blocks unauthorized canonical pushes and malformed
  task handoffs.

`AGENTS.md` and `CLAUDE.md` point all agents to the protocol and require the
guard commands at task boundaries.

## Handoff Contract

Each source task publishes one JSON handoff with these required fields:

- schema version and task ID;
- branch name and role;
- base commit and base tree;
- tip commit and tip tree;
- status: `in_progress`, `completed`, `superseded`, `rejected`, or `archive`;
- changed path scopes;
- verification commands and outcomes;
- server evidence paths when server verification was required;
- superseded task IDs or branches;
- aggregate eligibility and deployment eligibility;
- authoring timestamp and handoff timestamp.

The branch tip must contain the handoff for that branch's source state. To
avoid an impossible self-referential commit hash, the handoff records the
source commit immediately before the handoff-only commit. The guard derives
the handoff commit from the branch tip, verifies that its parent is the
recorded source commit, and confirms that it changes only the handoff path.

An aggregate manifest records every refreshed remote branch considered in the
cycle and assigns one disposition:

- `included`;
- `already_ancestor`;
- `patch_equivalent`;
- `superseded`;
- `rejected`;
- `archive_only`;
- `recovery_only`;
- `result_only`;
- `in_progress`;
- `needs_review`.

No branch may be silently omitted. A `needs_review` disposition blocks
canonical advancement.

## Agent Workflow

### Start

1. Run the guard's `agent-start` preflight from the canonical checkout.
2. Fetch every remote head using the all-head fetchspec.
3. Require the canonical checkout to be clean and equal to
   `origin/jk_standard_rl` by commit and tree.
4. Create a new `codex/task-*` branch and isolated worktree from that exact
   commit.
5. Create an `in_progress` handoff naming the base source state.

### Work

- Edit only in the task worktree.
- Keep generated transient files outside the canonical checkout.
- Commit intentional source and compact evidence changes.
- Push only the task branch.
- Re-fetch before completion and report whether the canonical branch advanced
  while the task was active.

### Finish

1. Run project-appropriate tests and static checks.
2. Require a clean task worktree.
3. Push the task source commit.
4. Change the handoff to `completed`, recording verification and any canonical
   advancement.
5. Commit and push the handoff-only commit.
6. Run `agent-finish` to verify local and remote task tips.

Completion means ready for aggregation, not deployed.

## Aggregate Workflow

1. Acquire the single-aggregator lock in a clean `codex/aggregate-*` worktree.
2. Fetch all remote heads and snapshot their full names and commit IDs.
3. Load and validate every relevant handoff.
4. Produce a draft aggregate manifest with a disposition for every head.
5. Stop if any current source branch is missing a handoff or marked
   `needs_review`.
6. Integrate only eligible completed branches. Resolve conflicts according to
   current code and tests; never mechanically merge recovery or archive state.
7. Run the required local static gates and server tests for affected paths.
8. Commit the immutable aggregate manifest.
9. Fetch all heads again. Stop if any included or current branch advanced.
10. Push the aggregate branch and verify its remote commit and tree.
11. With explicit aggregator authorization, fast-forward
    `jk_standard_rl` to the aggregate commit.
12. Fetch again and verify canonical local/remote parity.

The primary checkout is then updated only with `--ff-only`. Any divergence or
dirty state invokes recovery instead of merge or reset.

## Server Deployment Workflow

The aggregator supplies an expected canonical commit and tree. The server
runner executes the guard in `server-check` mode:

1. Verify that the requested commit is exactly the remote canonical tip.
2. Verify that the requested branch role is canonical, not task or archive.
3. Refuse a tracked-dirty canonical checkout.
4. Fetch through Git and check out the exact commit without source edits.
5. Verify `HEAD`, `HEAD^{tree}`, and tracked-clean state.
6. Write a compact parity receipt outside source-controlled paths or in the
   approved result area.
7. Start the approved run only after the receipt passes.

The server may retain ignored runtime files, but deployment and run guards use
tracked-clean checks. Any tracked result intended for Git is committed on a
`codex/result-*` branch and handed back to the aggregator.

## Guard CLI

`scripts/repo_sync_guard.py` provides deterministic subcommands:

- `status`: report branch role, dirty state, commit/tree, upstream, and
  divergence.
- `agent-start`: validate the canonical base and proposed task branch.
- `agent-finish`: validate task cleanliness, push parity, and handoff.
- `aggregate-preflight`: refresh/audit heads and validate a draft manifest.
- `aggregate-finalize`: require tests, stable head snapshot, aggregate push
  parity, and authorized canonical fast-forward readiness.
- `local-sync`: permit only clean fast-forward synchronization.
- `server-check`: validate exact canonical commit/tree and tracked-clean state.
- `result-check`: validate that a server result branch changes only approved
  artifact paths.

Commands return nonzero on every protocol violation and print one actionable
reason per failure. Read-only status commands never modify the worktree.

## Pre-Push Hook

The tracked pre-push hook delegates policy decisions to the guard. It rejects:

- ordinary pushes that update `refs/heads/jk_standard_rl`;
- task branches without a valid handoff at completion;
- pushes whose declared local tip does not match the handoff;
- archive or recovery branches claiming deployment eligibility.

Canonical pushes require an explicit, short-lived aggregator authorization
environment value plus a validated aggregate manifest. The hook does not store
credentials and does not infer authorization from the machine username.

Hooks are defense in depth, not the sole authority. GitHub branch protection
can later require the same guard in CI.

## Dirty-State Recovery

When the primary checkout is dirty:

1. Stop all edits and record status, commit, tree, and file hashes.
2. Create `codex/recovery-<date>` at the current commit.
3. Commit and push every potentially valuable changed or untracked file.
4. Verify local/remote recovery commit and tree parity.
5. Reclassify disposable files explicitly; never delete them by assumption.
6. Move the local canonical branch only after recovery is durable.
7. Reapply useful changes selectively from a clean current canonical base.

`git pull`, `git reset --hard`, `git clean`, and bulk checkout are not recovery
steps before the recovery branch is verified.

## Failure Handling

The protocol fails closed when:

- a canonical or task worktree is dirty at a protected boundary;
- local and remote commits or trees differ;
- the canonical branch advanced during aggregation;
- a branch lacks a valid role or handoff;
- a completed handoff names missing or mismatched commits;
- an aggregate omits a remote head disposition;
- a server checkout is dirty or not at the requested canonical tip;
- a result branch modifies source paths;
- an unauthorized process attempts to update the canonical branch.

Failures preserve all refs and files and provide recovery instructions. Guard
commands do not use destructive Git operations.

## Verification Strategy

Unit tests cover:

- branch-role classification;
- porcelain status parsing and tracked-dirty detection;
- commit/tree and upstream parity;
- handoff and aggregate schema validation;
- missing, stale, superseded, and patch-equivalent dispositions;
- pre-push stdin parsing and canonical authorization;
- server exact-SHA and dirty-check behavior;
- result-path allowlists;
- failures when remote heads change between aggregate snapshots.

Integration tests use temporary bare remotes and linked worktrees to prove:

- ordinary task pushes succeed;
- unauthorized canonical pushes fail;
- a clean authorized aggregate can fast-forward canonical;
- stale and dirty primary checkouts fail without data loss;
- a recovery branch preserves dirty files before synchronization;
- server parity succeeds only for the canonical commit and tree;
- result branches cannot alter source.

Shell changes receive `bash -n`; Python tooling receives focused unit tests,
`py_compile`, and the repository's configured lint gate when available.

## Migration

1. Add the protocol documentation, schema, guard, tests, and hooks on one task
   branch.
2. Validate against the current clean canonical checkout without changing
   existing historical branches.
3. Install the shared local hook path.
4. Create an initial aggregate manifest classifying current remote heads.
5. Push and aggregate the protocol branch through its own rules as far as
   bootstrapping permits.
6. Update the server only after the final canonical commit is available.
7. Verify local, GitHub, and server commit/tree parity.
8. Optionally enable GitHub branch protection and require the guard workflow.

## Non-Goals

- Automatically merging every `codex/*` branch.
- Deleting historical branches or worktrees.
- Storing server credentials in Git or hook configuration.
- Running experiments from hooks.
- Treating Git ancestry alone as proof that scientific results remain valid.
- Replacing code review, test evidence, or explicit branch disposition with an
  automatic heuristic.

## Completion Definition

The protocol is complete when:

- all tracked protocol files and tests are committed and pushed;
- ordinary agents are blocked from canonical pushes in the shared repository;
- task handoffs and aggregate manifests validate deterministically;
- the primary checkout remains clean and fast-forward-only;
- the final aggregate is the remote canonical tip;
- the server obtains that exact commit only through Git;
- local, remote, and server commit/tree IDs match with tracked-clean status;
- evidence records the initial disposition of all current remote heads.
