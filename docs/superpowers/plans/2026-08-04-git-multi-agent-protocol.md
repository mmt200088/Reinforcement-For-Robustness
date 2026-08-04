# Multi-Agent Git Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce a fail-closed multi-agent Git workflow that keeps the canonical local checkout, remote canonical branch, and server source on one reviewed aggregate commit and tree without destroying dirty work.

**Architecture:** Add one standard-library Python guard as the policy engine, repository-owned pre-push hooks as local enforcement, JSON handoffs/manifests as immutable provenance, and concise protocol documentation as the shared agent contract. Ordinary task branches publish handoff-only tip commits; one aggregate branch records a manifest covering every refreshed remote head; canonical and server synchronization are accepted only after exact commit, tree, cleanliness, and remote-head checks pass.

**Tech Stack:** Python 3 standard library, Git plumbing commands, POSIX shell, JSON Schema draft 2020-12, `unittest`.

---

## File Responsibilities

- `scripts/repo_sync_guard.py`: branch-role classification, repository snapshots, task-handoff validation, aggregate-manifest validation, pre-push policy, local parity, server synchronization, and result-branch checks.
- `tests/test_repo_sync_guard.py`: unit and temporary-repository integration coverage for all policy boundaries.
- `agent_handoffs/schema.json`: machine-readable task-handoff and aggregate-manifest schema.
- `agent_handoffs/README.md`: field semantics, immutable-commit convention, and examples.
- `agent_handoffs/tasks/.gitkeep`: tracked task-handoff directory.
- `agent_handoffs/aggregates/.gitkeep`: tracked aggregate-manifest directory.
- `.githooks/pre-push`: thin adapter from Git's pre-push stdin to the Python guard.
- `scripts/install_git_protocol_hooks.sh`: idempotently configure the shared repository to use `.githooks`.
- `docs/GIT_MULTI_AGENT_PROTOCOL.md`: human workflow, dirty recovery, aggregation, deployment, and incident procedures.
- `AGENTS.md`: mandatory entry-point rules for Codex-style agents.
- `CLAUDE.md`: mandatory entry-point rules for Claude-style agents.

## Task 1: Lock the Policy Contract in Tests and Schema

**Files:**
- Create: `tests/test_repo_sync_guard.py`
- Create: `agent_handoffs/schema.json`
- Create: `agent_handoffs/tasks/.gitkeep`
- Create: `agent_handoffs/aggregates/.gitkeep`

- [ ] Add branch-role tests for canonical, task, aggregate, result, archive, recovery, experiment, and legacy/unknown branches.
- [ ] Add temporary-Git-repository helpers that create a bare remote plus two working clones without touching the real repository.
- [ ] Add tests proving tracked dirtiness is detected while ignored runtime files do not count as tracked dirtiness.
- [ ] Add task-handoff fixtures covering every required field and status.
- [ ] Add failing tests for a handoff whose branch name, source parent, source tree, remote tip, or handoff-only changed path is wrong.
- [ ] Add aggregate fixtures and failing tests for omitted remote heads, `needs_review`, stale snapshot commits, invalid included handoffs, and archive/recovery/result branches marked `included`.
- [ ] Add pre-push tests proving ordinary canonical pushes fail, authorized manifest-backed aggregate pushes pass, task pushes remain possible, and archive/recovery branches cannot claim deployment eligibility.
- [ ] Add server/result tests proving exact canonical SHA/tree and allowed artifact scopes are mandatory.
- [ ] Define a JSON Schema `oneOf` contract for `task_handoff` and `aggregate_manifest`, including enums and 40-character lowercase Git object IDs.
- [ ] Commit and push the red contract:

```bash
git add tests/test_repo_sync_guard.py agent_handoffs/schema.json agent_handoffs/tasks/.gitkeep agent_handoffs/aggregates/.gitkeep
git commit -m "test: define multi-agent git protocol contract"
git push origin codex/task-git-protocol-20260804
```

Expected: the branch pushes successfully; the new tests cannot import `scripts.repo_sync_guard` yet and therefore fail when executed on the server candidate checkout.

## Task 2: Implement Repository State and Handoff Validation

**Files:**
- Create: `scripts/repo_sync_guard.py`
- Modify: `tests/test_repo_sync_guard.py`

- [ ] Implement a structured `GitError` boundary and Git command wrapper that never invokes reset, clean, force push, or bulk checkout.
- [ ] Implement commit/tree/ref lookup, tracked-dirty inspection, remote canonical lookup, remote-head snapshot, ancestor checks, and changed-path inspection.
- [ ] Implement deterministic branch-role classification with explicit legacy/unknown handling.
- [ ] Implement strict JSON loading and built-in structural validation so runtime does not require the third-party `jsonschema` package.
- [ ] Implement task-handoff validation: required fields, role/status eligibility, current branch tip, handoff-only tip commit, parent/source identity, source tree, remote parity, and changed-scope coverage.
- [ ] Implement `status`, `agent-start`, and `agent-finish` subcommands. `status` must be read-only; `agent-start` must fail unless canonical is tracked-clean and equals the remote canonical commit/tree; `agent-finish` must require the completed handoff and local/remote tip parity.
- [ ] Keep all diagnostics deterministic and include a machine-readable `--json` mode.
- [ ] Run local static checks only:

```bash
python3 -m py_compile scripts/repo_sync_guard.py tests/test_repo_sync_guard.py
git diff --check
```

Expected: both commands exit 0; no project code or test suite runs locally.

- [ ] Commit and push:

```bash
git add scripts/repo_sync_guard.py tests/test_repo_sync_guard.py
git commit -m "feat: validate repository and task handoff state"
git push origin codex/task-git-protocol-20260804
```

## Task 3: Implement Aggregate, Local, Server, and Result Gates

**Files:**
- Modify: `scripts/repo_sync_guard.py`
- Modify: `tests/test_repo_sync_guard.py`

- [ ] Implement `aggregate-preflight` with an explicit `--fetch` mode that refreshes the all-head refspec and an explicit `--write-draft` mode that writes every remote head once with a conservative disposition.
- [ ] Auto-classify only facts Git can prove: canonical/aggregate ancestry as `already_ancestor`, role-specific archive/recovery/result branches as their matching non-source disposition, and all other unresolved heads as `needs_review`.
- [ ] Implement aggregate-manifest validation with complete snapshot coverage, permitted disposition/role combinations, completed handoff checks for `included`, no duplicate branch entries, no unresolved entries, and source-parent/tree verification for a manifest-only aggregate tip commit.
- [ ] Implement `aggregate-finalize` to re-fetch all heads, reject any snapshot drift, verify aggregate local/remote parity, and emit the exact canonical fast-forward command without executing a force update.
- [ ] Implement `local-sync` as verify-only by default and `git merge --ff-only` only with explicit `--apply`, refusing all tracked dirtiness and divergence.
- [ ] Implement `server-check` as verify-only by default and exact Git fetch plus detached checkout only with explicit `--sync`, refusing tracked dirtiness and requiring the expected commit/tree to equal the remote canonical tip.
- [ ] Implement `result-check` for `codex/result-*` branches, clean remote parity, canonical base ancestry, and approved artifact path prefixes.
- [ ] Add transaction-style tests showing failed gates leave worktree content and branch refs unchanged.
- [ ] Run local static checks only:

```bash
python3 -m py_compile scripts/repo_sync_guard.py tests/test_repo_sync_guard.py
git diff --check
```

Expected: both commands exit 0.

- [ ] Commit and push:

```bash
git add scripts/repo_sync_guard.py tests/test_repo_sync_guard.py
git commit -m "feat: enforce aggregate and server source parity"
git push origin codex/task-git-protocol-20260804
```

## Task 4: Add Pre-Push Enforcement and Hook Installation

**Files:**
- Create: `.githooks/pre-push`
- Create: `scripts/install_git_protocol_hooks.sh`
- Modify: `scripts/repo_sync_guard.py`
- Modify: `tests/test_repo_sync_guard.py`

- [ ] Implement a `pre-push` guard subcommand that consumes Git's four-column stdin records without modifying refs or the worktree.
- [ ] Permit deletes and ordinary noncanonical task pushes while validating any handoff present at the pushed task tip.
- [ ] Block pushes to `jk_standard_rl` unless `RFR_AGGREGATOR_AUTHORIZED=1`, `RFR_AGGREGATE_MANIFEST` names a valid aggregate manifest in the pushed tip, and the remote old SHA is an ancestor of the new SHA.
- [ ] Block archive, recovery, result, and experiment branches when their protocol records claim source aggregate or deployment eligibility.
- [ ] Make `.githooks/pre-push` a minimal `exec` wrapper that resolves the repository root and forwards remote name, URL, and stdin.
- [ ] Make the installer verify the hook files, set `core.hooksPath=.githooks` in the shared Git config, and print the effective value. It must not overwrite user hooks elsewhere.
- [ ] Add hook integration tests using a temporary bare remote and environment-controlled authorized/unauthorized pushes.
- [ ] Run local static checks only:

```bash
python3 -m py_compile scripts/repo_sync_guard.py tests/test_repo_sync_guard.py
bash -n .githooks/pre-push scripts/install_git_protocol_hooks.sh
git diff --check
```

Expected: every command exits 0.

- [ ] Commit and push:

```bash
git add .githooks/pre-push scripts/install_git_protocol_hooks.sh scripts/repo_sync_guard.py tests/test_repo_sync_guard.py
git commit -m "feat: guard canonical pushes with repository hooks"
git push origin codex/task-git-protocol-20260804
```

## Task 5: Publish the Shared Human Protocol

**Files:**
- Create: `docs/GIT_MULTI_AGENT_PROTOCOL.md`
- Create: `agent_handoffs/README.md`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `tests/test_repo_sync_guard.py`

- [ ] Document the branch roles, authority model, ordinary-agent lifecycle, aggregate lifecycle, server lifecycle, result return path, exact guard commands, and fail-closed meanings.
- [ ] Document dirty recovery as preserve-first: record status/commit/tree and file hashes, create and push `codex/recovery-*`, then update canonical only after the recovery tip is verified remotely.
- [ ] Document that a clean tree is a boundary condition, not proof that a branch is current or complete.
- [ ] Document handoff and aggregate JSON examples whose fields pass the implemented validator.
- [ ] Add concise mandatory protocol pointers near the top of `AGENTS.md` and `CLAUDE.md`, including the prohibition on ordinary canonical pushes and direct server source edits.
- [ ] Add static tests that both agent instruction files point to the protocol and required guard commands.
- [ ] Run local static checks only:

```bash
python3 -m py_compile scripts/repo_sync_guard.py tests/test_repo_sync_guard.py
git diff --check
```

Expected: both commands exit 0.

- [ ] Commit and push:

```bash
git add docs/GIT_MULTI_AGENT_PROTOCOL.md agent_handoffs/README.md AGENTS.md CLAUDE.md tests/test_repo_sync_guard.py
git commit -m "docs: establish mandatory multi-agent git workflow"
git push origin codex/task-git-protocol-20260804
```

## Task 6: Publish This Task's Completed Handoff

**Files:**
- Create: `agent_handoffs/tasks/git-multi-agent-protocol-20260804.json`

- [ ] Re-fetch all remote heads and record whether `origin/jk_standard_rl` advanced from base `82f7b2486c15d7bba3db735076c50ea41fc0a165`.
- [ ] Record the implementation source commit/tree, changed scopes, every static verification command, and server verification intent in a completed task handoff.
- [ ] Commit only the handoff file so the branch tip is a handoff-only commit whose parent is the recorded source commit.
- [ ] Push the handoff tip and validate it:

```bash
python3 scripts/repo_sync_guard.py agent-finish --handoff agent_handoffs/tasks/git-multi-agent-protocol-20260804.json --remote origin
```

Expected: exit 0 and report local/remote handoff-tip parity.

## Task 7: Build a Complete Aggregate Candidate

**Files:**
- Create: `agent_handoffs/aggregates/2026-08-04-git-protocol-bootstrap.json`

- [ ] Fetch every remote head using `+refs/heads/*:refs/remotes/origin/*` immediately before aggregation.
- [ ] Create `codex/aggregate-20260804-git-protocol` from the current `origin/jk_standard_rl` tip in a clean isolated worktree.
- [ ] Integrate the completed protocol branch without merging recovery, archive, result, experiment, rejected, or in-progress source.
- [ ] Generate a draft manifest covering every refreshed remote head.
- [ ] Review every `needs_review` entry against Git ancestry, handoffs, branch role, prior aggregate provenance, and changed paths; assign an evidence-backed terminal disposition to every branch.
- [ ] Record the aggregate source commit/tree and commit only the immutable aggregate manifest as the aggregate tip.
- [ ] Re-fetch every remote head. If any snapshot head changed, regenerate and re-review the manifest before proceeding.
- [ ] Push the aggregate branch and run:

```bash
RFR_AGGREGATOR_AUTHORIZED=1 RFR_AGGREGATE_MANIFEST=agent_handoffs/aggregates/2026-08-04-git-protocol-bootstrap.json python3 scripts/repo_sync_guard.py aggregate-finalize --manifest agent_handoffs/aggregates/2026-08-04-git-protocol-bootstrap.json --remote origin --fetch
```

Expected: exit 0; no `needs_review`; aggregate local/remote commit and tree match; canonical fast-forward command is emitted but not yet executed.

## Task 8: Run the Server Verification Gate

**Files:**
- Server-generated evidence only under the approved experiment/result artifact area.

- [ ] Read-only inspect the latest server for the Git checkout path, running jobs, current commit/tree, tracked status, Python version, and remote URL. Do not stop jobs or edit source.
- [ ] Fetch the aggregate candidate through Git into an isolated clean server checkout and run the red/green contract history if retained, followed by the final focused suite:

```bash
python3 -m unittest -v tests.test_repo_sync_guard
python3 -m py_compile scripts/repo_sync_guard.py tests/test_repo_sync_guard.py
bash -n .githooks/pre-push scripts/install_git_protocol_hooks.sh
```

Expected: the final candidate passes all focused tests and static checks. Any intentionally retained red-contract commit fails for the expected missing/invalid implementation reason only.

- [ ] Run temporary-repository hook integration tests without touching the server canonical checkout or active experiment data.
- [ ] Save compact command output, server commit/tree, and SHA-256 evidence in an approved result directory; return that evidence through a `codex/result-*` branch only if it is intended for repository retention.
- [ ] If any server check fails, fix locally on the task branch, publish a new completed handoff, rebuild the aggregate from the newly refreshed canonical base, and repeat this task.

## Task 9: Advance Canonical and Establish Three-End Parity

**Files:**
- No new source files unless verification exposes a defect.

- [ ] Perform one final all-head fetch and validate the aggregate manifest again.
- [ ] Push the exact aggregate tip to canonical with explicit authorization and without force:

```bash
RFR_AGGREGATOR_AUTHORIZED=1 RFR_AGGREGATE_MANIFEST=agent_handoffs/aggregates/2026-08-04-git-protocol-bootstrap.json git push origin codex/aggregate-20260804-git-protocol:jk_standard_rl
```

Expected: fast-forward succeeds; the pre-push hook validates authorization and manifest.

- [ ] Update the primary local checkout with `git merge --ff-only origin/jk_standard_rl` only after confirming it is tracked-clean.
- [ ] Install the repository hook path in the shared local Git configuration:

```bash
bash scripts/install_git_protocol_hooks.sh
```

Expected: `git config --get core.hooksPath` prints the absolute `.githooks`
directory in the canonical primary checkout, so every linked worktree uses the
same policy implementation.

- [ ] On the server, run `server-check --sync` using the final canonical commit/tree, then rerun `server-check` verify-only.
- [ ] Verify and record all three endpoints:

```bash
git rev-parse HEAD
git rev-parse HEAD^{tree}
git status --porcelain=v2
git ls-remote --heads origin refs/heads/jk_standard_rl
```

Expected: local primary, remote canonical, and server canonical report one full commit SHA, one full tree SHA, and no tracked changes.

- [ ] Run the final guard smoke commands from the canonical local checkout:

```bash
python3 scripts/repo_sync_guard.py status --remote origin --canonical jk_standard_rl --json
python3 scripts/repo_sync_guard.py local-sync --remote origin --canonical jk_standard_rl
```

Expected: both exit 0 and report exact canonical parity without modifying files.

## Task 10: Completion Audit

- [ ] Review the final diff against `docs/superpowers/specs/2026-08-04-git-multi-agent-protocol-design.md`; verify every non-negotiable invariant has one documented workflow and one machine-enforced gate where enforcement is locally possible.
- [ ] Confirm no credentials, host passwords, runtime caches, large experiment outputs, or unrelated agent changes entered the aggregate.
- [ ] Confirm no destructive Git operation was used and the original dirty-state recovery branch remains reachable remotely.
- [ ] Confirm focused server tests, aggregate manifest completeness, authorized canonical fast-forward, local hook installation, and local/remote/server commit/tree/clean parity.
- [ ] Use `superpowers:finishing-a-development-branch` and `superpowers:verification-before-completion` before reporting completion.
