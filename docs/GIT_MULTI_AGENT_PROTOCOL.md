# Multi-Agent Git Protocol

This protocol is mandatory for every agent and server runner working on this
repository. Its purpose is to make one reviewed aggregate source state
identifiable and reproducible while multiple agents work concurrently.

## Invariants

- `jk_standard_rl` is the only canonical source branch.
- The primary local checkout of `jk_standard_rl` stays tracked-clean. It is not
  an agent development workspace and not an experiment output directory.
- Source work is local-only on isolated `codex/task-*` worktrees.
- Ordinary agents never update `jk_standard_rl` and never update the server
  canonical checkout.
- A task is aggregation-ready only when its branch tip is a valid handoff-only
  commit and the same tip exists remotely.
- Exactly one aggregator reviews all remote heads in an aggregate cycle.
- Archive, recovery, result, experiment, rejected, superseded, and in-progress
  branches are not source deployment candidates.
- The server obtains source only through Git at the exact remote canonical
  commit and tree. Source files are never patched or committed on the server.
- No run starts until local, remote, and server commit/tree parity is verified.
- Unexpected dirty state is preserved before any synchronization operation.

A clean worktree proves only that tracked files match its current `HEAD`. It
does not prove that `HEAD` is current, complete, reviewed, or deployable.

## Branch Roles

| Branch | Owner and purpose | Source aggregate | Direct deployment |
| --- | --- | --- | --- |
| `jk_standard_rl` | Aggregator-owned canonical source | N/A | Yes |
| `codex/task-<task>-<date>` | One local source task | Completed handoff only | No |
| `codex/aggregate-<date>` | One reviewed integration cycle | Becomes canonical | No |
| `codex/result-<run>-<date>` | Server-generated approved artifacts | No | No |
| `codex/archive-<name>` | Immutable historical archive | No | No |
| `codex/recovery-<name>` | Preserved dirty or incident state | Never directly | No |
| `codex/experiment-<name>` | Rejected or exploratory work | No by default | No |

Historical branches are not renamed or deleted mechanically. Each one receives
an explicit disposition in the next aggregate manifest that considers it.

## Install the Shared Hook

Install only from the clean primary checkout on `jk_standard_rl`, after the
protocol exists on canonical:

```bash
bash scripts/install_git_protocol_hooks.sh
git config --local --get core.hooksPath
```

The configured value is the absolute `.githooks` directory in the canonical
primary checkout. All linked worktrees therefore execute one current policy
implementation. The installer refuses to overwrite another configured hook
path.

The hook is defense in depth. The aggregate manifest, review, tests, and exact
server parity remain required even when the hook passes.

## Ordinary Agent Workflow

### Start

From the clean primary canonical checkout:

```bash
git fetch --prune origin '+refs/heads/*:refs/remotes/origin/*'
python3 scripts/repo_sync_guard.py agent-start \
  --task-id terminal-probe-cache-20260804 \
  --branch codex/task-terminal-probe-cache-20260804 \
  --remote origin \
  --canonical jk_standard_rl
git worktree add \
  -b codex/task-terminal-probe-cache-20260804 \
  ../rfr-terminal-probe-cache-20260804 \
  origin/jk_standard_rl
```

`agent-start` fails if canonical is dirty, stale, detached, or not equal to the
remote canonical tip. Do not create the task from a stale local branch.

Create an `in_progress` record under `agent_handoffs/tasks/` if the work will
span handoffs or coordination boundaries. It is provenance only and does not
make the branch aggregate-eligible.

### Work

- Edit only in the task worktree.
- Keep transient outputs outside tracked source paths.
- Commit intentional source and compact evidence changes.
- Push only the task branch.
- Never merge recovery/archive branches to make conflicts disappear.
- Re-fetch all heads before declaring completion and record canonical movement.

Task source commits may be pushed before completion. A task tip that changes a
handoff must change exactly one handoff file and pass the handoff validator.

### Finish

1. Run the required project gates on the server from Git-synchronized source.
2. Commit and push the final source commit.
3. Fill a completed task handoff with exact base/source commit and tree IDs,
   changed scopes, verification outcomes, and evidence paths.
4. Commit only that handoff file. Its parent must be `source_commit`.
5. Push the handoff-only tip.
6. Validate local/remote parity:

```bash
python3 scripts/repo_sync_guard.py agent-finish \
  --handoff agent_handoffs/tasks/terminal-probe-cache-20260804.json \
  --remote origin
```

Completion means ready for aggregator review. It does not mean merged,
canonical, synchronized to the primary checkout, or deployed to the server.

## Aggregator Workflow

Exactly one aggregator owns one cycle.

### Create the Candidate

```bash
git fetch --prune origin '+refs/heads/*:refs/remotes/origin/*'
git worktree add \
  -b codex/aggregate-20260804 \
  ../rfr-aggregate-20260804 \
  origin/jk_standard_rl
```

Review valid completed handoffs and integrate only selected source work. After
integration, generate the conservative draft:

```bash
python3 scripts/repo_sync_guard.py aggregate-preflight \
  --aggregate-id 20260804-runtime \
  --remote origin \
  --canonical jk_standard_rl \
  --fetch \
  --write-draft agent_handoffs/aggregates/20260804-runtime.json
```

The draft auto-classifies only facts Git can prove. Task, legacy, experiment,
or otherwise ambiguous branches remain `needs_review`. Review every such entry
and assign one terminal disposition:

- `included`: completed task handoff was integrated by ancestry;
- `already_ancestor`: content is already in the aggregate ancestry;
- `patch_equivalent`: independently verified equivalent patch already exists;
- `superseded`: a named newer task replaces it;
- `rejected`: reviewed and intentionally excluded;
- `archive_only`, `recovery_only`, or `result_only`: role-specific exclusion;
- `in_progress`: active work intentionally excluded from this cycle.

Each reason must say why the disposition is safe. No remote branch may be
omitted. `needs_review` blocks finalization.

Replace the draft verification placeholder with passed commands and evidence.
Record the aggregate source commit/tree, then commit only the manifest as the
aggregate tip.

### Stable Snapshot and Candidate Verification

Push the aggregate branch, run affected tests on the server from an isolated
Git checkout, and then re-fetch every head:

```bash
git push -u origin codex/aggregate-20260804
RFR_AGGREGATOR_AUTHORIZED=1 \
RFR_AGGREGATE_MANIFEST=agent_handoffs/aggregates/20260804-runtime.json \
python3 scripts/repo_sync_guard.py aggregate-finalize \
  --manifest agent_handoffs/aggregates/20260804-runtime.json \
  --remote origin \
  --fetch
```

If any snapshotted head advanced or a new head appeared, stop. Regenerate and
review the manifest. Do not deploy a candidate based on a partial refresh.

### Advance Canonical

Only after aggregate tests and stable-head validation pass:

```bash
RFR_AGGREGATOR_AUTHORIZED=1 \
RFR_AGGREGATE_MANIFEST=agent_handoffs/aggregates/20260804-runtime.json \
git push origin codex/aggregate-20260804:jk_standard_rl
```

The pre-push guard requires the old canonical SHA to equal the manifest base
and requires a fast-forward. Force pushes are not part of this protocol.

## Primary Local Synchronization

The primary checkout is verify-only by default:

```bash
python3 scripts/repo_sync_guard.py local-sync \
  --remote origin \
  --canonical jk_standard_rl
```

When remote canonical advanced and the primary checkout is clean:

```bash
python3 scripts/repo_sync_guard.py local-sync \
  --remote origin \
  --canonical jk_standard_rl \
  --apply
```

`--apply` fetches all heads and runs only `git merge --ff-only`. Dirtiness or
divergence is a recovery event, not permission to reset.

## Dirty-State Recovery

When the primary checkout is dirty, do not pull, reset, clean, stash by
assumption, or bulk-checkout paths.

1. Record `git status --porcelain=v2 --branch`, `HEAD`, `HEAD^{tree}`, and hashes
   of every valuable modified or untracked file.
2. Review untracked files for secrets and oversized runtime artifacts.
3. Create a recovery branch at the current commit:

Replace the two example paths below with the reviewed paths being preserved.

```bash
stamp="$(date +%Y%m%d-%H%M%S)"
git switch -c "codex/recovery-primary-${stamp}"
git add --intent-to-add -- AGENTS.md scripts/example.py
git diff -- AGENTS.md scripts/example.py
git add -- AGENTS.md scripts/example.py
git commit -m "recovery: preserve primary dirty state ${stamp}"
git push -u origin "codex/recovery-primary-${stamp}"
```

4. Verify the recovery branch local/remote commit and tree IDs.
5. Only after preservation is durable, return to `jk_standard_rl` and use
   `local-sync --apply` if the canonical history is a fast-forward.
6. Reapply wanted changes selectively in a fresh task worktree.

Recovery branches are never mechanically merged into source. The aggregator
reviews and reimplements or cherry-picks intentional changes through a normal
task with tests.

## Server Workflow

The aggregator supplies the exact canonical commit and tree. Inspect tracked
status before synchronization. Then run:

```bash
python3 scripts/repo_sync_guard.py server-check \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-tree "$EXPECTED_TREE" \
  --remote origin \
  --canonical jk_standard_rl \
  --sync
python3 scripts/repo_sync_guard.py server-check \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-tree "$EXPECTED_TREE" \
  --remote origin \
  --canonical jk_standard_rl
```

`--sync` fetches through Git and checks out the exact commit detached. It
refuses tracked changes and refuses any commit that is not the remote canonical
tip. Ignored runtime files may remain, but tracked source must be clean.

Do not start training or evaluation until the verify-only command passes.

## Returning Server Results

Create a `codex/result-*` branch from the exact run source commit. Include only
approved compact artifact paths. Before push:

```bash
python3 scripts/repo_sync_guard.py result-check \
  --base-commit "$RUN_SOURCE_COMMIT" \
  --allowed-prefix experiments/server_command_runs/ \
  --allowed-prefix rl_training_data_points/ \
  --remote origin
```

After push, repeat with `--require-remote`. A result branch that changes source,
tests, launchers, or configuration fails. Server-discovered source fixes return
as diagnosis; the actual fix is made locally on a new task branch.

## Final Parity Receipt

Before every run, collect from local primary, remote canonical, and server:

```bash
git rev-parse HEAD
git rev-parse HEAD^{tree}
git status --porcelain=v2
git ls-remote --heads origin refs/heads/jk_standard_rl
```

Acceptance requires one identical full commit SHA, one identical full tree SHA,
and no tracked changes at either checkout. A short SHA or branch name alone is
not sufficient evidence.

## Failure Rules

The guard fails closed on dirty protected boundaries, stale canonical state,
invalid handoffs, omitted heads, unresolved dispositions, snapshot drift,
unauthorized canonical pushes, non-fast-forward history, invalid server source,
or source changes on result branches.

Do not bypass a failure with `--no-verify`. Preserve evidence, fix the declared
state locally, publish a new handoff if source changed, and rebuild the
aggregate from a fresh all-head snapshot.
