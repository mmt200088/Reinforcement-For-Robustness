# Agent Handoffs and Aggregate Manifests

This directory contains immutable provenance records for the multi-agent Git
protocol in `docs/GIT_MULTI_AGENT_PROTOCOL.md`.

## Directory Layout

- `schema.json`: JSON Schema for both record types.
- `tasks/`: one task handoff per completed or coordination-relevant task.
- `aggregates/`: one immutable manifest per canonical aggregate cycle.

Protocol records use JSON with stable keys and full 40-character lowercase Git
object IDs. They contain no credentials.

## Task Handoff Commit Convention

The handoff records the source commit immediately before the handoff-only tip:

```text
base_commit ... source_commit -> handoff_commit (branch tip)
                                  changes one task JSON only
```

The guard derives `handoff_commit` from the branch tip, verifies its only parent
is `source_commit`, verifies `source_tree`, and confirms the tip changes only
the named handoff file. This avoids an impossible self-referential commit hash.

Status meanings:

- `in_progress`: not aggregate-eligible and `completed_at` is null.
- `completed`: verification passed and the branch may be reviewed for inclusion.
- `superseded`: replaced by named work; not aggregate-eligible.
- `rejected`: reviewed and intentionally excluded.
- `archive`: provenance only.

Completed does not mean deployed. Task branches always use
`deployment_eligible: false`.

Illustrative task record:

```json
{
  "aggregate_eligible": true,
  "base_commit": "1111111111111111111111111111111111111111",
  "base_tree": "2222222222222222222222222222222222222222",
  "branch": "codex/task-terminal-probe-cache-20260804",
  "branch_role": "task",
  "changed_scopes": [
    "blb_stage2_rl/",
    "tests/"
  ],
  "completed_at": "2026-08-04T12:00:00+08:00",
  "deployment_eligible": false,
  "record_type": "task_handoff",
  "schema_version": 1,
  "server_evidence": [
    "experiments/server_command_runs/terminal_probe_cache_20260804/"
  ],
  "source_commit": "3333333333333333333333333333333333333333",
  "source_tree": "4444444444444444444444444444444444444444",
  "started_at": "2026-08-04T09:00:00+08:00",
  "status": "completed",
  "supersedes": [],
  "task_id": "terminal-probe-cache-20260804",
  "verification": [
    {
      "command": "python3 -m unittest -v tests.test_terminal_probe_cache",
      "evidence": "server: focused suite passed",
      "outcome": "passed"
    }
  ]
}
```

## Aggregate Manifest Commit Convention

The aggregate manifest uses the same non-self-referential structure:

```text
canonical base ... aggregate source -> manifest_commit (aggregate tip)
                                      changes one aggregate JSON only
```

Every remote head from the preflight snapshot appears exactly once. The
aggregate branch itself may first appear remotely when the manifest tip is
pushed; the guard validates it separately as the expected aggregate tip.

Disposition meanings:

- `included`: completed task handoff integrated into aggregate ancestry.
- `already_ancestor`: branch content is already in source ancestry.
- `patch_equivalent`: review proved an equivalent patch is already present.
- `superseded`: named newer work replaces it.
- `rejected`: reviewed and intentionally excluded.
- `archive_only`, `recovery_only`, `result_only`: role-specific exclusion.
- `in_progress`: active and intentionally excluded from this cycle.
- `needs_review`: unresolved; blocks finalization and canonical push.

Illustrative aggregate record:

```json
{
  "aggregate_branch": "codex/aggregate-20260804",
  "aggregate_id": "20260804-runtime",
  "base_commit": "1111111111111111111111111111111111111111",
  "base_tree": "2222222222222222222222222222222222222222",
  "canonical_branch": "jk_standard_rl",
  "canonical_eligible": true,
  "created_at": "2026-08-04T15:00:00+08:00",
  "deployment_eligible": true,
  "heads": [
    {
      "branch": "jk_standard_rl",
      "commit": "1111111111111111111111111111111111111111",
      "disposition": "already_ancestor",
      "handoff": null,
      "reason": "canonical snapshot is the aggregate base",
      "role": "canonical"
    },
    {
      "branch": "codex/task-terminal-probe-cache-20260804",
      "commit": "5555555555555555555555555555555555555555",
      "disposition": "included",
      "handoff": "agent_handoffs/tasks/terminal-probe-cache-20260804.json",
      "reason": "completed handoff integrated and server tests passed",
      "role": "task"
    }
  ],
  "record_type": "aggregate_manifest",
  "remote": "origin",
  "schema_version": 1,
  "server_evidence": [
    "experiments/server_command_runs/aggregate_20260804/"
  ],
  "snapshot_at": "2026-08-04T14:45:00+08:00",
  "source_commit": "6666666666666666666666666666666666666666",
  "source_tree": "7777777777777777777777777777777777777777",
  "verification": [
    {
      "command": "python3 -m unittest -v tests.test_repo_sync_guard",
      "evidence": "server: focused suite passed",
      "outcome": "passed"
    }
  ]
}
```

The values above demonstrate structure only. Real records must name existing
Git objects and must pass `scripts/repo_sync_guard.py` against the repository.
