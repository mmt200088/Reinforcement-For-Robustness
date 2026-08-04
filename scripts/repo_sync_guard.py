#!/usr/bin/env python3
"""Fail-closed Git workflow guard for the RFR multi-agent repository."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, NamedTuple, Sequence


SCHEMA_VERSION = 1
DEFAULT_CANONICAL = "jk_standard_rl"
DEFAULT_REMOTE = "origin"
ZERO_SHA = "0" * 40
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

TASK_STATUSES = {"in_progress", "completed", "superseded", "rejected", "archive"}
VERIFICATION_OUTCOMES = {"passed", "failed", "not_run"}
HEAD_DISPOSITIONS = {
    "included",
    "already_ancestor",
    "patch_equivalent",
    "superseded",
    "rejected",
    "archive_only",
    "recovery_only",
    "result_only",
    "in_progress",
    "needs_review",
}
BRANCH_ROLES = {
    "canonical",
    "task",
    "aggregate",
    "result",
    "archive",
    "recovery",
    "experiment",
    "legacy",
}


class GuardError(RuntimeError):
    """Raised when repository state violates a protocol invariant."""


class PushUpdate(NamedTuple):
    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str


def _command_text(args: Sequence[str]) -> str:
    return " ".join(args)


def run_git(
    repo: Path | str,
    *args: str,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded Git command and preserve stderr for diagnostics."""

    cmd = ["git", *args]
    try:
        completed = subprocess.run(
            cmd,
            cwd=Path(repo),
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardError(f"Git command could not run: {_command_text(cmd)}: {exc}") from exc
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise GuardError(
            f"Git command failed ({completed.returncode}): {_command_text(cmd)}: {detail}"
        )
    return completed


def git_output(repo: Path | str, *args: str) -> str:
    return run_git(repo, *args).stdout.strip()


def repository_root(repo: Path | str = ".") -> Path:
    return Path(git_output(repo, "rev-parse", "--show-toplevel")).resolve()


def require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise GuardError(f"{field} must be a full lowercase 40-character Git object ID")
    return value


def git_commit(repo: Path | str, revision: str = "HEAD") -> str:
    value = git_output(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
    return require_sha(value, f"commit for {revision}")


def git_tree(repo: Path | str, revision: str = "HEAD") -> str:
    value = git_output(repo, "rev-parse", "--verify", f"{revision}^{{tree}}")
    return require_sha(value, f"tree for {revision}")


def current_branch(repo: Path | str) -> str:
    completed = run_git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    branch = completed.stdout.strip()
    if completed.returncode != 0 or not branch:
        raise GuardError("HEAD is detached; this operation requires an explicit local branch")
    return branch


def tracked_dirty_paths(repo: Path | str) -> list[str]:
    """Return changed tracked paths while intentionally ignoring untracked/ignored runtime files."""

    output = run_git(
        repo,
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        "HEAD",
        "--",
    ).stdout
    return sorted({path for path in output.split("\0") if path})


def require_tracked_clean(repo: Path | str, context: str) -> None:
    dirty = tracked_dirty_paths(repo)
    if dirty:
        preview = ", ".join(dirty[:10])
        suffix = "" if len(dirty) <= 10 else f" (+{len(dirty) - 10} more)"
        raise GuardError(f"{context} is tracked-dirty: {preview}{suffix}")


def branch_role(branch: str, canonical: str = DEFAULT_CANONICAL) -> str:
    if branch == canonical:
        return "canonical"
    prefixes = (
        ("codex/task-", "task"),
        ("codex/aggregate-", "aggregate"),
        ("codex/result-", "result"),
        ("codex/archive-", "archive"),
        ("codex/recovery-", "recovery"),
        ("codex/experiment-", "experiment"),
    )
    for prefix, role in prefixes:
        if branch.startswith(prefix) and len(branch) > len(prefix):
            return role
    return "legacy"


def is_ancestor(repo: Path | str, ancestor: str, descendant: str) -> bool:
    completed = run_git(
        repo,
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
        check=False,
    )
    if completed.returncode not in (0, 1):
        detail = completed.stderr.strip() or "unable to compare revisions"
        raise GuardError(f"Git ancestry check failed: {detail}")
    return completed.returncode == 0


def changed_paths(repo: Path | str, old: str, new: str) -> list[str]:
    output = run_git(
        repo,
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        old,
        new,
        "--",
    ).stdout
    return sorted({path for path in output.split("\0") if path})


def commit_parents(repo: Path | str, commit: str) -> list[str]:
    fields = git_output(repo, "rev-list", "--parents", "-n", "1", commit).split()
    if not fields or fields[0] != commit:
        raise GuardError(f"could not inspect parents for {commit}")
    return fields[1:]


def remote_head(repo: Path | str, remote: str, branch: str) -> str | None:
    completed = run_git(
        repo,
        "ls-remote",
        "--heads",
        remote,
        f"refs/heads/{branch}",
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "remote query failed"
        raise GuardError(f"could not query {remote}/{branch}: {detail}")
    rows = [line.split() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        return None
    if len(rows) != 1 or len(rows[0]) != 2:
        raise GuardError(f"unexpected ls-remote response for {remote}/{branch}")
    return require_sha(rows[0][0], f"remote head {remote}/{branch}")


def remote_heads(repo: Path | str, remote: str = DEFAULT_REMOTE) -> dict[str, str]:
    completed = run_git(repo, "ls-remote", "--heads", remote, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "remote query failed"
        raise GuardError(f"could not query remote heads for {remote}: {detail}")
    result: dict[str, str] = {}
    prefix = "refs/heads/"
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or not fields[1].startswith(prefix):
            raise GuardError(f"unexpected ls-remote head row: {line!r}")
        branch = fields[1][len(prefix) :]
        if branch in result:
            raise GuardError(f"remote returned duplicate branch {branch}")
        result[branch] = require_sha(fields[0], f"remote head {branch}")
    return dict(sorted(result.items()))


def fetch_all_heads(repo: Path | str, remote: str = DEFAULT_REMOTE) -> None:
    run_git(
        repo,
        "fetch",
        "--prune",
        remote,
        f"+refs/heads/*:refs/remotes/{remote}/*",
    )


def read_json_file(path: Path | str) -> dict[str, Any]:
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GuardError(f"could not read valid JSON from {file_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GuardError(f"protocol record must be a JSON object: {file_path}")
    return payload


def _json_from_commit(repo: Path | str, commit: str, relative_path: str) -> dict[str, Any]:
    completed = run_git(repo, "show", f"{commit}:{relative_path}", check=False)
    if completed.returncode != 0:
        raise GuardError(f"{relative_path} is not committed at {commit}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise GuardError(f"committed protocol record is invalid JSON: {relative_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GuardError(f"committed protocol record must be a JSON object: {relative_path}")
    return payload


def _require_keys(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    context: str,
) -> None:
    missing = sorted(required - payload.keys())
    extra = sorted(payload.keys() - allowed)
    if missing:
        raise GuardError(f"{context} is missing required fields: {', '.join(missing)}")
    if extra:
        raise GuardError(f"{context} contains unsupported fields: {', '.join(extra)}")


def _require_timestamp(value: Any, field: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value:
        raise GuardError(f"{field} must be an ISO-8601 timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GuardError(f"{field} must be an ISO-8601 timestamp") from exc


def _require_string_list(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise GuardError(f"{field} must be an array of strings")
    if nonempty and not value:
        raise GuardError(f"{field} must not be empty")
    if len(set(value)) != len(value):
        raise GuardError(f"{field} must not contain duplicates")
    return value


def _validate_verification(value: Any, *, require_passed: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise GuardError("verification must be an array")
    if require_passed and not value:
        raise GuardError("completed records require at least one verification result")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise GuardError(f"verification[{index}] must be an object")
        _require_keys(
            item,
            required={"command", "outcome"},
            allowed={"command", "outcome", "evidence"},
            context=f"verification[{index}]",
        )
        if not isinstance(item["command"], str) or not item["command"]:
            raise GuardError(f"verification[{index}].command must be a nonempty string")
        if item["outcome"] not in VERIFICATION_OUTCOMES:
            raise GuardError(f"verification[{index}].outcome is invalid")
        if "evidence" in item and not isinstance(item["evidence"], str):
            raise GuardError(f"verification[{index}].evidence must be a string")
        if require_passed and item["outcome"] != "passed":
            raise GuardError("completed records may contain only passed verification outcomes")
        normalized.append(dict(item))
    return normalized


def _validate_scope(scope: str) -> str:
    if not scope or scope.startswith("/"):
        raise GuardError(f"changed scope must be a nonempty repository-relative path: {scope!r}")
    pure = PurePosixPath(scope)
    if any(part in ("", ".", "..") for part in pure.parts):
        raise GuardError(f"changed scope is not normalized: {scope!r}")
    return scope


def _path_in_scope(path: str, scope: str) -> bool:
    normalized = scope.rstrip("/")
    return path == normalized or path.startswith(normalized + "/")


TASK_HANDOFF_FIELDS = {
    "schema_version",
    "record_type",
    "task_id",
    "branch",
    "branch_role",
    "base_commit",
    "base_tree",
    "source_commit",
    "source_tree",
    "status",
    "changed_scopes",
    "verification",
    "server_evidence",
    "supersedes",
    "aggregate_eligible",
    "deployment_eligible",
    "started_at",
    "completed_at",
}


def validate_task_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_keys(
        payload,
        required=TASK_HANDOFF_FIELDS,
        allowed=TASK_HANDOFF_FIELDS,
        context="task handoff",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise GuardError(f"unsupported task handoff schema_version: {payload['schema_version']!r}")
    if payload["record_type"] != "task_handoff":
        raise GuardError("task handoff record_type must be 'task_handoff'")
    task_id = payload["task_id"]
    if not isinstance(task_id, str) or TASK_ID_RE.fullmatch(task_id) is None:
        raise GuardError("task_id must contain lowercase letters, digits, and hyphens")
    branch = payload["branch"]
    if not isinstance(branch, str) or branch_role(branch) != "task":
        raise GuardError("task handoff branch must use the codex/task-* role")
    if payload["branch_role"] != "task":
        raise GuardError("task handoff branch_role must be 'task'")
    for field in ("base_commit", "base_tree", "source_commit", "source_tree"):
        require_sha(payload[field], field)
    status = payload["status"]
    if status not in TASK_STATUSES:
        raise GuardError(f"invalid task handoff status: {status!r}")
    scopes = _require_string_list(payload["changed_scopes"], "changed_scopes", nonempty=True)
    for scope in scopes:
        _validate_scope(scope)
    completed = status == "completed"
    _validate_verification(payload["verification"], require_passed=completed)
    _require_string_list(payload["server_evidence"], "server_evidence")
    _require_string_list(payload["supersedes"], "supersedes")
    if not isinstance(payload["aggregate_eligible"], bool):
        raise GuardError("aggregate_eligible must be boolean")
    if payload["deployment_eligible"] is not False:
        raise GuardError("task branches are never directly deployment eligible")
    if completed and payload["aggregate_eligible"] is not True:
        raise GuardError("completed task handoffs must be aggregate eligible")
    if not completed and payload["aggregate_eligible"] is True:
        raise GuardError("only completed task handoffs may be aggregate eligible")
    _require_timestamp(payload["started_at"], "started_at")
    _require_timestamp(payload["completed_at"], "completed_at", nullable=True)
    if completed and payload["completed_at"] is None:
        raise GuardError("completed task handoffs require completed_at")
    if not completed and payload["completed_at"] is not None:
        raise GuardError("non-completed task handoffs must use completed_at=null")
    return dict(payload)


def _relative_record_path(repo: Path, path: Path | str) -> str:
    root = repository_root(repo)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise GuardError(f"protocol record is outside repository: {candidate}") from exc
    if relative.startswith("../") or relative == ".":
        raise GuardError(f"invalid protocol record path: {relative}")
    return relative


def validate_task_handoff(
    repo: Path | str,
    handoff_path: Path | str,
    *,
    branch_tip: str | None = None,
    expected_branch: str | None = None,
    remote: str | None = None,
    require_remote: bool = False,
) -> dict[str, Any]:
    root = repository_root(repo)
    relative = _relative_record_path(root, handoff_path)
    if not relative.startswith("agent_handoffs/tasks/") or not relative.endswith(".json"):
        raise GuardError("task handoff must be under agent_handoffs/tasks/*.json")
    tip = git_commit(root, branch_tip or "HEAD")
    committed_payload = validate_task_payload(_json_from_commit(root, tip, relative))

    candidate_path = root / relative
    if branch_tip is None and candidate_path.exists():
        working_payload = validate_task_payload(read_json_file(candidate_path))
        if working_payload != committed_payload:
            raise GuardError("working handoff differs from the handoff committed at branch tip")

    branch = expected_branch or committed_payload["branch"]
    if committed_payload["branch"] != branch:
        raise GuardError(
            f"handoff branch {committed_payload['branch']} does not match expected branch {branch}"
        )
    if branch_tip is None and current_branch(root) != branch:
        raise GuardError(f"current branch does not match handoff branch {branch}")

    source_commit = require_sha(committed_payload["source_commit"], "source_commit")
    source_tree = require_sha(committed_payload["source_tree"], "source_tree")
    base_commit = require_sha(committed_payload["base_commit"], "base_commit")
    base_tree = require_sha(committed_payload["base_tree"], "base_tree")
    if git_tree(root, source_commit) != source_tree:
        raise GuardError("handoff source_tree does not match source_commit")
    if git_tree(root, base_commit) != base_tree:
        raise GuardError("handoff base_tree does not match base_commit")
    if not is_ancestor(root, base_commit, source_commit):
        raise GuardError("handoff base_commit is not an ancestor of source_commit")

    parents = commit_parents(root, tip)
    if parents != [source_commit]:
        raise GuardError("handoff tip must have exactly source_commit as its only parent")
    tip_paths = changed_paths(root, source_commit, tip)
    if tip_paths != [relative]:
        raise GuardError(
            "handoff tip must be a handoff-only commit; changed paths: " + ", ".join(tip_paths)
        )

    source_paths = changed_paths(root, base_commit, source_commit)
    scopes = committed_payload["changed_scopes"]
    outside = [path for path in source_paths if not any(_path_in_scope(path, scope) for scope in scopes)]
    if outside:
        raise GuardError("source changes fall outside declared changed scope: " + ", ".join(outside))

    if require_remote:
        if not remote:
            raise GuardError("remote is required when require_remote=True")
        remote_tip = remote_head(root, remote, branch)
        if remote_tip != tip:
            raise GuardError(
                f"local/remote task tip mismatch for {branch}: local={tip}, remote={remote_tip}"
            )
    return committed_payload


AGGREGATE_MANIFEST_FIELDS = {
    "schema_version",
    "record_type",
    "aggregate_id",
    "canonical_branch",
    "aggregate_branch",
    "base_commit",
    "base_tree",
    "source_commit",
    "source_tree",
    "remote",
    "snapshot_at",
    "created_at",
    "heads",
    "verification",
    "server_evidence",
    "canonical_eligible",
    "deployment_eligible",
}

HEAD_FIELDS = {"branch", "commit", "role", "disposition", "reason", "handoff"}

ROLE_DISPOSITIONS = {
    "canonical": {"already_ancestor"},
    "task": {
        "included",
        "already_ancestor",
        "patch_equivalent",
        "superseded",
        "rejected",
        "in_progress",
        "needs_review",
    },
    "aggregate": {"already_ancestor", "superseded", "rejected", "needs_review"},
    "result": {"result_only", "rejected", "needs_review"},
    "archive": {"archive_only", "rejected", "needs_review"},
    "recovery": {"recovery_only", "rejected", "needs_review"},
    "experiment": {"superseded", "rejected", "needs_review"},
    "legacy": {
        "already_ancestor",
        "patch_equivalent",
        "superseded",
        "rejected",
        "archive_only",
        "recovery_only",
        "result_only",
        "in_progress",
        "needs_review",
    },
}


def _validate_head_entry(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GuardError(f"heads[{index}] must be an object")
    _require_keys(
        value,
        required=HEAD_FIELDS - {"handoff"},
        allowed=HEAD_FIELDS,
        context=f"heads[{index}]",
    )
    branch = value["branch"]
    if not isinstance(branch, str) or not branch:
        raise GuardError(f"heads[{index}].branch must be a nonempty string")
    require_sha(value["commit"], f"heads[{index}].commit")
    role = value["role"]
    if role not in BRANCH_ROLES:
        raise GuardError(f"heads[{index}].role is invalid: {role!r}")
    actual_role = branch_role(branch)
    if role != actual_role:
        raise GuardError(
            f"heads[{index}] role mismatch for {branch}: declared={role}, actual={actual_role}"
        )
    disposition = value["disposition"]
    if disposition not in HEAD_DISPOSITIONS:
        raise GuardError(f"heads[{index}].disposition is invalid: {disposition!r}")
    if disposition not in ROLE_DISPOSITIONS[role]:
        raise GuardError(f"{role} branch {branch} cannot use disposition {disposition}")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise GuardError(f"heads[{index}].reason must be a nonempty string")
    handoff = value.get("handoff")
    if handoff is not None and (not isinstance(handoff, str) or not handoff):
        raise GuardError(f"heads[{index}].handoff must be a nonempty string or null")
    if disposition == "included" and not handoff:
        raise GuardError(f"included task branch {branch} requires a handoff path")
    if disposition != "included" and handoff is not None:
        raise GuardError(f"non-included branch {branch} must use handoff=null")
    return dict(value)


def validate_aggregate_payload(
    payload: Mapping[str, Any],
    *,
    current_heads: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    _require_keys(
        payload,
        required=AGGREGATE_MANIFEST_FIELDS,
        allowed=AGGREGATE_MANIFEST_FIELDS,
        context="aggregate manifest",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise GuardError(
            f"unsupported aggregate manifest schema_version: {payload['schema_version']!r}"
        )
    if payload["record_type"] != "aggregate_manifest":
        raise GuardError("aggregate manifest record_type must be 'aggregate_manifest'")
    aggregate_id = payload["aggregate_id"]
    if not isinstance(aggregate_id, str) or TASK_ID_RE.fullmatch(aggregate_id) is None:
        raise GuardError("aggregate_id must contain lowercase letters, digits, and hyphens")
    canonical = payload["canonical_branch"]
    if canonical != DEFAULT_CANONICAL:
        raise GuardError(f"canonical_branch must be {DEFAULT_CANONICAL}")
    aggregate_branch = payload["aggregate_branch"]
    if not isinstance(aggregate_branch, str) or branch_role(aggregate_branch) != "aggregate":
        raise GuardError("aggregate_branch must use codex/aggregate-*")
    for field in ("base_commit", "base_tree", "source_commit", "source_tree"):
        require_sha(payload[field], field)
    if not isinstance(payload["remote"], str) or not payload["remote"]:
        raise GuardError("remote must be a nonempty string")
    _require_timestamp(payload["snapshot_at"], "snapshot_at")
    _require_timestamp(payload["created_at"], "created_at")
    _validate_verification(payload["verification"], require_passed=True)
    _require_string_list(payload["server_evidence"], "server_evidence")
    if payload["canonical_eligible"] is not True:
        raise GuardError("final aggregate manifest must be canonical eligible")
    if payload["deployment_eligible"] is not True:
        raise GuardError("final aggregate manifest must be deployment eligible")
    raw_heads = payload["heads"]
    if not isinstance(raw_heads, list) or not raw_heads:
        raise GuardError("aggregate manifest heads must be a nonempty array")
    entries = [_validate_head_entry(item, index) for index, item in enumerate(raw_heads)]
    names = [entry["branch"] for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise GuardError("aggregate manifest contains duplicate branches: " + ", ".join(duplicates))
    unresolved = [entry["branch"] for entry in entries if entry["disposition"] == "needs_review"]
    if unresolved:
        raise GuardError("aggregate manifest contains needs_review branches: " + ", ".join(unresolved))

    if current_heads is not None:
        declared = {entry["branch"]: entry["commit"] for entry in entries}
        current = dict(current_heads)
        missing = sorted(set(current) - set(declared))
        extra = sorted(set(declared) - set(current))
        if missing:
            raise GuardError("aggregate manifest is missing remote heads: " + ", ".join(missing))
        if extra:
            raise GuardError("aggregate manifest contains heads absent from snapshot: " + ", ".join(extra))
        stale = sorted(branch for branch in current if current[branch] != declared[branch])
        if stale:
            details = ", ".join(
                f"{branch} declared={declared[branch]} current={current[branch]}" for branch in stale
            )
            raise GuardError("aggregate snapshot is stale; remote heads changed or advanced: " + details)
    return dict(payload)


def validate_aggregate_manifest(
    repo: Path | str,
    manifest_path: Path | str,
    *,
    aggregate_tip: str | None = None,
    expected_branch: str | None = None,
    remote: str | None = None,
    require_remote: bool = False,
    current_heads: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = repository_root(repo)
    relative = _relative_record_path(root, manifest_path)
    if not relative.startswith("agent_handoffs/aggregates/") or not relative.endswith(".json"):
        raise GuardError("aggregate manifest must be under agent_handoffs/aggregates/*.json")
    tip = git_commit(root, aggregate_tip or "HEAD")
    committed_raw = _json_from_commit(root, tip, relative)
    payload = validate_aggregate_payload(committed_raw)

    candidate_path = root / relative
    if aggregate_tip is None and candidate_path.exists():
        working_payload = validate_aggregate_payload(read_json_file(candidate_path))
        if working_payload != payload:
            raise GuardError("working aggregate manifest differs from the manifest committed at tip")

    aggregate_branch = expected_branch or payload["aggregate_branch"]
    if payload["aggregate_branch"] != aggregate_branch:
        raise GuardError(
            f"manifest aggregate_branch {payload['aggregate_branch']} does not match {aggregate_branch}"
        )
    if aggregate_tip is None and current_branch(root) != aggregate_branch:
        raise GuardError(f"current branch does not match aggregate branch {aggregate_branch}")

    source_commit = require_sha(payload["source_commit"], "source_commit")
    source_tree = require_sha(payload["source_tree"], "source_tree")
    base_commit = require_sha(payload["base_commit"], "base_commit")
    base_tree = require_sha(payload["base_tree"], "base_tree")
    if git_tree(root, source_commit) != source_tree:
        raise GuardError("aggregate source_tree does not match source_commit")
    if git_tree(root, base_commit) != base_tree:
        raise GuardError("aggregate base_tree does not match base_commit")
    if not is_ancestor(root, base_commit, source_commit):
        raise GuardError("aggregate base_commit is not an ancestor of source_commit")
    parents = commit_parents(root, tip)
    if parents != [source_commit]:
        raise GuardError("aggregate tip must have exactly source_commit as its only parent")
    tip_paths = changed_paths(root, source_commit, tip)
    if tip_paths != [relative]:
        raise GuardError(
            "aggregate tip must be a manifest-only commit; changed paths: " + ", ".join(tip_paths)
        )

    heads_for_validation = current_heads
    if require_remote:
        if not remote:
            raise GuardError("remote is required when require_remote=True")
        if payload["remote"] != remote:
            raise GuardError(
                f"manifest remote {payload['remote']} does not match requested remote {remote}"
            )
        aggregate_remote_tip = remote_head(root, remote, aggregate_branch)
        if aggregate_remote_tip != tip:
            raise GuardError(
                f"local/remote aggregate tip mismatch: local={tip}, remote={aggregate_remote_tip}"
            )
        fetched_heads = remote_heads(root, remote)
        fetched_heads.pop(aggregate_branch, None)
        heads_for_validation = fetched_heads
    validate_aggregate_payload(payload, current_heads=heads_for_validation)

    for entry in payload["heads"]:
        if entry["disposition"] != "included":
            continue
        handoff = validate_task_handoff(
            root,
            entry["handoff"],
            branch_tip=entry["commit"],
            expected_branch=entry["branch"],
        )
        if handoff["status"] != "completed" or handoff["aggregate_eligible"] is not True:
            raise GuardError(f"included branch {entry['branch']} lacks a completed eligible handoff")
        if not is_ancestor(root, entry["commit"], source_commit):
            raise GuardError(f"included branch {entry['branch']} is not integrated into aggregate source")
    return payload


def _default_head_disposition(
    repo: Path,
    *,
    branch: str,
    commit: str,
    source_commit: str,
    canonical: str,
) -> dict[str, Any]:
    role = branch_role(branch, canonical)
    if role == "canonical" and is_ancestor(repo, commit, source_commit):
        disposition = "already_ancestor"
        reason = "canonical snapshot is an ancestor of aggregate source"
    elif role == "aggregate" and is_ancestor(repo, commit, source_commit):
        disposition = "already_ancestor"
        reason = "prior aggregate is an ancestor of aggregate source"
    elif role == "archive":
        disposition = "archive_only"
        reason = "branch role excludes source integration"
    elif role == "recovery":
        disposition = "recovery_only"
        reason = "branch role requires manual recovery review and excludes direct integration"
    elif role == "result":
        disposition = "result_only"
        reason = "server result branch is artifact-only"
    else:
        disposition = "needs_review"
        reason = "no safe disposition can be inferred from Git role and ancestry alone"
    return {
        "branch": branch,
        "commit": commit,
        "role": role,
        "disposition": disposition,
        "reason": reason,
        "handoff": None,
    }


def build_aggregate_draft(
    repo: Path | str,
    *,
    aggregate_id: str,
    aggregate_branch: str,
    remote: str,
    canonical: str,
    heads: Mapping[str, str],
    timestamp: str,
) -> dict[str, Any]:
    root = repository_root(repo)
    if branch_role(aggregate_branch, canonical) != "aggregate":
        raise GuardError("aggregate draft branch must use codex/aggregate-*")
    if current_branch(root) != aggregate_branch:
        raise GuardError(f"aggregate draft must be created on {aggregate_branch}")
    source_commit = git_commit(root)
    source_tree = git_tree(root)
    if canonical not in heads:
        raise GuardError(f"remote snapshot does not contain canonical branch {canonical}")
    base_commit = heads[canonical]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "aggregate_manifest",
        "aggregate_id": aggregate_id,
        "canonical_branch": canonical,
        "aggregate_branch": aggregate_branch,
        "base_commit": base_commit,
        "base_tree": git_tree(root, base_commit),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "remote": remote,
        "snapshot_at": timestamp,
        "created_at": timestamp,
        "heads": [
            _default_head_disposition(
                root,
                branch=branch,
                commit=commit,
                source_commit=source_commit,
                canonical=canonical,
            )
            for branch, commit in sorted(heads.items())
            if branch != aggregate_branch
        ],
        "verification": [
            {
                "command": "aggregate verification pending",
                "outcome": "not_run",
                "evidence": "complete before manifest-only commit",
            }
        ],
        "server_evidence": [],
        "canonical_eligible": True,
        "deployment_eligible": True,
    }


def write_json_atomic(path: Path | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        temporary.write_text(data, encoding="utf-8")
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise GuardError(f"could not write aggregate draft {destination}: {exc}") from exc


def check_local_sync(
    repo: Path | str,
    *,
    remote: str = DEFAULT_REMOTE,
    canonical: str = DEFAULT_CANONICAL,
    apply: bool = False,
) -> dict[str, Any]:
    root = repository_root(repo)
    require_tracked_clean(root, "canonical checkout")
    branch = current_branch(root)
    if branch != canonical:
        raise GuardError(f"local-sync must run on canonical branch {canonical}, found {branch}")
    local_commit = git_commit(root)
    remote_commit = remote_head(root, remote, canonical)
    if remote_commit is None:
        raise GuardError(f"remote canonical branch does not exist: {remote}/{canonical}")
    if local_commit != remote_commit:
        if not apply:
            raise GuardError(
                f"local canonical is not current: local={local_commit}, remote={remote_commit}; "
                "rerun with --apply only after preserving dirty work"
            )
        fetch_all_heads(root, remote)
        tracked_remote = git_commit(root, f"refs/remotes/{remote}/{canonical}")
        if tracked_remote != remote_commit:
            raise GuardError("fetched remote-tracking canonical does not match remote canonical tip")
        if not is_ancestor(root, local_commit, remote_commit):
            raise GuardError("local canonical diverged from remote; fast-forward is impossible")
        run_git(root, "merge", "--ff-only", f"refs/remotes/{remote}/{canonical}")
        local_commit = git_commit(root)
    local_tree = git_tree(root)
    remote_tree = git_tree(root, remote_commit)
    if local_commit != remote_commit or local_tree != remote_tree:
        raise GuardError("local canonical commit/tree parity check failed")
    require_tracked_clean(root, "canonical checkout after local-sync")
    return {
        "ok": True,
        "branch": canonical,
        "commit": local_commit,
        "tree": local_tree,
        "remote": remote,
        "applied": apply,
        "tracked_clean": True,
    }


def check_server_state(
    repo: Path | str,
    *,
    expected_commit: str,
    expected_tree: str,
    remote: str = DEFAULT_REMOTE,
    canonical: str = DEFAULT_CANONICAL,
    sync: bool = False,
) -> dict[str, Any]:
    root = repository_root(repo)
    expected_commit = require_sha(expected_commit, "expected_commit")
    expected_tree = require_sha(expected_tree, "expected_tree")
    require_tracked_clean(root, "server source checkout")
    remote_commit = remote_head(root, remote, canonical)
    if remote_commit != expected_commit:
        raise GuardError(
            f"expected server commit is not remote canonical tip: expected={expected_commit}, "
            f"remote={remote_commit}"
        )
    if sync:
        fetch_all_heads(root, remote)
        fetched = git_commit(root, f"refs/remotes/{remote}/{canonical}")
        if fetched != expected_commit:
            raise GuardError("fetched canonical commit does not match expected commit")
        if git_tree(root, fetched) != expected_tree:
            raise GuardError("fetched canonical tree does not match expected tree")
        run_git(root, "switch", "--detach", expected_commit)
    head = git_commit(root)
    tree = git_tree(root)
    if head != expected_commit:
        raise GuardError(f"server HEAD mismatch: expected={expected_commit}, actual={head}")
    if tree != expected_tree:
        raise GuardError(f"server tree mismatch: expected={expected_tree}, actual={tree}")
    require_tracked_clean(root, "server source checkout after synchronization")
    return {
        "ok": True,
        "commit": head,
        "tree": tree,
        "remote_canonical_commit": remote_commit,
        "tracked_clean": True,
        "synchronized": sync,
    }


def check_result_branch(
    repo: Path | str,
    *,
    base_commit: str,
    allowed_prefixes: Sequence[str],
    remote: str | None = None,
    require_remote: bool = False,
) -> dict[str, Any]:
    root = repository_root(repo)
    require_tracked_clean(root, "result checkout")
    branch = current_branch(root)
    if branch_role(branch) != "result":
        raise GuardError("result-check requires a codex/result-* branch")
    base_commit = require_sha(base_commit, "base_commit")
    head = git_commit(root)
    if not is_ancestor(root, base_commit, head):
        raise GuardError("result branch base_commit is not an ancestor of HEAD")
    scopes = [_validate_scope(prefix) for prefix in allowed_prefixes]
    if not scopes:
        raise GuardError("result-check requires at least one allowed path prefix")
    paths = changed_paths(root, base_commit, head)
    disallowed = [path for path in paths if not any(_path_in_scope(path, scope) for scope in scopes)]
    if disallowed:
        raise GuardError("result branch changed paths outside allowed scopes: " + ", ".join(disallowed))
    if require_remote:
        if not remote:
            raise GuardError("remote is required when require_remote=True")
        remote_tip = remote_head(root, remote, branch)
        if remote_tip != head:
            raise GuardError(f"result branch local/remote mismatch: local={head}, remote={remote_tip}")
    return {
        "ok": True,
        "branch": branch,
        "base_commit": base_commit,
        "commit": head,
        "tree": git_tree(root),
        "changed_paths": paths,
        "allowed_prefixes": list(scopes),
        "remote_parity": require_remote,
    }


def parse_push_updates(text: str) -> list[PushUpdate]:
    updates: list[PushUpdate] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise GuardError(f"invalid pre-push input at line {line_number}: expected four columns")
        local_ref, local_sha, remote_ref, remote_sha = fields
        require_sha(local_sha, f"pre-push local SHA line {line_number}")
        require_sha(remote_sha, f"pre-push remote SHA line {line_number}")
        updates.append(PushUpdate(local_ref, local_sha, remote_ref, remote_sha))
    return updates


def _branch_from_head_ref(ref: str, *, field: str) -> str:
    prefix = "refs/heads/"
    if not ref.startswith(prefix) or len(ref) == len(prefix):
        raise GuardError(f"{field} must name refs/heads/*, found {ref!r}")
    return ref[len(prefix) :]


def _tip_changed_paths(repo: Path, commit: str) -> list[str]:
    parents = commit_parents(repo, commit)
    if len(parents) != 1:
        return []
    return changed_paths(repo, parents[0], commit)


def _validate_non_source_eligibility_claims(
    repo: Path,
    *,
    commit: str,
    branch: str,
) -> None:
    for path in _tip_changed_paths(repo, commit):
        if not path.startswith("agent_handoffs/") or not path.endswith(".json"):
            continue
        payload = _json_from_commit(repo, commit, path)
        if payload.get("aggregate_eligible") is True or payload.get("deployment_eligible") is True:
            raise GuardError(
                f"{branch_role(branch)} branch {branch} cannot claim aggregate or deployment eligibility"
            )


def _validate_task_push(repo: Path, *, branch: str, commit: str) -> None:
    tip_paths = _tip_changed_paths(repo, commit)
    handoffs = [
        path
        for path in tip_paths
        if path.startswith("agent_handoffs/tasks/") and path.endswith(".json")
    ]
    if not handoffs:
        return
    if len(handoffs) != 1 or tip_paths != handoffs:
        raise GuardError("task handoff tip must change exactly one task handoff file")
    validate_task_handoff(
        repo,
        handoffs[0],
        branch_tip=commit,
        expected_branch=branch,
    )


def _validate_aggregate_push(repo: Path, *, branch: str, commit: str) -> None:
    tip_paths = _tip_changed_paths(repo, commit)
    manifests = [
        path
        for path in tip_paths
        if path.startswith("agent_handoffs/aggregates/") and path.endswith(".json")
    ]
    if not manifests:
        return
    if len(manifests) != 1 or tip_paths != manifests:
        raise GuardError("aggregate manifest tip must change exactly one aggregate manifest file")
    validate_aggregate_manifest(
        repo,
        manifests[0],
        aggregate_tip=commit,
        expected_branch=branch,
    )


def validate_pre_push(
    repo: Path | str,
    updates: Sequence[PushUpdate],
    *,
    remote: str = DEFAULT_REMOTE,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    root = repository_root(repo)
    environment = os.environ if env is None else env
    results: list[dict[str, Any]] = []
    for update in updates:
        remote_branch = _branch_from_head_ref(update.remote_ref, field="remote_ref")
        target_role = branch_role(remote_branch)
        if update.local_sha == ZERO_SHA:
            if target_role == "canonical":
                raise GuardError("canonical branch deletion is forbidden")
            results.append({"branch": remote_branch, "role": target_role, "action": "delete"})
            continue
        local_commit = git_commit(root, update.local_sha)
        if local_commit != update.local_sha:
            raise GuardError(f"pre-push local SHA is not a commit: {update.local_sha}")

        if target_role == "canonical":
            if environment.get("RFR_AGGREGATOR_AUTHORIZED") != "1":
                raise GuardError("canonical push is not explicitly aggregator-authorized")
            manifest_path = environment.get("RFR_AGGREGATE_MANIFEST")
            if not manifest_path:
                raise GuardError("canonical push requires RFR_AGGREGATE_MANIFEST")
            if update.remote_sha == ZERO_SHA:
                raise GuardError("canonical branch creation through pre-push is forbidden")
            local_branch = _branch_from_head_ref(update.local_ref, field="local_ref")
            if branch_role(local_branch) != "aggregate":
                raise GuardError("canonical source must be pushed from a codex/aggregate-* branch")
            current_heads = remote_heads(root, remote)
            current_heads.pop(local_branch, None)
            payload = validate_aggregate_manifest(
                root,
                manifest_path,
                aggregate_tip=update.local_sha,
                expected_branch=local_branch,
                current_heads=current_heads,
            )
            if payload["canonical_branch"] != remote_branch:
                raise GuardError("aggregate manifest canonical branch does not match push target")
            if payload["base_commit"] != update.remote_sha:
                raise GuardError(
                    "aggregate base does not equal remote canonical old SHA: "
                    f"base={payload['base_commit']}, remote_old={update.remote_sha}"
                )
            if not is_ancestor(root, update.remote_sha, update.local_sha):
                raise GuardError("canonical update is not a fast-forward")
            results.append(
                {
                    "branch": remote_branch,
                    "role": "canonical",
                    "action": "fast_forward",
                    "manifest": manifest_path,
                }
            )
            continue

        local_branch = _branch_from_head_ref(update.local_ref, field="local_ref")
        if local_branch != remote_branch:
            raise GuardError("noncanonical pushes must preserve the local branch name")
        local_role = branch_role(local_branch)
        if local_role == "task":
            _validate_task_push(root, branch=local_branch, commit=update.local_sha)
        elif local_role == "aggregate":
            _validate_aggregate_push(root, branch=local_branch, commit=update.local_sha)
        elif local_role in {"archive", "recovery", "result", "experiment"}:
            _validate_non_source_eligibility_claims(
                root,
                commit=update.local_sha,
                branch=local_branch,
            )
        results.append({"branch": remote_branch, "role": local_role, "action": "update"})
    return results


def _status_payload(repo: Path, *, remote: str, canonical: str) -> dict[str, Any]:
    head = git_commit(repo)
    tree = git_tree(repo)
    branch = current_branch(repo)
    dirty = tracked_dirty_paths(repo)
    canonical_remote = remote_head(repo, remote, canonical)
    return {
        "repository": str(repository_root(repo)),
        "branch": branch,
        "role": branch_role(branch, canonical),
        "commit": head,
        "tree": tree,
        "tracked_clean": not dirty,
        "tracked_dirty_paths": dirty,
        "canonical_branch": canonical,
        "remote": remote,
        "remote_canonical_commit": canonical_remote,
        "canonical_commit_equal": head == canonical_remote if branch == canonical else None,
    }


def _emit(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, sort_keys=True)
        else:
            rendered = str(value)
        print(f"{key}: {rendered}")


def command_status(args: argparse.Namespace) -> dict[str, Any]:
    return _status_payload(repository_root(args.repo), remote=args.remote, canonical=args.canonical)


def command_agent_start(args: argparse.Namespace) -> dict[str, Any]:
    repo = repository_root(args.repo)
    require_tracked_clean(repo, "canonical checkout")
    branch = current_branch(repo)
    if branch != args.canonical:
        raise GuardError(f"agent-start must run from canonical branch {args.canonical}, found {branch}")
    task_branch = args.branch
    if branch_role(task_branch, args.canonical) != "task":
        raise GuardError("new agent branch must use codex/task-*")
    if args.task_id not in task_branch:
        raise GuardError("task branch must include the declared task_id")
    head = git_commit(repo)
    tree = git_tree(repo)
    remote_tip = remote_head(repo, args.remote, args.canonical)
    if remote_tip != head:
        raise GuardError(
            f"canonical local/remote mismatch: local={head}, remote={remote_tip}; synchronize first"
        )
    return {
        "ok": True,
        "task_id": args.task_id,
        "task_branch": task_branch,
        "base_commit": head,
        "base_tree": tree,
        "remote": args.remote,
        "canonical": args.canonical,
    }


def command_agent_finish(args: argparse.Namespace) -> dict[str, Any]:
    repo = repository_root(args.repo)
    require_tracked_clean(repo, "task checkout")
    payload = validate_task_handoff(
        repo,
        args.handoff,
        remote=args.remote,
        require_remote=True,
    )
    if payload["status"] != "completed" or payload["aggregate_eligible"] is not True:
        raise GuardError("agent-finish requires a completed, aggregate-eligible task handoff")
    return {
        "ok": True,
        "task_id": payload["task_id"],
        "branch": payload["branch"],
        "tip_commit": git_commit(repo),
        "tip_tree": git_tree(repo),
        "remote_parity": True,
    }


def command_aggregate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo = repository_root(args.repo)
    require_tracked_clean(repo, "aggregate source checkout")
    if args.fetch:
        fetch_all_heads(repo, args.remote)
    heads = remote_heads(repo, args.remote)
    if args.canonical not in heads:
        raise GuardError(f"remote snapshot does not contain canonical branch {args.canonical}")
    branch = current_branch(repo)
    if branch_role(branch, args.canonical) != "aggregate":
        raise GuardError("aggregate-preflight must run from a codex/aggregate-* branch")
    result: dict[str, Any] = {
        "ok": True,
        "aggregate_branch": branch,
        "remote": args.remote,
        "canonical": args.canonical,
        "remote_head_count": len(heads),
        "heads": heads,
    }
    if args.write_draft:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        draft = build_aggregate_draft(
            repo,
            aggregate_id=args.aggregate_id,
            aggregate_branch=branch,
            remote=args.remote,
            canonical=args.canonical,
            heads=heads,
            timestamp=timestamp,
        )
        destination = Path(args.write_draft)
        if not destination.is_absolute():
            destination = repo / destination
        relative = _relative_record_path(repo, destination)
        if not relative.startswith("agent_handoffs/aggregates/") or not relative.endswith(".json"):
            raise GuardError("aggregate draft must be written under agent_handoffs/aggregates/*.json")
        write_json_atomic(destination, draft)
        result["draft_manifest"] = relative
        result["needs_review"] = [
            entry["branch"] for entry in draft["heads"] if entry["disposition"] == "needs_review"
        ]
    return result


def command_aggregate_finalize(args: argparse.Namespace) -> dict[str, Any]:
    repo = repository_root(args.repo)
    require_tracked_clean(repo, "aggregate checkout")
    if args.fetch:
        fetch_all_heads(repo, args.remote)
    payload = validate_aggregate_manifest(
        repo,
        args.manifest,
        remote=args.remote,
        require_remote=True,
    )
    canonical_tip = remote_head(repo, args.remote, payload["canonical_branch"])
    if canonical_tip != payload["base_commit"]:
        raise GuardError(
            "remote canonical advanced after aggregate base: "
            f"base={payload['base_commit']}, remote={canonical_tip}"
        )
    aggregate_tip = git_commit(repo)
    if not is_ancestor(repo, canonical_tip, aggregate_tip):
        raise GuardError("aggregate tip is not a fast-forward of remote canonical")
    return {
        "ok": True,
        "aggregate_branch": payload["aggregate_branch"],
        "aggregate_commit": aggregate_tip,
        "aggregate_tree": git_tree(repo),
        "canonical_branch": payload["canonical_branch"],
        "canonical_old_commit": canonical_tip,
        "remote_parity": True,
        "canonical_fast_forward_command": (
            f"git push {args.remote} {payload['aggregate_branch']}:{payload['canonical_branch']}"
        ),
    }


def command_local_sync(args: argparse.Namespace) -> dict[str, Any]:
    return check_local_sync(
        repository_root(args.repo),
        remote=args.remote,
        canonical=args.canonical,
        apply=args.apply,
    )


def command_server_check(args: argparse.Namespace) -> dict[str, Any]:
    return check_server_state(
        repository_root(args.repo),
        expected_commit=args.expected_commit,
        expected_tree=args.expected_tree,
        remote=args.remote,
        canonical=args.canonical,
        sync=args.sync,
    )


def command_result_check(args: argparse.Namespace) -> dict[str, Any]:
    return check_result_branch(
        repository_root(args.repo),
        base_commit=args.base_commit,
        allowed_prefixes=args.allowed_prefix,
        remote=args.remote,
        require_remote=args.require_remote,
    )


def command_pre_push(args: argparse.Namespace) -> dict[str, Any]:
    updates = parse_push_updates(sys.stdin.read())
    validated = validate_pre_push(
        repository_root(args.repo),
        updates,
        remote=args.remote_name,
    )
    return {
        "ok": True,
        "remote_name": args.remote_name,
        "remote_url": args.remote_url,
        "updates": validated,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository checkout path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="read repository and remote canonical state")
    status.add_argument("--remote", default=DEFAULT_REMOTE)
    status.add_argument("--canonical", default=DEFAULT_CANONICAL)
    status.add_argument("--json", action="store_true", dest="as_json")
    status.set_defaults(handler=command_status)

    start = subparsers.add_parser("agent-start", help="validate a new task branch base")
    start.add_argument("--task-id", required=True)
    start.add_argument("--branch", required=True)
    start.add_argument("--remote", default=DEFAULT_REMOTE)
    start.add_argument("--canonical", default=DEFAULT_CANONICAL)
    start.add_argument("--json", action="store_true", dest="as_json")
    start.set_defaults(handler=command_agent_start)

    finish = subparsers.add_parser("agent-finish", help="validate a completed task handoff")
    finish.add_argument("--handoff", required=True)
    finish.add_argument("--remote", default=DEFAULT_REMOTE)
    finish.add_argument("--json", action="store_true", dest="as_json")
    finish.set_defaults(handler=command_agent_finish)

    aggregate_preflight = subparsers.add_parser(
        "aggregate-preflight",
        help="refresh and snapshot every remote head for one aggregate cycle",
    )
    aggregate_preflight.add_argument("--aggregate-id", required=True)
    aggregate_preflight.add_argument("--remote", default=DEFAULT_REMOTE)
    aggregate_preflight.add_argument("--canonical", default=DEFAULT_CANONICAL)
    aggregate_preflight.add_argument("--fetch", action="store_true")
    aggregate_preflight.add_argument("--write-draft")
    aggregate_preflight.add_argument("--json", action="store_true", dest="as_json")
    aggregate_preflight.set_defaults(handler=command_aggregate_preflight)

    aggregate_finalize = subparsers.add_parser(
        "aggregate-finalize",
        help="revalidate a pushed aggregate and emit its canonical fast-forward command",
    )
    aggregate_finalize.add_argument("--manifest", required=True)
    aggregate_finalize.add_argument("--remote", default=DEFAULT_REMOTE)
    aggregate_finalize.add_argument("--fetch", action="store_true")
    aggregate_finalize.add_argument("--json", action="store_true", dest="as_json")
    aggregate_finalize.set_defaults(handler=command_aggregate_finalize)

    local_sync = subparsers.add_parser(
        "local-sync",
        help="verify or explicitly fast-forward the clean local canonical checkout",
    )
    local_sync.add_argument("--remote", default=DEFAULT_REMOTE)
    local_sync.add_argument("--canonical", default=DEFAULT_CANONICAL)
    local_sync.add_argument("--apply", action="store_true")
    local_sync.add_argument("--json", action="store_true", dest="as_json")
    local_sync.set_defaults(handler=command_local_sync)

    server_check = subparsers.add_parser(
        "server-check",
        help="verify or Git-synchronize a clean server checkout to exact canonical source",
    )
    server_check.add_argument("--expected-commit", required=True)
    server_check.add_argument("--expected-tree", required=True)
    server_check.add_argument("--remote", default=DEFAULT_REMOTE)
    server_check.add_argument("--canonical", default=DEFAULT_CANONICAL)
    server_check.add_argument("--sync", action="store_true")
    server_check.add_argument("--json", action="store_true", dest="as_json")
    server_check.set_defaults(handler=command_server_check)

    result_check = subparsers.add_parser(
        "result-check",
        help="validate an artifact-only server result branch",
    )
    result_check.add_argument("--base-commit", required=True)
    result_check.add_argument("--allowed-prefix", action="append", required=True)
    result_check.add_argument("--remote", default=DEFAULT_REMOTE)
    result_check.add_argument("--require-remote", action="store_true")
    result_check.add_argument("--json", action="store_true", dest="as_json")
    result_check.set_defaults(handler=command_result_check)

    pre_push = subparsers.add_parser("pre-push", help="validate Git pre-push update records")
    pre_push.add_argument("--remote-name", required=True)
    pre_push.add_argument("--remote-url", required=True)
    pre_push.add_argument("--json", action="store_true", dest="as_json")
    pre_push.set_defaults(handler=command_pre_push)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.handler(args)
    except GuardError as exc:
        print(f"repo-sync-guard: FAIL: {exc}", file=sys.stderr)
        return 2
    _emit(payload, as_json=getattr(args, "as_json", False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
