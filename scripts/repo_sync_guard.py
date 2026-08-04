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
