from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = REPO_ROOT / "scripts" / "repo_sync_guard.py"
ZERO_SHA = "0" * 40
NOW = "2026-08-04T12:00:00+08:00"


def _load_guard_module():
    spec = importlib.util.spec_from_file_location("repo_sync_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(repo: Path, *args: str, input_text: str | None = None) -> str:
    completed = subprocess.run(
        list(args),
        cwd=repo,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def _git(repo: Path, *args: str) -> str:
    return _run(repo, "git", *args)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class GitFixture:
    def __init__(self, root: Path):
        self.root = root
        self.remote = root / "remote.git"
        self.repo = root / "repo"
        _run(root, "git", "init", "--bare", str(self.remote))
        _run(root, "git", "clone", str(self.remote), str(self.repo))
        _git(self.repo, "config", "user.name", "Protocol Test")
        _git(self.repo, "config", "user.email", "protocol@example.invalid")
        _git(self.repo, "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*")
        _git(self.repo, "switch", "-c", "jk_standard_rl")
        (self.repo / "README.md").write_text("base\n", encoding="utf-8")
        _git(self.repo, "add", "README.md")
        _git(self.repo, "commit", "-m", "base")
        _git(self.repo, "push", "-u", "origin", "jk_standard_rl")
        self.base_commit = _git(self.repo, "rev-parse", "HEAD")
        self.base_tree = _git(self.repo, "rev-parse", "HEAD^{tree}")

    def commit_file(self, relative: str, text: str, message: str) -> str:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        _git(self.repo, "add", relative)
        _git(self.repo, "commit", "-m", message)
        return _git(self.repo, "rev-parse", "HEAD")

    def make_completed_task(self, task_id: str = "speed-20260804") -> tuple[Path, dict]:
        branch = f"codex/task-{task_id}"
        _git(self.repo, "switch", "-c", branch, "origin/jk_standard_rl")
        source_commit = self.commit_file("src/runtime.py", "FAST = True\n", "optimize runtime")
        source_tree = _git(self.repo, "rev-parse", "HEAD^{tree}")
        payload = {
            "schema_version": 1,
            "record_type": "task_handoff",
            "task_id": task_id,
            "branch": branch,
            "branch_role": "task",
            "base_commit": self.base_commit,
            "base_tree": self.base_tree,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "status": "completed",
            "changed_scopes": ["src/"],
            "verification": [
                {"command": "python3 -m unittest", "outcome": "passed", "evidence": "server:test"}
            ],
            "server_evidence": ["experiments/server_command_runs/protocol-test/"],
            "supersedes": [],
            "aggregate_eligible": True,
            "deployment_eligible": False,
            "started_at": NOW,
            "completed_at": NOW,
        }
        path = self.repo / "agent_handoffs" / "tasks" / f"{task_id}.json"
        _write_json(path, payload)
        _git(self.repo, "add", str(path.relative_to(self.repo)))
        _git(self.repo, "commit", "-m", "handoff task")
        _git(self.repo, "push", "-u", "origin", branch)
        return path, payload

    def make_aggregate(self, task_path: Path, task_payload: dict) -> tuple[Path, dict]:
        task_tip = _git(self.repo, "rev-parse", "HEAD")
        branch = "codex/aggregate-20260804-test"
        _git(self.repo, "switch", "-c", branch, "origin/jk_standard_rl")
        _git(self.repo, "merge", "--ff-only", task_tip)
        source_commit = _git(self.repo, "rev-parse", "HEAD")
        source_tree = _git(self.repo, "rev-parse", "HEAD^{tree}")
        heads = {
            "jk_standard_rl": self.base_commit,
            task_payload["branch"]: task_tip,
        }
        payload = {
            "schema_version": 1,
            "record_type": "aggregate_manifest",
            "aggregate_id": "20260804-test",
            "canonical_branch": "jk_standard_rl",
            "aggregate_branch": branch,
            "base_commit": self.base_commit,
            "base_tree": self.base_tree,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "remote": "origin",
            "snapshot_at": NOW,
            "created_at": NOW,
            "heads": [
                {
                    "branch": "jk_standard_rl",
                    "commit": self.base_commit,
                    "role": "canonical",
                    "disposition": "already_ancestor",
                    "reason": "aggregate base",
                    "handoff": None,
                },
                {
                    "branch": task_payload["branch"],
                    "commit": task_tip,
                    "role": "task",
                    "disposition": "included",
                    "reason": "completed handoff integrated",
                    "handoff": str(task_path.relative_to(self.repo)),
                },
            ],
            "verification": [{"command": "python3 -m unittest", "outcome": "passed"}],
            "server_evidence": ["experiments/server_command_runs/protocol-test/"],
            "canonical_eligible": True,
            "deployment_eligible": True,
        }
        manifest = self.repo / "agent_handoffs" / "aggregates" / "20260804-test.json"
        _write_json(manifest, payload)
        _git(self.repo, "add", str(manifest.relative_to(self.repo)))
        _git(self.repo, "commit", "-m", "record aggregate manifest")
        _git(self.repo, "push", "-u", "origin", branch)
        return manifest, payload


class BranchRoleTest(unittest.TestCase):
    def test_branch_roles_are_explicit(self):
        guard = _load_guard_module()

        cases = {
            "jk_standard_rl": "canonical",
            "codex/task-speed-20260804": "task",
            "codex/aggregate-20260804": "aggregate",
            "codex/result-run-20260804": "result",
            "codex/archive-old": "archive",
            "codex/recovery-dirty": "recovery",
            "codex/experiment-ab": "experiment",
            "codex/old-unclassified": "legacy",
            "feature/external": "legacy",
        }
        for branch, expected in cases.items():
            with self.subTest(branch=branch):
                self.assertEqual(guard.branch_role(branch), expected)


class RepositoryStateTest(unittest.TestCase):
    def test_tracked_dirty_ignores_ignored_runtime_files(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            (fixture.repo / ".gitignore").write_text("runtime/\n", encoding="utf-8")
            _git(fixture.repo, "add", ".gitignore")
            _git(fixture.repo, "commit", "-m", "ignore runtime")
            (fixture.repo / "runtime").mkdir()
            (fixture.repo / "runtime" / "probe.log").write_text("ignored\n", encoding="utf-8")

            self.assertEqual(guard.tracked_dirty_paths(fixture.repo), [])

            (fixture.repo / "README.md").write_text("dirty\n", encoding="utf-8")
            dirty = guard.tracked_dirty_paths(fixture.repo)

        self.assertEqual(dirty, ["README.md"])


class TaskHandoffTest(unittest.TestCase):
    def test_completed_handoff_validates_tip_parent_tree_scope_and_remote(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            handoff, payload = fixture.make_completed_task()
            result = guard.validate_task_handoff(
                fixture.repo,
                handoff,
                remote="origin",
                require_remote=True,
            )

        self.assertEqual(result["task_id"], payload["task_id"])
        self.assertEqual(result["status"], "completed")

    def test_handoff_rejects_wrong_branch_source_tree_or_tip_content(self):
        guard = _load_guard_module()

        mutations = {
            "branch": "codex/task-other-20260804",
            "source_tree": "f" * 40,
            "source_commit": "e" * 40,
        }
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                fixture = GitFixture(Path(td))
                handoff, payload = fixture.make_completed_task()
                payload[field] = value
                _write_json(handoff, payload)
                with self.assertRaises(guard.GuardError):
                    guard.validate_task_handoff(fixture.repo, handoff)

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            handoff, _payload = fixture.make_completed_task()
            (fixture.repo / "unexpected.txt").write_text("not handoff only\n", encoding="utf-8")
            _git(fixture.repo, "add", "unexpected.txt")
            _git(fixture.repo, "commit", "--amend", "--no-edit")
            with self.assertRaises(guard.GuardError):
                guard.validate_task_handoff(fixture.repo, handoff)

    def test_handoff_rejects_source_changes_outside_declared_scope(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            handoff, payload = fixture.make_completed_task()
            payload["changed_scopes"] = ["docs/"]
            _write_json(handoff, payload)
            _git(fixture.repo, "add", str(handoff.relative_to(fixture.repo)))
            _git(fixture.repo, "commit", "--amend", "--no-edit")

            with self.assertRaisesRegex(guard.GuardError, "scope"):
                guard.validate_task_handoff(fixture.repo, handoff)


class AggregateManifestTest(unittest.TestCase):
    def test_manifest_requires_every_snapshot_head_and_no_needs_review(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            handoff, task_payload = fixture.make_completed_task()
            manifest, payload = fixture.make_aggregate(handoff, task_payload)
            current_heads = {
                "jk_standard_rl": fixture.base_commit,
                task_payload["branch"]: payload["heads"][1]["commit"],
                "codex/archive-old": "a" * 40,
            }

            with self.assertRaisesRegex(guard.GuardError, "missing"):
                guard.validate_aggregate_payload(payload, current_heads=current_heads)

            payload["heads"].append(
                {
                    "branch": "codex/archive-old",
                    "commit": "a" * 40,
                    "role": "archive",
                    "disposition": "needs_review",
                    "reason": "unreviewed",
                    "handoff": None,
                }
            )
            with self.assertRaisesRegex(guard.GuardError, "needs_review"):
                guard.validate_aggregate_payload(payload, current_heads=current_heads)

            payload["heads"][-1]["disposition"] = "archive_only"
            guard.validate_aggregate_payload(payload, current_heads=current_heads)

            payload["heads"][-1]["disposition"] = "included"
            with self.assertRaisesRegex(guard.GuardError, "archive"):
                guard.validate_aggregate_payload(payload, current_heads=current_heads)

            self.assertTrue(manifest.exists())

    def test_manifest_rejects_stale_snapshot_and_invalid_included_handoff(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            handoff, task_payload = fixture.make_completed_task()
            manifest, payload = fixture.make_aggregate(handoff, task_payload)
            task_branch = task_payload["branch"]
            current_heads = {entry["branch"]: entry["commit"] for entry in payload["heads"]}
            current_heads[task_branch] = "b" * 40

            with self.assertRaisesRegex(guard.GuardError, "advanced|changed|stale"):
                guard.validate_aggregate_payload(payload, current_heads=current_heads)

            payload["heads"][1]["handoff"] = "agent_handoffs/tasks/missing.json"
            _write_json(manifest, payload)
            with self.assertRaises(guard.GuardError):
                guard.validate_aggregate_manifest(fixture.repo, manifest)

    def test_manifest_only_tip_and_remote_parity_are_enforced(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            handoff, task_payload = fixture.make_completed_task()
            manifest, _payload = fixture.make_aggregate(handoff, task_payload)

            result = guard.validate_aggregate_manifest(
                fixture.repo,
                manifest,
                remote="origin",
                require_remote=True,
            )
            self.assertEqual(result["record_type"], "aggregate_manifest")

            (fixture.repo / "unexpected.txt").write_text("not manifest only\n", encoding="utf-8")
            _git(fixture.repo, "add", "unexpected.txt")
            _git(fixture.repo, "commit", "--amend", "--no-edit")
            with self.assertRaises(guard.GuardError):
                guard.validate_aggregate_manifest(fixture.repo, manifest)


class PushPolicyTest(unittest.TestCase):
    def test_push_record_parser_is_strict(self):
        guard = _load_guard_module()
        line = f"refs/heads/a {'a' * 40} refs/heads/a {ZERO_SHA}\n"

        updates = guard.parse_push_updates(line)

        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].remote_sha, ZERO_SHA)
        with self.assertRaises(guard.GuardError):
            guard.parse_push_updates("three columns only\n")

    def test_ordinary_canonical_push_fails_before_manifest_validation(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            update = guard.PushUpdate(
                local_ref="refs/heads/jk_standard_rl",
                local_sha=fixture.base_commit,
                remote_ref="refs/heads/jk_standard_rl",
                remote_sha=fixture.base_commit,
            )
            with self.assertRaisesRegex(guard.GuardError, "authorized"):
                guard.validate_pre_push(fixture.repo, [update], remote="origin", env={})

    def test_authorized_manifest_backed_fast_forward_passes(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            handoff, task_payload = fixture.make_completed_task()
            manifest, _payload = fixture.make_aggregate(handoff, task_payload)
            aggregate_tip = _git(fixture.repo, "rev-parse", "HEAD")
            update = guard.PushUpdate(
                local_ref="refs/heads/codex/aggregate-20260804-test",
                local_sha=aggregate_tip,
                remote_ref="refs/heads/jk_standard_rl",
                remote_sha=fixture.base_commit,
            )
            env = {
                "RFR_AGGREGATOR_AUTHORIZED": "1",
                "RFR_AGGREGATE_MANIFEST": str(manifest.relative_to(fixture.repo)),
            }

            result = guard.validate_pre_push(fixture.repo, [update], remote="origin", env=env)

        self.assertEqual(result[0]["role"], "canonical")

    def test_task_source_push_is_allowed_before_handoff_tip(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            branch = "codex/task-source-20260804"
            _git(fixture.repo, "switch", "-c", branch, "origin/jk_standard_rl")
            source_commit = fixture.commit_file("src/source.py", "VALUE = 1\n", "source work")
            update = guard.PushUpdate(
                local_ref=f"refs/heads/{branch}",
                local_sha=source_commit,
                remote_ref=f"refs/heads/{branch}",
                remote_sha=ZERO_SHA,
            )

            result = guard.validate_pre_push(fixture.repo, [update], remote="origin", env={})

        self.assertEqual(result, [{"branch": branch, "role": "task", "action": "update"}])

    def test_archive_branch_cannot_claim_deployment_eligibility(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            branch = "codex/archive-old"
            _git(fixture.repo, "switch", "-c", branch, "origin/jk_standard_rl")
            claim = fixture.repo / "agent_handoffs" / "archive-claim.json"
            _write_json(claim, {"deployment_eligible": True})
            _git(fixture.repo, "add", str(claim.relative_to(fixture.repo)))
            _git(fixture.repo, "commit", "-m", "invalid archive claim")
            tip = _git(fixture.repo, "rev-parse", "HEAD")
            update = guard.PushUpdate(
                local_ref=f"refs/heads/{branch}",
                local_sha=tip,
                remote_ref=f"refs/heads/{branch}",
                remote_sha=ZERO_SHA,
            )

            with self.assertRaisesRegex(guard.GuardError, "cannot claim"):
                guard.validate_pre_push(fixture.repo, [update], remote="origin", env={})


class HookIntegrationTest(unittest.TestCase):
    def test_real_hook_allows_task_push_and_blocks_ordinary_canonical_push(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            scripts = fixture.repo / "scripts"
            hooks = fixture.repo / ".githooks"
            scripts.mkdir()
            hooks.mkdir()
            shutil.copy2(GUARD_PATH, scripts / "repo_sync_guard.py")
            shutil.copy2(REPO_ROOT / ".githooks" / "pre-push", hooks / "pre-push")
            os.chmod(hooks / "pre-push", 0o755)
            _git(fixture.repo, "config", "core.hooksPath", ".githooks")

            branch = "codex/task-hook-20260804"
            _git(fixture.repo, "switch", "-c", branch, "origin/jk_standard_rl")
            fixture.commit_file("src/hook.py", "VALUE = 1\n", "task source")
            task_push = subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=fixture.repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(task_push.returncode, 0, task_push.stderr)

            _git(fixture.repo, "switch", "jk_standard_rl")
            fixture.commit_file("README.md", "unauthorized\n", "unauthorized canonical")
            canonical_push = subprocess.run(
                ["git", "push", "origin", "jk_standard_rl"],
                cwd=fixture.repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(canonical_push.returncode, 0)
        self.assertIn("aggregator-authorized", canonical_push.stderr)


class ServerAndResultPolicyTest(unittest.TestCase):
    def test_server_state_requires_exact_remote_canonical_commit_tree_and_cleanliness(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            result = guard.check_server_state(
                fixture.repo,
                expected_commit=fixture.base_commit,
                expected_tree=fixture.base_tree,
                remote="origin",
                canonical="jk_standard_rl",
                sync=False,
            )
            self.assertEqual(result["commit"], fixture.base_commit)

            with self.assertRaises(guard.GuardError):
                guard.check_server_state(
                    fixture.repo,
                    expected_commit="c" * 40,
                    expected_tree=fixture.base_tree,
                    remote="origin",
                    canonical="jk_standard_rl",
                    sync=False,
                )

            (fixture.repo / "README.md").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(guard.GuardError, "dirty"):
                guard.check_server_state(
                    fixture.repo,
                    expected_commit=fixture.base_commit,
                    expected_tree=fixture.base_tree,
                    remote="origin",
                    canonical="jk_standard_rl",
                    sync=False,
                )

    def test_result_branch_allows_only_explicit_artifact_scopes(self):
        guard = _load_guard_module()

        with tempfile.TemporaryDirectory() as td:
            fixture = GitFixture(Path(td))
            _git(fixture.repo, "switch", "-c", "codex/result-run-20260804")
            fixture.commit_file(
                "experiments/server_command_runs/run/summary.json",
                "{}\n",
                "record server result",
            )
            _git(fixture.repo, "push", "-u", "origin", "codex/result-run-20260804")

            result = guard.check_result_branch(
                fixture.repo,
                base_commit=fixture.base_commit,
                allowed_prefixes=["experiments/server_command_runs/"],
                remote="origin",
                require_remote=True,
            )
            self.assertEqual(result["changed_paths"], ["experiments/server_command_runs/run/summary.json"])

            fixture.commit_file("scripts/forbidden.py", "BAD = True\n", "bad source edit")
            with self.assertRaisesRegex(guard.GuardError, "allowed"):
                guard.check_result_branch(
                    fixture.repo,
                    base_commit=fixture.base_commit,
                    allowed_prefixes=["experiments/server_command_runs/"],
                    remote="origin",
                    require_remote=False,
                )


class InstructionPointerTest(unittest.TestCase):
    def test_agent_instruction_files_will_point_to_protocol_and_guards(self):
        for relative in ("AGENTS.md", "CLAUDE.md"):
            with self.subTest(relative=relative):
                text = (REPO_ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("docs/GIT_MULTI_AGENT_PROTOCOL.md", text)
                self.assertIn("repo_sync_guard.py", text)


if __name__ == "__main__":
    unittest.main()
