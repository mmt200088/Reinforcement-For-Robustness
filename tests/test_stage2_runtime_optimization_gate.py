import contextlib
import copy
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


_REPO = pathlib.Path(__file__).resolve().parents[1]
_GATE = _REPO / "scripts" / "stage2_runtime_optimization_gate.sh"
_COMPARATOR = _REPO / "scripts" / "stage2_ngpu_ab_compare.py"

_compare_spec = importlib.util.spec_from_file_location(
    "stage2_runtime_gate_ngpu_compare",
    str(_COMPARATOR),
)
compare_mod = importlib.util.module_from_spec(_compare_spec)
sys.modules[_compare_spec.name] = compare_mod
_compare_spec.loader.exec_module(compare_mod)


RUNTIME_TELEMETRY = {
    "candidate_bytes_written": 321,
    "process_rss_bytes": 10_000,
    "process_peak_rss_bytes": 20_000,
    "pool_id": "optimized-pool",
    "batch_set_key": "F1",
    "batch_count": 4,
    "process_count": 4,
    "worker_intraop_threads": 1,
    "worker_interop_threads": 1,
}


def _write_executable(path, source):
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def _run_checked(command, *, cwd=None):
    return subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        text=True,
        capture_output=True,
        check=True,
    )


def _init_clean_root(path):
    path.mkdir(parents=True)
    (path / "README.txt").write_text("fixture\n", encoding="utf-8")
    scripts = path / "scripts"
    scripts.mkdir()
    shutil.copy2(_COMPARATOR, scripts / _COMPARATOR.name)
    shutil.copy2(_REPO / "jsonl_utils.py", path / "jsonl_utils.py")
    shutil.copy2(_REPO / "jsonl_utils.py", scripts / "jsonl_utils.py")
    _run_checked(["git", "init", "-q"], cwd=path)
    _run_checked(["git", "add", "."], cwd=path)
    _run_checked(
        [
            "git",
            "-c",
            "user.name=Runtime Gate Test",
            "-c",
            "user.email=runtime-gate@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=path,
    )


def _git_status(path):
    return _run_checked(
        ["git", "status", "--porcelain"],
        cwd=path,
    ).stdout.strip()


def _make_fake_nvidia_smi(path):
    _write_executable(
        path,
        r'''
        #!/usr/bin/env bash
        set -eu

        printf '%s\n' "$*" >> "${FAKE_NVIDIA_LOG:?}"
        if [[ "$*" == *"query-compute-apps"* ]]; then
          scoped=0
          if [[ " $* " == *" -i "* ]] || [[ "$*" == *"--id="* ]] || \
             [[ " $* " == *" --id "* ]]; then
            scoped=1
          fi
          if [ -n "${FAKE_COMPUTE_PIDS:-}" ] && \
             { [ "$scoped" -eq 0 ] || [[ "$*" == *"4"* ]]; }; then
            printf '%s\n' "$FAKE_COMPUTE_PIDS"
          fi
          exit 0
        fi

        if [[ "$*" == *"query-gpu=index,memory.used"* ]]; then
          for gpu in 0 1 2 3 4; do
            printf '%s, 0\n' "$gpu"
          done
          exit 0
        fi

        for gpu in 0 1 2 3 4; do
          printf '2026-07-18 00:00:00, %s, Fake GPU, 0, 0\n' "$gpu"
        done
        ''',
    )


def _make_fake_case_launcher(path):
    # Test seam: CASE, SOURCE_ROOT, BATCH_SIZE, EPISODES, REWARD_DEVICES,
    # CASE_ARTIFACT_DIR. Production launches may use any command behind it.
    _write_executable(
        path,
        r'''
        #!/usr/bin/env bash
        set -euo pipefail

        if [ "$#" -ne 6 ]; then
          echo "fake launcher expected 6 arguments, got $#" >&2
          exit 64
        fi
        case_name="$1"
        source_root="$2"
        batch_size="$3"
        episodes="$4"
        reward_devices="$5"
        case_dir="$6"

        printf '%s|%s|%s|%s|%s|%s\n' \
          "$case_name" "$source_root" "$batch_size" "$episodes" \
          "$reward_devices" "$case_dir" >> "${FAKE_LAUNCH_LOG:?}"

        mkdir -p "$case_dir/diagnostics"
        printf '%s\n' "$$" > "$case_dir/worker_pids.txt"

        telemetry=''
        if [ "$case_name" != "base64" ]; then
          telemetry=',"candidate_bytes_written":321,"process_rss_bytes":10000,"process_peak_rss_bytes":20000,"pool_id":"optimized-pool","batch_set_key":"F1","batch_count":4,"process_count":4,"worker_intraop_threads":1,"worker_interop_threads":1'
        fi

        : > "$case_dir/diagnostics/episodes.jsonl"
        episode=1
        while [ "$episode" -le "$episodes" ]; do
          reward='10.0'
          if [ "${FAKE_SEMANTIC_MISMATCH_CASE:-}" = "$case_name" ] && \
             [ "$episode" -eq 1 ]; then
            reward='99.0'
          fi
          printf '{"episode":%s,"timestamp":%s,"total_reward":%s,"terminal_reward":8.0,"terminal_priority":3,"action_indices":[1,2],"action_hash":"action-1","terminal_loss_mean":0.1,"terminal_metric1_mean":0.9,"terminal_metric2_mean":0.8,"terminal_probe_devices":["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4"],"terminal_probe_trial_counts":[1,1,1,1,1],"terminal_probe_trial_indices":[[0],[1],[2],[3],[4]],"terminal_probe_trial_seeds":[101,102,103,104,105],"terminal_pareto_event_kind":"expanded"%s}\n' \
            "$episode" "$episode" "$reward" "$telemetry" \
            >> "$case_dir/diagnostics/episodes.jsonl"
          episode=$((episode + 1))
        done

        : > "$case_dir/diagnostics/ppo_updates.jsonl"
        updates=$((episodes / 120))
        if [ "$updates" -lt 1 ]; then
          updates=1
        fi
        update=1
        while [ "$update" -le "$updates" ]; do
          printf '{"update":%s,"timestamp":%s,"elapsed_sec":%s,"completed_episodes":%s,"policy_loss":-0.01,"value_loss":1.25,"entropy":0.1,"clip_fraction":0.02,"approx_kl":0.001,"n_samples":120,"window_mean_return":10.0,"best_reward_so_far":10.0%s}\n' \
            "$update" "$update" "$update" "$((update * 120))" "$telemetry" \
            >> "$case_dir/diagnostics/ppo_updates.jsonl"
          update=$((update + 1))
        done

        context_hash='215d3b7bb8cc42e90514045ee959cbce4f046e0cb52ca5f5e9757450eed24940'
        candidate_loss='0.1'
        if [ "${FAKE_CANDIDATE_MISMATCH_CASE:-}" = "$case_name" ]; then
          candidate_loss='0.2'
        fi
        if [ "$case_name" = "base64" ]; then
          printf '%s\n' \
            '{"record_type":"candidate_trial_group_v1","candidate_key":"candidate-1","action_indices":[1,2],"raw_action_indices":[1,2],"effective_action_indices":[1,2],"identity_context":{"profile":"mrpc"},"fidelity":"F1","valid":true,"trial_group":{"loss":[0.1],"metric1":[0.9],"metric2":[0.8],"seeds":[101]},"trial_group_metadata":{"fidelity":"F1","identity_context":{"profile":"mrpc"}}}' \
            > "$case_dir/diagnostics/candidate_store.jsonl"
        else
          printf '%s\n' \
            "{\"record_type\":\"candidate_identity_context_v1\",\"identity_context_hash\":\"${context_hash}\",\"identity_context\":{\"profile\":\"mrpc\"}}" \
            "{\"record_type\":\"candidate_trial_group_v2\",\"candidate_key\":\"candidate-1\",\"action_indices\":[1,2],\"identity_context_hash\":\"${context_hash}\",\"fidelity\":\"F1\",\"valid\":true,\"trial_group\":{\"loss\":[${candidate_loss}],\"metric1\":[0.9],\"metric2\":[0.8],\"seeds\":[101]},\"trial_group_metadata\":{\"fidelity\":\"F1\"}}" \
            > "$case_dir/diagnostics/candidate_store.jsonl"
        fi

        printf '{"pool_id":"%s","batch_set_key":"F1","batch_count":4,"process_count":4,"worker_intraop_threads":1,"worker_interop_threads":1}\n' \
          "$case_name-pool" > "$case_dir/diagnostics/diagnostics_summary.json"
        printf 'fake launch %s\n' "$case_name" > "$case_dir/launch.log"
        case "$case_name" in
          base64) wall=40 ;;
          opt64) wall=30 ;;
          opt128) wall=1 ;;
          opt256) wall=20 ;;
          *) exit 65 ;;
        esac
        printf '%s\n' "$wall" > "$case_dir/wall_seconds.txt"
        sleep "${FAKE_LAUNCH_DELAY_SECONDS:-0.3}"
        ''',
    )


def _base_gate_env(root, baseline, optimized, artifact_dir):
    fake_bin = root / "bin"
    fake_bin.mkdir()
    nvidia_log = root / "nvidia-smi.log"
    launch_log = root / "launches.log"
    launcher = root / "fake-case-launcher.sh"
    _make_fake_nvidia_smi(fake_bin / "nvidia-smi")
    _make_fake_case_launcher(launcher)

    env = os.environ.copy()
    env.update({
        "BASELINE_ROOT": str(baseline),
        "OPTIMIZED_ROOT": str(optimized),
        "ARTIFACT_DIR": str(artifact_dir),
        "STAGE2_GATE_CASE_LAUNCHER": str(launcher),
        "FAKE_NVIDIA_LOG": str(nvidia_log),
        "FAKE_LAUNCH_LOG": str(launch_log),
        "GPU_SAMPLE_INTERVAL_SECONDS": "0.01",
        "GATE_POLL_INTERVAL_SECONDS": "0.01",
        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
    })
    return env, nvidia_log, launch_log


def _run_gate(env, *, timeout=40):
    return subprocess.run(
        ["bash", str(_GATE)],
        cwd=str(_REPO),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _combined_output(proc):
    return proc.stdout + proc.stderr


def _load_normalized_rows(path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def _normalized_candidate_file(case_dir):
    matches = sorted(
        path
        for path in case_dir.rglob("*")
        if path.is_file()
        and "candidate" in path.name.lower()
        and "normal" in path.name.lower()
    )
    return matches[0] if matches else None


def _episode_row():
    return {
        "episode": 1,
        "timestamp": 1.0,
        "total_reward": 10.0,
        "terminal_reward": 8.0,
        "terminal_priority": 3,
        "action_indices": [1, 2],
        "action_hash": "action-1",
        "terminal_loss_mean": 0.1,
        "terminal_metric1_mean": 0.9,
        "terminal_metric2_mean": 0.8,
        "terminal_probe_devices": ["cuda:0"],
        "terminal_probe_trial_counts": [5],
        "terminal_probe_trial_indices": [[0, 1, 2, 3, 4]],
        "terminal_probe_trial_seeds": [101, 102, 103, 104, 105],
        "terminal_pareto_event_kind": "expanded",
    }


def _ppo_row():
    return {
        "update": 1,
        "timestamp": 1.0,
        "elapsed_sec": 5.0,
        "completed_episodes": 120,
        "policy_loss": -0.01,
        "value_loss": 1.25,
        "entropy": 0.1,
        "clip_fraction": 0.02,
        "approx_kl": 0.001,
        "n_samples": 120,
        "window_mean_return": 10.0,
        "best_reward_so_far": 10.0,
    }


def _invoke_require_equal(one_episode, many_episode, one_ppo, many_ppo):
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        paths = {}
        for name, row in (
            ("one.jsonl", one_episode),
            ("many.jsonl", many_episode),
            ("one_ppo.jsonl", one_ppo),
            ("many_ppo.jsonl", many_ppo),
        ):
            paths[name] = root / name
            paths[name].write_text(
                json.dumps(row, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        argv = [
            "stage2_ngpu_ab_compare.py",
            "--one", str(paths["one.jsonl"]),
            "--many", str(paths["many.jsonl"]),
            "--one-ppo", str(paths["one_ppo.jsonl"]),
            "--many-ppo", str(paths["many_ppo.jsonl"]),
            "--require-equal",
        ]
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
            return compare_mod.main(), stdout.getvalue()


class Stage2RuntimeOptimizationGateStaticTests(unittest.TestCase):
    def _source(self):
        self.assertTrue(
            _GATE.is_file(),
            "RED: scripts/stage2_runtime_optimization_gate.sh does not exist",
        )
        return _GATE.read_text(encoding="utf-8")

    def test_shell_interface_requires_roots_and_artifact_dir_with_stable_defaults(self):
        source = self._source()

        for name in ("BASELINE_ROOT", "OPTIMIZED_ROOT", "ARTIFACT_DIR"):
            self.assertRegex(source, rf"\$\{{{name}:\?[^}}]+\}}")
        defaults = {
            "BATCH_SIZES": "64 128 256",
            "EPISODES": "600",
            "REWARD_DEVICES": "0,1,2,3,4",
        }
        for name, value in defaults.items():
            self.assertRegex(
                source,
                rf"{name}\s*=\s*['\"]\$\{{{name}:-{re.escape(value)}\}}['\"]",
            )

    def test_static_contract_names_preflight_cases_evidence_and_verdict_stages(self):
        source = self._source()
        lowered = source.lower()

        for token in (
            "--porcelain",
            "nvidia-smi",
            "query-compute-apps",
            "base64",
            "opt64",
            "opt128",
            "opt256",
            "diagnostics",
            "candidate_store.jsonl",
            "candidate_trial_group_v1",
            "candidate_trial_group_v2",
            "candidate_identity_context_v1",
            "verdict.json",
            "verdict.md",
            "stage2_ngpu_ab_compare.py",
            "--require-equal",
        ):
            self.assertIn(token, lowered)
        self.assertRegex(lowered, r"normaliz")
        self.assertRegex(lowered, r"worker.*(pid|process)|(pid|process).*worker")
        self.assertIn("thread", lowered)
        self.assertIn("semantic", lowered)
        self.assertIn("speed", lowered)


class Stage2RuntimeOptimizationGateBehaviorTests(unittest.TestCase):
    def _roots(self, root):
        baseline = root / "baseline"
        optimized = root / "optimized"
        _init_clean_root(baseline)
        _init_clean_root(optimized)
        return baseline, optimized

    def test_required_variables_fail_before_case_launcher(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            baseline, optimized = self._roots(root)
            for missing in ("BASELINE_ROOT", "OPTIMIZED_ROOT", "ARTIFACT_DIR"):
                with self.subTest(missing=missing):
                    run_root = root / f"run-{missing.lower()}"
                    run_root.mkdir()
                    env, _nvidia_log, launch_log = _base_gate_env(
                        run_root,
                        baseline,
                        optimized,
                        run_root / "artifacts",
                    )
                    env.pop(missing)

                    proc = _run_gate(env)
                    output = _combined_output(proc)

                    self.assertNotEqual(proc.returncode, 0, output)
                    self.assertIn(missing, output)
                    self.assertFalse(launch_log.exists(), output)

    def test_dirty_baseline_and_optimized_roots_are_rejected_before_training(self):
        for dirty_name in ("baseline", "optimized"):
            with self.subTest(dirty_name=dirty_name), tempfile.TemporaryDirectory() as td:
                root = pathlib.Path(td)
                baseline, optimized = self._roots(root)
                dirty_root = baseline if dirty_name == "baseline" else optimized
                (dirty_root / "README.txt").write_text(
                    "fixture\ndirty\n",
                    encoding="utf-8",
                )
                env, _nvidia_log, launch_log = _base_gate_env(
                    root,
                    baseline,
                    optimized,
                    root / "artifacts",
                )

                proc = _run_gate(env)
                output = _combined_output(proc)

                self.assertNotEqual(proc.returncode, 0, output)
                self.assertTrue(
                    "dirty" in output.lower() or "not clean" in output.lower(),
                    output,
                )
                self.assertIn(dirty_name, output.lower())
                self.assertFalse(launch_log.exists(), output)

    def test_nonempty_compute_pid_fails_idle_gpu_preflight_before_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            baseline, optimized = self._roots(root)
            env, nvidia_log, launch_log = _base_gate_env(
                root,
                baseline,
                optimized,
                root / "artifacts",
            )
            env["FAKE_COMPUTE_PIDS"] = "4321"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            self.assertIn("4321", output)
            self.assertFalse(launch_log.exists(), output)
            query_log = nvidia_log.read_text(encoding="utf-8")
            self.assertIn("query-compute-apps", query_log)
            self.assertIn("pid", query_log)

    def test_fake_gate_runs_default_cases_from_clean_roots_and_collects_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            baseline, optimized = self._roots(root)
            artifact_dir = root / "artifacts"
            env, nvidia_log, launch_log = _base_gate_env(
                root,
                baseline,
                optimized,
                artifact_dir,
            )

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertEqual(proc.returncode, 0, output)
            launches = [
                line.split("|")
                for line in launch_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [parts[:5] for parts in launches],
                [
                    ["base64", str(baseline), "64", "600", "0,1,2,3,4"],
                    ["opt64", str(optimized), "64", "600", "0,1,2,3,4"],
                    ["opt128", str(optimized), "128", "600", "0,1,2,3,4"],
                    ["opt256", str(optimized), "256", "600", "0,1,2,3,4"],
                ],
            )
            self.assertEqual(_git_status(baseline), "")
            self.assertEqual(_git_status(optimized), "")
            self.assertTrue((artifact_dir / "verdict.json").is_file())
            self.assertTrue((artifact_dir / "verdict.md").is_file())
            verdict_json = (artifact_dir / "verdict.json").read_text(
                encoding="utf-8",
            ).lower()
            verdict_md = (artifact_dir / "verdict.md").read_text(
                encoding="utf-8",
            ).lower()
            for token in ("base64", "opt64", "opt128", "opt256", "semantic", "speed"):
                self.assertIn(token, verdict_json)
            self.assertTrue(
                any(
                    "opt128" in line and ("winner" in line or "fastest" in line)
                    for line in verdict_md.splitlines()
                ),
                verdict_md,
            )

            normalized = {}
            for parts in launches:
                case_name = parts[0]
                case_dir = pathlib.Path(parts[5])
                self.assertTrue(
                    case_dir.is_relative_to(artifact_dir),
                    f"{case_name} artifacts escaped ARTIFACT_DIR: {case_dir}",
                )
                self.assertTrue((case_dir / "diagnostics" / "episodes.jsonl").is_file())
                self.assertTrue((case_dir / "diagnostics" / "ppo_updates.jsonl").is_file())
                self.assertTrue((case_dir / "diagnostics" / "candidate_store.jsonl").is_file())
                gpu_samples = [
                    path
                    for path in case_dir.rglob("*")
                    if path.is_file()
                    and ("nvidia" in path.name.lower() or "gpu" in path.name.lower())
                    and ("sample" in path.name.lower() or path.suffix == ".csv")
                ]
                worker_inventory = [
                    path
                    for path in case_dir.rglob("*")
                    if path.is_file()
                    and "worker" in path.name.lower()
                    and ("inventory" in path.name.lower() or "thread" in path.name.lower())
                ]
                self.assertTrue(gpu_samples, f"{case_name} lacks nvidia-smi samples")
                self.assertTrue(
                    worker_inventory,
                    f"{case_name} lacks worker PID/thread inventory",
                )
                gpu_sample_text = "\n".join(
                    path.read_text(encoding="utf-8", errors="replace")
                    for path in gpu_samples
                )
                for gpu in ("0", "1", "2", "3", "4"):
                    self.assertRegex(
                        gpu_sample_text,
                        rf"(^|[,\s]){gpu}([,\s]|$)",
                        f"{case_name} lacks a sample for GPU {gpu}",
                    )
                worker_inventory_text = "\n".join(
                    path.read_text(encoding="utf-8", errors="replace")
                    for path in worker_inventory
                )
                self.assertRegex(worker_inventory_text, r"\b[0-9]+\b")
                self.assertRegex(
                    worker_inventory_text.lower(),
                    r"thread|tid|lwp|nlwp",
                )
                normalized_path = _normalized_candidate_file(case_dir)
                self.assertIsNotNone(
                    normalized_path,
                    f"{case_name} lacks normalized candidate evidence",
                )
                normalized[case_name] = _load_normalized_rows(normalized_path)

            for case_name in ("opt64", "opt128", "opt256"):
                self.assertEqual(normalized[case_name], normalized["base64"])
            nvidia_calls = nvidia_log.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(nvidia_calls), 5)

    def test_semantic_mismatch_exits_nonzero_and_excludes_faster_case_from_ranking(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            baseline, optimized = self._roots(root)
            artifact_dir = root / "artifacts"
            env, _nvidia_log, launch_log = _base_gate_env(
                root,
                baseline,
                optimized,
                artifact_dir,
            )
            env["FAKE_SEMANTIC_MISMATCH_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            self.assertEqual(
                [
                    line.split("|", 1)[0]
                    for line in launch_log.read_text(encoding="utf-8").splitlines()
                ],
                ["base64", "opt64", "opt128", "opt256"],
            )
            self.assertTrue((artifact_dir / "verdict.json").is_file())
            verdict_md = (artifact_dir / "verdict.md").read_text(encoding="utf-8")
            verdict_lines = verdict_md.lower().splitlines()
            self.assertTrue(
                any(
                    "opt128" in line and ("fail" in line or "mismatch" in line)
                    for line in verdict_lines
                ),
                verdict_md,
            )
            winner_lines = [
                line
                for line in verdict_lines
                if "winner" in line or "fastest" in line
            ]
            self.assertTrue(winner_lines, verdict_md)
            self.assertTrue(any("opt256" in line for line in winner_lines), verdict_md)
            self.assertFalse(any("opt128" in line for line in winner_lines), verdict_md)

    def test_candidate_v1_v2_normalization_detects_logical_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            baseline, optimized = self._roots(root)
            artifact_dir = root / "artifacts"
            env, _nvidia_log, _launch_log = _base_gate_env(
                root,
                baseline,
                optimized,
                artifact_dir,
            )
            env["FAKE_CANDIDATE_MISMATCH_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            self.assertTrue((artifact_dir / "verdict.json").is_file())
            verdict_md = (artifact_dir / "verdict.md").read_text(encoding="utf-8")
            lowered = verdict_md.lower()
            self.assertIn("opt128", lowered)
            self.assertIn("candidate", lowered)
            self.assertTrue("fail" in lowered or "mismatch" in lowered)
            winner_lines = [
                line
                for line in lowered.splitlines()
                if "winner" in line or "fastest" in line
            ]
            self.assertTrue(any("opt256" in line for line in winner_lines), verdict_md)
            self.assertFalse(any("opt128" in line for line in winner_lines), verdict_md)


class Stage2NgpuComparatorRuntimeTelemetryTests(unittest.TestCase):
    def test_require_equal_ignores_runtime_telemetry_key_presence_and_values(self):
        one_episode = _episode_row()
        many_episode = {**_episode_row(), **RUNTIME_TELEMETRY}
        one_ppo = _ppo_row()
        many_ppo = {**_ppo_row(), **RUNTIME_TELEMETRY}

        rc, report = _invoke_require_equal(
            one_episode,
            many_episode,
            one_ppo,
            many_ppo,
        )

        self.assertEqual(rc, 0, report)
        self.assertIn("quality/effect equality: PASS", report)
        self.assertIn("PPO update equality: PASS", report)

    def test_require_equal_rejects_reward_action_trial_and_frontier_drift(self):
        cases = (
            ("reward", "total_reward", 99.0),
            ("action", "action_indices", [2, 1]),
            ("trial", "terminal_probe_trial_indices", [[4, 3, 2, 1, 0]]),
            ("frontier", "terminal_pareto_event_kind", "dominated"),
        )
        for label, field, changed_value in cases:
            with self.subTest(label=label):
                many_episode = copy.deepcopy(_episode_row())
                many_episode[field] = changed_value

                rc, report = _invoke_require_equal(
                    _episode_row(),
                    many_episode,
                    _ppo_row(),
                    _ppo_row(),
                )

                self.assertNotEqual(rc, 0, report)
                self.assertIn(field, report)
                self.assertIn("[FATAL] equality requirement failed", report)

    def test_require_equal_rejects_ppo_scientific_drift(self):
        many_ppo = _ppo_row()
        many_ppo["entropy"] = 0.2

        rc, report = _invoke_require_equal(
            _episode_row(),
            _episode_row(),
            _ppo_row(),
            many_ppo,
        )

        self.assertNotEqual(rc, 0, report)
        self.assertIn("ppo_update[1].entropy", report)
        self.assertIn("[FATAL] equality requirement failed", report)


if __name__ == "__main__":
    unittest.main()
