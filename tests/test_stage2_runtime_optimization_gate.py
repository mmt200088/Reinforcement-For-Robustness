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
import time
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


def _make_fake_detached_llama(path):
    _write_executable(
        path,
        r'''
        #!/usr/bin/env bash
        set -euo pipefail

        if [ "${FAKE_REQUIRE_ORIGINAL_LAUNCHER_PATH:-0}" = "1" ]; then
          launcher_root="$(cd "$(dirname "$0")" && pwd -P)"
          [ -f "$launcher_root/presets/mrpc-blb-stage2-rl.conf" ] || {
            printf 'preset lookup failed relative to launcher path: %s\n' "$0" >&2
            exit 69
          }
        fi

        persistent_root=''
        while [ "$#" -gt 0 ]; do
          if [ "$1" = "--persistent-root" ]; then
            persistent_root="$2"
            shift 2
          else
            shift
          fi
        done
        [ -n "$persistent_root" ] || exit 64

        case_dir="$(dirname "$persistent_root")"
        case_name="$(basename "$case_dir")"
        mkdir -p "$persistent_root/rl/bert-base/mrpc"
        "${FAKE_DEFAULT_PAYLOAD_WRITER:?}" \
          "$case_name" "$PWD" 64 \
          "${FAKE_DEFAULT_EPISODES:-600}" \
          "${FAKE_DEFAULT_REWARD_DEVICES:-0,1,2,3,4}" \
          "$case_dir" &
        JOB_PID=$!
        if [ "${FAKE_PREWRITE_TRAIN_EXIT_CASE:-}" = "$case_name" ]; then
          printf '0\n' > "${STAGE2_GATE_TRAIN_EXIT_FILE:?}"
        fi
        printf '%s\n' "$JOB_PID" \
          > "$persistent_root/rl/bert-base/mrpc/LATEST_PID"
        printf '%s\n' "$JOB_PID" > "$persistent_root/rl.pid"
        printf 'fake detached training pid=%s\n' "$JOB_PID"
        ''',
    )


def _init_clean_root(path):
    path.mkdir(parents=True)
    (path / "README.txt").write_text("fixture\n", encoding="utf-8")
    presets = path / "presets"
    presets.mkdir()
    (presets / "mrpc-blb-stage2-rl.conf").write_text(
        "# fixture preset\n", encoding="utf-8",
    )
    scripts = path / "scripts"
    scripts.mkdir()
    shutil.copy2(_COMPARATOR, scripts / _COMPARATOR.name)
    shutil.copy2(_REPO / "jsonl_utils.py", path / "jsonl_utils.py")
    shutil.copy2(_REPO / "jsonl_utils.py", scripts / "jsonl_utils.py")
    _make_fake_detached_llama(path / "llama_7B_LayerImportance.sh")
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
          if [ -n "${FAKE_RUNTIME_COMPUTE_PID_CASE:-}" ] && \
             [ -s "${FAKE_CASE_RUNNING_FILE:-}" ] && \
             [ "${FAKE_RUNTIME_COMPUTE_PID_CASE:-}" = \
               "$(cat "$FAKE_CASE_RUNNING_FILE")" ]; then
            printf '%s, GPU-fake, 10\n' \
              "${FAKE_RUNTIME_COMPUTE_PID:-999999}"
          fi
          current_case=''
          if [ -s "${FAKE_CASE_RUNNING_FILE:-}" ]; then
            current_case="$(cat "$FAKE_CASE_RUNNING_FILE")"
          fi
          if [ -s "${FAKE_OWNED_COMPUTE_PIDS_FILE:-}" ] && \
             [ "${FAKE_SUPPRESS_OWNED_COMPUTE_CASE:-}" != "$current_case" ]; then
            gpu_index=0
            while IFS= read -r owned_pid; do
              [ -n "$owned_pid" ] || continue
              printf '%s, GPU-fake-%s, 10\n' "$owned_pid" "$gpu_index"
              gpu_index=$((gpu_index + 1))
            done < "$FAKE_OWNED_COMPUTE_PIDS_FILE"
          fi
          exit 0
        fi

        if [[ "$*" == *"query-gpu=index,memory.used"* ]]; then
          for gpu in 0 1 2 3 4; do
            printf '%s, 0\n' "$gpu"
          done
          exit 0
        fi

        sample_call=1
        if [ -n "${FAKE_NVIDIA_SAMPLE_COUNT_FILE:-}" ]; then
          sample_count_file="$FAKE_NVIDIA_SAMPLE_COUNT_FILE"
          if [ -s "${FAKE_CURRENT_CASE_FILE:-}" ]; then
            sample_count_file="${sample_count_file}.$(cat "$FAKE_CURRENT_CASE_FILE")"
          fi
          previous=0
          if [ -f "$sample_count_file" ]; then
            previous="$(cat "$sample_count_file")"
          fi
          sample_call=$((previous + 1))
          printf '%s\n' "$sample_call" > "$sample_count_file"
        fi
        for gpu in 0 1 2 3 4; do
          memory=1024
          utilization=50
          if [ "${FAKE_INACTIVE_GPU:-}" = "$gpu" ]; then
            memory=0
            utilization=0
          fi
          if [ "${FAKE_TRANSIENT_GPU:-}" = "$gpu" ] \
              && [ "$sample_call" -gt 1 ]; then
            memory=0
            utilization=0
          fi
          printf '2026-07-18 00:00:00, %s, Fake GPU, %s, %s\n' \
            "$gpu" "$memory" "$utilization"
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

        printf '%s\n' "$case_name" > "${FAKE_CURRENT_CASE_FILE:?}"
        if [ -n "${FAKE_CASE_RUNNING_FILE:-}" ]; then
          printf '%s\n' "$case_name" > "$FAKE_CASE_RUNNING_FILE"
        fi
        cleanup_fake_case() {
          for worker_pid in ${worker_pids:-}; do
            kill "$worker_pid" 2>/dev/null || true
            wait "$worker_pid" 2>/dev/null || true
          done
          rm -f "${FAKE_OWNED_COMPUTE_PIDS_FILE:-}"
          if [ -n "${FAKE_CASE_RUNNING_FILE:-}" ]; then
            rm -f "$FAKE_CASE_RUNNING_FILE"
          fi
        }
        trap cleanup_fake_case EXIT
        printf '%s|%s|%s|%s|%s|%s\n' \
          "$case_name" "$source_root" "$batch_size" "$episodes" \
          "$reward_devices" "$case_dir" >> "${FAKE_LAUNCH_LOG:?}"

        mkdir -p "$case_dir/diagnostics"
        worker_pids=''
        worker_pid_json=''
        worker_thread_json=''
        worker_index=1
        while [ "$worker_index" -le 4 ]; do
          sleep 30 &
          worker_pid=$!
          worker_pids="${worker_pids}${worker_pids:+ }${worker_pid}"
          worker_pid_json="${worker_pid_json}${worker_pid_json:+,}${worker_pid}"
          worker_thread_json="${worker_thread_json}${worker_thread_json:+,}1"
          worker_index=$((worker_index + 1))
        done
        printf '%s\n' $worker_pids > "$case_dir/worker_pids.txt"
        {
          printf '%s\n' "$$"
          printf '%s\n' $worker_pids
        } > "${FAKE_OWNED_COMPUTE_PIDS_FILE:?}"

        process_count=4
        if [ "${FAKE_INVALID_POOL_TELEMETRY_CASE:-}" = "$case_name" ]; then
          process_count=0
        fi
        telemetry=",\"pool_id\":\"${case_name}-pool\",\"batch_set_key\":\"F1\",\"batch_count\":4,\"process_count\":${process_count},\"worker_intraop_threads\":1,\"worker_interop_threads\":1"
        if [ "$case_name" != "base64" ]; then
          telemetry="${telemetry},\"candidate_bytes_written\":321,\"process_rss_bytes\":10000,\"process_peak_rss_bytes\":20000"
        fi
        probe_devices='["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4"]'
        probe_trial_counts='[1,1,1,1,1]'
        if [ "${FAKE_MISSING_PROBE_DEVICE_CASE:-}" = "$case_name" ]; then
          probe_devices='["cuda:0","cuda:1","cuda:2","cuda:3"]'
          probe_trial_counts='[1,1,1,1]'
        fi

        : > "$case_dir/diagnostics/episodes.jsonl"
        episode=1
        while [ "$episode" -le "$episodes" ]; do
          reward='10.0'
          if [ "${FAKE_SEMANTIC_MISMATCH_CASE:-}" = "$case_name" ] && \
             [ "$episode" -eq 1 ]; then
            reward='99.0'
          fi
          episode_probe_devices="$probe_devices"
          episode_probe_trial_counts="$probe_trial_counts"
          if [ "${FAKE_SPARSE_PROBE_DEVICE_CASE:-}" = "$case_name" ] && \
             [ "$episode" -gt 1 ]; then
            episode_probe_devices='["cuda:0","cuda:1","cuda:2","cuda:3"]'
            episode_probe_trial_counts='[1,1,1,1]'
          fi
          printf '{"episode":%s,"timestamp":%s,"total_reward":%s,"terminal_reward":8.0,"terminal_priority":3,"action_indices":[1,2],"action_hash":"action-1","terminal_loss_mean":0.1,"terminal_metric1_mean":0.9,"terminal_metric2_mean":0.8,"terminal_probe_devices":%s,"terminal_probe_trial_counts":%s,"terminal_probe_trial_indices":[[0],[1],[2],[3],[4]],"terminal_probe_trial_seeds":[101,102,103,104,105],"terminal_pareto_event_kind":"expanded"%s}\n' \
            "$episode" "$episode" "$reward" "$episode_probe_devices" \
            "$episode_probe_trial_counts" "$telemetry" \
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
        action_hash='49a64717d5d4cb19952e6eac2946415cf6879adacf9908e7d872332d32c6e684'
        candidate_key='3b1dcdbc7faf538e140e2e970438f8c39ae0d2ec56b13b77364850ebf0b01410'
        context_profile='mrpc'
        if [ "${FAKE_CONTEXT_HASH_MISMATCH_CASE:-}" = "$case_name" ]; then
          context_profile='corrupt-context'
        fi
        candidate_loss='0.1'
        if [ "${FAKE_CANDIDATE_MISMATCH_CASE:-}" = "$case_name" ]; then
          candidate_loss='0.2'
        fi
        if [ "${FAKE_CANDIDATE_KEY_MISMATCH_CASE:-}" = "$case_name" ]; then
          candidate_key='corrupt-candidate-key'
        fi
        if [ "$case_name" = "base64" ]; then
          printf '%s\n' \
            "{\"record_type\":\"candidate_trial_group_v1\",\"candidate_key\":\"${candidate_key}\",\"candidate_key_basis\":\"effective_action_hash + identity_context\",\"action_indices\":[1,2],\"raw_action_indices\":[1,2],\"effective_action_indices\":[1,2],\"raw_action_hash\":\"${action_hash}\",\"action_hash\":\"${action_hash}\",\"action_vector_hash\":\"${action_hash}\",\"effective_action_hash\":\"${action_hash}\",\"identity_context_hash\":\"${context_hash}\",\"identity_context\":{\"profile\":\"mrpc\"},\"fidelity\":\"F1\",\"valid\":true,\"trial_group\":{\"loss\":[0.1],\"metric1\":[0.9],\"metric2\":[0.8],\"seeds\":[101]},\"trial_group_metadata\":{\"fidelity\":\"F1\",\"identity_context\":{\"profile\":\"mrpc\"}}}" \
            > "$case_dir/diagnostics/candidate_store.jsonl"
        else
          printf '%s\n' \
            "{\"record_type\":\"candidate_identity_context_v1\",\"identity_context_hash\":\"${context_hash}\",\"identity_context\":{\"profile\":\"${context_profile}\"}}" \
            "{\"record_type\":\"candidate_trial_group_v2\",\"candidate_key\":\"${candidate_key}\",\"candidate_key_basis\":\"effective_action_hash + identity_context\",\"action_indices\":[1,2],\"raw_action_hash\":\"${action_hash}\",\"action_hash\":\"${action_hash}\",\"action_vector_hash\":\"${action_hash}\",\"effective_action_hash\":\"${action_hash}\",\"identity_context_hash\":\"${context_hash}\",\"fidelity\":\"F1\",\"valid\":true,\"trial_group\":{\"loss\":[${candidate_loss}],\"metric1\":[0.9],\"metric2\":[0.8],\"seeds\":[101]},\"trial_group_metadata\":{\"fidelity\":\"F1\"}}" \
            > "$case_dir/diagnostics/candidate_store.jsonl"
        fi

        if [ "${FAKE_ABANDONED_CANDIDATE_CASE:-}" = "$case_name" ]; then
          checkpoint_size="$(wc -c < "$case_dir/diagnostics/candidate_store.jsonl" | tr -d '[:space:]')"
          printf '%s\n' \
            "{\"record_type\":\"candidate_trial_group_v2\",\"candidate_key\":\"abandoned\",\"action_indices\":[1,2],\"identity_context_hash\":\"${context_hash}\",\"fidelity\":\"F1\",\"valid\":true,\"trial_group\":{\"loss\":[99.0],\"metric1\":[0.0],\"metric2\":[0.0],\"seeds\":[999]},\"trial_group_metadata\":{\"fidelity\":\"F1\"}}" \
            "{\"record_type\":\"candidate_store_recovery_v1\",\"checkpoint_size\":${checkpoint_size},\"logical_generation\":1}" \
            >> "$case_dir/diagnostics/candidate_store.jsonl"
        fi
        if [ -n "${FAKE_PROMOTION_ORDER_CASE:-}" ]; then
          if [ "$case_name" = "base64" ]; then
            promotion_type='candidate_promotion_status_v1'
            promotion_context=',"raw_action_indices":[1,2],"effective_action_indices":[1,2],"identity_context":{"profile":"mrpc"}'
          else
            promotion_type='candidate_promotion_status_v2'
            promotion_context=''
          fi
          promoted="{\"record_type\":\"${promotion_type}\",\"candidate_key\":\"${candidate_key}\",\"candidate_key_basis\":\"effective_action_hash + identity_context\",\"action_indices\":[1,2]${promotion_context},\"raw_action_hash\":\"${action_hash}\",\"action_hash\":\"${action_hash}\",\"action_vector_hash\":\"${action_hash}\",\"effective_action_hash\":\"${action_hash}\",\"identity_context_hash\":\"${context_hash}\",\"promotion_status\":\"promoted\",\"promotion_metadata\":{\"sequence\":1}}"
          rejected="{\"record_type\":\"${promotion_type}\",\"candidate_key\":\"${candidate_key}\",\"candidate_key_basis\":\"effective_action_hash + identity_context\",\"action_indices\":[1,2]${promotion_context},\"raw_action_hash\":\"${action_hash}\",\"action_hash\":\"${action_hash}\",\"action_vector_hash\":\"${action_hash}\",\"effective_action_hash\":\"${action_hash}\",\"identity_context_hash\":\"${context_hash}\",\"promotion_status\":\"rejected\",\"promotion_metadata\":{\"sequence\":2}}"
          if [ "$FAKE_PROMOTION_ORDER_CASE" = "$case_name" ]; then
            printf '%s\n' "$rejected" "$promoted" \
              >> "$case_dir/diagnostics/candidate_store.jsonl"
          else
            printf '%s\n' "$promoted" "$rejected" \
              >> "$case_dir/diagnostics/candidate_store.jsonl"
          fi
        fi
        if [ "${FAKE_UNTERMINATED_CANDIDATE_CASE:-}" = "$case_name" ]; then
          candidate_size="$(wc -c < "$case_dir/diagnostics/candidate_store.jsonl")"
          truncate -s "$((candidate_size - 1))" \
            "$case_dir/diagnostics/candidate_store.jsonl"
        fi

        printf '{"pool_id":"%s","batch_set_key":"F1","batch_count":4,"process_count":4,"worker_intraop_threads":1,"worker_interop_threads":1}\n' \
          "$case_name-pool" > "$case_dir/diagnostics/diagnostics_summary.json"
        f4_calls=1
        if [ "${FAKE_MISSING_F4_TOPOLOGY_CASE:-}" = "$case_name" ]; then
          f4_calls=0
        fi
        topology_worker_pid_json="$worker_pid_json"
        if [ "${FAKE_UNINVENTORIED_TOPOLOGY_PID_CASE:-}" = "$case_name" ]; then
          topology_worker_pid_json="999998,${topology_worker_pid_json}"
        fi
        printf '{"schema_version":"probe_pool_topology_v1","pool_id":"%s","backend":"process","devices":["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4"],"process_count":4,"primary_pid":%s,"worker_pids":[%s],"worker_intraop_threads":[1,%s],"worker_interop_threads":[1,%s],"batch_sets":{"F1":{"batch_count":4},"F4":{"batch_count":4}},"call_counts_by_batch_set":{"F1":1,"F4":%s},"trial_counts_by_batch_set":{"F1":5,"F4":%s}}\n' \
          "$case_name-pool" "$$" "$topology_worker_pid_json" \
          "$worker_thread_json" "$worker_thread_json" "$f4_calls" \
          "$((f4_calls * 25))" \
          > "$case_dir/diagnostics/probe_pool_topology.json"
        printf 'fake launch %s\n' "$case_name" > "$case_dir/launch.log"
        case "$case_name" in
          base64) wall=40 ;;
          opt64) wall=30 ;;
          opt128) wall=1 ;;
          opt256) wall=20 ;;
          *) exit 65 ;;
        esac
        printf '%s\n' "$wall" > "$case_dir/wall_seconds.txt"
        if [ "${FAKE_MUTATE_SOURCE_CASE:-}" = "$case_name" ]; then
          printf 'concurrent mutation\n' >> "${FAKE_MUTATE_SOURCE_PATH:?}"
        fi
        if [ "${FAKE_LAUNCH_HANG_CASE:-}" = "$case_name" ]; then
          trap 'exit 0' INT TERM
          while true; do sleep 1; done
        fi
        if [ "${FAKE_LEAK_CHILD_CASE:-}" = "$case_name" ]; then
          sleep "${FAKE_LEAK_CHILD_SECONDS:-30}" &
          printf '%s\n' "$!" > "$case_dir/leaked_child_pid.txt"
          exit 0
        fi
        if [ "${FAKE_LONG_DELAY_CASE:-}" = "$case_name" ]; then
          sleep "${FAKE_LONG_DELAY_SECONDS:-3}"
        fi
        sleep "${FAKE_LAUNCH_DELAY_SECONDS:-0.3}"
        if [ "${FAKE_LAUNCH_EXIT_CASE:-}" = "$case_name" ]; then
          exit "${FAKE_LAUNCH_EXIT_CODE:-17}"
        fi
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
    baseline_head = _run_checked(
        ["git", "rev-parse", "HEAD"], cwd=baseline,
    ).stdout.strip()
    optimized_head = _run_checked(
        ["git", "rev-parse", "HEAD"], cwd=optimized,
    ).stdout.strip()

    env = os.environ.copy()
    env.update({
        "BASELINE_ROOT": str(baseline),
        "OPTIMIZED_ROOT": str(optimized),
        "ARTIFACT_DIR": str(artifact_dir),
        "EXPECTED_BASELINE_SHA": baseline_head,
        "EXPECTED_OPTIMIZED_SHA": optimized_head,
        "STAGE2_GATE_CASE_LAUNCHER": str(launcher),
        "FAKE_DEFAULT_PAYLOAD_WRITER": str(launcher),
        "FAKE_DEFAULT_EPISODES": "600",
        "FAKE_DEFAULT_REWARD_DEVICES": "0,1,2,3,4",
        "FAKE_NVIDIA_LOG": str(nvidia_log),
        "FAKE_NVIDIA_SAMPLE_COUNT_FILE": str(root / "nvidia-smi-sample-count.txt"),
        "FAKE_CURRENT_CASE_FILE": str(root / "current-case.txt"),
        "FAKE_CASE_RUNNING_FILE": str(root / "running-case.txt"),
        "FAKE_OWNED_COMPUTE_PIDS_FILE": str(root / "owned-compute-pids.txt"),
        "FAKE_LAUNCH_LOG": str(launch_log),
        "GPU_SAMPLE_INTERVAL_SECONDS": "0.005",
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

    def test_static_contract_hardens_identity_gpu_exit_and_candidate_recovery(self):
        source = self._source()

        for token in (
            "48b03e869934aa8b3aa904a1fe8b611a1e2d618a",
            "EXPECTED_BASELINE_SHA",
            "EXPECTED_OPTIMIZED_SHA",
            "STAGE2_GATE_ARTIFACT_SCOPE",
            "STAGE2_GATE_TRAIN_EXIT_FILE",
            "CandidateStore",
            "iter_active_records",
            "gpu_utilization_report.py",
            "--require-all-visible-sampled-active",
        ):
            self.assertIn(token, source)
        self.assertRegex(source, r'wait\s+["\']?\$JOB_PID')
        self.assertIn("kill -KILL", source)


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

    def test_source_identity_mismatch_or_identical_roots_fail_before_training(self):
        cases = ("identical", "baseline_sha", "optimized_sha")
        for case_name in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as td:
                root = pathlib.Path(td)
                baseline, optimized = self._roots(root)
                selected_optimized = baseline if case_name == "identical" else optimized
                env, _nvidia_log, launch_log = _base_gate_env(
                    root,
                    baseline,
                    selected_optimized,
                    root / "artifacts",
                )
                if case_name == "baseline_sha":
                    env["EXPECTED_BASELINE_SHA"] = "0" * 40
                elif case_name == "optimized_sha":
                    env["EXPECTED_OPTIMIZED_SHA"] = "f" * 40

                proc = _run_gate(env)
                output = _combined_output(proc)

                self.assertNotEqual(proc.returncode, 0, output)
                self.assertRegex(output.lower(), r"same root|sha|commit|head")
                self.assertFalse(launch_log.exists(), output)

    def test_source_state_is_revalidated_between_serial_cases(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            baseline, optimized = self._roots(root)
            env, _nvidia_log, launch_log = _base_gate_env(
                root,
                baseline,
                optimized,
                root / "artifacts",
            )
            env["FAKE_MUTATE_SOURCE_CASE"] = "base64"
            env["FAKE_MUTATE_SOURCE_PATH"] = str(optimized / "README.txt")

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            self.assertRegex(output.lower(), r"source|dirty|head|sha")
            self.assertEqual(
                [
                    line.split("|", 1)[0]
                    for line in launch_log.read_text(encoding="utf-8").splitlines()
                ],
                ["base64"],
            )

    def test_existing_run_artifact_scope_is_allowed_but_other_dirtiness_is_not(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            baseline, optimized = self._roots(root)
            run_artifact = (
                optimized / "experiments" / "server_command_runs" / "existing"
            )
            cpu_tests = run_artifact / "cpu_tests"
            cpu_tests.mkdir(parents=True)
            (cpu_tests / "full_pytest.log").write_text(
                "existing Task 7 evidence\n", encoding="utf-8",
            )
            harness = root / "harness"
            harness.mkdir()
            env, _nvidia_log, launch_log = _base_gate_env(
                harness,
                baseline,
                optimized,
                run_artifact / "gpu_gate",
            )
            env["STAGE2_GATE_ARTIFACT_SCOPE"] = str(run_artifact)

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertEqual(proc.returncode, 0, output)
            self.assertTrue(launch_log.is_file(), output)
            self.assertTrue((cpu_tests / "full_pytest.log").is_file())

    def test_artifact_scope_cannot_hide_unrelated_worktree_dirtiness(self):
        cases = ("root_scope", "outside_scope")
        for case_name in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as td:
                root = pathlib.Path(td)
                baseline, optimized = self._roots(root)
                run_artifact = (
                    optimized / "experiments" / "server_command_runs" / "existing"
                )
                (run_artifact / "cpu_tests").mkdir(parents=True)
                (run_artifact / "cpu_tests" / "full_pytest.log").write_text(
                    "allowed evidence\n", encoding="utf-8",
                )
                if case_name == "outside_scope":
                    (optimized / "unexpected.tmp").write_text(
                        "unrelated\n", encoding="utf-8",
                    )
                harness = root / "harness"
                harness.mkdir()
                env, _nvidia_log, launch_log = _base_gate_env(
                    harness,
                    baseline,
                    optimized,
                    run_artifact / "gpu_gate",
                )
                env["STAGE2_GATE_ARTIFACT_SCOPE"] = str(
                    optimized if case_name == "root_scope" else run_artifact
                )

                proc = _run_gate(env)
                output = _combined_output(proc)

                self.assertNotEqual(proc.returncode, 0, output)
                self.assertRegex(output.lower(), r"scope|dirty|worktree root")
                self.assertFalse(launch_log.exists(), output)

    def test_artifact_scope_inside_source_tree_must_use_canonical_run_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            baseline, optimized = self._roots(root)
            unsafe_scope = optimized / "scripts" / "runtime-evidence"
            unsafe_scope.mkdir(parents=True)
            env, _nvidia_log, launch_log = _base_gate_env(
                root,
                baseline,
                optimized,
                unsafe_scope / "gpu_gate",
            )
            env["STAGE2_GATE_ARTIFACT_SCOPE"] = str(unsafe_scope)

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            self.assertRegex(output.lower(), r"artifact|scope|server_command_runs")
            self.assertFalse(launch_log.exists(), output)

    def test_ignored_noncache_source_file_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            baseline, optimized = self._roots(root)
            with (optimized / ".git" / "info" / "exclude").open(
                "a", encoding="utf-8",
            ) as handle:
                handle.write("unsafe-runtime.log\n")
            (optimized / "unsafe-runtime.log").write_text(
                "could alter runtime inputs\n", encoding="utf-8",
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
            self.assertRegex(output.lower(), r"ignored|dirty|unsafe-runtime")
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

    def test_duplicate_reward_devices_fail_before_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            baseline, optimized = self._roots(root)
            env, _nvidia_log, launch_log = _base_gate_env(
                root,
                baseline,
                optimized,
                root / "artifacts",
            )
            env["REWARD_DEVICES"] = "0,1,1,2,3"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            self.assertIn("duplicate", output.lower())
            self.assertFalse(launch_log.exists(), output)

    def test_inactive_requested_gpu_fails_evidence_after_all_cases_run(self):
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
            env["FAKE_INACTIVE_GPU"] = "4"

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
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            self.assertEqual(verdict["semantic_parity"], "FAIL")
            self.assertIsNone(verdict["fastest_eligible_case"])

    def test_transient_gpu_spike_does_not_count_as_sustained_five_gpu_use(self):
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
            env["FAKE_TRANSIENT_GPU"] = "4"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            self.assertEqual(verdict["semantic_parity"], "FAIL")
            self.assertIsNone(verdict["fastest_eligible_case"])

    def test_missing_probe_trials_on_one_gpu_fail_case_evidence(self):
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
            env["FAKE_MISSING_PROBE_DEVICE_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["evidence_pass"])
            self.assertFalse(by_case["opt128"]["semantic_pass"])
            self.assertEqual(verdict["fastest_eligible_case"], "opt256")

    def test_one_probe_episode_on_fifth_gpu_is_not_sufficient_coverage(self):
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
            env["FAKE_SPARSE_PROBE_DEVICE_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["evidence_pass"])
            self.assertEqual(verdict["fastest_eligible_case"], "opt256")

    def test_invalid_shared_pool_topology_fails_runtime_evidence(self):
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
            env["FAKE_INVALID_POOL_TELEMETRY_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["evidence_pass"])
            self.assertEqual(verdict["fastest_eligible_case"], "opt256")

    def test_unowned_runtime_compute_pid_fails_gpu_evidence(self):
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
            env["FAKE_RUNTIME_COMPUTE_PID_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["evidence_pass"])
            self.assertEqual(verdict["fastest_eligible_case"], "opt256")

    def test_empty_runtime_compute_pid_evidence_fails_closed(self):
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
            env["FAKE_SUPPRESS_OWNED_COMPUTE_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["evidence_pass"])
            self.assertEqual(verdict["fastest_eligible_case"], "opt256")

    def test_shared_pool_topology_requires_f4_calls(self):
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
            env["FAKE_MISSING_F4_TOPOLOGY_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["evidence_pass"])
            self.assertEqual(verdict["fastest_eligible_case"], "opt256")

    def test_shared_pool_topology_pids_must_match_runtime_inventory(self):
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
            env["FAKE_UNINVENTORIED_TOPOLOGY_PID_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["evidence_pass"])
            self.assertEqual(verdict["fastest_eligible_case"], "opt256")

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

    def test_default_launcher_reaps_detached_training_and_propagates_exit_code(self):
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
            env.pop("STAGE2_GATE_CASE_LAUNCHER")
            env["FAKE_LAUNCH_EXIT_CASE"] = "opt128"
            env["FAKE_LAUNCH_EXIT_CODE"] = "17"

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
            self.assertEqual(
                (artifact_dir / "cases" / "opt128" / "training_exit_code.txt")
                .read_text(encoding="utf-8")
                .strip(),
                "17",
            )
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["launch_pass"])
            self.assertFalse(by_case["opt128"]["semantic_pass"])
            self.assertEqual(verdict["fastest_eligible_case"], "opt256")

    def test_default_launcher_preserves_original_launcher_path_resolution(self):
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
            env.pop("STAGE2_GATE_CASE_LAUNCHER")
            env["FAKE_REQUIRE_ORIGINAL_LAUNCHER_PATH"] = "1"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertEqual(proc.returncode, 0, output)
            self.assertEqual(
                [
                    line.split("|", 1)[0]
                    for line in launch_log.read_text(encoding="utf-8").splitlines()
                ],
                ["base64", "opt64", "opt128", "opt256"],
            )
            for case_name in ("base64", "opt64", "opt128", "opt256"):
                self.assertEqual(
                    (artifact_dir / "cases" / case_name / "training_exit_code.txt")
                    .read_text(encoding="utf-8")
                    .strip(),
                    "0",
                )

    def test_default_launcher_uses_one_pre_registered_owned_process_group(self):
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
            env.pop("STAGE2_GATE_CASE_LAUNCHER")

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertEqual(proc.returncode, 0, output)
            for case_name in ("base64", "opt64", "opt128", "opt256"):
                case_dir = artifact_dir / "cases" / case_name
                self.assertEqual(
                    (case_dir / "gate_launcher_pid.txt")
                    .read_text(encoding="utf-8")
                    .strip(),
                    (case_dir / "training_process_group.txt")
                    .read_text(encoding="utf-8")
                    .strip(),
                )

    def test_custom_launcher_timeout_is_bounded_and_authoritative(self):
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
            env["FAKE_LONG_DELAY_CASE"] = "opt128"
            env["FAKE_LONG_DELAY_SECONDS"] = "3"
            env["STAGE2_GATE_CASE_TIMEOUT_SECONDS"] = "1"
            env["STAGE2_GATE_TERMINATION_GRACE_SECONDS"] = "1"

            started = time.monotonic()
            proc = _run_gate(env, timeout=20)
            elapsed = time.monotonic() - started
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            self.assertLess(elapsed, 10.0, output)
            self.assertEqual(
                (artifact_dir / "cases" / "opt128" / "launcher_exit_code.txt")
                .read_text(encoding="utf-8")
                .strip(),
                "124",
            )

    def test_custom_launcher_residual_process_group_is_reaped_and_fails_case(self):
        leaked_pid = None
        try:
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
                env["FAKE_LEAK_CHILD_CASE"] = "opt128"

                proc = _run_gate(env)
                output = _combined_output(proc)

                leaked_pid = int(
                    (artifact_dir / "cases" / "opt128" / "leaked_child_pid.txt")
                    .read_text(encoding="utf-8")
                    .strip()
                )
                self.assertNotEqual(proc.returncode, 0, output)
                with self.assertRaises(ProcessLookupError):
                    os.kill(leaked_pid, 0)
        finally:
            if leaked_pid is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(leaked_pid, 9)

    def test_timeout_cannot_be_overridden_by_graceful_training_exit(self):
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
            env.pop("STAGE2_GATE_CASE_LAUNCHER")
            env["FAKE_LAUNCH_HANG_CASE"] = "opt128"
            env["FAKE_PREWRITE_TRAIN_EXIT_CASE"] = "opt128"
            env["STAGE2_GATE_CASE_TIMEOUT_SECONDS"] = "1"
            env["STAGE2_GATE_TERMINATION_GRACE_SECONDS"] = "1"

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
            opt128 = artifact_dir / "cases" / "opt128"
            self.assertEqual(
                (opt128 / "launcher_exit_code.txt").read_text(encoding="utf-8").strip(),
                "124",
            )
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["launch_pass"])
            worker_pid = int(
                (opt128 / "persistent" / "rl" / "bert-base" / "mrpc" / "LATEST_PID")
                .read_text(encoding="utf-8")
                .strip()
            )
            with self.assertRaises(ProcessLookupError):
                os.kill(worker_pid, 0)

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

    def test_candidate_recovery_ignores_abandoned_physical_branch(self):
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
            env["FAKE_ABANDONED_CANDIDATE_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertTrue(by_case["opt128"]["candidate_pass"])
            self.assertTrue(by_case["opt128"]["semantic_pass"])
            self.assertEqual(verdict["semantic_parity"], "PASS")

    def test_candidate_context_hash_mismatch_fails_normalization(self):
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
            env["FAKE_CONTEXT_HASH_MISMATCH_CASE"] = "opt128"

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
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["evidence_pass"])
            self.assertFalse(by_case["opt128"]["candidate_pass"])
            self.assertFalse(by_case["opt128"]["semantic_pass"])
            self.assertEqual(verdict["fastest_eligible_case"], "opt256")

    def test_candidate_key_mismatch_fails_normalization(self):
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
            env["FAKE_CANDIDATE_KEY_MISMATCH_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["evidence_pass"])
            self.assertFalse(by_case["opt128"]["candidate_pass"])
            self.assertFalse(by_case["opt128"]["semantic_pass"])

    def test_candidate_append_order_mismatch_fails_semantic_parity(self):
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
            env["FAKE_PROMOTION_ORDER_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertTrue(by_case["opt128"]["evidence_pass"])
            self.assertFalse(by_case["opt128"]["candidate_pass"])
            self.assertFalse(by_case["opt128"]["semantic_pass"])
            self.assertEqual(verdict["fastest_eligible_case"], "opt256")

    def test_candidate_validation_never_repairs_original_evidence_in_place(self):
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
            env["FAKE_UNTERMINATED_CANDIDATE_CASE"] = "opt128"

            proc = _run_gate(env)
            output = _combined_output(proc)

            self.assertNotEqual(proc.returncode, 0, output)
            candidate_path = (
                artifact_dir / "cases" / "opt128" / "diagnostics"
                / "candidate_store.jsonl"
            )
            self.assertFalse(candidate_path.read_bytes().endswith(b"\n"))
            verdict = json.loads(
                (artifact_dir / "verdict.json").read_text(encoding="utf-8")
            )
            by_case = {row["case"]: row for row in verdict["cases"]}
            self.assertFalse(by_case["opt128"]["evidence_pass"])
            self.assertFalse(by_case["opt128"]["semantic_pass"])


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
