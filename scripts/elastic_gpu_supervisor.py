#!/usr/bin/env python3
"""Run RL against CUDA-verified physical GPUs without importing Torch.

Normal startup performs one batched ``nvidia-smi`` query, then requires every
candidate to execute an isolated CUDA canary. Passing devices are remapped
through ``CUDA_VISIBLE_DEVICES`` and the child receives dense logical device
IDs. A reserved child exit can quarantine one failed device and resume the same
run directory.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Iterable, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from device_utils import split_device_spec_tokens
from elastic_gpu import (
    ELASTIC_GPU_FAILURE_FILENAME,
    ELASTIC_GPU_RESTART_EXIT_CODE,
    ELASTIC_GPU_RESTART_REQUEST_ENV,
)


NVIDIA_SMI_HEALTH_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,uuid,gpu_recovery_action",
    "--format=csv,noheader,nounits",
)
_AUTO_DEVICE_FLAGS = frozenset(
    {
        "--blb-v3-reward-devices",
        "--blb_v3_reward_devices",
        "--stage1-rl-devices",
        "--stage1_rl_devices",
        "--stage2-rl-devices",
        "--stage2_rl_devices",
    }
)
_RESUME_FLAGS = frozenset({"--resume-run-dir", "--resume_run_dir"})
_HEALTHY_RECOVERY_ACTIONS = frozenset({"", "none"})
_FAILURE_RECORD_TYPE = "elastic_gpu_failure_v1"
_RESTART_RECORD_TYPE = "elastic_gpu_restart_request_v1"
_RESTART_REQUEST_FILENAME = "elastic_gpu_restart_request.json"
_HEALTH_EVENTS_FILENAME = "elastic_gpu_health_events.jsonl"


@dataclass(frozen=True)
class GPUHealthRecord:
    index: str
    uuid: str
    recovery_action: str

    @property
    def is_healthy(self) -> bool:
        return self.recovery_action.strip().lower() in (
            _HEALTHY_RECOVERY_ACTIONS
        )


@dataclass(frozen=True)
class HealthSnapshot:
    candidate_tokens: tuple[str, ...]
    healthy_tokens: tuple[str, ...]
    quarantined_tokens: tuple[str, ...]
    records: tuple[GPUHealthRecord, ...]
    cuda_verified_tokens: tuple[str, ...] = ()

    @property
    def logical_device_spec(self) -> str:
        return ",".join(str(index) for index in range(len(self.healthy_tokens)))

    @property
    def visibility_mode(self) -> str:
        return "index" if self.cuda_verified_tokens else "uuid"

    @property
    def healthy_visibility_tokens(self) -> tuple[str, ...]:
        attribute = self.visibility_mode
        return tuple(
            getattr(_record_for_token(self.records, token), attribute)
            for token in self.healthy_tokens
        )

    @property
    def quarantined_visibility_tokens(self) -> tuple[str, ...]:
        attribute = self.visibility_mode
        return tuple(
            getattr(_record_for_token(self.records, token), attribute)
            for token in self.quarantined_tokens
        )

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_tokens": list(self.candidate_tokens),
            "healthy_tokens": list(self.healthy_tokens),
            "quarantined_tokens": list(self.quarantined_tokens),
            "healthy_visibility_tokens": list(
                self.healthy_visibility_tokens
            ),
            "quarantined_visibility_tokens": list(
                self.quarantined_visibility_tokens
            ),
            "cuda_verified_tokens": list(self.cuda_verified_tokens),
            "visibility_mode": self.visibility_mode,
            "logical_device_spec": self.logical_device_spec,
            "devices": [
                {
                    "index": record.index,
                    "uuid": record.uuid,
                    "recovery_action": record.recovery_action,
                    "healthy": token in self.healthy_tokens,
                    "health_source": (
                        (
                            "cuda_canary_override"
                            if not record.is_healthy
                            else "cuda_canary"
                        )
                        if token in self.cuda_verified_tokens
                        else "nvidia_smi"
                        if record.is_healthy
                        else "quarantined"
                    ),
                }
                for token, record in zip(
                    self.candidate_tokens,
                    self.records,
                )
            ],
        }


def parse_nvidia_smi_csv(text: str) -> tuple[GPUHealthRecord, ...]:
    """Parse index, UUID, and recovery-action rows from ``nvidia-smi``."""
    records: list[GPUHealthRecord] = []
    seen_indices: set[str] = set()
    seen_uuids: set[str] = set()
    for row_number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not row or all(not str(value).strip() for value in row):
            continue
        if len(row) != 3:
            raise ValueError(
                f"nvidia-smi health row {row_number} has {len(row)} fields; "
                "expected index, uuid, gpu_recovery_action"
            )
        index, uuid, recovery_action = (str(value).strip() for value in row)
        if not index or not uuid:
            raise ValueError(
                f"nvidia-smi health row {row_number} has an empty index or UUID"
            )
        if index in seen_indices or uuid in seen_uuids:
            raise ValueError(
                f"nvidia-smi health snapshot contains duplicate GPU {index}/{uuid}"
            )
        seen_indices.add(index)
        seen_uuids.add(uuid)
        records.append(
            GPUHealthRecord(
                index=index,
                uuid=uuid,
                recovery_action=recovery_action,
            )
        )
    if not records:
        raise RuntimeError("nvidia-smi returned no GPU health records")
    return tuple(records)


def query_nvidia_smi_records(
    *,
    timeout_seconds: float = 10.0,
) -> tuple[GPUHealthRecord, ...]:
    completed = subprocess.run(
        list(NVIDIA_SMI_HEALTH_COMMAND),
        check=True,
        capture_output=True,
        text=True,
        timeout=max(float(timeout_seconds), 0.1),
    )
    return parse_nvidia_smi_csv(completed.stdout)


def _record_for_token(
    records: Sequence[GPUHealthRecord],
    token: str,
) -> GPUHealthRecord:
    matches = [
        record
        for record in records
        if token == record.index or token == record.uuid
    ]
    if not matches:
        raise ValueError(
            f"candidate GPU token {token!r} is not present in nvidia-smi"
        )
    if len(matches) != 1:
        raise ValueError(f"candidate GPU token {token!r} is ambiguous")
    return matches[0]


def resolve_health_snapshot(
    records: Sequence[GPUHealthRecord],
    *,
    candidate_tokens: Iterable[object],
    cuda_verified_tokens: Iterable[object] = (),
    require_cuda_verified: bool = False,
) -> HealthSnapshot:
    candidates = tuple(str(token).strip() for token in candidate_tokens)
    if not candidates or any(not token for token in candidates):
        raise ValueError("at least one non-empty candidate GPU token is required")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate GPU tokens must be unique")
    cuda_verified = tuple(
        str(token).strip() for token in cuda_verified_tokens
    )
    if len(set(cuda_verified)) != len(cuda_verified):
        raise ValueError("CUDA-verified GPU tokens must be unique")
    if any(token not in candidates for token in cuda_verified):
        raise ValueError("CUDA-verified GPU tokens must be candidates")

    selected = tuple(
        _record_for_token(records, token) for token in candidates
    )
    healthy = tuple(
        token
        for token, record in zip(candidates, selected)
        if (
            token in cuda_verified
            if require_cuda_verified
            else record.is_healthy or token in cuda_verified
        )
    )
    quarantined = tuple(
        token
        for token in candidates
        if token not in healthy
    )
    if not healthy:
        raise RuntimeError(
            "no healthy GPU remains after recovery-action filtering"
        )
    return HealthSnapshot(
        candidate_tokens=candidates,
        healthy_tokens=healthy,
        quarantined_tokens=quarantined,
        records=selected,
        cuda_verified_tokens=cuda_verified,
    )


def resolve_startup_health_snapshot(
    records: Sequence[GPUHealthRecord],
    *,
    candidate_tokens: Iterable[object],
    canary: Callable[[str], bool],
) -> HealthSnapshot:
    """Require isolated CUDA execution for every startup candidate."""
    candidates = tuple(str(token).strip() for token in candidate_tokens)
    selected = tuple(
        _record_for_token(records, token) for token in candidates
    )
    cuda_verified = tuple(
        token
        for token, record in zip(candidates, selected)
        if canary(record.index)
    )
    return resolve_health_snapshot(
        records,
        candidate_tokens=candidates,
        cuda_verified_tokens=cuda_verified,
        require_cuda_verified=True,
    )


def _visibility_tokens_for_candidates(
    records: Sequence[GPUHealthRecord],
    candidate_tokens: Iterable[object],
) -> tuple[str, ...]:
    return tuple(
        _record_for_token(records, str(token).strip()).uuid
        for token in candidate_tokens
    )


def _candidate_token_for_device(
    records: Sequence[GPUHealthRecord],
    candidate_tokens: Iterable[object],
    device_token: object,
) -> str:
    device_record = _record_for_token(records, str(device_token).strip())
    matches = [
        str(candidate).strip()
        for candidate in candidate_tokens
        if _record_for_token(
            records, str(candidate).strip()
        ).uuid == device_record.uuid
    ]
    if len(matches) != 1:
        raise ValueError(
            f"device token {device_token!r} does not map to exactly one "
            "candidate GPU"
        )
    return matches[0]


def build_child_command(
    command: Sequence[object],
    *,
    logical_device_spec: str,
    resume_run_dir: Optional[os.PathLike[str] | str],
    original_to_current_logical: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Rewrite only known auto-device and resume arguments."""
    source = [str(value) for value in command]
    rewritten: list[str] = []
    resume_value = (
        str(Path(resume_run_dir))
        if resume_run_dir is not None
        else None
    )
    saw_resume = False

    def rewrite_device_value(value: str) -> str:
        if value == "auto":
            return logical_device_spec
        if original_to_current_logical is None:
            return value
        mapped: list[str] = []
        for raw_token in split_device_spec_tokens(value):
            token = raw_token.lower()
            if token.startswith("cuda:"):
                token = token.split("cuda:", 1)[1].strip()
            if not token.isdigit():
                raise ValueError(
                    f"GPU device flag contains a non-logical ID: {raw_token!r}"
                )
            current = original_to_current_logical.get(token)
            if current is not None and current not in mapped:
                mapped.append(current)
        if not mapped:
            raise RuntimeError(
                f"all explicitly requested GPU devices are quarantined: {value}"
            )
        return ",".join(mapped)

    index = 0
    while index < len(source):
        token = source[index]
        if token in _AUTO_DEVICE_FLAGS:
            if index + 1 >= len(source):
                raise ValueError(f"{token} requires a value")
            value = source[index + 1]
            rewritten.extend([token, rewrite_device_value(value)])
            index += 2
            continue
        auto_equals = next(
            (
                flag
                for flag in _AUTO_DEVICE_FLAGS
                if token.startswith(f"{flag}=")
            ),
            None,
        )
        if auto_equals is not None:
            value = token.split("=", 1)[1]
            rewritten_value = rewrite_device_value(value)
            if rewritten_value != value:
                token = f"{auto_equals}={rewritten_value}"
            rewritten.append(token)
            index += 1
            continue
        if token in _RESUME_FLAGS:
            if index + 1 >= len(source):
                raise ValueError(f"{token} requires a value")
            value = source[index + 1]
            if resume_value is not None:
                value = resume_value
                saw_resume = True
            rewritten.extend([token, value])
            index += 2
            continue
        resume_equals = next(
            (
                flag
                for flag in _RESUME_FLAGS
                if token.startswith(f"{flag}=")
            ),
            None,
        )
        if resume_equals is not None:
            value = token.split("=", 1)[1]
            if resume_value is not None:
                rewritten.extend(["--resume_run_dir", resume_value])
                saw_resume = True
            else:
                rewritten.append(f"{resume_equals}={value}")
            index += 1
            continue
        rewritten.append(token)
        index += 1
    if resume_value is not None and not saw_resume:
        rewritten.extend(["--resume_run_dir", resume_value])
    return rewritten


def isolated_cuda_canary(
    physical_token: str,
    *,
    timeout_seconds: float = 30.0,
) -> bool:
    """Run a synchronized CUDA matrix operation in a disposable process."""
    canary = (
        "import torch\n"
        "assert torch.cuda.is_available()\n"
        "x = torch.ones((2048, 2048), device='cuda:0', dtype=torch.float16)\n"
        "y = x @ x\n"
        "assert float(y[0, 0].item()) == 2048.0\n"
        "torch.cuda.synchronize()\n"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_token)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", canary],
            check=False,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(float(timeout_seconds), 0.1),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return int(completed.returncode) == 0


class RecoveryMonitor:
    """Low-frequency recovery check for quarantined GPUs only."""

    def __init__(
        self,
        *,
        quarantined_tokens: Iterable[object],
        query_records: Callable[[], Sequence[GPUHealthRecord]],
        canary: Callable[[str], bool],
        on_recovered: Callable[[tuple[str, ...]], None],
        interval_seconds: float,
    ) -> None:
        self._tokens = tuple(str(token) for token in quarantined_tokens)
        self._query_records = query_records
        self._canary = canary
        self._on_recovered = on_recovered
        self._interval_seconds = max(float(interval_seconds), 0.0)
        self._reported: set[str] = set()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def poll_once(self) -> tuple[str, ...]:
        records = tuple(self._query_records())
        recovered: list[str] = []
        for token in self._tokens:
            if token in self._reported:
                continue
            record = _record_for_token(records, token)
            if not record.is_healthy:
                continue
            if not self._canary(record.uuid):
                continue
            self._reported.add(token)
            recovered.append(token)
        result = tuple(recovered)
        if result:
            self._on_recovered(result)
        return result

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                self.poll_once()
            except Exception:
                # Recovery is opportunistic. Query/canary failures keep the
                # device quarantined and must not interrupt healthy training.
                continue

    def start(self) -> None:
        if (
            not self._tokens
            or self._interval_seconds <= 0.0
            or self._thread is not None
        ):
            return
        self._thread = threading.Thread(
            target=self._run,
            name="elastic-gpu-recovery",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(
                timeout=max(min(self._interval_seconds, 1.0), 0.1)
            )


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_event(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **dict(payload),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
        )
        handle.flush()


def _consume_failure_record(path: Path) -> Optional[dict[str, object]]:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    path.unlink()
    if not isinstance(payload, dict):
        raise ValueError(f"elastic GPU failure record is not an object: {path}")
    return payload


def run_child_foreground(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    check: bool,
) -> subprocess.CompletedProcess[object]:
    """Run the learner in front and forward launcher stop signals to it."""
    lock_fd_text = str(env.get("BLB_STAGE2_RUN_LOCK_FD", "")).strip()
    pass_fds = (int(lock_fd_text),) if lock_fd_text else ()
    previous_handlers: dict[int, object] = {}
    pending_signals: list[int] = []
    in_main_thread = threading.current_thread() is threading.main_thread()

    def capture_until_child_starts(signum: int, _frame: object) -> None:
        pending_signals.append(signum)

    if in_main_thread:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, capture_until_child_starts)
    try:
        child = subprocess.Popen(
            list(command),
            env=dict(env),
            pass_fds=pass_fds,
        )
    except BaseException:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        raise

    def forward(signum: int, _frame: object) -> None:
        try:
            child.send_signal(signum)
        except ProcessLookupError:
            pass

    if in_main_thread:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, forward)
        for signum in pending_signals:
            forward(signum, None)
    try:
        return_code = child.wait()
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    if check and return_code:
        raise subprocess.CalledProcessError(return_code, list(command))
    return subprocess.CompletedProcess(list(command), return_code)


def run_supervised(
    *,
    child_command: Sequence[object],
    run_dir: os.PathLike[str] | str,
    candidate_tokens: Iterable[object],
    query_records: Callable[[], Sequence[GPUHealthRecord]] = (
        query_nvidia_smi_records
    ),
    process_runner: Callable[
        ..., subprocess.CompletedProcess[object]
    ] = run_child_foreground,
    max_restarts: int = 8,
    recovery_interval: float = 60.0,
    canary: Callable[[str], bool] = isolated_cuda_canary,
) -> int:
    """Run and, only for reserved elastic exits, resume the RL child."""
    command = tuple(str(value) for value in child_command)
    if not command:
        raise ValueError("child command must not be empty")
    output_dir = Path(run_dir).expanduser().resolve()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    failure_path = logs_dir / ELASTIC_GPU_FAILURE_FILENAME
    request_path = logs_dir / _RESTART_REQUEST_FILENAME
    events_path = logs_dir / _HEALTH_EVENTS_FILENAME
    candidates = tuple(str(token).strip() for token in candidate_tokens)
    if max_restarts < 0:
        raise ValueError("max_restarts must be non-negative")

    permanently_quarantined: set[str] = set()
    canary_recovered: set[str] = set()
    restart_count = 0
    should_resume = False

    while True:
        records = tuple(query_records())
        eligible_candidates = tuple(
            token
            for token in candidates
            if token not in permanently_quarantined
        )
        if not eligible_candidates:
            _append_event(
                events_path,
                {
                    "event": "no_healthy_gpu",
                    "quarantined_tokens": sorted(permanently_quarantined),
                },
            )
            return ELASTIC_GPU_RESTART_EXIT_CODE
        snapshot = resolve_startup_health_snapshot(
            records,
            candidate_tokens=eligible_candidates,
            canary=canary,
        )
        permanently_quarantined.update(snapshot.quarantined_tokens)
        live_tokens = tuple(
            token
            for token in snapshot.healthy_tokens
            if token not in permanently_quarantined
        )
        if not live_tokens:
            return ELASTIC_GPU_RESTART_EXIT_CODE
        logical_spec = ",".join(
            str(index) for index in range(len(live_tokens))
        )
        current_index = {
            token: str(index) for index, token in enumerate(live_tokens)
        }
        original_to_current = {
            str(original_index): current_index[token]
            for original_index, token in enumerate(candidates)
            if token in current_index
        }
        visibility_tokens = tuple(
            getattr(
                _record_for_token(records, token),
                snapshot.visibility_mode,
            )
            for token in live_tokens
        )
        launch_command = build_child_command(
            command,
            logical_device_spec=logical_spec,
            resume_run_dir=output_dir if should_resume else None,
            original_to_current_logical=original_to_current,
        )
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = ",".join(visibility_tokens)
        child_env[ELASTIC_GPU_RESTART_REQUEST_ENV] = str(request_path)

        failure_path.unlink(missing_ok=True)
        request_path.unlink(missing_ok=True)
        _append_event(
            events_path,
            {
                "event": "launch",
                "restart_count": restart_count,
                "healthy_tokens": list(live_tokens),
                "cuda_visible_devices": list(visibility_tokens),
                "quarantined_tokens": sorted(permanently_quarantined),
                "cuda_verified_tokens": list(
                    snapshot.cuda_verified_tokens
                ),
                "logical_device_spec": logical_spec,
                "resume": should_resume,
            },
        )

        def notify_recovered(tokens: tuple[str, ...]) -> None:
            canary_recovered.update(tokens)
            _atomic_write_json(
                request_path,
                {
                    "record_type": _RESTART_RECORD_TYPE,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "reason": "device_recovery",
                    "physical_devices": list(tokens),
                    "payload": {"restart_count": restart_count},
                },
            )

        monitor = RecoveryMonitor(
            quarantined_tokens=tuple(
                token
                for token in candidates
                if token in permanently_quarantined
            ),
            query_records=query_records,
            canary=canary,
            on_recovered=notify_recovered,
            interval_seconds=recovery_interval,
        )
        monitor.start()
        try:
            completed = process_runner(
                launch_command,
                env=child_env,
                check=False,
            )
        finally:
            monitor.stop()
        return_code = int(completed.returncode)
        if return_code != ELASTIC_GPU_RESTART_EXIT_CODE:
            _append_event(
                events_path,
                {
                    "event": "exit",
                    "return_code": return_code,
                    "restart_count": restart_count,
                },
            )
            return return_code
        if restart_count >= max_restarts:
            _append_event(
                events_path,
                {
                    "event": "restart_budget_exhausted",
                    "restart_count": restart_count,
                },
            )
            return return_code

        failure = _consume_failure_record(failure_path)
        if failure is None:
            _append_event(
                events_path,
                {
                    "event": "missing_failure_record",
                    "restart_count": restart_count,
                },
            )
            return return_code
        record_type = str(failure.get("record_type", ""))
        if record_type == _FAILURE_RECORD_TYPE:
            failed_token = str(failure.get("physical_device", "")).strip()
            if not failed_token:
                return return_code
            try:
                failed_candidate = _candidate_token_for_device(
                    records,
                    live_tokens,
                    failed_token,
                )
            except (RuntimeError, ValueError):
                return return_code
            permanently_quarantined.add(failed_candidate)
        elif record_type == _RESTART_RECORD_TYPE:
            recovered_tokens = tuple(
                str(token)
                for token in failure.get("physical_devices", [])
            )
            if (
                not recovered_tokens
                or any(
                    token not in canary_recovered
                    for token in recovered_tokens
                )
            ):
                return return_code
            permanently_quarantined.difference_update(recovered_tokens)
            canary_recovered.difference_update(recovered_tokens)
        else:
            return return_code

        restart_count += 1
        should_resume = True
        _append_event(
            events_path,
            {
                "event": "restart",
                "record_type": record_type,
                "restart_count": restart_count,
                "quarantined_tokens": sorted(permanently_quarantined),
            },
        )


def _candidate_tokens(
    records: Sequence[GPUHealthRecord],
    explicit: str,
) -> tuple[str, ...]:
    if str(explicit).strip():
        return tuple(split_device_spec_tokens(explicit))
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible:
        return tuple(split_device_spec_tokens(visible))
    return tuple(record.index for record in records)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RL on the healthy physical GPU set",
    )
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--candidate-devices", default="")
    parser.add_argument("--recovery-interval", type=float, default=60.0)
    parser.add_argument("--max-restarts", type=int, default=8)
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("child_command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    started = time.perf_counter()
    records = query_nvidia_smi_records()
    candidates = _candidate_tokens(records, args.candidate_devices)
    snapshot = resolve_startup_health_snapshot(
        records,
        candidate_tokens=candidates,
        canary=isolated_cuda_canary,
    )
    elapsed = time.perf_counter() - started
    health_payload = {
        "record_type": "elastic_gpu_health_snapshot_v1",
        "elapsed_seconds": elapsed,
        **snapshot.to_record(),
    }
    if args.health_only:
        if args.json:
            print(json.dumps(health_payload, ensure_ascii=True, sort_keys=True))
        else:
            print(
                "healthy="
                f"{','.join(snapshot.healthy_tokens)} "
                "quarantined="
                f"{','.join(snapshot.quarantined_tokens)} "
                f"elapsed={elapsed:.6f}s"
            )
        return 0
    if not str(args.run_dir).strip():
        raise SystemExit("--run-dir is required unless --health-only is used")
    child_command = list(args.child_command)
    if child_command[:1] == ["--"]:
        child_command = child_command[1:]
    if not child_command:
        raise SystemExit("a child command is required after --")

    first_records: Optional[tuple[GPUHealthRecord, ...]] = tuple(records)

    def query_with_cached_startup() -> tuple[GPUHealthRecord, ...]:
        nonlocal first_records
        if first_records is not None:
            cached = first_records
            first_records = None
            return cached
        return query_nvidia_smi_records()

    return run_supervised(
        child_command=child_command,
        run_dir=args.run_dir,
        candidate_tokens=candidates,
        query_records=query_with_cached_startup,
        max_restarts=args.max_restarts,
        recovery_interval=args.recovery_interval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
