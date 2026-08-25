"""N-GPU parallel reward-probe runner.

The PPO reward in BLB Stage-2 RL is computed by repeating the model forward
``K`` times (``--stage2-k-trials``) with independent CKKS noise seeds. Without
this runner, ``BLBStage2Env._eval_on_probe`` runs those K trials sequentially
on a single GPU. With N workers, each worker owns one GPU and executes its
assigned trials concurrently.

Design:

* **One PPO learner, one action stream.** The runner is invisible to the PPO
  loop; it slots in behind ``BLBStage2Env`` only at probe time.
* **Each device owns its own model + handler + bridge + probe_batches.**
  Worker 0 reuses the env's existing primary model (no extra allocation).
  Workers 1+ deepcopy the primary model onto their device. Each worker
  installs the same BLB cfg via its own bridge before running trials.
* **Process fan-out.** The primary GPU runs in the learner process. Replica
  GPUs live in persistent ``spawn`` children, avoiding Python/GIL contention
  between the model wrappers while retaining one model per GPU. Set
  ``BLB_STAGE2_PROBE_BACKEND=thread`` for the in-process fallback.
* **Single-device fallback.** A 1-worker ``ProbeRunner`` is a thin no-op
  wrapper. ``BLBStage2Env`` only constructs one when there are 2+ devices,
  so existing single-GPU runs keep the original codepath bitwise.
* **Determinism.** Trial seed = ``base_seed XOR (trial_idx * 2654435761)``,
  derived once per action from ``(episode/step counter)``. Independent of
  wall clock so repro is feasible. Each worker reseeds only the independent
  BLB-noise generator for its current device and leaves global Torch/NumPy RNG
  streams untouched, avoiding cross-thread interference.
"""
from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
import weakref
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import torch
import torch.nn as nn

from rfr.search.runtime.elastic_gpu import ElasticGPUFailure, is_recoverable_gpu_failure
from rfr.search.runtime.model_handler import ReversibleLayerHandler, reseed_noise_rng_for_device

from rfr.search.common.action_space import ActionDecodeResult
from .inference_eval import run_installed_probe_trial


try:
    from rfr.search.runtime.blb_bridge import BLBNoiseRLBridge
except Exception:  # pragma: no cover — torch-free import path
    BLBNoiseRLBridge = None  # type: ignore


_PROCESS_STARTUP_TIMEOUT_SECONDS = 300.0
_DEFAULT_PROCESS_COMMAND_TIMEOUT_SECONDS = 300.0


def resolve_probe_backend(spec: Optional[str] = None) -> str:
    """Resolve the multi-GPU probe execution backend.

    Persistent processes are the default because same-process BERT forwards
    contend in Python even when their CUDA kernels target separate devices.
    The thread backend remains available as an operational rollback.
    """
    raw = (
        os.environ.get("BLB_STAGE2_PROBE_BACKEND", "process")
        if spec is None else spec
    )
    value = str(raw or "process").strip().lower()
    if value not in {"process", "thread"}:
        raise ValueError(
            "BLB_STAGE2_PROBE_BACKEND must be 'process' or 'thread', "
            f"got {raw!r}"
        )
    return value


def resolve_probe_command_timeout_seconds() -> float:
    """Bound a stalled replica command without changing healthy execution."""
    env_name = "BLB_STAGE2_PROBE_COMMAND_TIMEOUT_SECONDS"
    raw = os.environ.get(
        env_name,
        str(_DEFAULT_PROCESS_COMMAND_TIMEOUT_SECONDS),
    )
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{env_name} must be a positive finite number, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{env_name} must be a positive finite number, got {raw!r}"
        )
    return value


def _resolve_probe_thread_count(env_name: str) -> int:
    raw = os.environ.get(env_name, "1")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{env_name} must be a positive integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{env_name} must be a positive integer, got {raw!r}")
    return value


def resolve_probe_intraop_threads() -> int:
    return _resolve_probe_thread_count("BLB_STAGE2_PROBE_INTRAOP_THREADS")


def resolve_probe_interop_threads() -> int:
    return _resolve_probe_thread_count("BLB_STAGE2_PROBE_INTEROP_THREADS")


def enable_cuda_reward_probe_fast_math() -> None:
    """Enable fast FP32 matmul modes that are appropriate for reward probes.

    On Ampere/Ada GPUs, TF32 keeps the tensors in FP32 while using Tensor Core
    matmul kernels. This is a throughput setting, not a change to the BLB action
    or Rescale_optimizer semantics. It is process-global and idempotent.
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _trial_seed(base_seed: int, trial_idx: int) -> int:
    """Deterministic per-trial seed.

    Two GPUs running with the same ``base_seed`` but different ``trial_idx``
    values get truly independent noise streams (and reruns of the same
    (base_seed, trial_idx) reproduce the same noise — useful for diagnosis).
    """
    from blb_stage2_rl.seed_utils import derive_probe_trial_seed

    return derive_probe_trial_seed(base_seed, trial_idx)


@lru_cache(maxsize=64)
def _split_round_robin_cached(k: int, n_workers: int) -> Tuple[Tuple[int, ...], ...]:
    """Return an immutable worker-to-trial assignment template."""
    k = max(0, int(k))
    n = max(1, int(n_workers))
    out: List[List[int]] = [[] for _ in range(n)]
    for trial_idx in range(k):
        out[trial_idx % n].append(trial_idx)
    return tuple(tuple(trials) for trials in out)


def _split_round_robin(k: int, n_workers: int) -> List[List[int]]:
    """Return per-worker trial-index lists. Round-robin balances variance.

    Example for k=5, n_workers=2: ``[[0, 2, 4], [1, 3]]``.
    Example for k=5, n_workers=3: ``[[0, 3], [1, 4], [2]]``.
    """
    return [list(trials) for trials in _split_round_robin_cached(k, n_workers)]


@lru_cache(maxsize=64)
def _split_action_trial_tasks_cached(
        action_count: int,
        k: int,
        n_workers: int,
        ) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """Assign action-major trial tasks round-robin across workers."""
    action_count = max(0, int(action_count))
    k = max(0, int(k))
    n = max(1, int(n_workers))
    out: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    flat_index = 0
    for action_index in range(action_count):
        for trial_index in range(k):
            out[flat_index % n].append((action_index, trial_index))
            flat_index += 1
    return tuple(tuple(tasks) for tasks in out)


def _normalize_probe_trial_result(
        raw_result: Sequence[Any],
        ) -> Tuple[float, float, float]:
    if len(raw_result) != 3:
        raise ValueError(
            "probe trial result must contain loss, metric1, and metric2"
        )
    loss, metric1, metric2 = raw_result
    return (
        float("nan") if loss is None else float(loss),
        float(metric1),
        float(metric2),
    )


def _normalize_trial_indices(
        trial_indices: Sequence[int],
        ) -> Tuple[int, ...]:
    normalized = tuple(int(trial_index) for trial_index in trial_indices)
    if any(trial_index < 0 for trial_index in normalized):
        raise ValueError("trial_indices must be nonnegative")
    if len(set(normalized)) != len(normalized):
        raise ValueError("trial_indices must be unique")
    return normalized


def _split_trial_indices_round_robin(
        trial_indices: Sequence[int],
        n_workers: int,
        ) -> Tuple[Tuple[int, ...], ...]:
    """Assign explicit trial indices using the cached round-robin template."""
    indices = _normalize_trial_indices(trial_indices)
    positions = _split_round_robin_cached(len(indices), n_workers)
    if indices == tuple(range(len(indices))):
        return positions
    return tuple(
        tuple(indices[position] for position in worker_positions)
        for worker_positions in positions
    )


def _split_action_trial_index_tasks(
        action_count: int,
        trial_indices: Sequence[int],
        n_workers: int,
        ) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """Assign explicit action/trial-index tasks round-robin across workers."""
    action_count = max(0, int(action_count))
    indices = _normalize_trial_indices(trial_indices)
    worker_count = max(1, int(n_workers))
    out: List[List[Tuple[int, int]]] = [[] for _ in range(worker_count)]
    flat_index = 0
    for action_index in range(action_count):
        for trial_index in indices:
            out[flat_index % worker_count].append(
                (action_index, trial_index)
            )
            flat_index += 1
    return tuple(tuple(tasks) for tasks in out)


def _group_action_trial_tasks(
        tasks: Sequence[Tuple[int, int]],
        actions: Sequence[ActionDecodeResult],
        base_seeds: Sequence[int],
        ) -> List[Dict[str, Any]]:
    """Build compact action groups for one worker's ordered task list."""
    groups: List[Dict[str, Any]] = []
    for action_index, trial_index in tasks:
        if not groups or int(groups[-1]["action_index"]) != int(action_index):
            groups.append({
                "action_index": int(action_index),
                "decoded": actions[int(action_index)],
                "base_seed": int(base_seeds[int(action_index)]),
                "trial_indices": [],
            })
        groups[-1]["trial_indices"].append(int(trial_index))
    return groups


def _move_probe_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Return a copy of ``batch`` with every tensor field moved to ``device``.

    ``ProbeBatch`` is a small dataclass with input_ids / attention_mask /
    labels / token_type_ids. We don't want to import its symbol here to keep
    this module's import graph thin, so we duck-type on attribute presence.
    """
    fields = ("input_ids", "attention_mask", "labels", "token_type_ids")
    moved = {}
    for f in fields:
        t = getattr(batch, f, None)
        if t is None:
            moved[f] = None
            continue
        if isinstance(t, torch.Tensor):
            moved[f] = t.to(device, non_blocking=True)
        else:
            moved[f] = t


    cls = batch.__class__
    try:
        return cls(**moved)
    except TypeError:
        clone = copy.copy(batch)
        for f, v in moved.items():
            try:
                setattr(clone, f, v)
            except Exception:
                pass
        return clone


def _normalize_batch_set_key(key: Any) -> str:
    normalized = str(key).strip()
    if not normalized:
        raise ValueError("probe batch-set key must be nonempty")
    return normalized


def _freeze_probe_batches(batches: Sequence[Any]) -> Tuple[Any, ...]:
    frozen = tuple(batches)
    if not frozen:
        raise ValueError("probe batch set must contain at least one batch")
    return frozen


@dataclass
class ProbeWorker:
    """Per-GPU state: replicated model + its own handler/bridge/probe_batches."""
    device: torch.device
    model: nn.Module
    handler: Any
    bridge: Any
    probe_batches: Sequence[Any]
    is_regression: bool
    metric_profile: str = ""
    role: str = "primary"
    probe_batch_sets: Dict[str, Tuple[Any, ...]] = field(
        init=False, repr=False,
    )

    def __post_init__(self) -> None:
        initial_batches = _freeze_probe_batches(self.probe_batches)
        self.probe_batch_sets = {"F1": initial_batches}

    def register_batch_set(self, key: str, batches: Sequence[Any]) -> None:
        normalized = _normalize_batch_set_key(key)
        frozen = _freeze_probe_batches(batches)
        if normalized in self.probe_batch_sets:
            raise ValueError(
                f"probe batch-set {normalized!r} is already registered"
            )
        self.probe_batch_sets[normalized] = frozen

    def install(self, decoded: ActionDecodeResult) -> None:
        """Install the BLB cfg on this worker's model via its bridge."""
        with torch.cuda.device(self.device):
            self.bridge.apply(
                block1_cfgs=decoded.block1_cfgs,
                block2_cfgs=decoded.block2_cfgs,
                block3_cfgs=decoded.block3_cfgs,
                block4_cfgs=decoded.block4_cfgs,
                block5_cfgs=decoded.block5_cfgs,
            )

    def clear(self) -> None:
        """Reverse install (called even on exception so the model can be reused)."""
        with torch.cuda.device(self.device):
            self.bridge.clear()

    def run_trial(
            self,
            trial_idx: int,
            base_seed: int,
            batch_set_key: str = "F1",
            ) -> Tuple[float, float, float]:
        """Run one trial and return (loss, m1, m2) for the selected batches."""
        normalized = _normalize_batch_set_key(batch_set_key)
        try:
            probe_batches = self.probe_batch_sets[normalized]
        except KeyError as exc:
            raise KeyError(
                f"unknown probe batch-set {normalized!r}; "
                f"registered={sorted(self.probe_batch_sets)}"
            ) from exc
        with torch.cuda.device(self.device):
            seed = _trial_seed(base_seed, trial_idx)
            reseed_noise_rng_for_device(self.device, seed)

            return run_installed_probe_trial(
                self.model,
                probe_batches,
                metric_profile=str(self.metric_profile),
                is_regression=bool(self.is_regression),
            )


def _probe_process_main(
        connection: Any,
        device_id: int,
        model_template: nn.Module,
        probe_batches_cpu: Sequence[Any],
        layers_attribute: str,
        is_regression: bool,
        metric_profile: str,
        ) -> None:
    """Own one replica GPU and serve synchronous probe commands."""
    worker: Optional[ProbeWorker] = None
    try:
        device = torch.device(f"cuda:{int(device_id)}")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.set_num_threads(resolve_probe_intraop_threads())
        try:
            torch.set_num_interop_threads(resolve_probe_interop_threads())
        except RuntimeError:
            pass
        if BLBNoiseRLBridge is None:
            raise RuntimeError("BLBNoiseRLBridge is unavailable in probe child")
        enable_cuda_reward_probe_fast_math()
        with torch.cuda.device(device):
            model = model_template.to(device)
            model.eval()
            handler = ReversibleLayerHandler(model)
            bridge = BLBNoiseRLBridge(
                handler, layers_attribute=str(layers_attribute),
            )
            probe_batches = [
                _move_probe_batch_to_device(batch, device)
                for batch in probe_batches_cpu
            ]
            torch.cuda.synchronize(device)
        worker = ProbeWorker(
            device=device,
            model=model,
            handler=handler,
            bridge=bridge,
            probe_batches=probe_batches,
            is_regression=bool(is_regression),
            metric_profile=str(metric_profile),
            role="replica_process",
        )
        connection.send({
            "status": "ready",
            "operation": "startup",
            "device": str(device),
        })
    except BaseException as exc:  # noqa: BLE001
        try:
            connection.send({
                "status": "error",
                "operation": "startup",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        finally:
            connection.close()
        return

    while True:
        try:
            message = connection.recv()
        except EOFError:
            break
        operation = str(message.get("operation", ""))
        payload = dict(message.get("payload") or {})
        started = time.perf_counter()
        should_close = operation == "close"
        try:
            if operation == "install":
                worker.install(payload["decoded"])
                result: dict = {}
            elif operation == "clear":
                worker.clear()
                result = {}
            elif operation == "register_batch_set":
                batch_set_key = _normalize_batch_set_key(
                    payload["batch_set_key"]
                )
                if batch_set_key in worker.probe_batch_sets:
                    raise ValueError(
                        f"probe batch-set {batch_set_key!r} is already registered"
                    )
                with torch.cuda.device(worker.device):
                    registered_batches = tuple(
                        _move_probe_batch_to_device(batch, worker.device)
                        for batch in payload["probe_batches_cpu"]
                    )
                worker.register_batch_set(batch_set_key, registered_batches)
                result = {
                    "batch_set_key": batch_set_key,
                    "batch_count": len(registered_batches),
                }
            elif operation == "run_trials":
                base_seed = int(payload["base_seed"])
                batch_set_key = str(payload["batch_set_key"])
                for raw_trial_index in payload["trial_indices"]:
                    trial_index = int(raw_trial_index)
                    connection.send({
                        "status": "result",
                        "operation": operation,
                        "payload": {
                            "trial_index": trial_index,
                            "result": worker.run_trial(
                                trial_index, base_seed, batch_set_key,
                            ),
                        },
                    })
                result = {"results": []}
            elif operation == "run_action_trial":
                trial_idx = int(payload["trial_idx"])
                base_seed = int(payload["base_seed"])
                batch_set_key = str(payload["batch_set_key"])
                worker.install(payload["decoded"])
                result = {
                    "trial_idx": trial_idx,
                    "result": worker.run_trial(
                        trial_idx, base_seed, batch_set_key,
                    ),
                }
            elif operation == "run_action_trial_groups":
                batch_set_key = str(payload["batch_set_key"])
                for group in payload["action_groups"]:
                    action_index = int(group["action_index"])
                    base_seed = int(group["base_seed"])
                    worker.install(group["decoded"])
                    for raw_trial_index in group["trial_indices"]:
                        trial_index = int(raw_trial_index)
                        connection.send({
                            "status": "result",
                            "operation": operation,
                            "payload": {
                                "action_index": action_index,
                                "trial_index": trial_index,
                                "result": worker.run_trial(
                                    trial_index, base_seed, batch_set_key,
                                ),
                            },
                        })
                result = {"results": []}
            elif operation == "close":
                try:
                    worker.clear()
                except Exception:
                    pass
                result = {}
            else:
                raise ValueError(f"unknown probe child operation {operation!r}")
            result["wall_seconds"] = float(time.perf_counter() - started)
            connection.send({
                "status": "ok",
                "operation": operation,
                "payload": result,
            })
        except BaseException as exc:  # noqa: BLE001
            connection.send({
                "status": "error",
                "operation": operation,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        if should_close:
            break
    connection.close()


class _ProcessProbeWorker:
    """Parent-side handle for one persistent replica process."""

    def __init__(self, *, device: torch.device, connection: Any, process: Any):
        self.device = device
        self.connection = connection
        self.process = process
        self.role = "replica_process"
        self._pending_operation: Optional[str] = None
        self._closed = False

    def _receive_message(self, timeout: float) -> dict:
        deadline = time.monotonic() + max(0.1, float(timeout))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"probe child {self.device} timed out after {timeout:.1f}s"
                )
            if self.connection.poll(min(0.1, remaining)):
                return dict(self.connection.recv())
            if not self.process.is_alive():
                raise RuntimeError(
                    f"probe child {self.device} exited with code "
                    f"{self.process.exitcode}"
                )

    def wait_until_ready(self) -> None:
        message = self._receive_message(_PROCESS_STARTUP_TIMEOUT_SECONDS)
        if message.get("status") != "ready":
            details = message.get("traceback") or message.get("error") or message
            raise RuntimeError(
                f"probe child {self.device} startup failed: {details}"
            )

    def submit(self, operation: str, payload: dict) -> None:
        if self._closed:
            raise RuntimeError(f"probe child {self.device} is closed")
        if self._pending_operation is not None:
            raise RuntimeError(
                f"probe child {self.device} already has pending operation "
                f"{self._pending_operation!r}"
            )
        if not self.process.is_alive():
            raise RuntimeError(
                f"probe child {self.device} is not alive "
                f"(exitcode={self.process.exitcode})"
            )
        self.connection.send({
            "operation": str(operation),
            "payload": dict(payload),
        })
        self._pending_operation = str(operation)

    def receive(
            self,
            operation: str,
            timeout: float | None = None,
            result_handler: Callable[[Dict[str, Any]], None] | None = None,
            ) -> dict:
        expected = str(operation)
        if self._pending_operation != expected:
            raise RuntimeError(
                f"probe child {self.device} expected pending {expected!r}, "
                f"got {self._pending_operation!r}"
            )
        command_timeout = (
            resolve_probe_command_timeout_seconds()
            if timeout is None else float(timeout)
        )
        deadline = time.monotonic() + command_timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"probe child {self.device} timed out after "
                        f"{command_timeout:.1f}s"
                    )
                message = self._receive_message(remaining)
                if message.get("operation") != expected:
                    raise RuntimeError(
                        f"probe child {self.device} returned operation "
                        f"{message.get('operation')!r}, expected {expected!r}"
                    )
                if message.get("status") != "result":
                    break
                if result_handler is None:
                    raise RuntimeError(
                        f"probe child {self.device} returned an unexpected "
                        f"result for {expected!r}"
                    )
                result_handler(dict(message.get("payload") or {}))
        finally:
            self._pending_operation = None
        if message.get("status") != "ok":
            details = message.get("traceback") or message.get("error") or message
            raise RuntimeError(
                f"probe child {self.device} {expected} failed: {details}"
            )
        return dict(message.get("payload") or {})

    def _terminate_stubborn_process(self) -> None:
        if not self.process.is_alive():
            return
        self.process.terminate()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            kill = getattr(self.process, "kill", None)
            if callable(kill):
                kill()
            else:
                pid = getattr(self.process, "pid", None)
                if (
                        isinstance(pid, bool)
                        or not isinstance(pid, int)
                        or pid <= 0
                ):
                    raise RuntimeError(
                        f"probe child {self.device} has no callable kill() "
                        f"and no valid pid for SIGKILL: {pid!r}"
                    )
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError as exc:
                    raise RuntimeError(
                        f"failed to SIGKILL probe child {self.device} "
                        f"(pid={pid}): {exc}"
                    ) from exc
            self.process.join(timeout=5.0)
        if self.process.is_alive():
            raise RuntimeError(
                f"probe child {self.device} did not exit after "
                f"terminate/kill (pending operation "
                f"{self._pending_operation!r})"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._pending_operation is not None and self.process.is_alive():
                self._terminate_stubborn_process()
            else:
                if self._pending_operation is None and self.process.is_alive():
                    try:
                        self.connection.send({
                            "operation": "close", "payload": {},
                        })
                        self._pending_operation = "close"
                        self.receive("close", timeout=5.0)
                    except Exception:
                        pass
                self.process.join(timeout=5.0)
                if self.process.is_alive():
                    self._terminate_stubborn_process()
        finally:
            try:
                self.connection.close()
            except Exception:
                pass


@dataclass
class ProbeRunnerDiagnostics:
    """Per-call timing snapshot. Captured each ``run_trials`` invocation;
    callers can sample these (e.g. every 100 episodes) for the speedup log line."""
    k: int = 0
    wall_seconds: float = 0.0
    per_worker_seconds: List[float] = field(default_factory=list)
    per_worker_trial_counts: List[int] = field(default_factory=list)
    per_worker_trial_indices: List[List[int]] = field(default_factory=list)
    per_worker_trial_seeds: List[List[int]] = field(default_factory=list)
    devices: List[str] = field(default_factory=list)
    multi_action: bool = False
    action_count: int = 0
    trials_per_action: int = 0
    per_worker_action_trial_indices: List[
        List[Tuple[int, int]]
    ] = field(default_factory=list)
    pool_generation: int = 0
    retry_count: int = 0
    quarantined_devices: List[str] = field(default_factory=list)
    retried_trial_indices: List[int] = field(default_factory=list)
    retried_action_trial_indices: List[Tuple[int, int]] = field(
        default_factory=list
    )

    @property
    def speedup_vs_sequential(self) -> float:
        if not self.per_worker_seconds or self.wall_seconds <= 0:
            return 1.0
        return float(sum(self.per_worker_seconds)) / float(self.wall_seconds)


class ProbeRunner:
    """Fan trials across N workers, aggregate results in trial order."""

    def __init__(
            self,
            workers: List[ProbeWorker],
            *,
            process_workers: Optional[Sequence[Any]] = None,
            ):
        if not workers:
            raise ValueError("ProbeRunner requires at least one worker")
        self._process_workers = list(process_workers or [])
        if self._process_workers and len(workers) != 1:
            raise ValueError(
                "process probe backend requires exactly one local primary worker"
            )
        self.workers = list(workers) + self._process_workers
        self.pool_id = f"probe-pool-{uuid4().hex}"
        self._batch_sets: Dict[str, Tuple[Any, ...]] = {
            "F1": tuple(getattr(workers[0], "probe_batches", ())),
        }
        self.last_diagnostics: Optional[ProbeRunnerDiagnostics] = None
        self._closed = False
        self._poisoned_reason: Optional[str] = None
        self.pool_generation = 0
        self._quarantine_events: List[Dict[str, Any]] = []
        self._deferred_gpu_failures: List[ElasticGPUFailure] = []
        self._process_finalizer: Optional[weakref.finalize] = None
        self._refresh_process_finalizer()

    @staticmethod
    def _close_worker_handles(workers: Sequence[Any]) -> None:
        def close_one(worker: Any) -> None:
            try:
                worker.close()
            except Exception:
                pass

        if len(workers) == 1:
            close_one(workers[0])
            return

        threads = [
            threading.Thread(target=close_one, args=(worker,), daemon=True)
            for worker in workers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def _refresh_process_finalizer(self) -> None:
        if (
                self._process_finalizer is not None
                and self._process_finalizer.alive
        ):
            self._process_finalizer.detach()
        self._process_finalizer = (
            weakref.finalize(
                self,
                ProbeRunner._close_worker_handles,
                tuple(self._process_workers),
            )
            if self._process_workers else None
        )

    @property
    def quarantine_events(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(dict(event) for event in self._quarantine_events)

    def pop_deferred_gpu_failure(self) -> Optional[ElasticGPUFailure]:
        """Return one recovered replica failure at the next PPO checkpoint."""
        if not self._deferred_gpu_failures:
            return None
        return self._deferred_gpu_failures.pop(0)

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    @property
    def devices(self) -> List[torch.device]:
        return [w.device for w in self.workers]

    @property
    def backend(self) -> str:
        return "process" if self._process_workers else "thread"

    def _require_open(self) -> None:
        if self._poisoned_reason is not None:
            raise RuntimeError(
                f"probe runner pool is poisoned: {self._poisoned_reason}"
            )
        if self._closed:
            raise RuntimeError("probe runner pool is closed")

    def _require_batch_set(self, key: str) -> str:
        normalized = _normalize_batch_set_key(key)
        if normalized not in self._batch_sets:
            raise KeyError(
                f"unknown probe batch-set {normalized!r}; "
                f"registered={sorted(self._batch_sets)}"
            )
        return normalized

    def register_batch_set(self, key: str, batches: Sequence[Any]) -> None:
        """Register one immutable batch set on every worker in this pool."""
        self._require_open()
        normalized = _normalize_batch_set_key(key)
        frozen = _freeze_probe_batches(batches)
        if normalized in self._batch_sets:
            raise ValueError(
                f"probe batch-set {normalized!r} is already registered"
            )

        try:
            if self._process_workers:
                cpu_batches = tuple(
                    _move_probe_batch_to_device(batch, torch.device("cpu"))
                    for batch in frozen
                )
                self.workers[0].register_batch_set(normalized, frozen)
                submitted: List[Tuple[int, Any]] = []
                for worker_index, worker in enumerate(
                        self._process_workers, start=1,
                        ):
                    try:
                        worker.submit("register_batch_set", {
                            "batch_set_key": normalized,
                            "probe_batches_cpu": cpu_batches,
                        })
                        submitted.append((worker_index, worker))
                    except BaseException as exc:  # noqa: BLE001
                        self._raise_process_error(
                            worker_index, "register_batch_set", exc,
                        )
                for worker_index, worker in submitted:
                    try:
                        worker.receive("register_batch_set")
                    except BaseException as exc:  # noqa: BLE001
                        self._raise_process_error(
                            worker_index, "register_batch_set", exc,
                        )
            else:
                worker_batches = [frozen]
                worker_batches.extend(
                    tuple(
                        _move_probe_batch_to_device(batch, worker.device)
                        for batch in frozen
                    )
                    for worker in self.workers[1:]
                )
                for worker, batches_for_worker in zip(
                        self.workers, worker_batches,
                        ):
                    worker.register_batch_set(normalized, batches_for_worker)

            self._batch_sets[normalized] = frozen
        except BaseException as exc:  # noqa: BLE001
            self._poisoned_reason = (
                f"register_batch_set {normalized!r} failed: {exc}"
            )
            self.close()
            raise RuntimeError(
                f"probe-runner register_batch_set {normalized!r} failed; "
                f"pool closed: {exc}"
            ) from exc

    def view(self, batch_set_key: str) -> "ProbeRunnerView":
        self._require_open()
        normalized = self._require_batch_set(batch_set_key)
        return ProbeRunnerView(self, normalized)

    def _raise_process_error(
            self,
            worker_index: int,
            operation: str,
            exc: BaseException,
            ) -> None:
        raise RuntimeError(
            f"probe-runner worker {worker_index} "
            f"(device {self.workers[worker_index].device}) "
            f"{operation} failed: {exc}"
        ) from exc

    def _quarantine_process_worker(
            self,
            worker_index: int,
            operation: str,
            exc: BaseException,
            ) -> str:
        if worker_index <= 0 or worker_index >= len(self.workers):
            raise RuntimeError(
                f"cannot quarantine invalid process worker {worker_index}"
            )
        worker = self.workers[worker_index]
        if worker not in self._process_workers:
            raise RuntimeError(
                f"worker {worker_index} is not a process replica"
            )
        device = str(worker.device)
        failure = ElasticGPUFailure(
            device=worker.device,
            role="reward-probe-replica",
            operation=operation,
            cause=exc,
        )
        self.workers.pop(worker_index)
        self._process_workers.remove(worker)
        try:
            worker.close()
        finally:
            self._deferred_gpu_failures.append(failure)
            self.pool_generation += 1
            self._quarantine_events.append({
                "pool_generation": int(self.pool_generation),
                "device": device,
                "operation": str(operation),
                "cause_type": type(exc).__name__,
                "cause": str(exc),
            })
            self._refresh_process_finalizer()
        return device

    def _handle_process_errors(
            self,
            errors: Sequence[Tuple[int, BaseException]],
            operation: str,
            ) -> List[str]:
        if not errors:
            return []
        for worker_index, exc in errors:
            if not is_recoverable_gpu_failure(exc):
                self._raise_process_error(worker_index, operation, exc)
        for worker_index, exc in errors:
            if worker_index == 0:
                raise ElasticGPUFailure(
                    device=self.workers[0].device,
                    role="learner-primary",
                    operation=operation,
                    cause=exc,
                ) from exc

        quarantined: List[str] = []
        unique_errors: Dict[int, BaseException] = {}
        for worker_index, exc in errors:
            unique_errors.setdefault(int(worker_index), exc)
        for worker_index in sorted(unique_errors, reverse=True):
            quarantined.append(
                self._quarantine_process_worker(
                    worker_index,
                    operation,
                    unique_errors[worker_index],
                )
            )
        quarantined.reverse()
        return quarantined

    def close(self) -> None:
        """Stop persistent replica processes. Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        if self._process_finalizer is not None and self._process_finalizer.alive:
            self._process_finalizer()

    def _for_each_worker(self, fn) -> None:
        """Run a worker-local operation on every worker.

        Install and clear touch independent model replicas and CUDA devices, so
        worker-local setup runs concurrently with the same failure aggregation.
        """
        if len(self.workers) == 1:
            fn(self.workers[0])
            return

        errors: List[Tuple[int, BaseException]] = []
        lock = threading.Lock()

        def task(w_idx: int) -> None:
            try:
                fn(self.workers[w_idx])
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append((w_idx, exc))

        threads = [
            threading.Thread(target=task, args=(i,), daemon=True)
            for i in range(len(self.workers))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            w_idx, exc = errors[0]
            raise RuntimeError(
                f"probe-runner worker {w_idx} (device {self.workers[w_idx].device}) "
                f"setup failed: {exc!r}"
            ) from exc

    def install_action(self, decoded: ActionDecodeResult) -> None:
        """Apply the same cfg on every worker's model."""
        self._require_open()
        if self._process_workers:
            submitted: List[Tuple[int, Any]] = []
            errors: List[Tuple[int, BaseException]] = []
            for worker_index, worker in enumerate(self._process_workers, start=1):
                try:
                    worker.submit("install", {"decoded": decoded})
                    submitted.append((worker_index, worker))
                except BaseException as exc:  # noqa: BLE001
                    errors.append((worker_index, exc))
            try:
                self.workers[0].install(decoded)
            except BaseException as exc:  # noqa: BLE001
                errors.append((0, exc))
            for worker_index, worker in submitted:
                try:
                    worker.receive("install")
                except BaseException as exc:  # noqa: BLE001
                    errors.append((worker_index, exc))
            if errors:
                self._handle_process_errors(errors, "install")
            return
        self._for_each_worker(lambda w: w.install(decoded))

    def clear(self) -> None:
        """Reverse install on every worker. Always safe (clear is idempotent)."""
        self._require_open()
        if self._process_workers:
            submitted = []
            for worker in self._process_workers:
                try:
                    worker.submit("clear", {})
                    submitted.append(worker)
                except Exception:
                    pass
            try:
                self.workers[0].clear()
            except Exception:
                pass
            for worker in submitted:
                try:
                    worker.receive("clear")
                except Exception:
                    pass
            return

        def clear_one(w: ProbeWorker) -> None:
            try:
                w.clear()
            except Exception:


                pass
        self._for_each_worker(clear_one)

    @staticmethod
    def _accept_trial_result(
            *,
            trial_index: int,
            raw_result: Sequence[float],
            expected_indices: Sequence[int],
            results: Dict[int, Tuple[float, float, float] | None],
            ) -> None:
        index = int(trial_index)
        expected = {int(value) for value in expected_indices}
        if (
                index not in expected
                or index not in results
                or results[index] is not None
        ):
            raise RuntimeError(
                "probe-runner received duplicate or out-of-range trial "
                f"identity {index}; expected={sorted(expected)}"
            )
        if len(raw_result) != 3:
            raise RuntimeError(
                f"probe-runner trial {index} returned {len(raw_result)} metrics"
            )
        results[index] = _normalize_probe_trial_result(raw_result)

    @classmethod
    def _accept_trial_payload(
            cls,
            *,
            payload: Dict[str, Any],
            expected_indices: Sequence[int],
            results: Dict[int, Tuple[float, float, float] | None],
            ) -> None:
        for raw_trial_index, raw_result in payload.get("results", []):
            cls._accept_trial_result(
                trial_index=int(raw_trial_index),
                raw_result=raw_result,
                expected_indices=expected_indices,
                results=results,
            )

    def _run_trial_indices_processes(
            self,
            *,
            trial_indices: Sequence[int],
            base_seed: int,
            batch_set_key: str,
            ) -> List[Tuple[float, float, float]]:
        indices = _normalize_trial_indices(trial_indices)
        results: Dict[int, Tuple[float, float, float] | None] = {
            trial_index: None for trial_index in indices
        }
        initial_assignments = _split_trial_indices_round_robin(
            indices, self.num_workers,
        )

        diagnostic_devices = [str(worker.device) for worker in self.workers]
        seconds_by_device = {device: 0.0 for device in diagnostic_devices}
        assignments = initial_assignments
        retry_count = 0
        quarantined: List[str] = []
        retried: List[int] = []
        retried_set: set[int] = set()
        wall_started = time.perf_counter()

        while True:
            errors: List[Tuple[int, BaseException]] = []
            submitted: List[Tuple[int, Any]] = []
            for worker_index, worker in enumerate(
                    self._process_workers, start=1,
            ):
                trials = assignments[worker_index]
                if not trials:
                    continue
                try:
                    worker.submit("run_trials", {
                        "trial_indices": list(trials),
                        "base_seed": int(base_seed),
                        "batch_set_key": batch_set_key,
                    })
                    submitted.append((worker_index, worker))
                except BaseException as exc:  # noqa: BLE001
                    errors.append((worker_index, exc))

            local_device = str(self.workers[0].device)
            local_started = time.perf_counter()
            try:
                for trial_index in assignments[0]:
                    self._accept_trial_result(
                        trial_index=trial_index,
                        raw_result=self.workers[0].run_trial(
                            trial_index, base_seed, batch_set_key,
                        ),
                        expected_indices=assignments[0],
                        results=results,
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append((0, exc))
            finally:
                seconds_by_device[local_device] = (
                    seconds_by_device.get(local_device, 0.0)
                    + time.perf_counter() - local_started
                )

            for worker_index, worker in submitted:
                try:
                    def accept_result(
                            payload: Dict[str, Any],
                            worker_index: int = worker_index,
                    ) -> None:
                        self._accept_trial_result(
                            trial_index=int(payload["trial_index"]),
                            raw_result=payload["result"],
                            expected_indices=assignments[worker_index],
                            results=results,
                        )

                    payload = worker.receive(
                        "run_trials", result_handler=accept_result,
                    )
                    device = str(worker.device)
                    seconds_by_device[device] = (
                        seconds_by_device.get(device, 0.0)
                        + float(payload.get("wall_seconds", 0.0) or 0.0)
                    )
                    self._accept_trial_payload(
                        payload=payload,
                        expected_indices=assignments[worker_index],
                        results=results,
                    )
                except BaseException as exc:  # noqa: BLE001
                    errors.append((worker_index, exc))

            if not errors:
                break
            quarantined.extend(
                self._handle_process_errors(errors, "run_trials")
            )
            missing = [
                trial_index for trial_index in indices
                if results[trial_index] is None
            ]
            if not missing:
                break
            retry_count += 1
            for trial_index in missing:
                if trial_index not in retried_set:
                    retried_set.add(trial_index)
                    retried.append(trial_index)
            assignments = _split_trial_indices_round_robin(
                missing, self.num_workers,
            )

        wall_elapsed = time.perf_counter() - wall_started
        self.last_diagnostics = ProbeRunnerDiagnostics(
            k=len(indices),
            wall_seconds=float(wall_elapsed),
            per_worker_seconds=[
                float(seconds_by_device.get(device, 0.0))
                for device in diagnostic_devices
            ],
            per_worker_trial_counts=[
                len(trials) for trials in initial_assignments
            ],
            per_worker_trial_indices=[
                list(trials) for trials in initial_assignments
            ],
            per_worker_trial_seeds=[
                [_trial_seed(base_seed, trial_index) for trial_index in trials]
                for trials in initial_assignments
            ],
            devices=diagnostic_devices,
            pool_generation=int(self.pool_generation),
            retry_count=int(retry_count),
            quarantined_devices=quarantined,
            retried_trial_indices=retried,
        )
        ordered: List[Tuple[float, float, float]] = []
        for trial_index in indices:
            result = results[trial_index]
            if result is None:
                raise RuntimeError(
                    f"probe-runner missing trial {trial_index}"
                )
            ordered.append(result)
        return ordered

    def run_trials_at_indices(
            self,
            *,
            trial_indices: Sequence[int],
            base_seed: int,
            batch_set_key: str = "F1",
            ) -> List[Tuple[float, float, float]]:
        """Run exact installed-action trial indices in caller-provided order."""
        self._require_open()
        normalized_batch_set_key = self._require_batch_set(batch_set_key)
        indices = _normalize_trial_indices(trial_indices)
        assignments = _split_trial_indices_round_robin(
            indices, self.num_workers,
        )
        if not indices:
            self.last_diagnostics = ProbeRunnerDiagnostics(
                k=0,
                wall_seconds=0.0,
                per_worker_trial_counts=[0 for _ in self.workers],
                per_worker_trial_indices=[[] for _ in self.workers],
                per_worker_trial_seeds=[[] for _ in self.workers],
                devices=[str(device) for device in self.devices],
                pool_generation=int(self.pool_generation),
            )
            return []
        if self._process_workers:
            return self._run_trial_indices_processes(
                trial_indices=indices,
                base_seed=int(base_seed),
                batch_set_key=normalized_batch_set_key,
            )

        results: Dict[int, Tuple[float, float, float] | None] = {
            trial_index: None for trial_index in indices
        }
        per_worker_seconds = [0.0] * self.num_workers
        errors: List[Tuple[int, BaseException]] = []
        error_lock = threading.Lock()
        result_lock = threading.Lock()

        def task(worker_index: int) -> None:
            worker = self.workers[worker_index]
            started = time.perf_counter()
            try:
                for trial_index in assignments[worker_index]:
                    raw_result = worker.run_trial(
                        trial_index, int(base_seed), normalized_batch_set_key,
                    )
                    with result_lock:
                        self._accept_trial_result(
                            trial_index=trial_index,
                            raw_result=raw_result,
                            expected_indices=assignments[worker_index],
                            results=results,
                        )
            except BaseException as exc:  # noqa: BLE001
                with error_lock:
                    errors.append((worker_index, exc))
            finally:
                per_worker_seconds[worker_index] = (
                    time.perf_counter() - started
                )

        wall_started = time.perf_counter()
        if self.num_workers == 1:
            task(0)
        else:
            threads = [
                threading.Thread(target=task, args=(index,), daemon=True)
                for index in range(self.num_workers)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        wall_elapsed = time.perf_counter() - wall_started
        self.last_diagnostics = ProbeRunnerDiagnostics(
            k=len(indices),
            wall_seconds=float(wall_elapsed),
            per_worker_seconds=[float(value) for value in per_worker_seconds],
            per_worker_trial_counts=[len(trials) for trials in assignments],
            per_worker_trial_indices=[list(trials) for trials in assignments],
            per_worker_trial_seeds=[
                [_trial_seed(int(base_seed), trial_index) for trial_index in trials]
                for trials in assignments
            ],
            devices=[str(device) for device in self.devices],
            pool_generation=int(self.pool_generation),
        )
        if errors:
            worker_index, exc = errors[0]
            raise RuntimeError(
                f"probe-runner worker {worker_index} "
                f"(device {self.workers[worker_index].device}) failed: {exc!r}"
            ) from exc
        ordered: List[Tuple[float, float, float]] = []
        for trial_index in indices:
            result = results[trial_index]
            if result is None:
                raise RuntimeError(
                    f"probe-runner missing trial {trial_index}"
                )
            ordered.append(result)
        return ordered

    def run_trials(
            self,
            k: int,
            base_seed: int,
            batch_set_key: str = "F1",
            ) -> List[Tuple[float, float, float]]:
        """Run trials [0..k-1] through the exact-index implementation."""
        return self.run_trials_at_indices(
            trial_indices=range(max(0, int(k))),
            base_seed=base_seed,
            batch_set_key=batch_set_key,
        )

    def _run_action_trials_once_processes(
            self,
            actions: Sequence[ActionDecodeResult],
            base_seed: int,
            batch_set_key: str,
            ) -> List[Tuple[float, float, float]]:
        k = len(actions)
        results_per_trial: List[Optional[Tuple[float, float, float]]] = [None] * k
        per_worker_seconds: List[float] = [0.0] * self.num_workers
        assignments: List[List[int]] = [[] for _ in self.workers]
        seed_assignments: List[List[int]] = [[] for _ in self.workers]
        for trial_idx in range(k):
            assignments[trial_idx].append(trial_idx)
            seed_assignments[trial_idx].append(_trial_seed(base_seed, trial_idx))

        errors: List[Tuple[int, BaseException]] = []
        submitted: List[Tuple[int, Any]] = []
        wall_started = time.perf_counter()
        for worker_index in range(1, k):
            worker = self._process_workers[worker_index - 1]
            try:
                worker.submit("run_action_trial", {
                    "trial_idx": int(worker_index),
                    "base_seed": int(base_seed),
                    "decoded": actions[worker_index],
                    "batch_set_key": batch_set_key,
                })
                submitted.append((worker_index, worker))
            except BaseException as exc:  # noqa: BLE001
                errors.append((worker_index, exc))

        local_started = time.perf_counter()
        try:
            self.workers[0].install(actions[0])
            results_per_trial[0] = self.workers[0].run_trial(
                0, base_seed, batch_set_key,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append((0, exc))
        finally:
            per_worker_seconds[0] = time.perf_counter() - local_started

        for worker_index, worker in submitted:
            try:
                payload = worker.receive("run_action_trial")
                per_worker_seconds[worker_index] = float(
                    payload.get("wall_seconds", 0.0) or 0.0
                )
                trial_idx = int(payload["trial_idx"])
                if (
                        trial_idx != worker_index
                        or not 0 <= trial_idx < k
                        or results_per_trial[trial_idx] is not None
                ):
                    raise RuntimeError(
                        "probe-runner received duplicate or out-of-range "
                        f"multi-action trial identity {trial_idx}; "
                        f"expected={worker_index}"
                    )
                results_per_trial[trial_idx] = tuple(payload["result"])
            except BaseException as exc:  # noqa: BLE001
                errors.append((worker_index, exc))
        wall_elapsed = time.perf_counter() - wall_started

        self.last_diagnostics = ProbeRunnerDiagnostics(
            k=k,
            wall_seconds=float(wall_elapsed),
            per_worker_seconds=[float(value) for value in per_worker_seconds],
            per_worker_trial_counts=[len(trials) for trials in assignments],
            per_worker_trial_indices=[list(trials) for trials in assignments],
            per_worker_trial_seeds=[list(seeds) for seeds in seed_assignments],
            devices=[str(device) for device in self.devices],
            multi_action=True,
            pool_generation=int(self.pool_generation),
        )
        if errors:
            quarantined = self._handle_process_errors(
                errors,
                "run_action_trial",
            )
            retry_count, retry_quarantined, retried = (
                self._retry_missing_action_trials_processes(
                    actions=actions,
                    base_seed=base_seed,
                    batch_set_key=batch_set_key,
                    results=results_per_trial,
                )
            )
            quarantined.extend(retry_quarantined)
            self.last_diagnostics.pool_generation = int(
                self.pool_generation
            )
            self.last_diagnostics.retry_count = int(retry_count)
            self.last_diagnostics.quarantined_devices = quarantined
            self.last_diagnostics.retried_trial_indices = retried

        ordered: List[Tuple[float, float, float]] = []
        for trial_idx, result in enumerate(results_per_trial):
            if result is None:
                raise RuntimeError(
                    f"probe-runner missing multi-action trial {trial_idx}"
                )
            ordered.append(result)
        return ordered

    def _retry_missing_action_trials_processes(
            self,
            *,
            actions: Sequence[ActionDecodeResult],
            base_seed: int,
            batch_set_key: str,
            results: List[Optional[Tuple[float, float, float]]],
            ) -> Tuple[int, List[str], List[int]]:
        retried = [
            index for index, result in enumerate(results) if result is None
        ]
        retry_rounds = 0
        quarantined: List[str] = []
        while True:
            missing = [
                index for index, result in enumerate(results)
                if result is None
            ]
            if not missing:
                return retry_rounds, quarantined, retried
            retry_rounds += 1
            active_tasks = missing[:self.num_workers]
            errors: List[Tuple[int, BaseException]] = []
            submitted: List[Tuple[int, Any, int]] = []
            for worker_index, worker in enumerate(
                    self._process_workers, start=1,
            ):
                if worker_index >= len(active_tasks):
                    continue
                trial_index = active_tasks[worker_index]
                try:
                    worker.submit("run_action_trial", {
                        "trial_idx": int(trial_index),
                        "base_seed": int(base_seed),
                        "decoded": actions[trial_index],
                        "batch_set_key": batch_set_key,
                    })
                    submitted.append(
                        (worker_index, worker, trial_index)
                    )
                except BaseException as exc:  # noqa: BLE001
                    errors.append((worker_index, exc))

            local_trial = active_tasks[0]
            try:
                self.workers[0].install(actions[local_trial])
                results[local_trial] = self.workers[0].run_trial(
                    local_trial,
                    base_seed,
                    batch_set_key,
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append((0, exc))

            for worker_index, worker, expected_trial in submitted:
                try:
                    payload = worker.receive("run_action_trial")
                    trial_index = int(payload["trial_idx"])
                    if (
                            trial_index != expected_trial
                            or results[trial_index] is not None
                    ):
                        raise RuntimeError(
                            "probe-runner received duplicate or out-of-range "
                            "retried multi-action trial identity "
                            f"{trial_index}; expected={expected_trial}"
                        )
                    results[trial_index] = tuple(payload["result"])
                except BaseException as exc:  # noqa: BLE001
                    errors.append((worker_index, exc))
            if errors:
                quarantined.extend(
                    self._handle_process_errors(
                        errors,
                        "run_action_trial",
                    )
                )
                continue

    def run_action_trials_once(
            self,
            decoded_by_trial: Sequence[ActionDecodeResult],
            base_seed: int,
            batch_set_key: str = "F1",
            ) -> List[Tuple[float, float, float]]:
        """Run one independent trial for each distinct decoded action.

        This is the fast online RL path: instead of evaluating one action with
        K repeated trials, a rollout batch can hand us up to N completed actions
        and each GPU evaluates one of them. Results are returned in the same
        order as ``decoded_by_trial``.
        """
        self._require_open()
        normalized_batch_set_key = self._require_batch_set(batch_set_key)
        actions = list(decoded_by_trial)
        k = len(actions)
        if k == 0:
            self.last_diagnostics = ProbeRunnerDiagnostics(
                k=0, wall_seconds=0.0,
                per_worker_trial_indices=[[] for _ in self.workers],
                per_worker_trial_seeds=[[] for _ in self.workers],
                devices=[str(d) for d in self.devices],
                multi_action=True,
                pool_generation=int(self.pool_generation),
            )
            return []
        if k > len(self.workers):
            raise ValueError(
                f"run_action_trials_once received {k} actions for "
                f"{len(self.workers)} workers"
            )

        if self._process_workers:
            return self._run_action_trials_once_processes(
                actions, base_seed, normalized_batch_set_key,
            )

        results_per_trial: List[Optional[Tuple[float, float, float]]] = [None] * k
        per_worker_seconds: List[float] = [0.0] * len(self.workers)
        assignments: List[List[int]] = [[] for _ in self.workers]
        seed_assignments: List[List[int]] = [[] for _ in self.workers]
        for idx in range(k):
            assignments[idx].append(idx)
            seed_assignments[idx].append(_trial_seed(base_seed, idx))

        errors: List[Tuple[int, BaseException]] = []
        lock = threading.Lock()

        def task(w_idx: int) -> None:
            if w_idx >= k:
                return
            worker = self.workers[w_idx]
            t0 = time.perf_counter()
            try:


                decoded = actions[w_idx]
                worker.install(decoded)
                res = worker.run_trial(
                    w_idx, base_seed, normalized_batch_set_key,
                )
                results_per_trial[w_idx] = res
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append((w_idx, exc))
            finally:
                per_worker_seconds[w_idx] = time.perf_counter() - t0

        wall_t0 = time.perf_counter()
        if len(self.workers) == 1:
            task(0)
        else:
            threads = [
                threading.Thread(target=task, args=(i,), daemon=True)
                for i in range(k)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        wall_elapsed = time.perf_counter() - wall_t0

        self.last_diagnostics = ProbeRunnerDiagnostics(
            k=k,
            wall_seconds=float(wall_elapsed),
            per_worker_seconds=[float(x) for x in per_worker_seconds],
            per_worker_trial_counts=[len(a) for a in assignments],
            per_worker_trial_indices=[list(a) for a in assignments],
            per_worker_trial_seeds=[list(a) for a in seed_assignments],
            devices=[str(d) for d in self.devices],
            multi_action=True,
            pool_generation=int(self.pool_generation),
        )

        if errors:
            w_idx, exc = errors[0]
            raise RuntimeError(
                f"probe-runner multi-action worker {w_idx} "
                f"(device {self.workers[w_idx].device}) failed: {exc!r}"
            ) from exc

        ordered: List[Tuple[float, float, float]] = []
        for ti in range(k):
            result = results_per_trial[ti]
            if result is None:
                raise RuntimeError(
                    f"probe-runner missing multi-action trial {ti}"
                )
            ordered.append(result)
        return ordered

    @staticmethod
    def _accept_grouped_result(
            *,
            action_index: int,
            trial_index: int,
            raw_result: Sequence[float],
            expected_tasks: Sequence[Tuple[int, int]],
            results: List[
                List[Tuple[float, float, float] | None]
            ],
            position_by_trial: Dict[int, int],
            ) -> None:
        task = (int(action_index), int(trial_index))
        expected = {
            (int(expected_action), int(expected_trial))
            for expected_action, expected_trial in expected_tasks
        }
        if (
                task not in expected
                or not 0 <= task[0] < len(results)
                or task[1] not in position_by_trial
                or results[task[0]][position_by_trial[task[1]]] is not None
        ):
            raise RuntimeError(
                "probe-runner received duplicate or out-of-range grouped "
                f"task {task}; expected={sorted(expected)}"
            )
        if len(raw_result) != 3:
            raise RuntimeError(
                f"probe-runner grouped task {task} returned "
                f"{len(raw_result)} metrics"
            )
        results[task[0]][position_by_trial[task[1]]] = (
            _normalize_probe_trial_result(raw_result)
        )

    @classmethod
    def _accept_grouped_payload(
            cls,
            *,
            payload: Dict[str, Any],
            expected_tasks: Sequence[Tuple[int, int]],
            results: List[
                List[Tuple[float, float, float] | None]
            ],
            position_by_trial: Dict[int, int],
            ) -> None:
        for raw_action_index, raw_trial_index, raw_result in (
                payload.get("results", [])
        ):
            cls._accept_grouped_result(
                action_index=int(raw_action_index),
                trial_index=int(raw_trial_index),
                raw_result=raw_result,
                expected_tasks=expected_tasks,
                results=results,
                position_by_trial=position_by_trial,
            )

    def _retry_missing_grouped_processes(
            self,
            *,
            actions: Sequence[ActionDecodeResult],
            base_seeds: Sequence[int],
            trial_indices: Sequence[int],
            batch_set_key: str,
            results: List[
                List[Optional[Tuple[float, float, float]]]
            ],
            ) -> Tuple[int, List[str], List[Tuple[int, int]]]:
        indices = _normalize_trial_indices(trial_indices)
        position_by_trial = {
            trial_index: position
            for position, trial_index in enumerate(indices)
        }

        def missing_tasks() -> List[Tuple[int, int]]:
            return [
                (action_index, trial_index)
                for action_index in range(len(actions))
                for trial_index in indices
                if results[action_index][position_by_trial[trial_index]]
                is None
            ]

        retried = missing_tasks()
        retry_rounds = 0
        quarantined: List[str] = []
        while True:
            missing = missing_tasks()
            if not missing:
                return retry_rounds, quarantined, retried
            retry_rounds += 1
            assignments: List[List[Tuple[int, int]]] = [
                [] for _ in range(self.num_workers)
            ]
            for position, task in enumerate(missing):
                assignments[position % self.num_workers].append(task)

            errors: List[Tuple[int, BaseException]] = []
            submitted: List[Tuple[int, Any]] = []
            for worker_index, worker in enumerate(
                    self._process_workers, start=1,
            ):
                tasks = assignments[worker_index]
                if not tasks:
                    continue
                try:
                    worker.submit("run_action_trial_groups", {
                        "action_groups": _group_action_trial_tasks(
                            tasks, actions, base_seeds,
                        ),
                        "batch_set_key": batch_set_key,
                    })
                    submitted.append((worker_index, worker))
                except BaseException as exc:  # noqa: BLE001
                    errors.append((worker_index, exc))

            try:
                for group in _group_action_trial_tasks(
                        assignments[0], actions, base_seeds,
                ):
                    action_index = int(group["action_index"])
                    self.workers[0].install(group["decoded"])
                    for trial_index in group["trial_indices"]:
                        self._accept_grouped_result(
                            action_index=action_index,
                            trial_index=int(trial_index),
                            raw_result=self.workers[0].run_trial(
                                int(trial_index),
                                int(group["base_seed"]),
                                batch_set_key,
                            ),
                            expected_tasks=assignments[0],
                            results=results,
                            position_by_trial=position_by_trial,
                        )
            except BaseException as exc:  # noqa: BLE001
                errors.append((0, exc))

            for worker_index, worker in submitted:
                try:
                    def accept_result(
                            payload: Dict[str, Any],
                            worker_index: int = worker_index,
                    ) -> None:
                        self._accept_grouped_result(
                            action_index=int(payload["action_index"]),
                            trial_index=int(payload["trial_index"]),
                            raw_result=payload["result"],
                            expected_tasks=assignments[worker_index],
                            results=results,
                            position_by_trial=position_by_trial,
                        )

                    payload = worker.receive(
                        "run_action_trial_groups",
                        result_handler=accept_result,
                    )
                    self._accept_grouped_payload(
                        payload=payload,
                        expected_tasks=assignments[worker_index],
                        results=results,
                        position_by_trial=position_by_trial,
                    )
                except BaseException as exc:  # noqa: BLE001
                    errors.append((worker_index, exc))
            if errors:
                quarantined.extend(
                    self._handle_process_errors(
                        errors, "run_action_trial_groups",
                    )
                )
                continue
            omitted = missing_tasks()
            if omitted:
                raise RuntimeError(
                    "probe-runner grouped retry completed without errors but "
                    f"omitted tasks {omitted}"
                )

    def _run_action_trial_groups_processes(
            self,
            actions: Sequence[ActionDecodeResult],
            base_seeds: Sequence[int],
            trial_indices: Sequence[int],
            batch_set_key: str,
            ) -> List[List[Tuple[float, float, float]]]:
        action_count = len(actions)
        indices = _normalize_trial_indices(trial_indices)
        position_by_trial = {
            trial_index: position
            for position, trial_index in enumerate(indices)
        }
        assignments = _split_action_trial_index_tasks(
            action_count, indices, self.num_workers,
        )
        results: List[List[Optional[Tuple[float, float, float]]]] = [
            [None] * len(indices) for _ in range(action_count)
        ]
        per_worker_seconds = [0.0] * self.num_workers
        errors: List[Tuple[int, BaseException]] = []
        submitted: List[Tuple[int, Any]] = []

        wall_started = time.perf_counter()
        for worker_index, worker in enumerate(self._process_workers, start=1):
            tasks = assignments[worker_index]
            if not tasks:
                continue
            try:
                worker.submit("run_action_trial_groups", {
                    "action_groups": _group_action_trial_tasks(
                        tasks, actions, base_seeds,
                    ),
                    "batch_set_key": batch_set_key,
                })
                submitted.append((worker_index, worker))
            except BaseException as exc:  # noqa: BLE001
                errors.append((worker_index, exc))

        local_started = time.perf_counter()
        try:
            for group in _group_action_trial_tasks(
                    assignments[0], actions, base_seeds,
            ):
                action_index = int(group["action_index"])
                self.workers[0].install(group["decoded"])
                for trial_index in group["trial_indices"]:
                    self._accept_grouped_result(
                        action_index=action_index,
                        trial_index=int(trial_index),
                        raw_result=self.workers[0].run_trial(
                            int(trial_index),
                            int(group["base_seed"]),
                            batch_set_key,
                        ),
                        expected_tasks=assignments[0],
                        results=results,
                        position_by_trial=position_by_trial,
                    )
        except BaseException as exc:  # noqa: BLE001
            errors.append((0, exc))
        finally:
            per_worker_seconds[0] = time.perf_counter() - local_started

        for worker_index, worker in submitted:
            try:
                def accept_result(
                        payload: Dict[str, Any],
                        worker_index: int = worker_index,
                ) -> None:
                    self._accept_grouped_result(
                        action_index=int(payload["action_index"]),
                        trial_index=int(payload["trial_index"]),
                        raw_result=payload["result"],
                        expected_tasks=assignments[worker_index],
                        results=results,
                        position_by_trial=position_by_trial,
                    )

                payload = worker.receive(
                    "run_action_trial_groups",
                    result_handler=accept_result,
                )
                per_worker_seconds[worker_index] = float(
                    payload.get("wall_seconds", 0.0) or 0.0
                )
                self._accept_grouped_payload(
                    payload=payload,
                    expected_tasks=assignments[worker_index],
                    results=results,
                    position_by_trial=position_by_trial,
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append((worker_index, exc))
        wall_elapsed = time.perf_counter() - wall_started
        self._set_group_diagnostics(
            assignments, base_seeds, indices,
            per_worker_seconds, wall_elapsed,
        )
        if errors:
            quarantined = self._handle_process_errors(
                errors, "run_action_trial_groups",
            )
            retry_count, retry_quarantined, retried = (
                self._retry_missing_grouped_processes(
                    actions=actions,
                    base_seeds=base_seeds,
                    trial_indices=indices,
                    batch_set_key=batch_set_key,
                    results=results,
                )
            )
            quarantined.extend(retry_quarantined)
            self.last_diagnostics.pool_generation = int(
                self.pool_generation
            )
            self.last_diagnostics.retry_count = int(retry_count)
            self.last_diagnostics.quarantined_devices = quarantined
            self.last_diagnostics.retried_action_trial_indices = retried
        return self._ordered_group_results(results, assignments)

    def _set_group_diagnostics(
            self,
            assignments: Sequence[Sequence[Tuple[int, int]]],
            base_seeds: Sequence[int],
            trial_indices: Sequence[int],
            per_worker_seconds: Sequence[float],
            wall_seconds: float,
            ) -> None:
        indices = _normalize_trial_indices(trial_indices)
        position_by_trial = {
            trial_index: position
            for position, trial_index in enumerate(indices)
        }
        trials_per_action = len(indices)
        self.last_diagnostics = ProbeRunnerDiagnostics(
            k=len(base_seeds) * trials_per_action,
            wall_seconds=float(wall_seconds),
            per_worker_seconds=[
                float(value) for value in per_worker_seconds
            ],
            per_worker_trial_counts=[
                len(tasks) for tasks in assignments
            ],
            per_worker_trial_indices=[
                [
                    int(action_index) * trials_per_action
                    + position_by_trial[int(trial_index)]
                    for action_index, trial_index in tasks
                ]
                for tasks in assignments
            ],
            per_worker_trial_seeds=[
                [
                    _trial_seed(base_seeds[int(action_index)], int(trial_index))
                    for action_index, trial_index in tasks
                ]
                for tasks in assignments
            ],
            devices=[str(device) for device in self.devices],
            multi_action=True,
            action_count=len(base_seeds),
            trials_per_action=trials_per_action,
            per_worker_action_trial_indices=[
                [(int(action_index), int(trial_index))
                 for action_index, trial_index in tasks]
                for tasks in assignments
            ],
            pool_generation=int(self.pool_generation),
        )

    @staticmethod
    def _ordered_group_results(
            results: Sequence[
                Sequence[Optional[Tuple[float, float, float]]]
            ],
            assignments: Sequence[Sequence[Tuple[int, int]]],
            ) -> List[List[Tuple[float, float, float]]]:
        ordered: List[List[Tuple[float, float, float]]] = []
        for action_index, action_results in enumerate(results):
            ordered_action: List[Tuple[float, float, float]] = []
            for trial_index, result in enumerate(action_results):
                if result is None:
                    raise RuntimeError(
                        "probe-runner missing grouped task "
                        f"({action_index}, {trial_index}); "
                        f"assignments={assignments}"
                    )
                ordered_action.append(result)
            ordered.append(ordered_action)
        return ordered

    def run_action_trial_groups(
            self,
            decoded_by_action: Sequence[ActionDecodeResult],
            *,
            base_seeds: Sequence[int],
            k: int,
            batch_set_key: str = "F1",
            ) -> List[List[Tuple[float, float, float]]]:
        """Run K exact seeded trials for each action and preserve both orders."""
        return self.run_action_trial_groups_at_indices(
            decoded_by_action,
            base_seeds=base_seeds,
            trial_indices=range(max(0, int(k))),
            batch_set_key=batch_set_key,
        )

    def run_action_trial_groups_at_indices(
            self,
            decoded_by_action: Sequence[ActionDecodeResult],
            *,
            base_seeds: Sequence[int],
            trial_indices: Sequence[int],
            batch_set_key: str = "F1",
            ) -> List[List[Tuple[float, float, float]]]:
        """Run exact seeded trial indices for each action in the given order."""
        self._require_open()
        normalized_batch_set_key = self._require_batch_set(batch_set_key)
        actions = list(decoded_by_action)
        seeds = [int(seed) for seed in base_seeds]
        if len(actions) != len(seeds):
            raise ValueError(
                "run_action_trial_groups_at_indices requires one base seed "
                "per action"
            )
        indices = _normalize_trial_indices(trial_indices)
        position_by_trial = {
            trial_index: position
            for position, trial_index in enumerate(indices)
        }
        if not actions or not indices:
            assignments = _split_action_trial_index_tasks(
                len(actions), indices, self.num_workers,
            )
            self._set_group_diagnostics(
                assignments, seeds, indices,
                [0.0] * self.num_workers, 0.0,
            )
            return [[] for _ in actions]
        if self._process_workers:
            return self._run_action_trial_groups_processes(
                actions, seeds, indices, normalized_batch_set_key,
            )

        assignments = _split_action_trial_index_tasks(
            len(actions), indices, self.num_workers,
        )
        results: List[List[Optional[Tuple[float, float, float]]]] = [
            [None] * len(indices) for _ in actions
        ]
        per_worker_seconds = [0.0] * self.num_workers
        errors: List[Tuple[int, BaseException]] = []
        lock = threading.Lock()

        def task(worker_index: int) -> None:
            worker = self.workers[worker_index]
            started = time.perf_counter()
            try:
                for group in _group_action_trial_tasks(
                        assignments[worker_index], actions, seeds,
                ):
                    action_index = int(group["action_index"])
                    worker.install(group["decoded"])
                    for trial_index in group["trial_indices"]:
                        results[action_index][
                            position_by_trial[int(trial_index)]
                        ] = (
                            worker.run_trial(
                                int(trial_index),
                                int(group["base_seed"]),
                                normalized_batch_set_key,
                            )
                        )
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append((worker_index, exc))
            finally:
                per_worker_seconds[worker_index] = (
                    time.perf_counter() - started
                )

        wall_started = time.perf_counter()
        if self.num_workers == 1:
            task(0)
        else:
            threads = [
                threading.Thread(target=task, args=(index,), daemon=True)
                for index in range(self.num_workers)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        wall_elapsed = time.perf_counter() - wall_started
        self._set_group_diagnostics(
            assignments, seeds, indices,
            per_worker_seconds, wall_elapsed,
        )
        if errors:
            worker_index, exc = errors[0]
            raise RuntimeError(
                f"probe-runner grouped worker {worker_index} "
                f"(device {self.workers[worker_index].device}) failed: {exc!r}"
            ) from exc
        return self._ordered_group_results(results, assignments)


class ProbeRunnerView:
    """Non-owning fidelity view over one shared ProbeRunner."""

    def __init__(self, owner: ProbeRunner, batch_set_key: str):
        self._owner = owner
        self.batch_set_key = _normalize_batch_set_key(batch_set_key)

    @property
    def pool_id(self) -> str:
        return self._owner.pool_id

    @property
    def num_workers(self) -> int:
        return self._owner.num_workers

    @property
    def devices(self) -> List[torch.device]:
        return self._owner.devices

    @property
    def backend(self) -> str:
        return self._owner.backend

    @property
    def pool_generation(self) -> int:
        return self._owner.pool_generation

    @property
    def quarantine_events(self) -> Tuple[Dict[str, Any], ...]:
        return self._owner.quarantine_events

    @property
    def last_diagnostics(self) -> Optional[ProbeRunnerDiagnostics]:
        return self._owner.last_diagnostics

    def install_action(self, decoded: ActionDecodeResult) -> None:
        self._owner.install_action(decoded)

    def clear(self) -> None:
        self._owner.clear()

    def run_trials(
            self, k: int, base_seed: int,
            ) -> List[Tuple[float, float, float]]:
        return self._owner.run_trials(
            k, base_seed, batch_set_key=self.batch_set_key,
        )

    def run_trials_at_indices(
            self,
            *,
            trial_indices: Sequence[int],
            base_seed: int,
            ) -> List[Tuple[float, float, float]]:
        return self._owner.run_trials_at_indices(
            trial_indices=trial_indices,
            base_seed=base_seed,
            batch_set_key=self.batch_set_key,
        )

    def run_action_trials_once(
            self,
            decoded_by_trial: Sequence[ActionDecodeResult],
            base_seed: int,
            ) -> List[Tuple[float, float, float]]:
        return self._owner.run_action_trials_once(
            decoded_by_trial,
            base_seed,
            batch_set_key=self.batch_set_key,
        )

    def run_action_trial_groups(
            self,
            decoded_by_action: Sequence[ActionDecodeResult],
            *,
            base_seeds: Sequence[int],
            k: int,
            ) -> List[List[Tuple[float, float, float]]]:
        return self._owner.run_action_trial_groups(
            decoded_by_action,
            base_seeds=base_seeds,
            k=k,
            batch_set_key=self.batch_set_key,
        )

    def run_action_trial_groups_at_indices(
            self,
            decoded_by_action: Sequence[ActionDecodeResult],
            *,
            base_seeds: Sequence[int],
            trial_indices: Sequence[int],
            ) -> List[List[Tuple[float, float, float]]]:
        return self._owner.run_action_trial_groups_at_indices(
            decoded_by_action,
            base_seeds=base_seeds,
            trial_indices=trial_indices,
            batch_set_key=self.batch_set_key,
        )

    def close(self) -> None:
        return None


def parse_device_ids(spec: Any) -> List[int]:
    """Parse reward-probe device ids into ``[0, 1]`` style integers.

    The launcher passes ``--blb_v3_reward_devices 0,1`` through Python Fire.
    Fire eagerly parses that value as the tuple ``(0, 1)``; downstream code may
    then preserve the tuple or stringify it to ``"(0, 1)"``. Accept all of
    those forms so the server does not silently fall back to single-GPU mode.
    """
    if spec is None:
        return []

    if isinstance(spec, bool):
        raise ValueError(
            f"invalid device id {spec!r}; expected comma-separated ints"
        )

    if isinstance(spec, int):
        tokens = [spec]
    elif isinstance(spec, (list, tuple)):
        tokens = list(spec)
    else:
        s = str(spec).strip()
        if not s:
            return []
        if (s.startswith("(") and s.endswith(")")) or (
            s.startswith("[") and s.endswith("]")
        ):
            s = s[1:-1].strip()
        tokens = [tok.strip() for tok in s.split(",") if tok.strip()]

    out: List[int] = []
    for tok in tokens:
        if isinstance(tok, bool):
            raise ValueError(
                f"invalid device id {tok!r} in spec {spec!r}; expected ints"
            )
        try:
            out.append(int(tok))
        except ValueError as exc:
            raise ValueError(
                f"invalid device id {tok!r} in spec {spec!r}; expected comma-separated ints"
            ) from exc
    return out


def build_probe_runner(
        *,
        primary_model: nn.Module,
        primary_handler: Any,
        primary_bridge: Any,
        primary_probe_batches: Sequence[Any],
        layers_attribute: str,
        is_regression: bool,
        device_ids: Sequence[int],
        metric_profile: str = "",
        log_fn: Optional[Callable[[str], None]] = None,
        ) -> ProbeRunner:
    """Construct a ProbeRunner with one worker per device id.

    Worker 0 reuses the env's existing model + handler + bridge + probe_batches
    (zero extra GPU allocation, and avoids the "two bridges, one handler" trap
    — only the env's bridge tracks what's installed on the primary model).
    Workers 1+ own a deepcopy of the primary model plus their own handler,
    bridge, and probe batches. By default those replicas live in persistent
    spawn children; the explicit thread fallback keeps them in this process.

    Caller guarantees:
      * ``primary_model`` is the model the env already holds (so worker 0
        sees the same parameters PPO data collection sees).
      * ``primary_bridge`` is ``env.bridge`` (the only bridge wrapping
        ``primary_handler``; reusing it prevents install-tracking corruption).
      * ``primary_probe_batches`` are already on the primary device.
      * ``device_ids[0]`` is the primary device id.

    Raises:
        ValueError: empty device_ids.
        RuntimeError: deepcopy / device move failed for a replica.
    """
    if BLBNoiseRLBridge is None:
        raise RuntimeError(
            "blb_rl_bridge import failed earlier; cannot build ProbeRunner. "
            "Likely cause: function_handler.py failed to import (torch/transformers missing)."
        )
    if not device_ids:
        raise ValueError("build_probe_runner requires at least one device id")

    enable_cuda_reward_probe_fast_math()
    log = log_fn or (lambda _msg: None)

    workers: List[ProbeWorker] = []


    primary_device = torch.device(f"cuda:{int(device_ids[0])}")
    workers.append(ProbeWorker(
        device=primary_device,
        model=primary_model,
        handler=primary_handler,
        bridge=primary_bridge,
        probe_batches=list(primary_probe_batches),
        is_regression=bool(is_regression),
        metric_profile=str(metric_profile),
        role="primary",
    ))
    log(f"[probe-runner] worker 0: {primary_device} (primary, reusing env.bridge)")

    backend = resolve_probe_backend()
    log(f"[probe-runner] backend={backend}")

    if backend == "process" and len(device_ids) >= 2:
        process_workers: List[_ProcessProbeWorker] = []
        context = mp.get_context("spawn")
        try:


            model_template = copy.deepcopy(primary_model).to(torch.device("cpu"))
            model_template.eval()
            probe_batches_cpu = [
                _move_probe_batch_to_device(batch, torch.device("cpu"))
                for batch in primary_probe_batches
            ]
            for device_id in device_ids[1:]:
                device = torch.device(f"cuda:{int(device_id)}")
                parent_connection, child_connection = context.Pipe(duplex=True)
                process = context.Process(
                    target=_probe_process_main,
                    args=(
                        child_connection,
                        int(device_id),
                        model_template,
                        probe_batches_cpu,
                        str(layers_attribute),
                        bool(is_regression),
                        str(metric_profile),
                    ),
                    name=f"blb-probe-{device}",
                    daemon=True,
                )
                process_worker = _ProcessProbeWorker(
                    device=device,
                    connection=parent_connection,
                    process=process,
                )
                process_workers.append(process_worker)
                process.start()
                child_connection.close()

            for worker_index, worker in enumerate(process_workers, start=1):
                worker.wait_until_ready()
                log(
                    f"[probe-runner] worker {worker_index}: {worker.device} "
                    f"(persistent process pid={worker.process.pid})"
                )
        except BaseException as exc:  # noqa: BLE001
            ProbeRunner._close_worker_handles(tuple(process_workers))
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(
                f"failed to start persistent probe processes: {exc!r}"
            ) from exc
        finally:


            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return ProbeRunner(workers, process_workers=process_workers)


    for d in device_ids[1:]:
        device = torch.device(f"cuda:{int(d)}")
        try:
            with torch.cuda.device(device):
                replica = copy.deepcopy(primary_model)
                replica = replica.to(device)
                replica.eval()
                replica_handler = ReversibleLayerHandler(replica)
                replica_bridge = BLBNoiseRLBridge(
                    replica_handler, layers_attribute=layers_attribute,
                )
                replica_batches = [
                    _move_probe_batch_to_device(b, device)
                    for b in primary_probe_batches
                ]
        except Exception as exc:
            raise RuntimeError(
                f"failed to deepcopy primary model onto {device}: {exc!r}"
            ) from exc
        workers.append(ProbeWorker(
            device=device,
            model=replica,
            handler=replica_handler,
            bridge=replica_bridge,
            probe_batches=replica_batches,
            is_regression=bool(is_regression),
            metric_profile=str(metric_profile),
            role="replica",
        ))
        log(f"[probe-runner] worker {len(workers)-1}: {device} (deepcopy replica)")

    return ProbeRunner(workers)


def format_diagnostics_line(diag: ProbeRunnerDiagnostics) -> str:
    """One-line summary suitable for ``pruning_search_log.txt``.

    Example:
        ``[probe-runner] k=4 split=[1, 1, 1, 1] devices=[cuda:0, ...]
          wall=0.42s worker_seconds=[0.41, 0.40] speedup=1.95x``
    """
    if diag.k == 0:
        return "[probe-runner] k=0 (no trials)"
    ws = ", ".join(f"{s:.3f}" for s in diag.per_worker_seconds)
    devs = ", ".join(diag.devices)
    counts = ", ".join(str(n) for n in diag.per_worker_trial_counts)
    trial_map = "; ".join(
        f"{dev}:{idxs}"
        for dev, idxs in zip(diag.devices, diag.per_worker_trial_indices)
    )
    mode = " multi_action=1" if bool(getattr(diag, "multi_action", False)) else ""
    if int(getattr(diag, "action_count", 0) or 0) > 0:
        mode += (
            f" actions={int(diag.action_count)}"
            f" trials_per_action={int(diag.trials_per_action)}"
        )
    if int(getattr(diag, "pool_generation", 0) or 0) > 0:
        mode += (
            f" pool_generation={int(diag.pool_generation)}"
            f" retries={int(diag.retry_count)}"
        )
    return (
        f"[probe-runner]{mode} k={diag.k} split=[{counts}] devices=[{devs}]  "
        f"wall={diag.wall_seconds:.3f}s worker_seconds=[{ws}]  "
        f"speedup={diag.speedup_vs_sequential:.2f}x  trials=[{trial_map}]"
    )


def diagnostics_payload(diag: ProbeRunnerDiagnostics) -> dict:
    """Canonical JSON-ready payload for ProbeRunner diagnostics."""
    payload = {
        "k": int(diag.k),
        "wall_seconds": float(diag.wall_seconds),
        "per_worker_seconds": [float(x) for x in diag.per_worker_seconds],
        "per_worker_trial_counts": [int(x) for x in diag.per_worker_trial_counts],
        "per_worker_trial_indices": [
            list(map(int, x)) for x in diag.per_worker_trial_indices
        ],
        "per_worker_trial_seeds": [
            list(map(int, x)) for x in diag.per_worker_trial_seeds
        ],
        "devices": [str(x) for x in diag.devices],
        "speedup_vs_sequential": float(diag.speedup_vs_sequential),
        "line": format_diagnostics_line(diag),
    }
    if bool(getattr(diag, "multi_action", False)):
        payload["multi_action"] = True
    if int(getattr(diag, "action_count", 0) or 0) > 0:
        payload.update({
            "action_count": int(diag.action_count),
            "trials_per_action": int(diag.trials_per_action),
            "per_worker_action_trial_indices": [
                [[int(action_index), int(trial_index)]
                 for action_index, trial_index in tasks]
                for tasks in diag.per_worker_action_trial_indices
            ],
        })
    if int(getattr(diag, "pool_generation", 0) or 0) > 0:
        payload.update({
            "pool_generation": int(diag.pool_generation),
            "retry_count": int(diag.retry_count),
            "quarantined_devices": [
                str(device) for device in diag.quarantined_devices
            ],
            "retried_trial_indices": [
                int(trial_index)
                for trial_index in diag.retried_trial_indices
            ],
            "retried_action_trial_indices": [
                [int(action_index), int(trial_index)]
                for action_index, trial_index
                in diag.retried_action_trial_indices
            ],
        })
    return payload
