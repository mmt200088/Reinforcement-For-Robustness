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
  ``BLB_STAGE2_PROBE_BACKEND=thread`` for the legacy in-process fallback.
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

from function_handler import ReversibleLayerHandler, reseed_noise_rng_for_device

from .action_space import ActionDecodeResult
from .inference_eval import run_installed_probe_trial
# We use BLBNoiseRLBridge for noise install/clear; defer import to avoid the
# heavy chain at module-load time when this file is imported by tests.
try:
    from blb_rl_bridge import BLBNoiseRLBridge
except Exception:  # pragma: no cover — torch-free import path
    BLBNoiseRLBridge = None  # type: ignore


_PROCESS_STARTUP_TIMEOUT_SECONDS = 300.0
_PROCESS_COMMAND_TIMEOUT_SECONDS = 3600.0


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


# ---------------------------------------------------------------------------
# Trial / split helpers
# ---------------------------------------------------------------------------

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
    from .seed_utils import derive_probe_trial_seed

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


# ---------------------------------------------------------------------------
# Probe batch transfer (cheap one-time copy per worker)
# ---------------------------------------------------------------------------

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
    # Reconstruct via the original class so downstream code reads identical
    # attributes (handles ProbeBatch as a dataclass / namedtuple / plain obj).
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


# ---------------------------------------------------------------------------
# ProbeWorker — one per device
# ---------------------------------------------------------------------------

@dataclass
class ProbeWorker:
    """Per-GPU state: replicated model + its own handler/bridge/probe_batches."""
    device: torch.device
    model: nn.Module
    handler: Any  # ReversibleLayerHandler
    bridge: Any   # BLBNoiseRLBridge
    probe_batches: Sequence[Any]
    is_regression: bool
    metric_profile: str = ""
    role: str = "primary"  # "primary" (worker 0, reuses env model) or "replica"
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
                result = {
                    "results": [
                        (
                            int(trial_idx),
                            worker.run_trial(
                                int(trial_idx), base_seed, batch_set_key,
                            ),
                        )
                        for trial_idx in payload["trial_indices"]
                    ]
                }
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

    def receive(self, operation: str, timeout: Optional[float] = None) -> dict:
        expected = str(operation)
        if self._pending_operation != expected:
            raise RuntimeError(
                f"probe child {self.device} expected pending {expected!r}, "
                f"got {self._pending_operation!r}"
            )
        try:
            message = self._receive_message(
                _PROCESS_COMMAND_TIMEOUT_SECONDS if timeout is None else timeout
            )
        finally:
            self._pending_operation = None
        if message.get("operation") != expected:
            raise RuntimeError(
                f"probe child {self.device} returned operation "
                f"{message.get('operation')!r}, expected {expected!r}"
            )
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


# ---------------------------------------------------------------------------
# ProbeRunner — distributes k trials across workers
# ---------------------------------------------------------------------------

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
        self._process_finalizer = (
            weakref.finalize(
                self,
                ProbeRunner._close_worker_handles,
                tuple(self._process_workers),
            )
            if self._process_workers else None
        )

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

    def close(self) -> None:
        """Stop persistent replica processes. Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        if self._process_finalizer is not None and self._process_finalizer.alive:
            self._process_finalizer()

    def _for_each_worker(self, fn) -> None:
        """Run a worker-local operation on every worker.

        Install/clear touches separate model replicas and CUDA devices. The old
        serial loop made multi-GPU reward probes pay that setup cost N times
        before the actual parallel forward even began.
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
                worker_index, exc = errors[0]
                self._raise_process_error(worker_index, "install", exc)
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
                # Defensive: clearing should never fail, but if it does we
                # still want to attempt the other workers.
                pass
        self._for_each_worker(clear_one)

    def _run_trials_processes(
            self,
            k: int,
            base_seed: int,
            batch_set_key: str,
            ) -> List[Tuple[float, float, float]]:
        assignments = _split_round_robin_cached(k, self.num_workers)
        seed_assignments = [
            [_trial_seed(base_seed, trial_idx) for trial_idx in trials]
            for trials in assignments
        ]
        results_per_trial: List[Optional[Tuple[float, float, float]]] = [None] * k
        per_worker_seconds: List[float] = [0.0] * self.num_workers
        errors: List[Tuple[int, BaseException]] = []
        submitted: List[Tuple[int, Any]] = []

        wall_started = time.perf_counter()
        for worker_index, worker in enumerate(self._process_workers, start=1):
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

        local_started = time.perf_counter()
        try:
            for trial_idx in assignments[0]:
                results_per_trial[trial_idx] = self.workers[0].run_trial(
                    trial_idx, base_seed, batch_set_key,
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append((0, exc))
        finally:
            per_worker_seconds[0] = time.perf_counter() - local_started

        for worker_index, worker in submitted:
            try:
                payload = worker.receive("run_trials")
                per_worker_seconds[worker_index] = float(
                    payload.get("wall_seconds", 0.0) or 0.0
                )
                for trial_idx, result in payload.get("results", []):
                    results_per_trial[int(trial_idx)] = tuple(result)
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
        )
        if errors:
            worker_index, exc = errors[0]
            self._raise_process_error(worker_index, "run_trials", exc)

        ordered: List[Tuple[float, float, float]] = []
        for trial_idx, result in enumerate(results_per_trial):
            if result is None:
                raise RuntimeError(
                    f"probe-runner missing trial {trial_idx} "
                    f"(assignments={assignments})"
                )
            ordered.append(result)
        return ordered

    def run_trials(
            self,
            k: int,
            base_seed: int,
            batch_set_key: str = "F1",
            ) -> List[Tuple[float, float, float]]:
        """Run trials [0..k-1] in parallel across workers; return in trial order."""
        self._require_open()
        normalized_batch_set_key = self._require_batch_set(batch_set_key)
        k = max(0, int(k))
        if k == 0:
            self.last_diagnostics = ProbeRunnerDiagnostics(
                k=0, wall_seconds=0.0,
                per_worker_trial_indices=[[] for _ in self.workers],
                per_worker_trial_seeds=[[] for _ in self.workers],
                devices=[str(d) for d in self.devices],
            )
            return []

        if self._process_workers:
            return self._run_trials_processes(
                k, base_seed, normalized_batch_set_key,
            )

        assignments = _split_round_robin_cached(k, len(self.workers))
        seed_assignments = [
            [_trial_seed(base_seed, ti) for ti in trials]
            for trials in assignments
        ]
        results_per_trial: List[Optional[Tuple[float, float, float]]] = [None] * k
        per_worker_seconds: List[float] = [0.0] * len(self.workers)
        errors: List[Tuple[int, BaseException]] = []
        lock = threading.Lock()

        def task(w_idx: int) -> None:
            trials = assignments[w_idx]
            if not trials:
                return
            worker = self.workers[w_idx]
            t0 = time.perf_counter()
            try:
                for ti in trials:
                    res = worker.run_trial(
                        ti, base_seed, normalized_batch_set_key,
                    )
                    results_per_trial[ti] = res
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append((w_idx, exc))
            finally:
                per_worker_seconds[w_idx] = time.perf_counter() - t0

        wall_t0 = time.perf_counter()
        if len(self.workers) == 1:
            # Single-worker mode: no thread overhead, just call directly.
            task(0)
        else:
            threads = [
                threading.Thread(target=task, args=(i,), daemon=True)
                for i in range(len(self.workers))
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
        )

        if errors:
            w_idx, exc = errors[0]
            raise RuntimeError(
                f"probe-runner worker {w_idx} (device {self.workers[w_idx].device}) "
                f"failed: {exc!r}"
            ) from exc

        ordered: List[Tuple[float, float, float]] = []
        for ti in range(k):
            result = results_per_trial[ti]
            if result is None:
                raise RuntimeError(
                    f"probe-runner missing trial {ti} (assignments={assignments})"
                )
            ordered.append(result)
        return ordered

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
        )
        if errors:
            worker_index, exc = errors[0]
            self._raise_process_error(worker_index, "run_action_trial", exc)

        ordered: List[Tuple[float, float, float]] = []
        for trial_idx, result in enumerate(results_per_trial):
            if result is None:
                raise RuntimeError(
                    f"probe-runner missing multi-action trial {trial_idx}"
                )
            ordered.append(result)
        return ordered

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
                # Each worker installs the decoded cfg for its own action, then
                # runs exactly one seeded trial for that action.
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

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Factory: build_probe_runner
# ---------------------------------------------------------------------------

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
        primary_handler: Any,                 # ReversibleLayerHandler
        primary_bridge: Any,                  # BLBNoiseRLBridge owned by the env
        primary_probe_batches: Sequence[Any], # List[ProbeBatch]
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

    # ---- worker 0: reuse env's existing primary model + handler + bridge ----
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
            # Children must not inherit an initialized CUDA context. Build one
            # CPU template and let torch multiprocessing share its storages
            # while each spawn child moves its own copy to its assigned GPU.
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
            # Release the temporary GPU allocation created by deepcopy before
            # it was moved to CPU. This does not affect the primary model.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return ProbeRunner(workers, process_workers=process_workers)

    # ---- workers 1+: deepcopy the primary model onto each extra device ----
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


# ---------------------------------------------------------------------------
# Diagnostic helpers (for the env to format the speedup log line)
# ---------------------------------------------------------------------------

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
    return payload
