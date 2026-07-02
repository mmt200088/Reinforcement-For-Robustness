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
* **Threaded fan-out.** Workers run their trial subsets in Python threads;
  PyTorch CUDA forwards release the GIL so the cards run concurrently.
  Aggregation happens on the main thread after join.
* **Single-device fallback.** A 1-worker ``ProbeRunner`` is a thin no-op
  wrapper. ``BLBStage2Env`` only constructs one when there are 2+ devices,
  so existing single-GPU runs keep the original codepath bitwise.
* **Determinism.** Trial seed = ``base_seed XOR (trial_idx * 2654435761)``,
  derived once per action from ``(episode/step counter)``. Independent of
  wall clock so repro is feasible. Each worker seeds only its current CUDA
  device; it must not call ``torch.cuda.manual_seed_all`` because that creates
  cross-thread RNG interference on multi-GPU reward probes.
"""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from function_handler import ReversibleLayerHandler

from .action_space import ActionDecodeResult
from .eval_metrics import (
    finalize_probe_trial_metrics as _finalize_probe_trial_metrics_local,
    probe_batch_sample_count as _probe_batch_sample_count_local,
)
# We use BLBNoiseRLBridge for noise install/clear; defer import to avoid the
# heavy chain at module-load time when this file is imported by tests.
try:
    from blb_rl_bridge import BLBNoiseRLBridge
except Exception:  # pragma: no cover — torch-free import path
    BLBNoiseRLBridge = None  # type: ignore


# ---------------------------------------------------------------------------
# Trial / split helpers
# ---------------------------------------------------------------------------

_TRIAL_SEED_MULTIPLIER = 2654435761  # Knuth's multiplicative-hash constant


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
    return int((int(base_seed) ^ (int(trial_idx) * _TRIAL_SEED_MULTIPLIER)) & 0x7FFFFFFFFFFFFFFF)


def _split_round_robin(k: int, n_workers: int) -> List[List[int]]:
    """Return per-worker trial-index lists. Round-robin balances variance.

    Example for k=5, n_workers=2: ``[[0, 2, 4], [1, 3]]``.
    Example for k=5, n_workers=3: ``[[0, 3], [1, 4], [2]]``.
    """
    k = max(0, int(k))
    n = max(1, int(n_workers))
    out: List[List[int]] = [[] for _ in range(n)]
    for i in range(k):
        out[i % n].append(i)
    return out


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


# ---------------------------------------------------------------------------
# Metric compute (kept local to avoid env↔probe_runner circular import)
# ---------------------------------------------------------------------------

def _compute_metrics_on_batch_local(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        is_regression: bool,
        ) -> Tuple[float, float, float]:
    """Mirror of ``blb_stage2_rl.env._compute_metrics_on_batch``.

    Local copy so this module is import-safe even when env.py is mid-load
    (matters at runner boot time when env constructs the probe runner).
    Returns (mean_loss, metric1, metric2) — all floats — using the same
    cross-entropy / accuracy convention as the existing path.
    """
    if is_regression:
        preds = logits.view(-1).float()
        targets = labels.view(-1).float()
        loss = float(torch.nn.functional.mse_loss(preds, targets).item())
        # Pearson r as m1 (clamp to [-1, 1])
        if preds.numel() > 1:
            pm = preds - preds.mean()
            tm = targets - targets.mean()
            denom = float((pm.pow(2).sum() * tm.pow(2).sum()).sqrt().item()) + 1e-12
            r = float((pm * tm).sum().item()) / denom
        else:
            r = 0.0
        return (loss, float(np.clip(r, -1.0, 1.0)), 0.0)
    # Classification: cross-entropy + accuracy.
    loss_t = torch.nn.functional.cross_entropy(
        logits, labels.long(), reduction="mean",
    )
    loss = float(loss_t.item())
    preds = logits.argmax(dim=-1)
    acc = float((preds == labels.long()).float().mean().item())
    return (loss, acc, acc)


def _logits_to_classes_local(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 1:
        return (logits > 0.5).long()
    return logits.argmax(dim=-1)


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
    probe_batches: List[Any]
    is_regression: bool
    metric_profile: str = ""
    role: str = "primary"  # "primary" (worker 0, reuses env model) or "replica"

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
            ) -> Tuple[float, float, float]:
        """Run one trial on this worker. Returns (loss, m1, m2) averaged over probe_batches."""
        with torch.cuda.device(self.device):
            seed = _trial_seed(base_seed, trial_idx)
            # CUDA generator state is per-device; pinning to self.device above
            # ensures these seeds land on the right GPU.
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            losses: List[float] = []
            m1s: List[float] = []
            m2s: List[float] = []
            counts: List[int] = []
            trial_preds: List[np.ndarray] = []
            trial_labels: List[np.ndarray] = []
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.inference_mode():
                    for batch in self.probe_batches:
                        kwargs = {
                            "input_ids": batch.input_ids,
                            "attention_mask": batch.attention_mask,
                            "labels": batch.labels,
                        }
                        tti = getattr(batch, "token_type_ids", None)
                        if tti is not None:
                            kwargs["token_type_ids"] = tti
                        out = self.model(**kwargs)
                        logits = out.logits if hasattr(out, "logits") else out[1]
                        loss, m1, m2 = _compute_metrics_on_batch_local(
                            logits, batch.labels, is_regression=self.is_regression,
                        )
                        losses.append(loss)
                        m1s.append(m1)
                        m2s.append(m2)
                        counts.append(_probe_batch_sample_count_local(batch.labels))
                        if not self.is_regression:
                            trial_preds.append(
                                _logits_to_classes_local(logits.detach()).detach().cpu().numpy()
                            )
                            trial_labels.append(batch.labels.detach().cpu().numpy())
            finally:
                if was_training:
                    self.model.train()

            trial_metrics = _finalize_probe_trial_metrics_local(
                losses,
                m1s,
                m2s,
                counts,
                metric_profile=self.metric_profile,
                is_regression=bool(self.is_regression),
                preds=trial_preds,
                labels=trial_labels,
            )
            if trial_metrics is None:
                return (float("nan"), float("nan"), float("nan"))
            return trial_metrics


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

    def __init__(self, workers: List[ProbeWorker]):
        if not workers:
            raise ValueError("ProbeRunner requires at least one worker")
        self.workers = workers
        self.last_diagnostics: Optional[ProbeRunnerDiagnostics] = None

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    @property
    def devices(self) -> List[torch.device]:
        return [w.device for w in self.workers]

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
        self._for_each_worker(lambda w: w.install(decoded))

    def clear(self) -> None:
        """Reverse install on every worker. Always safe (clear is idempotent)."""
        def clear_one(w: ProbeWorker) -> None:
            try:
                w.clear()
            except Exception:
                # Defensive: clearing should never fail, but if it does we
                # still want to attempt the other workers.
                pass
        self._for_each_worker(clear_one)

    def run_trials(self, k: int, base_seed: int) -> List[Tuple[float, float, float]]:
        """Run trials [0..k-1] in parallel across workers; return in trial order."""
        k = max(0, int(k))
        if k == 0:
            self.last_diagnostics = ProbeRunnerDiagnostics(
                k=0, wall_seconds=0.0,
                per_worker_trial_indices=[[] for _ in self.workers],
                per_worker_trial_seeds=[[] for _ in self.workers],
                devices=[str(d) for d in self.devices],
            )
            return []

        assignments = _split_round_robin(k, len(self.workers))
        seed_assignments = [
            [_trial_seed(base_seed, ti) for ti in trials]
            for trials in assignments
        ]
        results_per_trial: dict = {}
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
                local_results: List[Tuple[int, Tuple[float, float, float]]] = []
                for ti in trials:
                    res = worker.run_trial(ti, base_seed)
                    local_results.append((ti, res))
                with lock:
                    for ti, res in local_results:
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
            if ti not in results_per_trial:
                raise RuntimeError(
                    f"probe-runner missing trial {ti} (assignments={assignments})"
                )
            ordered.append(results_per_trial[ti])
        return ordered

    def run_action_trials_once(
            self,
            decoded_by_trial: Sequence[ActionDecodeResult],
            base_seed: int,
            ) -> List[Tuple[float, float, float]]:
        """Run one independent trial for each distinct decoded action.

        This is the fast online RL path: instead of evaluating one action with
        K repeated trials, a rollout batch can hand us up to N completed actions
        and each GPU evaluates one of them. Results are returned in the same
        order as ``decoded_by_trial``.
        """
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

        results_per_trial: dict = {}
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
                res = worker.run_trial(w_idx, base_seed)
                with lock:
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
            if ti not in results_per_trial:
                raise RuntimeError(
                    f"probe-runner missing multi-action trial {ti}"
                )
            ordered.append(results_per_trial[ti])
        return ordered


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
    Workers 1+ deepcopy the primary model onto their device and construct
    their own handler / bridge / probe_batches copies.

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
        metric_profile=str(metric_profile or ""),
        role="primary",
    ))
    log(f"[probe-runner] worker 0: {primary_device} (primary, reusing env.bridge)")

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
            metric_profile=str(metric_profile or ""),
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
