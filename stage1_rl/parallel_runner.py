"""N-GPU parallel rollout runner for Stage-1 RL.

Stage-2 BLB has ``blb_stage2_rl/probe_runner.py``: K reward trials for one
action run concurrently across N GPUs. Stage-1 RL's expensive unit is
different — every episode ends with one BERT forward over the proxy split.
So Stage-1's parallelism is **data-parallel rollout collection**: each of
the N GPUs runs ``episodes_per_worker`` complete episodes (per-layer policy
decisions + final eval) independently. After all workers finish, the
per-episode rollouts merge into the central ``RecurrentRolloutBuffer`` and
one PPO update happens on the shared Stage-1 policy.

Design (mirrors Stage-2 conventions where they fit):

* **One PPO learner, one action stream.** Caller owns the Stage-1 policy on
  the primary device. Worker threads transfer state tensors to that device
  for action sampling under a lock; the policy forward is tiny so lock
  serialization is negligible.
* **Worker 0 reuses the evaluator's primary model.** No extra GPU memory.
* **Workers 1..N-1 deepcopy the BERT model onto their device** plus their
  own ``ReversibleLayerHandler`` for GELU/Softmax replacement. Each worker
  owns its own ``TransformerOptEnv`` instance so episode-local state never
  collides across workers.
* **Threaded fan-out.** BERT forward releases the GIL, so a single Python
  process with N threads saturates N GPUs. No multiprocessing overhead.
* **GPU-count-independent seeding.** ``derive_episode_seed(base, window, g)``
  is keyed on the GLOBAL episode index ``g`` (not on the worker), and rollouts
  are returned in global order, so a window runs the identical seeded episodes
  in the identical order for any GPU count — results don't change with #GPUs.
* **Single-device path.** A 1-worker runner drives the same global-seeded path,
  so ``--stage1-rl-devices 0`` on a 1-GPU box reproduces what 4 GPUs produce.
  Build the runner whenever ``len(device_ids) >= 1``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .seed_utils import assign_global_episodes, derive_episode_seed

# Defer the heavy import so this module is importable in torch-free CI without
# pulling transformers / function_handler at module load.
try:
    from function_handler import ReversibleLayerHandler  # noqa: F401  (worker uses it via factory)
    _HANDLER_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover — only matters on import-broken envs
    ReversibleLayerHandler = None  # type: ignore[assignment]
    _HANDLER_IMPORT_ERROR = _exc


# Seed derivation + global-episode assignment live in the torch-free
# ``seed_utils`` module (imported above) so the GPU-count-independence contract
# is unit-testable without torch. See ``stage1_rl/seed_utils.py``.


# ---------------------------------------------------------------------------
# Device-id parsing (cloned from Stage-2's probe_runner for consistency)
# ---------------------------------------------------------------------------

def parse_device_ids(spec: Any) -> List[int]:
    """Parse rollout device ids into a clean ``[0, 1, 2, 3]`` integer list.

    Accepts None, int, list/tuple, or comma-separated string. Mirrors
    ``blb_stage2_rl.probe_runner.parse_device_ids`` so the launcher's
    ``--stage1-rl-devices`` flag behaves the same as
    ``--blb-v3-reward-devices``.
    """
    if spec is None:
        return []
    if isinstance(spec, bool):
        raise ValueError(f"invalid device id {spec!r}; expected ints")
    if isinstance(spec, int):
        tokens: List[Any] = [spec]
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
            raise ValueError(f"invalid device id {tok!r} in spec {spec!r}; expected ints")
        try:
            out.append(int(tok))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid device id {tok!r} in spec {spec!r}; expected comma-separated ints"
            ) from exc
    return out


# ---------------------------------------------------------------------------
# Per-episode rollout container (mirrors RecurrentRolloutBuffer's per-episode dict)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRollout:
    """One episode's recorded transitions + summary metrics.

    Field names match ``RecurrentRolloutBuffer._current`` so the caller can
    splat directly via ``buffer.episodes.append(rollout.to_buffer_dict())``.

    Recorded fields stay on CPU; the Stage-1 PPO update path moves the whole
    buffer to the target device via ``RecurrentRolloutBuffer.get_batch``.
    """
    cont_features: List[np.ndarray]
    layer_indices: List[int]
    prev_g_actions: List[int]
    actions_g: List[int]
    logprobs: List[float]
    rewards: List[float]
    values: List[float]
    dones: List[float]
    gelu_masks: List[np.ndarray]
    # Per-episode summary used by the central loop's bookkeeping
    episode_reward: float
    episode_loss: float
    episode_metric1: float
    episode_metric2: float
    episode_cost: float
    gelu_config: List[int]
    softmax_config: List[int]
    final_config_metrics: Optional[Dict[str, float]] = None
    # Optional per-step info dicts (for the details/ writer, if enabled)
    step_infos: List[Dict[str, Any]] = field(default_factory=list)

    def to_buffer_dict(self) -> Dict[str, Any]:
        """Return a dict shaped like ``RecurrentRolloutBuffer._current``."""
        return {
            "cont_features": list(self.cont_features),
            "layer_indices": list(self.layer_indices),
            "prev_g_actions": list(self.prev_g_actions),
            "actions_g": list(self.actions_g),
            "logprobs": list(self.logprobs),
            "rewards": list(self.rewards),
            "values": list(self.values),
            "dones": list(self.dones),
            "gelu_masks": list(self.gelu_masks),
        }


# ---------------------------------------------------------------------------
# Stage1RolloutWorker — one per device
# ---------------------------------------------------------------------------

@dataclass
class Stage1RolloutWorker:
    """Per-GPU rollout state.

    Holds the BERT replica + handler + a private ``TransformerOptEnv``; the
    env's evaluator wrapper delegates its single expensive call
    (``evaluate_model``) into this worker's replica via
    ``evaluator._stage1_evaluate_on_model(...)``. All other env state lives
    on the worker itself, so two workers running concurrently never share
    mutable env state.
    """
    worker_idx: int
    device: torch.device
    model: nn.Module
    handler: Any                      # ReversibleLayerHandler
    evaluator: Any                    # primary LayerImportanceEvaluator (read-only helpers)
    env: Any                          # TransformerOptEnv with the per-worker eval wrapper
    eval_split_name: str              # which dataloader split (e.g. "train" / proxy)
    role: str = "primary"             # "primary" (worker 0) | "replica"
    gtrxl_replica: Optional[nn.Module] = None  # per-worker eval policy on .device


# ---------------------------------------------------------------------------
# Stage1ParallelRunnerDiagnostics
# ---------------------------------------------------------------------------

@dataclass
class Stage1ParallelRunnerDiagnostics:
    """Per-window timing snapshot, captured at the end of ``run_window``."""
    window_idx: int = 0
    episodes_per_worker: int = 0
    wall_seconds: float = 0.0
    per_worker_seconds: List[float] = field(default_factory=list)
    per_worker_episode_counts: List[int] = field(default_factory=list)
    devices: List[str] = field(default_factory=list)

    @property
    def speedup_vs_sequential(self) -> float:
        if not self.per_worker_seconds or self.wall_seconds <= 0:
            return 1.0
        return float(sum(self.per_worker_seconds)) / float(self.wall_seconds)


# ---------------------------------------------------------------------------
# Stage1ParallelRunner — orchestrates the worker threads
# ---------------------------------------------------------------------------

class Stage1ParallelRunner:
    """Fan ``episodes_per_worker`` rollouts across N worker threads per PPO window."""

    def __init__(
            self,
            workers: List[Stage1RolloutWorker],
            primary_device: torch.device,
            collect_episode_fn: Callable[..., EpisodeRollout],
            log_fn: Optional[Callable[[str], None]] = None,
    ):
        if not workers:
            raise ValueError("Stage1ParallelRunner requires at least one worker")
        self.workers = workers
        self.primary_device = torch.device(primary_device)
        # The episode-collection routine is supplied by the caller (lives in
        # layer_importance_evaluator.py so it can read all of the existing
        # private state — RL_OPT_FLAGS, _write_step_info, etc. — without
        # circular import.) Signature: collect_episode_fn(worker, episode_seed).
        self._collect_episode = collect_episode_fn
        self._log = log_fn or (lambda _msg: None)
        self.last_diagnostics: Optional[Stage1ParallelRunnerDiagnostics] = None

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    def _sync_policy_replicas(self, gtrxl_net: nn.Module) -> None:
        """Give every worker its own eval-mode copy of the central policy on its
        own device (built once, then weight-synced each window).

        The policy is frozen during a window's collection (PPO updates the
        central net only afterwards), so per-worker replicas remove the old
        shared-policy lock WITHOUT changing what gets sampled: episode ``g``
        draws the same action on any device (identical weights + global-index
        seed + device-independent CUDA Philox). This is the speedup — the 12-step
        GTrXL rollout now runs on each worker's GPU in parallel instead of
        serializing through one lock on ``cuda:0``.
        """
        state_dict = None
        for w in self.workers:
            if w.gtrxl_replica is None:
                if w.device == self.primary_device:
                    replica = copy.deepcopy(gtrxl_net).to(w.device)
                else:
                    with torch.cuda.device(w.device):
                        replica = copy.deepcopy(gtrxl_net).to(w.device)
                w.gtrxl_replica = replica
            else:
                if state_dict is None:
                    state_dict = gtrxl_net.state_dict()
                w.gtrxl_replica.load_state_dict(state_dict)
            w.gtrxl_replica.eval()

    def run_window(
            self,
            *,
            gtrxl_net: nn.Module,
            total_episodes: int,
            window_idx: int,
            base_seed: int,
    ) -> List[EpisodeRollout]:
        """Collect ``total_episodes`` episodes for one PPO window, in parallel.

        Returns rollouts in **global episode order** ``[0 .. total_episodes-1]``,
        independent of how many workers ran them: episode ``g`` is always seeded
        by ``derive_episode_seed(base_seed, window_idx, g)`` and stored at index
        ``g``. So filling the central buffer from this list yields identical
        ordering — and identical PPO updates — for any GPU count.
        """
        if total_episodes <= 0:
            return []

        # Build/refresh each worker's own eval-mode policy replica (frozen for
        # this window) so the rollout runs lock-free — one policy per GPU.
        self._sync_policy_replicas(gtrxl_net)

        n_workers = len(self.workers)
        assignments = assign_global_episodes(total_episodes, n_workers)
        results: List[Optional[EpisodeRollout]] = [None] * total_episodes
        per_worker_wall: List[float] = [0.0] * n_workers
        threads: List[threading.Thread] = []
        errors: List[Tuple[int, BaseException]] = []
        errors_lock = threading.Lock()

        def worker_thread(w_idx: int) -> None:
            worker = self.workers[w_idx]
            t0 = time.time()
            try:
                for g in assignments[w_idx]:
                    ep_seed = derive_episode_seed(base_seed, window_idx, g)
                    rollout = self._collect_episode(
                        worker=worker,
                        episode_seed=ep_seed,
                    )
                    results[g] = rollout
            except BaseException as exc:  # noqa: BLE001 — propagate any failure
                with errors_lock:
                    errors.append((w_idx, exc))
            finally:
                per_worker_wall[w_idx] = time.time() - t0

        wall_t0 = time.time()
        for w_idx in range(n_workers):
            t = threading.Thread(
                target=worker_thread,
                args=(w_idx,),
                name=f"stage1-rollout-w{w_idx}",
                daemon=False,
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        wall_seconds = time.time() - wall_t0

        if errors:
            # Re-raise the first error so the caller sees the same traceback
            # path as a single-GPU failure would produce.
            _, exc = errors[0]
            raise exc

        flat: List[EpisodeRollout] = []
        for g, r in enumerate(results):
            if r is None:
                raise RuntimeError(
                    f"internal: no rollout for global episode {g} without raising; "
                    "this should be unreachable"
                )
            flat.append(r)

        self.last_diagnostics = Stage1ParallelRunnerDiagnostics(
            window_idx=window_idx,
            episodes_per_worker=(total_episodes // n_workers if n_workers else 0),
            wall_seconds=wall_seconds,
            per_worker_seconds=list(per_worker_wall),
            per_worker_episode_counts=[len(a) for a in assignments],
            devices=[str(w.device) for w in self.workers],
        )

        # Determinism signature over the window's rollouts in GLOBAL order: a hash
        # of every episode's (actions, rewards). The 1-GPU-vs-N-GPU harness diffs
        # these — identical sequences prove the rollout is GPU-count-independent.
        sig = hashlib.sha1(
            repr([(r.actions_g, r.rewards) for r in flat]).encode("utf-8")
        ).hexdigest()[:16]
        self._log(f"[stage1-rollout] window={window_idx} episodes={len(flat)} rollout_sig={sig}")

        return flat


# ---------------------------------------------------------------------------
# Factory: build_stage1_parallel_runner
# ---------------------------------------------------------------------------

def build_stage1_parallel_runner(
        *,
        primary_model: nn.Module,
        primary_handler: Any,
        evaluator: Any,                     # primary LayerImportanceEvaluator
        env_factory: Callable[..., Any],    # build_env_for_worker(model, handler, device, eval_wrapper)
        collect_episode_fn: Callable[..., EpisodeRollout],
        device_ids: Sequence[int],
        eval_split_name: str,
        log_fn: Optional[Callable[[str], None]] = None,
) -> Stage1ParallelRunner:
    """Construct a Stage1ParallelRunner with one worker per device id.

    Worker 0 reuses the evaluator's existing model + handler (no extra GPU
    allocation). Workers 1..N-1 deepcopy the BERT model onto their device
    and build their own ``ReversibleLayerHandler``.

    Arguments:
        primary_model:      The evaluator's existing BERT model (worker 0 reuses).
        primary_handler:    The evaluator's existing ReversibleLayerHandler
                            wrapping ``primary_model``.
        evaluator:          The primary LayerImportanceEvaluator; workers hold a
                            back-reference so the per-worker eval helper can call
                            ``evaluator._stage1_evaluate_on_model(...)``.
        env_factory:        ``build_env_for_worker(model, handler, device, eval_wrapper)``
                            returning a fresh TransformerOptEnv whose eval wrapper
                            calls into the worker's replica.
        collect_episode_fn: One-episode rollout routine (lives in
                            layer_importance_evaluator.py so it can use the
                            existing per-step state-building / step-info writer
                            helpers without circular import).
        device_ids:         e.g. ``[0, 1, 2, 3]``. ``device_ids[0]`` is the
                            primary device.
        eval_split_name:    Dataloader split used for the per-episode reward.
                            Stage-1 protocol uses ``"validation_full"``.
        log_fn:             Optional log sink; defaults to silent.

    Raises:
        ValueError:   empty device_ids.
        RuntimeError: deepcopy onto a replica device failed.
    """
    if ReversibleLayerHandler is None or _HANDLER_IMPORT_ERROR is not None:
        raise RuntimeError(
            "function_handler import failed; cannot build Stage1ParallelRunner. "
            "Original error: "
            f"{_HANDLER_IMPORT_ERROR!r}"
        )
    ids = list(device_ids)
    if not ids:
        raise ValueError("build_stage1_parallel_runner requires at least one device id")

    log = log_fn or (lambda _msg: None)

    workers: List[Stage1RolloutWorker] = []

    # Build worker 0 — reuses the primary model/handler/evaluator. No extra
    # GPU memory; the env wrapper still routes through the replica path so
    # single-GPU and multi-GPU codepaths agree bit-for-bit.
    primary_device = torch.device(f"cuda:{int(ids[0])}")
    primary_eval_wrapper = _build_per_worker_eval_wrapper(
        evaluator=evaluator,
        model=primary_model,
        handler=primary_handler,
        device=primary_device,
        eval_split_name=eval_split_name,
    )
    primary_env = env_factory(primary_model, primary_handler, primary_device, primary_eval_wrapper)
    workers.append(Stage1RolloutWorker(
        worker_idx=0,
        device=primary_device,
        model=primary_model,
        handler=primary_handler,
        evaluator=evaluator,
        env=primary_env,
        eval_split_name=eval_split_name,
        role="primary",
    ))
    log(f"[stage1-rollout] worker 0: {primary_device} (primary, reusing evaluator.model)")

    # Workers 1..N-1 — deepcopy the model onto each extra device.
    for slot, dev_id in enumerate(ids[1:], start=1):
        device = torch.device(f"cuda:{int(dev_id)}")
        try:
            with torch.cuda.device(device):
                replica = copy.deepcopy(primary_model)
                replica = replica.to(device)
                replica.eval()
                replica_handler = ReversibleLayerHandler(replica)
        except Exception as exc:
            raise RuntimeError(
                f"failed to deepcopy primary model onto {device}: {exc!r}"
            ) from exc
        replica_eval_wrapper = _build_per_worker_eval_wrapper(
            evaluator=evaluator,
            model=replica,
            handler=replica_handler,
            device=device,
            eval_split_name=eval_split_name,
        )
        replica_env = env_factory(replica, replica_handler, device, replica_eval_wrapper)
        workers.append(Stage1RolloutWorker(
            worker_idx=slot,
            device=device,
            model=replica,
            handler=replica_handler,
            evaluator=evaluator,
            env=replica_env,
            eval_split_name=eval_split_name,
            role="replica",
        ))
        log(f"[stage1-rollout] worker {slot}: {device} (deepcopy replica)")

    return Stage1ParallelRunner(
        workers=workers,
        primary_device=primary_device,
        collect_episode_fn=collect_episode_fn,
        log_fn=log,
    )


def _build_per_worker_eval_wrapper(
        *,
        evaluator: Any,
        model: nn.Module,
        handler: Any,
        device: torch.device,
        eval_split_name: str,
) -> Any:
    """Return an object with ``evaluate_model(gelu_arr, softmax_arr) -> tuple``
    that delegates into the worker's replica via the evaluator's stateless
    helper. Matches the single-GPU ``RLEvaluatorWrapper`` contract used by
    ``TransformerOptEnv``."""

    class _WorkerEvalWrapper:
        def evaluate_model(self, gelu_arr: Sequence[int], softmax_arr: Sequence[int]):
            return evaluator._stage1_evaluate_on_model(
                model=model,
                handler=handler,
                device=device,
                gelu_degrees=gelu_arr,
                softmax_degrees=softmax_arr,
                split_name=eval_split_name,
            )

    return _WorkerEvalWrapper()


# ---------------------------------------------------------------------------
# Diagnostics formatter
# ---------------------------------------------------------------------------

def format_diagnostics_line(diag: Stage1ParallelRunnerDiagnostics) -> str:
    """One-line summary suitable for ``pruning_search_log.txt``."""
    if not diag.devices:
        return "[stage1-rollout] (no workers)"
    ws = ", ".join(f"{s:.3f}" for s in diag.per_worker_seconds)
    devs = ", ".join(diag.devices)
    counts = ", ".join(str(n) for n in diag.per_worker_episode_counts)
    return (
        f"[stage1-rollout] window={diag.window_idx} eps_per_worker={diag.episodes_per_worker}  "
        f"devices=[{devs}] counts=[{counts}]  "
        f"wall={diag.wall_seconds:.3f}s worker_seconds=[{ws}]  "
        f"speedup={diag.speedup_vs_sequential:.2f}x"
    )
