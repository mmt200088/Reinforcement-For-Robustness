"""COINN-style Optimization Space Reduction (OSR) for BLB Stage-2 RL.

COINN ("Crypto/ML Codesign for Oblivious Inference via Neural Networks",
Hussain et al., CCS 2021) pre-prunes its mixed-precision search space by
testing each (layer, bitwidth-tuple) candidate in isolation (other layers at
full precision) and removing those that violate the user-set accuracy
constraint. Survivors form a much smaller candidate set for the global
optimization that follows.

Translating to our setting:

* "layer" → ``(block, layer)`` pair (block 3 is held at baseline throughout).
* "bitwidth tuple" → per-slot action-index tuple for that ``(block, layer)``.
* "full precision elsewhere" → all other ``(block, layer)`` slots at the
  ``static_skeletons`` baseline action.
* "accuracy constraint" → the same ``acc_orig - stage2_limit_tolerance``
  threshold the RL reward uses, plus the stability threshold
  ``noisy_baseline_loss_std × (1 + stage2_stability_tolerance)``.
* "follow-on global optimization" → our PPO over the reduced action space.

The per-(block, layer) combination space is astronomical for our blocks
(block 2 alone has ~57 trillion per layer), so full enumeration is
impossible. We use a two-layer hybrid (settled during grilling on
2026-05-27):

  Layer 1 — per-slot exhaustive: every ``(block, layer, slot, level)`` is
            tested with all other slots at baseline. Catches all per-slot
            disasters (invalid_chain / accuracy / stability).
  Layer 2 — per-(block, layer) sampled combinations: ``num_combo_samples``
            random slot combinations per (block, layer) are tested in full.
            Catches the most egregious combination failures.

Results are persisted to JSON with a fingerprint over
``(profile, num_layers, stage1_gelu, stage1_softmax, K, probe_size,
tol_acc, tol_stab, baseline_action_vec_hash, osr_version)``. Loading an OSR
results file with a fingerprint mismatch aborts by default; pass
``allow_fingerprint_mismatch=True`` to override.

At training time the resulting :class:`OSRPrePruneMask` lives alongside the
existing three masks (``StaticInvalidLevelMask``, ``EmpiricalInvalidLevelMask``,
``ForbiddenActionMask``) — it does not replace them. The grilling session
chose this "keep all masks, OSR is a fourth layer" design to keep the
training loop defensive against sampling leaks in Layer 2.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple,
)

import numpy as np


OSR_VERSION = "v1"
DEFAULT_OSR_BLOCKS: Tuple[int, ...] = (1, 2, 4, 5)
"""Blocks scanned by OSR. Block 3 is excluded: per the 2026-05-27 sub-stage
decision, every layer's block-3 action is pinned to the ``static_skeletons``
baseline throughout training, so its action space contributes nothing."""


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def compute_fingerprint(
        *,
        profile: str,
        num_layers: int,
        stage1_gelu_per_layer: Sequence[int],
        stage1_softmax_per_layer: Sequence[int],
        k_trials: int,
        probe_size: int,
        tol_acc: float,
        tol_stab: float,
        baseline_action_vec: Sequence[int],
        osr_version: str = OSR_VERSION,
        ) -> Dict[str, Any]:
    """Return a JSON-serialisable fingerprint dict.

    Any change to the underlying scan inputs flips the fingerprint, so a stale
    cache file is rejected at load time. ``baseline_action_vec`` is hashed (not
    embedded) to keep the fingerprint compact.
    """
    bvec = np.asarray(baseline_action_vec, dtype=np.int64).reshape(-1).tolist()
    payload = json.dumps(bvec, separators=(",", ":")).encode("utf-8")
    bvec_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    return {
        "profile": str(profile),
        "num_layers": int(num_layers),
        "stage1_gelu_per_layer": [int(x) for x in stage1_gelu_per_layer],
        "stage1_softmax_per_layer": [int(x) for x in stage1_softmax_per_layer],
        "k_trials": int(k_trials),
        "probe_size": int(probe_size),
        "tol_acc": float(tol_acc),
        "tol_stab": float(tol_stab),
        "baseline_action_vec_hash": str(bvec_hash),
        "osr_version": str(osr_version),
    }


def fingerprints_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for key in a:
        if a[key] != b[key]:
            return False
    return True


# ---------------------------------------------------------------------------
# Results object + mask
# ---------------------------------------------------------------------------
@dataclass
class OSRPrePruneResults:
    """Pre-prune scan outputs serialised to ``osr_results.json``.

    Attributes:
        fingerprint:    See :func:`compute_fingerprint`.
        disabled_cells: ``{(layer_idx, block_idx): set of (slot_idx, level_idx)}``
            from Layer 1 — per-slot tests that failed invalid_chain /
            accuracy / stability.
        pruned_combos:  ``{(layer_idx, block_idx): set of tuple(int)}`` from
            Layer 2 — per-(block, layer) sampled combinations that failed.
        scan_summary:   Bookkeeping for the scan run (evaluated counts,
            elapsed seconds, aborted flag, etc.).
    """
    fingerprint: Dict[str, Any] = field(default_factory=dict)
    disabled_cells: Dict[Tuple[int, int], Set[Tuple[int, int]]] = field(default_factory=dict)
    pruned_combos: Dict[Tuple[int, int], Set[Tuple[int, ...]]] = field(default_factory=dict)
    scan_summary: Dict[str, Any] = field(default_factory=dict)

    def add_disabled_cell(self, layer_idx: int, block_idx: int,
                          slot_idx: int, level_idx: int) -> None:
        self.disabled_cells.setdefault(
            (int(layer_idx), int(block_idx)), set(),
        ).add((int(slot_idx), int(level_idx)))

    def add_pruned_combo(self, layer_idx: int, block_idx: int,
                         action_tuple: Sequence[int]) -> None:
        tup = tuple(int(x) for x in action_tuple)
        self.pruned_combos.setdefault(
            (int(layer_idx), int(block_idx)), set(),
        ).add(tup)

    def total_disabled_cells(self) -> int:
        return sum(len(v) for v in self.disabled_cells.values())

    def total_pruned_combos(self) -> int:
        return sum(len(v) for v in self.pruned_combos.values())

    def summary(self) -> str:
        return (
            f"OSR results: disabled_cells={self.total_disabled_cells()} "
            f"across {len(self.disabled_cells)} (layer,block); "
            f"pruned_combos={self.total_pruned_combos()} "
            f"across {len(self.pruned_combos)} (layer,block)"
        )


@dataclass
class OSRPrePruneMask:
    """Runtime mask backed by :class:`OSRPrePruneResults`.

    Lives alongside ``StaticInvalidLevelMask`` / ``EmpiricalInvalidLevelMask``
    / ``ForbiddenActionMask`` (see :mod:`blb_stage2_rl.action_mask`). The PPO
    rollout loop calls :meth:`apply_per_slot` after the other per-slot masks
    and treats any disabled cell as "this level cannot be sampled for this
    slot". For sampled action tuples, :meth:`is_combo_pruned` adds a final
    reject check on top of ``ForbiddenActionMask.is_forbidden``.

    Both checks short-circuit cheaply (set membership / dict lookup).
    """
    results: OSRPrePruneResults = field(default_factory=OSRPrePruneResults)

    def apply_per_slot(
            self,
            layer_idx: int,
            block_idx: int,
            action_level_mask: Sequence[Sequence[bool]],
            *,
            protected_actions: Sequence[Sequence[int]] = (),
            ) -> np.ndarray:
        """Disable OSR-flagged ``(slot, level)`` cells. Mirror's
        :meth:`StaticInvalidLevelMask.apply` signature."""
        mask = np.asarray(action_level_mask, dtype=bool).copy()
        if mask.ndim != 2:
            raise ValueError("action_level_mask must be a 2-D boolean array")
        protected: Set[Tuple[int, int]] = set()
        for action in protected_actions or ():
            for slot_idx, level_idx in enumerate(action):
                protected.add((int(slot_idx), int(level_idx)))
        disabled = self.results.disabled_cells.get(
            (int(layer_idx), int(block_idx)), set(),
        )
        for slot_idx, level_idx in disabled:
            if (slot_idx, level_idx) in protected:
                continue
            if 0 <= slot_idx < mask.shape[0] and 0 <= level_idx < mask.shape[1]:
                mask[slot_idx, level_idx] = False
        for slot_idx, level_idx in protected:
            if 0 <= slot_idx < mask.shape[0] and 0 <= level_idx < mask.shape[1]:
                mask[slot_idx, level_idx] = True
        # Guarantee at least one level per slot survives.
        for slot_idx in range(mask.shape[0]):
            if not bool(mask[slot_idx].any()):
                for protected_slot, protected_level in protected:
                    if protected_slot == slot_idx and 0 <= protected_level < mask.shape[1]:
                        mask[slot_idx, protected_level] = True
                        break
        return mask

    def is_combo_pruned(
            self,
            layer_idx: int,
            block_idx: int,
            action_tuple: Sequence[int],
            ) -> bool:
        key = (int(layer_idx), int(block_idx))
        if key not in self.results.pruned_combos:
            return False
        tup = tuple(int(x) for x in action_tuple)
        return tup in self.results.pruned_combos[key]

    def total_disabled(self) -> int:
        return int(self.results.total_disabled_cells())

    def total_pruned_combos(self) -> int:
        return int(self.results.total_pruned_combos())

    def summary(self) -> str:
        return self.results.summary()


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def _serialise_disabled(d: Mapping[Tuple[int, int], Set[Tuple[int, int]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (layer_idx, block_idx), cells in sorted(d.items()):
        out.append({
            "layer": int(layer_idx),
            "block": int(block_idx),
            "cells": [
                {"slot": int(s), "level": int(l)}
                for (s, l) in sorted(cells)
            ],
        })
    return out


def _serialise_combos(d: Mapping[Tuple[int, int], Set[Tuple[int, ...]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (layer_idx, block_idx), combos in sorted(d.items()):
        out.append({
            "layer": int(layer_idx),
            "block": int(block_idx),
            "combos": [
                [int(x) for x in tup]
                for tup in sorted(combos)
            ],
        })
    return out


def _deserialise_disabled(records: Iterable[Mapping[str, Any]]) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    out: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    for row in records or ():
        key = (int(row["layer"]), int(row["block"]))
        for cell in row.get("cells", []) or []:
            out.setdefault(key, set()).add((int(cell["slot"]), int(cell["level"])))
    return out


def _deserialise_combos(records: Iterable[Mapping[str, Any]]) -> Dict[Tuple[int, int], Set[Tuple[int, ...]]]:
    out: Dict[Tuple[int, int], Set[Tuple[int, ...]]] = {}
    for row in records or ():
        key = (int(row["layer"]), int(row["block"]))
        for tup in row.get("combos", []) or []:
            out.setdefault(key, set()).add(tuple(int(x) for x in tup))
    return out


def save_osr_results(path: str, results: OSRPrePruneResults) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "fingerprint": dict(results.fingerprint),
        "disabled_cells": _serialise_disabled(results.disabled_cells),
        "pruned_combos": _serialise_combos(results.pruned_combos),
        "scan_summary": dict(results.scan_summary),
    }
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, str(path))
    return str(path)


def load_osr_results(
        path: str,
        *,
        expected_fingerprint: Optional[Mapping[str, Any]] = None,
        allow_fingerprint_mismatch: bool = False,
        ) -> OSRPrePruneResults:
    """Read OSR results from JSON.

    Raises ValueError if ``expected_fingerprint`` is provided and differs from
    the stored fingerprint, unless ``allow_fingerprint_mismatch=True``.
    """
    with open(str(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    fingerprint = dict(payload.get("fingerprint") or {})
    if expected_fingerprint is not None and not allow_fingerprint_mismatch:
        if not fingerprints_equal(fingerprint, expected_fingerprint):
            diffs = []
            keys = set(fingerprint) | set(expected_fingerprint)
            for k in sorted(keys):
                if fingerprint.get(k) != expected_fingerprint.get(k):
                    diffs.append(f"  {k}: stored={fingerprint.get(k)!r} "
                                 f"expected={expected_fingerprint.get(k)!r}")
            raise ValueError(
                "OSR results fingerprint mismatch (pass "
                "allow_fingerprint_mismatch=True to override):\n"
                + "\n".join(diffs)
            )
    return OSRPrePruneResults(
        fingerprint=fingerprint,
        disabled_cells=_deserialise_disabled(payload.get("disabled_cells") or []),
        pruned_combos=_deserialise_combos(payload.get("pruned_combos") or []),
        scan_summary=dict(payload.get("scan_summary") or {}),
    )


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------
@dataclass
class _SlotLayout:
    layer_idx: int
    block_idx: int
    slot_dims: Tuple[int, ...]
    full_vec_offsets: Tuple[int, ...]


def _enumerate_slot_layouts(
        *,
        num_layers: int,
        profile: str,
        stage1_gelu_per_layer: Sequence[int],
        stage1_softmax_per_layer: Sequence[int],
        blocks_to_scan: Sequence[int] = DEFAULT_OSR_BLOCKS,
        ) -> List[_SlotLayout]:
    """Flatten the (block, layer) schedule into a list of slot layouts.

    Block 3 is excluded by default. Block 1 skips layer 0 (no block 1 there)
    naturally because :func:`step_schedule` does so.
    """
    from .action_space import step_schedule

    sched = step_schedule(
        int(num_layers),
        profile=str(profile),
        attn_degree_per_layer=list(stage1_softmax_per_layer),
        gelu_degree_per_layer=list(stage1_gelu_per_layer),
    )
    layouts: List[_SlotLayout] = []
    for spec in sched:
        if int(spec.block_idx) not in {int(b) for b in blocks_to_scan}:
            continue
        layouts.append(_SlotLayout(
            layer_idx=int(spec.layer_idx),
            block_idx=int(spec.block_idx),
            slot_dims=tuple(int(x) for x in spec.slot_dims),
            full_vec_offsets=tuple(int(x) for x in spec.full_vec_offsets),
        ))
    return layouts


def _classify_step_outcome(
        info: Mapping[str, Any],
        *,
        acc_threshold: float,
        stab_threshold: float,
        baseline_metric1: float,
        ) -> Tuple[bool, str]:
    """Return ``(fail, reason)`` from a base_env.step info dict."""
    if bool(info.get("any_invalid", False)):
        return True, "invalid_chain"
    if bool(info.get("forward_skipped", False)):
        # invalid chain causes forward to be skipped; treat as fail.
        return True, "forward_skipped"
    metrics = info.get("metrics")
    if metrics is None:
        return True, "no_metrics"
    metric1 = float(getattr(metrics, "metric1_mean", baseline_metric1))
    loss_std = float(getattr(metrics, "loss_std", 0.0) or 0.0)
    if metric1 < float(acc_threshold):
        return True, f"acc_below_threshold(m1={metric1:.4f}<thr={acc_threshold:.4f})"
    if loss_std > float(stab_threshold):
        return True, f"stab_above_threshold(std={loss_std:.4f}>thr={stab_threshold:.4f})"
    return False, ""


def run_osr_scan(
        *,
        base_env: Any,
        baseline_action_vec: Sequence[int],
        fingerprint: Mapping[str, Any],
        num_layers: int,
        profile: str,
        stage1_gelu_per_layer: Sequence[int],
        stage1_softmax_per_layer: Sequence[int],
        acc_orig: float,
        tol_acc: float,
        loss_std_orig: float,
        tol_stab: float,
        num_combo_samples: int = 300,
        blocks_to_scan: Sequence[int] = DEFAULT_OSR_BLOCKS,
        rng_seed: int = 17,
        log_fn: Optional[Callable[[str], None]] = None,
        save_path: Optional[str] = None,
        save_every: int = 200,
        stop_flag: Optional[Callable[[], bool]] = None,
        resume_from: Optional["OSRPrePruneResults"] = None,
        ) -> OSRPrePruneResults:
    """Run the two-layer OSR scan and return :class:`OSRPrePruneResults`.

    Args:
        base_env:           Fully-built :class:`BLBStage2Env` (same env the RL
            runner uses). Must accept ``step(full_action_vec)`` and report
            ``info['any_invalid']`` and ``info['metrics']``.
        baseline_action_vec: 577-dim baseline (static_skeletons all-max action).
            Layer-1 tests perturb a single slot; Layer-2 tests randomise the
            full ``(block, layer)`` slot tuple.
        acc_orig, loss_std_orig: from the noisy preflight on the baseline.
        tol_acc, tol_stab:   matched to ``stage2_limit_tolerance`` /
            ``stage2_stability_tolerance``.
        num_combo_samples:   per-(block, layer) Layer-2 sample budget. 0 skips
            Layer 2 entirely.
        save_path:           if given, partial results saved every
            ``save_every`` evaluations and on graceful stop.
        stop_flag:           callable returning True to request graceful stop
            (e.g. SIGINT handler).
        resume_from:         optional existing results to continue from; the
            scan skips ``(layer, block, slot, level)`` cells already recorded
            as disabled and avoids re-scanning fully-covered ``(layer, block)``
            layouts.
    """
    log = log_fn or (lambda _msg: None)
    rng = np.random.default_rng(int(rng_seed))
    baseline = np.asarray(baseline_action_vec, dtype=np.int64).reshape(-1).copy()
    results = resume_from or OSRPrePruneResults(fingerprint=dict(fingerprint))
    if not results.fingerprint:
        results.fingerprint = dict(fingerprint)
    acc_threshold = float(acc_orig) - float(tol_acc) - 1.0 / max(
        1, int(getattr(base_env, "probe_size", 256) or 256),
    )
    stab_threshold = float(loss_std_orig) * (1.0 + float(tol_stab))
    stab_threshold = max(stab_threshold, 0.01)

    layouts = _enumerate_slot_layouts(
        num_layers=int(num_layers),
        profile=str(profile),
        stage1_gelu_per_layer=stage1_gelu_per_layer,
        stage1_softmax_per_layer=stage1_softmax_per_layer,
        blocks_to_scan=blocks_to_scan,
    )
    log(
        f"[OSR] scanning {len(layouts)} (layer,block) layouts; "
        f"acc_threshold={acc_threshold:.4f} stab_threshold={stab_threshold:.4f} "
        f"num_combo_samples={num_combo_samples}"
    )

    summary = results.scan_summary
    summary.setdefault("started_at", time.time())
    summary.setdefault("layer1_evaluated", 0)
    summary.setdefault("layer1_pruned", 0)
    summary.setdefault("layer2_evaluated", 0)
    summary.setdefault("layer2_pruned", 0)
    summary.setdefault("layer1_done_layouts", [])
    summary.setdefault("layer2_done_layouts", [])
    summary["acc_threshold"] = float(acc_threshold)
    summary["stab_threshold"] = float(stab_threshold)
    summary["acc_orig"] = float(acc_orig)
    summary["loss_std_orig"] = float(loss_std_orig)

    layer1_done = {tuple(x) for x in summary.get("layer1_done_layouts", [])}
    layer2_done = {tuple(x) for x in summary.get("layer2_done_layouts", [])}

    def _maybe_save_and_stop(idx: int) -> bool:
        if save_path and idx > 0 and idx % max(1, int(save_every)) == 0:
            try:
                save_osr_results(save_path, results)
                log(f"[OSR] partial save @ {idx} evals  ({results.summary()})")
            except Exception as exc:
                log(f"[OSR] partial save FAILED: {exc}")
        if stop_flag is not None and bool(stop_flag()):
            log(f"[OSR] stop_flag set — saving and exiting after {idx} evals")
            if save_path:
                try:
                    save_osr_results(save_path, results)
                except Exception as exc:
                    log(f"[OSR] final save FAILED: {exc}")
            return True
        return False

    # ----- Layer 1: per-slot exhaustive -----
    eval_idx = 0
    for layout in layouts:
        key = (layout.layer_idx, layout.block_idx)
        if key in layer1_done:
            continue
        baseline_slice = baseline[list(layout.full_vec_offsets)].astype(np.int64).copy()
        for slot_idx, dim in enumerate(layout.slot_dims):
            base_level = int(baseline_slice[slot_idx])
            for level_idx in range(int(dim)):
                if level_idx == base_level:
                    continue
                # Skip cells already known disabled (resume support).
                if (slot_idx, level_idx) in results.disabled_cells.get(key, set()):
                    continue
                # Splice into a full vec copy.
                full_vec = baseline.copy()
                modified_slice = baseline_slice.copy()
                modified_slice[slot_idx] = int(level_idx)
                for off, val in zip(layout.full_vec_offsets, modified_slice):
                    full_vec[int(off)] = int(val)
                try:
                    _state, _reward, _done, info = base_env.step(full_vec)
                except Exception as exc:
                    info = {"any_invalid": True, "step_exception": str(exc)}
                fail, reason = _classify_step_outcome(
                    info,
                    acc_threshold=acc_threshold,
                    stab_threshold=stab_threshold,
                    baseline_metric1=float(acc_orig),
                )
                summary["layer1_evaluated"] = int(summary["layer1_evaluated"]) + 1
                eval_idx += 1
                if fail:
                    results.add_disabled_cell(
                        layout.layer_idx, layout.block_idx, slot_idx, level_idx,
                    )
                    summary["layer1_pruned"] = int(summary["layer1_pruned"]) + 1
                if _maybe_save_and_stop(eval_idx):
                    return results
        layer1_done.add(key)
        summary["layer1_done_layouts"] = sorted(list(layer1_done))

    log(
        f"[OSR] Layer 1 done: {summary['layer1_evaluated']} evals, "
        f"{summary['layer1_pruned']} cells pruned"
    )

    # ----- Layer 2: per-(block, layer) sampled combinations -----
    if int(num_combo_samples) > 0:
        for layout in layouts:
            key = (layout.layer_idx, layout.block_idx)
            if key in layer2_done:
                continue
            baseline_slice = baseline[list(layout.full_vec_offsets)].astype(np.int64).copy()
            already_sampled: Set[Tuple[int, ...]] = set()
            attempts = 0
            samples_done = 0
            max_attempts = int(num_combo_samples) * 4  # avoid infinite loop
            while samples_done < int(num_combo_samples) and attempts < max_attempts:
                attempts += 1
                tup = tuple(int(rng.integers(0, int(dim))) for dim in layout.slot_dims)
                if tup in already_sampled:
                    continue
                if tup == tuple(int(x) for x in baseline_slice):
                    # Baseline tuple is trivially valid; skip.
                    continue
                already_sampled.add(tup)
                if tup in results.pruned_combos.get(key, set()):
                    samples_done += 1
                    continue
                full_vec = baseline.copy()
                for off, val in zip(layout.full_vec_offsets, tup):
                    full_vec[int(off)] = int(val)
                try:
                    _state, _reward, _done, info = base_env.step(full_vec)
                except Exception as exc:
                    info = {"any_invalid": True, "step_exception": str(exc)}
                fail, reason = _classify_step_outcome(
                    info,
                    acc_threshold=acc_threshold,
                    stab_threshold=stab_threshold,
                    baseline_metric1=float(acc_orig),
                )
                summary["layer2_evaluated"] = int(summary["layer2_evaluated"]) + 1
                samples_done += 1
                eval_idx += 1
                if fail:
                    results.add_pruned_combo(
                        layout.layer_idx, layout.block_idx, tup,
                    )
                    summary["layer2_pruned"] = int(summary["layer2_pruned"]) + 1
                if _maybe_save_and_stop(eval_idx):
                    return results
            layer2_done.add(key)
            summary["layer2_done_layouts"] = sorted(list(layer2_done))

    summary["finished_at"] = time.time()
    summary["elapsed_seconds"] = float(
        summary["finished_at"] - summary.get("started_at", summary["finished_at"])
    )
    log(
        f"[OSR] scan complete: {results.summary()}  "
        f"elapsed={summary['elapsed_seconds']:.1f}s"
    )
    if save_path:
        save_osr_results(save_path, results)
    return results


# ---------------------------------------------------------------------------
# Helper used by run_substage_via_runner / run_sequential_via_runner to
# either load existing OSR results or run a fresh scan into the same path.
# Returns (mask_or_None, scan_only_should_exit).
# ---------------------------------------------------------------------------
def prepare_osr_mask(
        *,
        base_env: Any,
        results_path: str,
        scan_only: bool,
        num_combo_samples: int,
        allow_fingerprint_mismatch: bool,
        num_layers: int,
        profile: str,
        stage1_gelu_per_layer: Sequence[int],
        stage1_softmax_per_layer: Sequence[int],
        baseline_action_vec: Sequence[int],
        acc_orig: float,
        loss_std_orig: float,
        k_trials: int,
        probe_size: int,
        tol_acc: float,
        tol_stab: float,
        log_fn: Optional[Callable[[str], None]] = None,
        stop_flag: Optional[Callable[[], bool]] = None,
        ) -> Tuple[Optional[OSRPrePruneMask], bool]:
    """Resolve the OSR mask for a training run.

    Behaviour:
      * ``results_path`` empty/None → no OSR layer, returns ``(None, False)``.
      * file exists with matching fingerprint → load + return mask.
      * file exists but fingerprint differs → abort unless
        ``allow_fingerprint_mismatch`` (then load with warning).
      * file does NOT exist → run scan, save to ``results_path``, return mask
        (or trigger early exit if ``scan_only=True``).
    """
    log = log_fn or (lambda _msg: None)
    if not results_path:
        return None, False
    fingerprint = compute_fingerprint(
        profile=profile,
        num_layers=num_layers,
        stage1_gelu_per_layer=stage1_gelu_per_layer,
        stage1_softmax_per_layer=stage1_softmax_per_layer,
        k_trials=k_trials,
        probe_size=probe_size,
        tol_acc=tol_acc,
        tol_stab=tol_stab,
        baseline_action_vec=baseline_action_vec,
    )
    if os.path.exists(results_path):
        try:
            results = load_osr_results(
                results_path,
                expected_fingerprint=fingerprint,
                allow_fingerprint_mismatch=allow_fingerprint_mismatch,
            )
            log(f"[OSR] loaded results from {results_path}: {results.summary()}")
            return OSRPrePruneMask(results), bool(scan_only)
        except ValueError as exc:
            log(f"[OSR] existing results unusable ({exc}); will re-scan")
        except Exception as exc:
            log(f"[OSR] failed to load results ({exc}); will re-scan")
    log(f"[OSR] scanning → save to {results_path}")
    results = run_osr_scan(
        base_env=base_env,
        baseline_action_vec=baseline_action_vec,
        fingerprint=fingerprint,
        num_layers=num_layers,
        profile=profile,
        stage1_gelu_per_layer=stage1_gelu_per_layer,
        stage1_softmax_per_layer=stage1_softmax_per_layer,
        acc_orig=acc_orig,
        tol_acc=tol_acc,
        loss_std_orig=loss_std_orig,
        tol_stab=tol_stab,
        num_combo_samples=int(num_combo_samples),
        save_path=results_path,
        log_fn=log,
        stop_flag=stop_flag,
    )
    return OSRPrePruneMask(results), bool(scan_only)
