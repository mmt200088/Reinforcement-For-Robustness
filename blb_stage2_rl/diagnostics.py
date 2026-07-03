"""Long-term RL diagnostic recorder for the BLB Stage-2 sequential PPO loop.

Purpose: every run leaves enough on-disk evidence behind that we can answer
"what went wrong / what's stalled / where is the policy stuck" *after* a run
finishes (or crashes mid-way). The recorder writes to
``<blb_progress_dir>/diagnostics/`` and is append-only for streaming-style
artifacts (JSONL files) so a ``tail -f`` style monitor works even mid-run.

Files written
-------------
* ``episodes.jsonl``        – one JSONL row per episode (reward, valid steps,
                              first-invalid, cost signals, …)
* ``ppo_updates.jsonl``     – one JSONL row per PPO update (losses, entropy,
                              clip fraction, rolling window stats)
* ``top_candidates.jsonl``  – the top-K best actions seen so far. Rewritten on
                              every periodic flush; sorted descending by reward.
* ``pareto_frontier.jsonl`` – non-dominated candidates seen so far across
                              quality/cost objectives. Rewritten on every
                              periodic flush.
* ``pareto_frontier.json``  – compact metadata plus the same frontier rows.
* ``pareto_frontier.html``  – lightweight table for browser inspection.
* ``first_invalid_counts.json`` – Counter of ``"L<ii>-B<b>"`` → how many
                              episodes hit their *first* invalid step at that
                              (layer, block) tile. Identifies persistent
                              failure points fast.
* ``action_histogram.npz``  – per-slot action-index count matrix, periodically
                              flushed. Shape ``(num_slots, max_levels)`` int64.
                              Reveals "policy collapsed onto action 0 / 5".
* ``diagnostics_summary.md`` – **human-readable Chinese summary** regenerated
                              at every periodic flush. **Read this first when
                              debugging.** Contains training progress, Top-K
                              table, first-invalid heatmap, recent PPO update
                              dynamics, and auto-flag warnings.
* ``best_action_vec.json``  – action_grid-compatible JSON of the current best
                              action. Written on every new best so the user
                              can run::

                                bash Paean/run_final_eval.sh \\
                                    --action-config <progress_dir>/diagnostics/best_action_vec.json

                              to evaluate the in-flight champion without waiting
                              for the run to finish.

Usage
-----
.. code-block:: python

    recorder = RLDiagnosticsRecorder(
        output_dir=blb_progress_dir,
        num_layers=12,
        num_action_slots=len(action_dims_for_config(12)),
        max_action_levels=6,
        top_k=20,
        log_fn=log,
    )
    recorder.set_meta({"profile": "mrpc", "fixed_label": "..."})

    # in episode callback:
    recorder.record_episode(
        episode_stats=EpisodeStats(...), full_action_vec=..., is_new_best=True,
        best_reward_so_far=..., baseline_avg_k=...,
    )

    # in ppo-update callback:
    recorder.record_ppo_update(PPOUpdateStats(...))

    # every save_interval episodes:
    recorder.flush_periodic()

    # end of training:
    recorder.finalize()

Design notes
------------
* The recorder owns ZERO knowledge about the launcher / runner / policy
  internals. Callers pass in plain dataclass payloads. This keeps the recorder
  testable and the runner integration shallow.
* All writes are best-effort with try/except: a recorder failure must never
  abort training.
* JSONL files are append-only so a crash leaves a tail-able record. The
  ``.json`` / ``.npz`` / ``.md`` files are atomically rewritten (tmp + rename)
  on each periodic flush.
"""
from __future__ import annotations

import heapq
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


@dataclass
class EpisodeStats:
    """Per-episode payload persisted to ``episodes.jsonl``.

    Kept narrow on purpose — additional per-step detail goes to other files
    (e.g. ``top_candidates.jsonl`` carries the action_vec).
    """
    episode: int                           # global episode index (post-resume offset already applied)
    total_reward: float
    terminal_reward: float
    per_step_sum: float
    valid_steps: int
    invalid_steps: int
    steps_taken: int
    total_bits: int
    fusion_count: int
    first_invalid_step: Optional[int]
    first_invalid_block: Optional[int]
    first_invalid_layer: Optional[int]
    early_terminated: bool
    # 2026-06-13: per-block-TYPE fusion split (b2/b4/b5) so a runaway block
    # type (e.g. the accuracy-toxic block4 in the 3rd-60k hot collapse) is a
    # one-glance read instead of a re-derivation from the slot vector.
    fusion_count_b2: int = 0
    fusion_count_b4: int = 0
    fusion_count_b5: int = 0
    terminal_priority: int = 0
    terminal_loss_mean: float = 0.0
    terminal_loss_std: float = 0.0
    terminal_metric1_mean: float = 0.0
    terminal_metric2_mean: float = 0.0
    terminal_metric1_std: float = 0.0
    terminal_metric2_std: float = 0.0
    terminal_stab_excess_m1: float = 0.0
    terminal_stab_excess_m2: float = 0.0
    terminal_stab_excess_loss: float = 0.0
    terminal_stab_violation: float = 0.0
    terminal_bits_gain: float = 0.0
    terminal_k_gain: float = 0.0
    terminal_fusion_gain: float = 0.0
    terminal_cost_score: float = 0.0
    terminal_p3_metric_margin_reward: float = 0.0
    # ADR-014 (2026-06-14) DEBUG: ADR-013 barrier/margin, now persisted (was a
    # black box). worst_signed_margin=mu (barrier input); acc_barrier_sat/vio
    # (outputs); margin_m1/m2 (per-channel); fusion_norm_raw vs _saturated shows
    # the anti-runaway saturation. These make a collapse diagnosable from
    # episodes.jsonl alone.
    terminal_worst_signed_margin: float = 0.0
    terminal_acc_barrier_sat: float = 0.0
    terminal_acc_barrier_vio: float = 0.0
    terminal_near_miss: bool = False
    terminal_margin_m1: float = 0.0
    terminal_margin_m2: float = 0.0
    terminal_fusion_norm_raw: float = 0.0
    terminal_fusion_norm_saturated: float = 0.0
    terminal_cost_fusion_bonus: float = 0.0
    terminal_cost_truncation_bonus: float = 0.0
    terminal_cost_bits_tiebreaker: float = 0.0
    terminal_cost_truncation_step_gain: float = 0.0
    terminal_cost_rank_score: float = 0.0
    terminal_cost_rank_fusion: float = 0.0
    terminal_cost_rank_truncation: float = 0.0
    terminal_cost_rank_bits: float = 0.0
    terminal_pareto_event_kind: str = ""
    terminal_pareto_action_hash: str = ""
    terminal_pareto_frontier_removed: int = 0
    terminal_probe_wall_seconds: float = 0.0
    terminal_probe_devices: List[str] = field(default_factory=list)
    terminal_probe_trial_counts: List[int] = field(default_factory=list)
    terminal_probe_trial_indices: List[List[int]] = field(default_factory=list)
    terminal_probe_speedup: float = 1.0
    fusion_action_steps: List[Dict[str, Any]] = field(default_factory=list)
    per_step_optimizer_wall_seconds: float = 0.0
    policy_rollout_wall_seconds: float = 0.0
    terminal_cost_eval_wall_seconds: float = 0.0
    terminal_probe_install_wall_seconds: float = 0.0
    terminal_probe_clear_wall_seconds: float = 0.0
    terminal_probe_install_skipped: bool = False
    terminal_probe_clear_skipped: bool = False
    safe_neighbor_active: bool = False
    safe_neighbor_mutation_count: int = 0
    safe_neighbor_radius: int = 0
    exploration_mode: str = ""
    guarded_radius2_active: bool = False
    guarded_radius2_recent_frontier_expansions: int = 0
    guarded_radius2_recent_duplicate_rate: float = 0.0
    guarded_radius2_recent_dominated_rate: float = 0.0
    guarded_radius2_cooldown_remaining: int = 0
    guarded_radius2_safe_offset_count: int = 0
    guarded_radius2_episode_count: int = 0
    guarded_radius2_failure_count: int = 0
    guarded_radius2_frontier_expansion_count: int = 0
    samples_rejected_by_mask: int = 0
    samples_rejected_by_optimizer: int = 0
    steps_fallen_back_to_baseline: int = 0
    forbidden_mask_total: int = 0
    static_invalid_level_disabled: int = 0
    static_invalid_level_applied: int = 0
    static_invalid_level_scan_evaluated: int = 0
    static_invalid_level_scan_invalid: int = 0
    empirical_invalid_level_disabled: int = 0
    empirical_invalid_level_applied: int = 0
    rejection_optimizer_wall_seconds: float = 0.0
    baseline_prior_scale: float = 0.0
    base_action_source: str = ""
    proposal_direction: str = ""
    empirical_offset_success_rate: float = 0.0
    empirical_offset_failure_rate: float = 0.0
    frontier_seed_episode: int = -1
    timestamp: float = field(default_factory=time.time)


@dataclass
class PPOUpdateStats:
    """Per-PPO-update payload persisted to ``ppo_updates.jsonl``."""
    update: int
    completed_episodes: int                # cumulative episodes finished
    policy_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    n_samples: int
    window_mean_return: float
    window_max_return: float
    window_min_return: float
    window_mean_invalid: float
    best_reward_so_far: float
    elapsed_sec: float
    # 2026-05-18: effective ent_coef for this update (per the anchor → ramp →
    # steady schedule in sequential_runner._resolve_ent_coef_schedule).
    # Surfaced in diagnostics_summary.md "PPO 学习动态" table so operators
    # can verify the schedule is firing as expected.
    ent_coef: float = 0.0
    approx_kl: float = 0.0
    kl_early_stop: bool = False
    lr: float = 0.0
    lr_scale: float = 1.0
    entropy_recovery_delta: float = 0.0
    return_mean: float = 0.0
    return_std: float = 1.0
    timestamp: float = field(default_factory=time.time)


class RLDiagnosticsRecorder:
    def __init__(
            self,
            *,
            output_dir: str,
            num_layers: int,
            num_action_slots: int,
            max_action_levels: int = 6,
            top_k: int = 20,
            log_fn=None,
            slots_view_builder=None,
            schema_version: str = "blb_v3_slots_human_v1",
            data_point_writer=None,
            ) -> None:
        """Long-term diagnostics recorder.

        Args:
            slots_view_builder: optional callable
                ``action_vec → list[dict]`` that decodes an action vec into
                the human-readable slot list (label + scaling_factor /
                truncation_bits). Injected by the caller so this module
                stays standalone — typical wiring uses
                :func:`blb_stage2_rl.action_io.action_vec_to_slots_list`.
            schema_version: emitted in ``best_action_vec.json`` so future
                Paean parsers can pick the right reader.
        """
        self.output_dir = os.path.join(str(output_dir), "diagnostics")
        os.makedirs(self.output_dir, exist_ok=True)
        self.num_layers = int(num_layers)
        self.num_slots = int(num_action_slots)
        self.max_levels = max(1, int(max_action_levels))
        self.top_k = max(1, int(top_k))
        self.log = log_fn or (lambda _msg: None)
        self._slots_view_builder = slots_view_builder
        self.schema_version = str(schema_version)
        self._data_point_writer = data_point_writer

        self.episodes_path = os.path.join(self.output_dir, "episodes.jsonl")
        self.ppo_updates_path = os.path.join(self.output_dir, "ppo_updates.jsonl")
        self.top_path = os.path.join(self.output_dir, "top_candidates.jsonl")
        self.pareto_jsonl_path = os.path.join(self.output_dir, "pareto_frontier.jsonl")
        self.pareto_json_path = os.path.join(self.output_dir, "pareto_frontier.json")
        self.pareto_html_path = os.path.join(self.output_dir, "pareto_frontier.html")
        self.first_inv_path = os.path.join(self.output_dir, "first_invalid_counts.json")
        self.action_hist_path = os.path.join(self.output_dir, "action_histogram.npz")
        self.summary_md_path = os.path.join(self.output_dir, "diagnostics_summary.md")
        # ADR-014 (2026-06-14): in-repo rolling-health log. The 4th-60k collapse
        # trajectory (rolling600 P1/P2/P3 + fusion + margin) was only computed by
        # a server-side bash script (long60k_health.log) — invisible in the repo.
        # This makes it a reproducible artifact in the run dir.
        self.health_log_path = os.path.join(self.output_dir, "blb_stage2_health.log")
        self.best_json_path = os.path.join(self.output_dir, "best_action_vec.json")
        self.baseline_slots_path = os.path.join(self.output_dir, "baseline_action_vec.json")

        # in-memory accumulators (rebuilt from scratch each run — resume from
        # the on-disk JSONL is left to post-hoc tooling, kept simple here).
        self._first_invalid_counts: Counter = Counter()
        self._action_hist: np.ndarray = np.zeros(
            (self.num_slots, self.max_levels), dtype=np.int64,
        )
        # heap entries: (rank_key, tiebreaker, payload_dict). min-heap by rank.
        self._top_candidates: List = []
        self._pareto_candidates: List[Dict[str, Any]] = []
        self._pareto_seen_hashes: set = set()
        self._next_topcand_id = 0
        self._last_episode_stats: Optional[EpisodeStats] = None
        self._all_episode_returns: List[float] = []
        self._all_invalid_counts: List[int] = []
        self._all_terminal: List[float] = []
        # ADR-014: rolling-health accumulators (priority / fusion / margin).
        self._all_priority: List[int] = []
        self._all_fusion: List[int] = []
        self._all_fusion_b2: List[int] = []
        self._all_fusion_b4: List[int] = []
        self._all_fusion_b5: List[int] = []
        self._all_margin: List[float] = []
        self._ppo_history: List[PPOUpdateStats] = []
        self._t_start = time.time()
        self._meta: Dict[str, Any] = {}
        self._baseline_avg_k: Optional[float] = None
        self._baseline_action_vec: Optional[np.ndarray] = None
        self._baseline_slots: Optional[List[Dict[str, Any]]] = None

    @staticmethod
    def _candidate_rank_key(payload: Mapping[str, Any]) -> tuple:
        """Higher-is-better runtime candidate rank with P1/P2 hard gated."""
        try:
            priority = int(payload.get("terminal_priority", 0) or 0)
        except Exception:
            priority = 0
        try:
            invalid_steps = int(payload.get("invalid_steps", 0) or 0)
        except Exception:
            invalid_steps = 0

        total_reward = float(payload.get("total_reward", 0.0) or 0.0)
        terminal_reward = float(payload.get("terminal_reward", 0.0) or 0.0)
        metric1 = float(payload.get("terminal_metric1_mean", 0.0) or 0.0)
        metric2 = float(payload.get("terminal_metric2_mean", 0.0) or 0.0)
        stab = float(payload.get("terminal_stab_violation", 0.0) or 0.0)

        if priority == 3 and invalid_steps == 0:
            return (
                3.0,
                terminal_reward,
                total_reward,
                float(payload.get("terminal_cost_rank_score", 0.0) or 0.0),
                float(payload.get("terminal_fusion_gain", 0.0) or 0.0),
                float(payload.get("terminal_k_gain", 0.0) or 0.0),
                float(payload.get("terminal_bits_gain", 0.0) or 0.0),
            )
        if priority == 2:
            return (2.0, -max(0.0, stab), metric1, metric2, terminal_reward, total_reward)
        if priority == 1:
            return (
                1.0,
                -float(max(0, invalid_steps)),
                metric1,
                metric2,
                terminal_reward,
                total_reward,
            )
        return (0.0, terminal_reward, total_reward)

    def set_meta(self, meta: Mapping[str, Any]) -> None:
        """Stash run-level metadata that will be embedded in best_action_vec.json
        and the summary header (profile, fixed_label, hyperparams, etc.)."""
        self._meta = dict(meta or {})
        if self._data_point_writer is not None:
            try:
                baseline_preflight = self._meta.get("baseline_preflight_metrics")
                if not isinstance(baseline_preflight, Mapping):
                    baseline_preflight = self._meta.get("trainer_gate_baseline")
                manifest_payload = {
                    "source_diagnostics_dir": self.output_dir,
                    "num_layers": int(self.num_layers),
                    "num_action_slots": int(self.num_slots),
                    "max_action_levels": int(self.max_levels),
                    "top_k": int(self.top_k),
                    "schema_version": self.schema_version,
                    "meta": dict(self._meta),
                    "files": {
                        "episodes": self.episodes_path,
                        "ppo_updates": self.ppo_updates_path,
                        "top_candidates": self.top_path,
                        "pareto_frontier": self.pareto_jsonl_path,
                        "summary": self.summary_md_path,
                        "health_log": self.health_log_path,
                    },
                }
                if isinstance(baseline_preflight, Mapping):
                    baseline_preflight_dict = dict(baseline_preflight)
                    manifest_payload["trainer_gate_baseline"] = baseline_preflight_dict
                    manifest_payload["baseline_preflight_metrics"] = baseline_preflight_dict
                    manifest_payload["baseline_preflight_trial_count"] = int(
                        baseline_preflight_dict.get("trial_count", 0) or 0
                    )
                    manifest_payload["stage2_k_trials"] = int(
                        self._meta.get(
                            "stage2_k_trials",
                            baseline_preflight_dict.get("trial_count", 0),
                        )
                        or 0
                    )
                    for src, dst in (
                        ("limit_tolerance", "precision_tolerance"),
                        ("stability_tolerance", "stability_tolerance"),
                        ("loss_threshold", "loss_threshold"),
                        ("metric1_threshold", "metric1_threshold"),
                        ("metric2_threshold", "metric2_threshold"),
                        ("loss_std_threshold", "loss_std_threshold"),
                        ("metric1_std_threshold", "metric1_std_threshold"),
                        ("metric2_std_threshold", "metric2_std_threshold"),
                    ):
                        if src in baseline_preflight_dict:
                            manifest_payload[dst] = baseline_preflight_dict[src]
                if "reward_design" in self._meta:
                    manifest_payload["reward_design"] = self._meta["reward_design"]
                for key in (
                    "borderline_retest_enabled",
                    "borderline_retest_trials_multiplier",
                ):
                    if key in self._meta:
                        manifest_payload[key] = self._meta[key]
                self._data_point_writer.write_manifest(manifest_payload)
            except Exception as exc:
                self.log(f"  [diag][warning] structured data manifest write failed: {exc}")

    def set_baseline_avg_k(self, value: float) -> None:
        try:
            self._baseline_avg_k = float(value)
        except Exception:
            self._baseline_avg_k = None

    def set_baseline_action_vec(self, action_vec) -> None:
        """Record the all-max / static_skeletons baseline so top-K rows in the
        summary can show *diffs* against it (which slots changed and by how much)."""
        try:
            self._baseline_action_vec = np.asarray(action_vec, dtype=np.int64).copy()
        except Exception:
            self._baseline_action_vec = None
        if self._baseline_action_vec is not None and self._slots_view_builder is not None:
            try:
                self._baseline_slots = list(self._slots_view_builder(self._baseline_action_vec))
                # eagerly dump a copy so it lives even if the run crashes early
                _atomic_json_dump(self.baseline_slots_path, {
                    "schema_version": self.schema_version,
                    "num_layers": int(self.num_layers),
                    "source": "static_skeletons_baseline",
                    "meta": dict(self._meta),
                    "slots": self._baseline_slots,
                    "action_vec": self._baseline_action_vec.tolist(),
                })
            except Exception as exc:
                self.log(f"  [diag][warning] baseline_action_vec.json write failed: {exc}")
                self._baseline_slots = None

    # ------------------------------------------------------------------
    # Recording APIs
    # ------------------------------------------------------------------

    def record_episode(
            self,
            *,
            episode_stats: EpisodeStats,
            full_action_vec: Optional[np.ndarray],
            is_new_best: bool,
            best_reward_so_far: float,
            ) -> None:
        """Append episode JSONL row and update in-memory accumulators."""
        # 1) append JSONL row
        try:
            with open(self.episodes_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode_stats.__dict__, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            self.log(f"  [diag][warning] episodes.jsonl write failed: {exc}")
        if self._data_point_writer is not None:
            try:
                mirror_payload = dict(episode_stats.__dict__)
                mirror_payload["is_new_best"] = bool(is_new_best)
                mirror_payload["best_reward_so_far"] = float(best_reward_so_far)
                if full_action_vec is not None:
                    mirror_payload["full_action_vec"] = (
                        np.asarray(full_action_vec, dtype=int).reshape(-1).tolist()
                    )
                self._data_point_writer.write_episode(mirror_payload)
            except Exception as exc:
                self.log(f"  [diag][warning] structured episode write failed: {exc}")

        # 2) accumulate stats
        self._all_episode_returns.append(float(episode_stats.total_reward))
        self._all_invalid_counts.append(int(episode_stats.invalid_steps))
        self._all_terminal.append(float(episode_stats.terminal_reward))
        # ADR-014 rolling-health accumulators.
        self._all_priority.append(int(episode_stats.terminal_priority))
        self._all_fusion.append(int(episode_stats.fusion_count))
        self._all_fusion_b2.append(int(episode_stats.fusion_count_b2))
        self._all_fusion_b4.append(int(episode_stats.fusion_count_b4))
        self._all_fusion_b5.append(int(episode_stats.fusion_count_b5))
        self._all_margin.append(float(episode_stats.terminal_worst_signed_margin))
        self._last_episode_stats = episode_stats

        # 3) first-invalid counter
        if (
            episode_stats.first_invalid_layer is not None
            and episode_stats.first_invalid_block is not None
        ):
            key = (
                f"L{int(episode_stats.first_invalid_layer):02d}"
                f"-B{int(episode_stats.first_invalid_block)}"
            )
            self._first_invalid_counts[key] += 1

        # 4) action histogram (per-slot index counts)
        if full_action_vec is not None:
            arr = np.asarray(full_action_vec, dtype=int).reshape(-1)
            n = min(arr.size, self.num_slots)
            for slot_idx in range(n):
                a = int(arr[slot_idx])
                if 0 <= a < self.max_levels:
                    self._action_hist[slot_idx, a] += 1

        # 5) top-K candidates
        if full_action_vec is not None:
            payload = {
                "episode": int(episode_stats.episode),
                "total_reward": float(episode_stats.total_reward),
                "terminal_reward": float(episode_stats.terminal_reward),
                "per_step_sum": float(episode_stats.per_step_sum),
                "valid_steps": int(episode_stats.valid_steps),
                "invalid_steps": int(episode_stats.invalid_steps),
                "total_bits": int(episode_stats.total_bits),
                "fusion_count": int(episode_stats.fusion_count),
                "terminal_priority": int(episode_stats.terminal_priority),
                "terminal_loss_mean": float(episode_stats.terminal_loss_mean),
                "terminal_loss_std": float(episode_stats.terminal_loss_std),
                "terminal_metric1_mean": float(episode_stats.terminal_metric1_mean),
                "terminal_metric2_mean": float(episode_stats.terminal_metric2_mean),
                "terminal_metric1_std": float(episode_stats.terminal_metric1_std),
                "terminal_metric2_std": float(episode_stats.terminal_metric2_std),
                "terminal_bits_gain": float(episode_stats.terminal_bits_gain),
                "terminal_k_gain": float(episode_stats.terminal_k_gain),
                "terminal_fusion_gain": float(episode_stats.terminal_fusion_gain),
                "terminal_cost_score": float(episode_stats.terminal_cost_score),
                "terminal_p3_metric_margin_reward": float(
                    episode_stats.terminal_p3_metric_margin_reward
                ),
                "terminal_cost_fusion_bonus": float(episode_stats.terminal_cost_fusion_bonus),
                "terminal_cost_truncation_bonus": float(
                    episode_stats.terminal_cost_truncation_bonus
                ),
                "terminal_cost_bits_tiebreaker": float(
                    episode_stats.terminal_cost_bits_tiebreaker
                ),
                "terminal_cost_truncation_step_gain": float(
                    episode_stats.terminal_cost_truncation_step_gain
                ),
                "terminal_cost_rank_score": float(episode_stats.terminal_cost_rank_score),
                "terminal_cost_rank_fusion": float(episode_stats.terminal_cost_rank_fusion),
                "terminal_cost_rank_truncation": float(
                    episode_stats.terminal_cost_rank_truncation
                ),
                "terminal_cost_rank_bits": float(episode_stats.terminal_cost_rank_bits),
                "terminal_pareto_event_kind": str(episode_stats.terminal_pareto_event_kind),
                "terminal_pareto_action_hash": str(episode_stats.terminal_pareto_action_hash),
                "terminal_pareto_frontier_removed": int(episode_stats.terminal_pareto_frontier_removed),
                "safe_neighbor_active": bool(episode_stats.safe_neighbor_active),
                "safe_neighbor_mutation_count": int(episode_stats.safe_neighbor_mutation_count),
                "safe_neighbor_radius": int(episode_stats.safe_neighbor_radius),
                "exploration_mode": str(episode_stats.exploration_mode),
                "guarded_radius2_active": bool(episode_stats.guarded_radius2_active),
                "guarded_radius2_recent_frontier_expansions": int(
                    episode_stats.guarded_radius2_recent_frontier_expansions
                ),
                "guarded_radius2_recent_duplicate_rate": float(
                    episode_stats.guarded_radius2_recent_duplicate_rate
                ),
                "guarded_radius2_recent_dominated_rate": float(
                    episode_stats.guarded_radius2_recent_dominated_rate
                ),
                "guarded_radius2_cooldown_remaining": int(
                    episode_stats.guarded_radius2_cooldown_remaining
                ),
                "guarded_radius2_safe_offset_count": int(
                    episode_stats.guarded_radius2_safe_offset_count
                ),
                "guarded_radius2_episode_count": int(episode_stats.guarded_radius2_episode_count),
                "guarded_radius2_failure_count": int(episode_stats.guarded_radius2_failure_count),
                "guarded_radius2_frontier_expansion_count": int(
                    episode_stats.guarded_radius2_frontier_expansion_count
                ),
                "static_invalid_level_disabled": int(
                    episode_stats.static_invalid_level_disabled
                ),
                "static_invalid_level_applied": int(
                    episode_stats.static_invalid_level_applied
                ),
                "baseline_prior_scale": float(episode_stats.baseline_prior_scale),
                "base_action_source": str(episode_stats.base_action_source),
                "proposal_direction": str(episode_stats.proposal_direction),
                "empirical_offset_success_rate": float(
                    episode_stats.empirical_offset_success_rate
                ),
                "empirical_offset_failure_rate": float(
                    episode_stats.empirical_offset_failure_rate
                ),
                "frontier_seed_episode": int(episode_stats.frontier_seed_episode),
                "action_vec": np.asarray(full_action_vec, dtype=int).tolist(),
            }
            if not payload["terminal_pareto_action_hash"]:
                payload["terminal_pareto_action_hash"] = _action_vec_hash(payload["action_vec"])
            self._consider_pareto_candidate(payload)
            rank_key = self._candidate_rank_key(payload)
            payload["top_candidate_rank_key"] = [float(x) for x in rank_key]
            entry = (
                rank_key,
                self._next_topcand_id,
                payload,
            )
            self._next_topcand_id += 1
            if len(self._top_candidates) < self.top_k:
                heapq.heappush(self._top_candidates, entry)
            elif entry[0] > self._top_candidates[0][0]:
                heapq.heapreplace(self._top_candidates, entry)

        # 6) on new best, persist best_action_vec.json in a *human-readable*
        # format: every slot listed with its model location + the actual
        # scaling_factor / truncation_bits value. The flat ``action_vec`` is
        # kept as a fallback field so Paean's old reader keeps working.
        if is_new_best and full_action_vec is not None:
            try:
                vec_int = np.asarray(full_action_vec, dtype=int)
                slots_view: Optional[List[Dict[str, Any]]] = None
                if self._slots_view_builder is not None:
                    try:
                        slots_view = list(self._slots_view_builder(vec_int))
                    except Exception as exc:
                        self.log(f"  [diag][warning] slots_view_builder failed: {exc}")
                        slots_view = None
                diff_vs_baseline = self._diff_against_baseline(slots_view)
                payload = {
                    "schema_version": self.schema_version,
                    "num_layers": int(self.num_layers),
                    "source": "blb_v3_sequential_runtime_best",
                    "episode": int(episode_stats.episode),
                    "total_reward": float(episode_stats.total_reward),
                    "terminal_reward": float(episode_stats.terminal_reward),
                    "valid_steps": int(episode_stats.valid_steps),
                    "invalid_steps": int(episode_stats.invalid_steps),
                    "terminal_cost_rank_score": float(episode_stats.terminal_cost_rank_score),
                    "terminal_cost_rank_fusion": float(episode_stats.terminal_cost_rank_fusion),
                    "terminal_cost_rank_truncation": float(
                        episode_stats.terminal_cost_rank_truncation
                    ),
                    "terminal_cost_rank_bits": float(episode_stats.terminal_cost_rank_bits),
                    "best_reward_so_far": float(best_reward_so_far),
                    "wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "meta": dict(self._meta),
                    # Human view (primary)
                    "slots": slots_view,
                    "diff_vs_baseline": diff_vs_baseline,
                    # Machine view (Paean fallback, kept for back-compat)
                    "action_vec": vec_int.tolist(),
                    "decode_rules": {
                        "scaling_factor_kinds": "F W M S R",
                        "truncation_kind": "K",
                        "note": (
                            "SF slots: select 'scaling_factor' (snapped to noise table). "
                            "K slots: select 'truncation_bits' ∈ K_LEVELS. "
                            "Rescale (R) slots: scaling_factor=null disables that rescale point."
                        ),
                    },
                }
                _atomic_json_dump(self.best_json_path, payload)
            except Exception as exc:
                self.log(f"  [diag][warning] best_action_vec.json write failed: {exc}")

    def _diff_against_baseline(
            self,
            slots_view: Optional[List[Dict[str, Any]]],
            ) -> List[Dict[str, Any]]:
        """For each slot whose value differs from the baseline, emit a diff row.

        Diff entries:
          ``{"label": "L05.B3.K", "baseline": 13, "best": 10, "delta": -3, "kind": "K"}``

        If no baseline has been set, returns ``[]``.
        """
        if slots_view is None or self._baseline_slots is None:
            return []
        base_by_label: Dict[str, Dict[str, Any]] = {
            row["label"]: row for row in self._baseline_slots
        }
        diff: List[Dict[str, Any]] = []
        for row in slots_view:
            label = row.get("label", "")
            base = base_by_label.get(label)
            if base is None:
                continue
            kind = str(row.get("kind", ""))
            if kind == "K":
                a = row.get("truncation_bits")
                b = base.get("truncation_bits")
                if a is not None and b is not None and int(a) != int(b):
                    diff.append({
                        "label": label, "kind": kind,
                        "baseline_truncation_bits": int(b),
                        "best_truncation_bits": int(a),
                        "delta": int(a) - int(b),
                    })
            else:
                a = row.get("scaling_factor")
                b = base.get("scaling_factor")
                # treat None as "off" — comparing None vs int counts as a change.
                if (a is None) != (b is None) or (
                    a is not None and b is not None and int(a) != int(b)
                ):
                    diff.append({
                        "label": label, "kind": kind,
                        "baseline_scaling_factor": (None if b is None else int(b)),
                        "best_scaling_factor": (None if a is None else int(a)),
                        "delta": (
                            None
                            if (a is None or b is None)
                            else int(a) - int(b)
                        ),
                    })
        return diff

    def record_ppo_update(self, stats: PPOUpdateStats) -> None:
        """Append PPO update JSONL row."""
        try:
            with open(self.ppo_updates_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats.__dict__, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            self.log(f"  [diag][warning] ppo_updates.jsonl write failed: {exc}")
        if self._data_point_writer is not None:
            try:
                self._data_point_writer.write_ppo_update(dict(stats.__dict__))
            except Exception as exc:
                self.log(f"  [diag][warning] structured ppo update write failed: {exc}")
        self._ppo_history.append(stats)

    def flush_periodic(self) -> None:
        """Flush action histogram + first-invalid counts + top candidates +
        regenerate summary.md. Call at ``save_interval`` cadence (default 200
        episodes) — cheap enough but not free."""
        try:
            np.savez(self.action_hist_path, counts=self._action_hist)
        except Exception as exc:
            self.log(f"  [diag][warning] action_histogram.npz write failed: {exc}")
        try:
            _atomic_json_dump(self.first_inv_path, dict(self._first_invalid_counts))
        except Exception as exc:
            self.log(f"  [diag][warning] first_invalid_counts.json write failed: {exc}")
        try:
            tmp = self.top_path + ".tmp"
            sorted_top = sorted(self._top_candidates, key=lambda e: e[0], reverse=True)
            with open(tmp, "w", encoding="utf-8") as f:
                for rank, entry in enumerate(sorted_top, start=1):
                    out_row = dict(entry[2])
                    # Augment top-K rows with a human slots view (best-effort).
                    if self._slots_view_builder is not None and "action_vec" in out_row:
                        try:
                            slots_view = list(self._slots_view_builder(np.asarray(
                                out_row["action_vec"], dtype=int,
                            )))
                            out_row["slots"] = slots_view
                            out_row["diff_vs_baseline"] = self._diff_against_baseline(slots_view)
                        except Exception:
                            pass
                    out_row["rank"] = int(rank)
                    f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            os.replace(tmp, self.top_path)
        except Exception as exc:
            self.log(f"  [diag][warning] top_candidates.jsonl write failed: {exc}")
        try:
            self._write_pareto_artifacts()
        except Exception as exc:
            self.log(f"  [diag][warning] pareto frontier write failed: {exc}")
        try:
            self._write_summary_md()
        except Exception as exc:
            self.log(f"  [diag][warning] diagnostics_summary.md write failed: {exc}")
        try:
            self._write_health_log()
        except Exception as exc:
            self.log(f"  [diag][warning] blb_stage2_health.log write failed: {exc}")

    def _write_health_log(self, window: int = 600) -> None:
        """Append one rolling-window health line to ``blb_stage2_health.log``.

        Mirrors the server-side ``long60k_health.log`` format so the collapse
        trajectory (P1/P2/P3 + fusion + per-block + margin over a rolling window)
        is a reproducible in-repo artifact: the smoking gun for a hot collapse is
        ``fusion`` rising while ``P3`` falls and ``margin`` goes negative.
        """
        n = len(self._all_episode_returns)
        if n == 0:
            return
        w = min(int(window), n)

        def _tm(xs: List[float]) -> float:
            t = xs[-w:]
            return (sum(t) / len(t)) if t else 0.0

        pri = self._all_priority[-w:]
        p1 = sum(1 for p in pri if int(p) == 1)
        p2 = sum(1 for p in pri if int(p) == 2)
        p3 = sum(1 for p in pri if int(p) == 3)
        line = (
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')} ep={n} rolling{w}: "
            f"reward={_tm(self._all_episode_returns):.3f} "
            f"P1={p1} P2={p2} P3={p3} "
            f"fusion={_tm([float(x) for x in self._all_fusion]):.2f} "
            f"(b2={_tm([float(x) for x in self._all_fusion_b2]):.1f} "
            f"b4={_tm([float(x) for x in self._all_fusion_b4]):.1f} "
            f"b5={_tm([float(x) for x in self._all_fusion_b5]):.1f}) "
            f"margin={_tm(self._all_margin):.4f}"
        )
        with open(self.health_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def finalize(self) -> None:
        """Flush + leave behind the summary in its final form."""
        self.flush_periodic()
        if self._data_point_writer is not None:
            try:
                last_episode = (
                    dict(self._last_episode_stats.__dict__)
                    if self._last_episode_stats is not None
                    else None
                )
                self._data_point_writer.write_summary({
                    "status": "completed",
                    "episode_count": int(len(self._all_episode_returns)),
                    "ppo_update_count": int(len(self._ppo_history)),
                    "meta": dict(self._meta),
                    "last_episode": last_episode,
                    "diagnostics_dir": self.output_dir,
                    "diagnostics_summary": self.summary_md_path,
                    "top_candidates": self.top_path,
                    "pareto_frontier": self.pareto_jsonl_path,
                })
                self._data_point_writer.close()
            except Exception as exc:
                self.log(f"  [diag][warning] structured summary write failed: {exc}")

    # ------------------------------------------------------------------
    # Summary.md writer
    # ------------------------------------------------------------------

    def _consider_pareto_candidate(self, payload: Mapping[str, Any]) -> None:
        candidate = dict(payload)
        if int(candidate.get("terminal_priority", 0) or 0) != 3:
            return
        if int(candidate.get("invalid_steps", 0) or 0) != 0:
            return
        action_hash = str(candidate.get("terminal_pareto_action_hash", "") or "")
        if action_hash and action_hash in self._pareto_seen_hashes:
            return
        if action_hash:
            self._pareto_seen_hashes.add(action_hash)
        if any(_pareto_dominates(existing, candidate) for existing in self._pareto_candidates):
            return
        self._pareto_candidates = [
            existing
            for existing in self._pareto_candidates
            if not _pareto_dominates(candidate, existing)
        ]
        self._pareto_candidates.append(candidate)

    def _sorted_pareto_candidates(self) -> List[Dict[str, Any]]:
        rows = [dict(row) for row in self._pareto_candidates]
        rows.sort(key=lambda r: (
            -float(r.get("terminal_fusion_gain", 0.0)),
            -float(r.get("terminal_k_gain", 0.0)),
            -float(r.get("terminal_bits_gain", 0.0)),
            -float(r.get("total_reward", -1e9)),
            int(r.get("episode", 10**12)),
        ))
        for rank, row in enumerate(rows, start=1):
            row["pareto_rank"] = int(rank)
            if self._slots_view_builder is not None and "action_vec" in row:
                try:
                    slots_view = list(self._slots_view_builder(np.asarray(
                        row["action_vec"], dtype=int,
                    )))
                    row["slots"] = slots_view
                    row["diff_vs_baseline"] = self._diff_against_baseline(slots_view)
                except Exception:
                    pass
        return rows

    def _write_pareto_artifacts(self) -> None:
        rows = self._sorted_pareto_candidates()
        tmp = self.pareto_jsonl_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, self.pareto_jsonl_path)
        _atomic_json_dump(self.pareto_json_path, {
            "schema_version": "blb_stage2_pareto_frontier_v1",
            "source": "blb_v3_sequential_runtime_diagnostics",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "meta": dict(self._meta),
            "objectives": {
                "maximize": [
                    "terminal_fusion_gain",
                    "terminal_k_gain",
                    "terminal_bits_gain",
                ],
            },
            "dominance_note": (
                "Only P3 candidates enter this archive. A candidate dominates "
                "another only when it is no worse on fusion_gain, k_gain, and "
                "bits_gain, and strictly better on at least one axis."
            ),
            "count": int(len(rows)),
            "frontier": rows,
        })
        self._write_pareto_html(rows)

    def _write_pareto_html(self, rows: List[Mapping[str, Any]]) -> None:
        def esc(value: Any) -> str:
            return (
                str(value)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )

        cols = [
            "pareto_rank",
            "episode",
            "terminal_priority",
            "invalid_steps",
            "terminal_metric1_mean",
            "terminal_metric2_mean",
            "terminal_loss_mean",
            "terminal_fusion_gain",
            "terminal_k_gain",
            "terminal_bits_gain",
            "terminal_cost_score",
            "terminal_cost_rank_score",
            "terminal_cost_rank_fusion",
            "terminal_cost_rank_truncation",
            "terminal_cost_rank_bits",
            "terminal_p3_metric_margin_reward",
            "terminal_cost_fusion_bonus",
            "terminal_cost_truncation_bonus",
            "terminal_cost_bits_tiebreaker",
            "terminal_cost_truncation_step_gain",
            "terminal_pareto_event_kind",
            "total_bits",
            "total_reward",
            "safe_neighbor_mutation_count",
            "safe_neighbor_radius",
            "exploration_mode",
            "guarded_radius2_active",
            "guarded_radius2_recent_frontier_expansions",
            "guarded_radius2_recent_duplicate_rate",
            "guarded_radius2_recent_dominated_rate",
            "guarded_radius2_cooldown_remaining",
            "baseline_prior_scale",
            "base_action_source",
            "proposal_direction",
            "empirical_offset_success_rate",
            "empirical_offset_failure_rate",
            "frontier_seed_episode",
        ]
        lines = [
            "<!doctype html>",
            "<html><head><meta charset=\"utf-8\">",
            "<title>BLB Stage-2 Pareto Frontier</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;color:#111827}",
            "table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #d1d5db;padding:6px 8px;text-align:right}",
            "th{background:#f3f4f6;position:sticky;top:0}td:nth-child(2),th:nth-child(2){text-align:left}",
            "code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}",
            "</style></head><body>",
            "<h1>BLB Stage-2 Pareto Frontier</h1>",
            "<p>Non-dominated runtime candidates across quality, stability, and cost objectives.</p>",
            "<table><thead><tr>",
        ]
        lines.extend(f"<th>{esc(c)}</th>" for c in cols)
        lines.append("</tr></thead><tbody>")
        for row in rows:
            lines.append("<tr>")
            for col in cols:
                value = row.get(col, "")
                if isinstance(value, float):
                    value = f"{value:.6g}"
                lines.append(f"<td>{esc(value)}</td>")
            lines.append("</tr>")
        lines.extend(["</tbody></table>", "</body></html>"])
        tmp = self.pareto_html_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        os.replace(tmp, self.pareto_html_path)

    def _write_summary_md(self) -> None:
        last = self._last_episode_stats
        if last is None:
            return
        elapsed = time.time() - self._t_start
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

        lines: List[str] = []
        lines.append(
            f"# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode={last.episode + 1}）"
        )
        lines.append("")
        lines.append(
            f"_更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}_  ·  "
            f"累计用时: **{elapsed_str}**"
        )
        if self._meta:
            lines.append("")
            lines.append("**Run meta**：")
            for k, v in self._meta.items():
                lines.append(f"- `{k}` = `{v}`")
        lines.append("")

        # 1. 训练进度
        lines.append("## 1. 训练进度（training progress）")
        lines.append("")
        n_ep = len(self._all_episode_returns)
        if n_ep == 0:
            lines.append("_no episodes recorded yet._")
            self._dump(lines)
            return
        last50 = self._all_episode_returns[-50:]
        last50_inv = self._all_invalid_counts[-50:]
        last50_term = self._all_terminal[-50:]
        best_so_far = max(self._all_episode_returns)
        worst_so_far = min(self._all_episode_returns)
        lines.append(f"- 已完成回合数: **{n_ep}**")
        lines.append(
            f"- 最近 50 回合 mean return: **{np.mean(last50):+.4f}** "
            f"(min={np.min(last50):+.4f}, max={np.max(last50):+.4f})"
        )
        lines.append(
            f"- 最近 50 回合 mean terminal reward: **{np.mean(last50_term):+.4f}**"
        )
        lines.append(
            # horizon = 3 + (L-1)*4 (C 2026-05-30: block 3 removed from schedule)
            f"- 最近 50 回合 mean invalid 子步数: **{np.mean(last50_inv):.2f}** / "
            f"{3 + (self.num_layers - 1) * 4}"
        )
        lines.append(f"- 训练期 best reward: **{best_so_far:+.4f}**")
        lines.append(f"- 训练期 worst reward: **{worst_so_far:+.4f}**")
        lines.append(f"- PPO 更新次数: **{len(self._ppo_history)}**")
        if self._baseline_avg_k is not None:
            lines.append(f"- baseline avg_k (per-block 加权): **{self._baseline_avg_k:.3f}**")
        lines.append("")

        # 2. Top-K candidates
        lines.append(f"## 2. 训练期 Top-{self.top_k} candidates")
        lines.append("")
        lines.append(
            "**说明**：按 hard-priority + unbounded P3 cost rank 排序。"
            "P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。"
            "每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。"
        )
        lines.append("")
        lines.append(
            "| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |"
        )
        lines.append(
            "|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|"
        )
        topk_sorted = sorted(self._top_candidates, key=lambda e: e[0], reverse=True)
        for rank, entry in enumerate(topk_sorted, start=1):
            p = entry[2]
            lines.append(
                f"| {rank} | {p['episode']} | {p.get('terminal_priority', 0)} | "
                f"{float(p.get('terminal_cost_rank_score', 0.0)):+.4f} | "
                f"{p['total_reward']:+.4f} | "
                f"{p['terminal_reward']:+.4f} | {p['per_step_sum']:+.4f} | "
                f"{p['valid_steps']} | {p['invalid_steps']} | "
                f"{p['total_bits']} | {p['fusion_count']} |"
            )
        lines.append("")

        # 2.1 Best vs baseline — slot-level diff (the actionable view)
        best_entry = topk_sorted[0] if topk_sorted else None
        if (
            best_entry is not None
            and self._slots_view_builder is not None
            and self._baseline_slots is not None
            and "action_vec" in best_entry[2]
        ):
            try:
                best_slots = list(self._slots_view_builder(np.asarray(
                    best_entry[2]["action_vec"], dtype=int,
                )))
                diff = self._diff_against_baseline(best_slots)
                lines.append("### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）")
                lines.append("")
                if not diff:
                    lines.append("_best action 与 baseline 完全相同（warmstart 还没动）。_")
                else:
                    # Separate SF diffs from K diffs for readability.
                    sf_diffs = [d for d in diff if d.get("kind") != "K"]
                    k_diffs = [d for d in diff if d.get("kind") == "K"]
                    lines.append(
                        f"_共 {len(diff)} 个槽与 baseline 不同_"
                        f"（{len(sf_diffs)} SF + {len(k_diffs)} K）"
                    )
                    lines.append("")
                    if k_diffs:
                        lines.append("**Truncation K diffs**：")
                        lines.append("")
                        lines.append("| Slot | Baseline K | Best K | Δ |")
                        lines.append("|:-----|----------:|------:|--:|")
                        for d in sorted(k_diffs, key=lambda r: r["label"]):
                            lines.append(
                                f"| `{d['label']}` | {d['baseline_truncation_bits']} | "
                                f"{d['best_truncation_bits']} | {d['delta']:+d} |"
                            )
                        lines.append("")
                    if sf_diffs:
                        lines.append("**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：")
                        lines.append("")
                        lines.append("| Slot | Kind | Baseline SF | Best SF | Δ |")
                        lines.append("|:-----|:----:|------------:|--------:|--:|")
                        def _abs_delta(r):
                            d_ = r.get("delta")
                            return -1 if d_ is None else abs(int(d_))
                        for d in sorted(sf_diffs, key=_abs_delta, reverse=True)[:20]:
                            b_ = d.get("baseline_scaling_factor")
                            a_ = d.get("best_scaling_factor")
                            delta_s = "off→on" if b_ is None else (
                                "on→off" if a_ is None else f"{int(d['delta']):+d}"
                            )
                            lines.append(
                                f"| `{d['label']}` | {d['kind']} | "
                                f"{'off' if b_ is None else b_} | "
                                f"{'off' if a_ is None else a_} | {delta_s} |"
                            )
                        lines.append("")
                    lines.append(
                        "完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。"
                    )
                lines.append("")
            except Exception as exc:
                self.log(f"  [diag][warning] best/baseline diff section failed: {exc}")

        # 3. first-invalid heatmap
        lines.append("## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）")
        lines.append("")
        if not self._first_invalid_counts:
            lines.append("_暂无 invalid 记录（所有 episode 完整通过）。_")
        else:
            top_inv = self._first_invalid_counts.most_common(15)
            total_invalid_eps = sum(self._first_invalid_counts.values())
            invalid_rate = total_invalid_eps / max(n_ep, 1) * 100
            lines.append(
                f"_包含至少一个 invalid 步的 episode 数：**{total_invalid_eps}** "
                f"({invalid_rate:.1f}% 的总回合)_"
            )
            lines.append("")
            lines.append("| Rank | (L, B) | 频次 | 占 invalid 比 |")
            lines.append("|-----:|:------:|----:|-------------:|")
            for rank, (key, count) in enumerate(top_inv, start=1):
                lines.append(
                    f"| {rank} | {key} | {count} | "
                    f"{count / max(total_invalid_eps, 1) * 100:.1f}% |"
                )
        lines.append("")

        # 4. PPO learning dynamics
        lines.append("## 4. PPO 学习动态（最近 10 次更新）")
        lines.append("")
        if self._ppo_history:
            recent = self._ppo_history[-10:]
            lines.append(
                "| Update | Eps | policy_loss | value_loss | entropy | clip_frac "
                "| ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |"
            )
            lines.append(
                "|------:|----:|------------:|-----------:|--------:|----------:"
                "|---------:|----------:|---------:|-----------------:|-------------:|-------------:|"
            )
            for u in recent:
                ec = float(getattr(u, "ent_coef", 0.0))
                lines.append(
                    f"| {u.update} | {u.completed_episodes} | {u.policy_loss:+.4f} | "
                    f"{u.value_loss:+.4f} | {u.entropy:+.4f} | {u.clip_fraction:.3f} | "
                    f"{ec:.5f} | {float(getattr(u, 'approx_kl', 0.0)):.5f} | "
                    f"{float(getattr(u, 'lr_scale', 1.0)):.3f} | "
                    f"{float(getattr(u, 'entropy_recovery_delta', 0.0)):.5f} | "
                    f"{u.window_mean_return:+.4f} | {u.window_mean_invalid:.2f} |"
                )
            if len(self._ppo_history) >= 2:
                ent0 = self._ppo_history[0].entropy
                ent1 = self._ppo_history[-1].entropy
                trend = "下降（policy 在收敛）" if ent1 < ent0 else "上升（policy 在分散）"
                lines.append("")
                lines.append(
                    f"_Entropy 趋势：{ent0:+.4f} → {ent1:+.4f}（{trend}）_"
                )
        else:
            lines.append("_暂无 PPO 更新（episode 数 < rollout_size）_")
        lines.append("")

        # 5. Action histogram quick overview
        lines.append("## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）")
        lines.append("")
        hist = self._action_hist
        n_samples_per_slot = max(int(hist.sum(axis=1).max()), 1)
        collapsed_slots = []
        spread_slots = []
        for slot_idx in range(self.num_slots):
            row = hist[slot_idx]
            total = int(row.sum())
            if total == 0:
                continue
            max_idx = int(np.argmax(row))
            max_share = float(row[max_idx]) / total
            if max_share > 0.85:
                collapsed_slots.append((slot_idx, max_idx, max_share, total))
            else:
                # entropy proxy
                p = row[row > 0].astype(np.float64) / total
                ent = float(-(p * np.log(p + 1e-12)).sum())
                spread_slots.append((slot_idx, ent, total))
        lines.append(
            f"- **已收敛 slot**（top-1 占比 > 85%）：**{len(collapsed_slots)}** / {self.num_slots}"
        )
        lines.append(
            f"- **未收敛 slot**：**{len(spread_slots)}** / {self.num_slots}"
        )
        if collapsed_slots and len(self._ppo_history) >= 5:
            sample = collapsed_slots[:8]
            lines.append("")
            lines.append("已收敛 slot 示例（前 8 个）：")
            for slot_idx, max_idx, share, _ in sample:
                lines.append(
                    f"  - slot[{slot_idx:03d}] → action_index={max_idx} "
                    f"（占比 {share*100:.1f}%）"
                )
        if spread_slots and len(self._ppo_history) >= 5:
            spread_slots.sort(key=lambda x: -x[1])
            sample = spread_slots[:8]
            lines.append("")
            lines.append("最分散 slot 示例（前 8 个）：")
            for slot_idx, ent, _ in sample:
                lines.append(
                    f"  - slot[{slot_idx:03d}] entropy={ent:.3f} "
                    f"(uniform≈{np.log(self.max_levels):.3f})"
                )
        lines.append("")

        # 6. auto-flags
        lines.append("## 6. 自动诊断（auto-flags）")
        lines.append("")
        flags: List[str] = []
        if n_ep >= 20:
            mean_last20 = float(np.mean(self._all_episode_returns[-20:]))
            mean_first20 = float(np.mean(self._all_episode_returns[:20]))
            if mean_last20 < mean_first20 - 0.05:
                flags.append(
                    f"⚠ **学习退化**：最近 20 回合平均回报 {mean_last20:+.4f} 低于前 "
                    f"20 回合 {mean_first20:+.4f}（Δ={mean_last20 - mean_first20:+.4f}）。"
                    "建议：降低 lr / 增加 ent_coef / 检查 invalid_penalty 是否过强。"
                )
            elif (
                abs(mean_last20 - mean_first20) < 0.05
                and n_ep > 200
                and len(self._ppo_history) > 5
            ):
                flags.append(
                    f"⚠ **训练停滞**：训练 {n_ep} 回合后回报变化 "
                    f"Δ={mean_last20 - mean_first20:+.4f}。"
                    "建议：检查 reward shaping 是否被 invalid_penalty dominat；"
                    "尝试提高 lr 或减小 invalid_penalty。"
                )
        if last50 and float(np.mean(last50_inv)) > 30:
            flags.append(
                f"⚠ **invalid 子步偏多**：最近 50 回合 mean invalid="
                f"{np.mean(last50_inv):.1f}/{3 + (self.num_layers - 1) * 4} (>30)。"
                "建议：检查 stage1 GELU/Softmax degree 是否过低、或 max_sfs 表损坏。"
            )
        if self._first_invalid_counts:
            top1 = self._first_invalid_counts.most_common(1)[0]
            total = sum(self._first_invalid_counts.values())
            if top1[1] > total * 0.3 and total > 30:
                key = top1[0]
                # extract LL and B
                try:
                    layer = int(key[1:3])
                    block = int(key[-1])
                except Exception:
                    layer = -1
                    block = -1
                flags.append(
                    f"⚠ **first-invalid 集中**：{top1[1] / max(total, 1) * 100:.0f}% "
                    f"的 invalid 都首先发生在 {key}。"
                    f"建议：查 stage1_degree[{layer}] / max_sfs 表对应 block {block} 的项。"
                )
        if self._ppo_history and len(self._ppo_history) >= 5:
            recent_ent = float(np.mean([u.entropy for u in self._ppo_history[-3:]]))
            if recent_ent < 0.1:
                flags.append(
                    f"⚠ **熵过低**：最近 3 次 PPO 更新平均 entropy={recent_ent:.3f} "
                    "(< 0.1)。policy 已经几乎确定性输出，可能过早收敛 — "
                    "增大 ent_coef 或 clip_range。"
                )
            recent_clip = float(np.mean([u.clip_fraction for u in self._ppo_history[-3:]]))
            if recent_clip > 0.40:
                flags.append(
                    f"⚠ **clip_fraction 偏高**：最近 3 次 PPO clip_frac="
                    f"{recent_clip:.2f}（>0.40）。lr 可能过大，"
                    "建议降低 lr 一档。"
                )
        # uniform-collapse check
        if n_ep > 200 and self._action_hist.sum() > 0:
            collapsed_pct = len(collapsed_slots) / max(self.num_slots, 1)
            if collapsed_pct > 0.95 and best_so_far < 0:
                flags.append(
                    f"⚠ **policy collapse**：{collapsed_pct*100:.0f}% 的 slot 已经收敛，"
                    f"但 best_reward 仍为负 ({best_so_far:+.4f})。"
                    "policy 可能锁死在差解；考虑重置或加大 ent_coef。"
                )
        if not flags:
            flags.append("✓ 暂无异常。")
        for fl in flags:
            lines.append(f"- {fl}")
        lines.append("")

        # 7. file index
        lines.append("## 7. 原始数据文件（machine-readable）")
        lines.append("")
        lines.append("| 文件 | 内容 |")
        lines.append("|------|------|")
        lines.append("| `episodes.jsonl` | 完整 per-episode 记录（append-only） |")
        lines.append("| `ppo_updates.jsonl` | 完整 per-PPO-update 记录（append-only） |")
        lines.append(f"| `top_candidates.jsonl` | Top-{self.top_k} 训练期 best：含每条候选的完整 `slots` 列表（人类可读） |")
        lines.append("| `pareto_frontier.jsonl` | 训练期非支配候选（质量 / 稳定性 / cost 多目标） |")
        lines.append("| `pareto_frontier.json` | Pareto frontier 元数据 + 完整候选列表 |")
        lines.append("| `pareto_frontier.html` | 可直接用浏览器打开的 Pareto frontier 表格 |")
        lines.append("| `first_invalid_counts.json` | (L, B) → 首次 invalid 计数 |")
        lines.append("| `action_histogram.npz` | (num_slots, max_levels) 频次矩阵 |")
        lines.append("| `baseline_action_vec.json` | static_skeletons baseline 的完整 `slots` 视图（参照系） |")
        lines.append("| `best_action_vec.json` | **训练期最优**：`slots` 列表（按 SF/K 选）+ `action_vec` 兜底字段。**可直接喂给 `Paean/run_final_eval.sh --action-config`** |")
        lines.append("")
        lines.append(
            "**重跑 final eval 的最简命令**（无需等训练结束）："
        )
        lines.append("")
        lines.append("```bash")
        lines.append(
            "bash Paean/run_final_eval.sh \\\n"
            f"    --preset mrpc-final-eval-only \\\n"
            f"    --action-config {self.best_json_path}"
        )
        lines.append("```")
        lines.append("")
        lines.append(
            "**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表"
            "中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。"
            "支持简写 `{\"base\":\"max\", \"overrides\":[{\"label\":\"L05.B3.K\", \"truncation_bits\":10}]}`。"
        )

        self._dump(lines)

    def _dump(self, lines: List[str]) -> None:
        tmp = self.summary_md_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        os.replace(tmp, self.summary_md_path)


def _atomic_json_dump(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _pareto_dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    objectives = [
        ("terminal_fusion_gain", "max"),
        ("terminal_k_gain", "max"),
        ("terminal_bits_gain", "max"),
    ]
    strictly_better = False
    for key, direction in objectives:
        av = _pareto_number(a.get(key), direction)
        bv = _pareto_number(b.get(key), direction)
        if direction == "min":
            if av > bv:
                return False
            if av < bv:
                strictly_better = True
        else:
            if av < bv:
                return False
            if av > bv:
                strictly_better = True
    return strictly_better


def _action_vec_hash(action_vec: Any) -> str:
    if hasattr(action_vec, "tolist"):
        action_vec = action_vec.tolist()
    payload = json.dumps(
        [int(x) for x in np.asarray(action_vec, dtype=int).reshape(-1).tolist()],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pareto_number(value: Any, direction: str) -> float:
    try:
        out = float(value)
    except Exception:
        return float("inf") if direction == "min" else float("-inf")
    if not np.isfinite(out):
        return float("inf") if direction == "min" else float("-inf")
    return out
