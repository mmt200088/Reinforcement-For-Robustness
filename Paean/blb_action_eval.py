from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from rfr.search.runtime.blb_bridge import BLBNoiseRLBridge
from rfr.search.runtime.model_handler import reseed_noise_rng
from rfr.search.common.action_space import (
    ActionDecodeResult,
    _block_default_N,
    _decode_block_field_values,
    action_vector_to_cfgs,
    avg_truncation_k_in_action,
    build_block_cfg_from_field_values,
    build_optimizer_requests,
    step_schedule,
    sum_truncation_k_in_action,
    validate_action_vector,
)
from rfr.preparation.rescale.baseline_bootstrap import (
    load_calibrated_stage2_action_context,
)
from rfr.search.common.feasibility import build_final_eval_feasibility
from rfr.search.common.eval_metrics import (
    pack_repeat_evaluation,
    summarize_selected_vs_random_results,
)
from rfr.preparation.fusion.count_map import FusionCountMap
from rfr.preparation.fusion.fixed_action import select_fusion_eval_metadata
from rfr.preparation.rescale.optimizer_cost import materialize_decoded_action
from final_evaluation_module import (
    UnifiedFinalEvaluationModule,
    require_final_evaluation_protocol,
)
from rfr.preparation.data.protocol import FINAL_EVAL_SPLIT
from rfr.common.json_utils import read_json_file, to_jsonable
from rfr.preparation.rescale.bridge import (
    RescaleOptimizerBridge,
    aggregate_optimizer_signals,
    build_rescale_invoker,
)

from .action_grid import (
    ActionCandidate,
    CostMatchedSamplingDiagnostics,
    build_action_candidates,
    build_cost_matched_random_action_candidates,
    coerce_spec_list,
)

_PLOT_RENDER_FALSE_VALUES = {"0", "false", "no", "off", "skip", "none"}


def _atomic_json(path: str, payload: Any) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(payload, stringify_unknown=True),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class BLBActionFinalEvaluationModule:
    """Focused final-eval for BLB action vectors.

    This module is intentionally separate from ``UnifiedFinalEvaluationModule``:
    it evaluates concrete BLB action vectors and optional cartesian ranges over
    action fields such as ``truncation`` and ``wffn1``.
    """

    def __init__(
        self,
        *,
        evaluator,
        config_source: str = "search",
        config_path: str = "glue_final_configs_best_ppo.json",
        manual_stage1_gelu: Optional[Sequence[int]] = None,
        manual_stage1_softmax: Optional[Sequence[int]] = None,
        random_seed: int = 42,
        random_enabled: bool = False,
        random_count: int = 0,
        repeat_n: int = 1,
        results_dir: Optional[str] = None,
        action_config_path: str = "",
        action_ranges=(),
        action_fixed=(),
        cost_match_count: int = 50,
        cost_match_max_attempts: int = 5000,
    ):
        self.evaluator = evaluator
        self.config_source = (config_source or "search").lower()
        self.config_path = config_path
        self.manual_stage1_gelu = manual_stage1_gelu
        self.manual_stage1_softmax = manual_stage1_softmax
        self.random_seed = int(random_seed)
        self.random_enabled = bool(random_enabled)
        self.random_count = max(0, int(random_count))
        self.repeat_n = max(1, int(repeat_n))
        self.cost_match_count = max(0, int(cost_match_count))
        self.cost_match_max_attempts = max(0, int(cost_match_max_attempts))
        default_results_dir = getattr(
            evaluator, "final_eval_dir", os.path.join("rl_results", "final_eval")
        )
        self.results_dir = results_dir or default_results_dir
        self.action_config_path = str(action_config_path or "").strip()
        self.action_ranges = coerce_spec_list(action_ranges)
        self.action_fixed = coerce_spec_list(action_fixed)
        self.rescale_optimizer_mode = self._load_rescale_optimizer_mode()

    @staticmethod
    def _render_plots_enabled() -> bool:
        raw = os.environ.get("RFR_PAEAN_RENDER_PLOTS", "1")
        return raw.strip().lower() not in _PLOT_RENDER_FALSE_VALUES

    @staticmethod
    def _capture_isolated_candidate_rng_state() -> Dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    def _restore_isolated_candidate_rng_state(
        self,
        metadata: Mapping[str, Any],
        state: Optional[Mapping[str, Any]],
    ) -> None:
        if not bool((metadata or {}).get("isolate_random_seed", False)):
            return
        if state is None:
            raise RuntimeError("isolated batch candidate RNG state was not captured")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        reseed_noise_rng(self.random_seed)

    def _evaluate_candidate_with_seed_lifecycle(
        self,
        *,
        metadata: Mapping[str, Any],
        isolated_candidate_rng_state: Optional[Mapping[str, Any]],  # noqa: UP045
        evaluate: Callable[[], Any],
    ) -> Any:
        candidate_metadata = dict(metadata or {})
        isolate_noise_rng = (
            candidate_metadata.get("isolate_random_seed") is True
        )
        self._restore_isolated_candidate_rng_state(
            candidate_metadata,
            isolated_candidate_rng_state,
        )
        try:
            return evaluate()
        finally:
            if isolate_noise_rng:
                reseed_noise_rng(None)

    def _validate_stage2_final_eval_handoff(
            self,
            search_best_stage2,
            *,
            expected_profile,
            ):
        comparator_backends = {"bo_rf", "greedy", "coinn_ga"}
        if not isinstance(search_best_stage2, Mapping):
            return None
        backend = str(search_best_stage2.get("search_backend") or "").lower()
        rl_variant = str(search_best_stage2.get("rl_variant") or "")
        variant_prefix = "blb_v3_layerwise_search_"
        variant_backend = ""
        if rl_variant.startswith(variant_prefix):
            variant_backend = rl_variant[len(variant_prefix):]
            if variant_backend.endswith("_smoke"):
                variant_backend = variant_backend[:-6]
        is_comparator = bool(
            backend in comparator_backends
            or variant_backend in comparator_backends
        )
        if not is_comparator:
            return None
        if backend not in comparator_backends or variant_backend != backend:
            raise ValueError(
                "comparator final-eval backend identity mismatch"
            )
        if search_best_stage2.get("status") != "completed":
            raise ValueError(
                "comparator final-eval requires a completed Stage-2 result"
            )
        if search_best_stage2.get("strict_feasible") is not True:
            raise ValueError(
                "comparator final-eval requires a strict-feasible Stage-2 result"
            )
        profile = str(search_best_stage2.get("blb_v3_profile") or "")
        if profile != str(expected_profile):
            raise ValueError(
                "comparator final-eval profile does not match the active dataset"
            )
        if search_best_stage2.get("blb_v3_fusion_count_action") is not True:
            raise ValueError(
                "comparator final-eval requires fusion-count action metadata"
            )
        group = search_best_stage2.get("blb_v3_best_action_group")
        if not isinstance(group, Mapping):
            raise ValueError(
                "comparator final-eval selected action group is missing"
            )
        raw_matrix = group.get("policy_actions")
        raw_vector = search_best_stage2.get("blb_v3_best_action_vec")
        raw_overrides = group.get("boosted_overrides")
        if not isinstance(raw_matrix, (list, tuple)) or not all(
                isinstance(row, (list, tuple)) for row in raw_matrix
        ):
            raise ValueError(
                "comparator final-eval action matrix is missing"
            )
        if raw_vector is None:
            raise ValueError(
                "comparator final-eval full vector is missing"
            )
        if not isinstance(raw_overrides, (list, tuple)):
            raise ValueError(
                "comparator final-eval boosted overrides are missing"
            )
        fingerprint = str(
            search_best_stage2.get("final_config_fingerprint") or ""
        )
        if (
                len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise ValueError(
                "comparator final-eval config fingerprint is invalid"
            )
        return {
            "status": "completed",
            "search_backend": backend,
            "rl_variant": rl_variant,
            "strict_feasible": True,
            "blb_v3_profile": profile,
            "blb_v3_fusion_count_action": True,
            "blb_v3_best_action_vec": [int(item) for item in raw_vector],
            "blb_v3_best_action_group": {
                **dict(group),
                "policy_actions": [list(row) for row in raw_matrix],
                "boosted_overrides": [dict(item) for item in raw_overrides],
            },
            "final_config_fingerprint": fingerprint,
        }

    def _validate_prepared_materialization(
            self,
            final_eval_handoff,
            *,
            materialized,
            ):
        if not isinstance(final_eval_handoff, Mapping):
            raise ValueError(
                "comparator final-eval handoff is missing"
            )
        if not bool(getattr(materialized, "model_ready", False)):
            reason = str(getattr(materialized, "failure_reason", "") or "")
            raise ValueError(
                "comparator final-eval selected action is not model-ready"
                + (f": {reason}" if reason else "")
            )
        expected_fingerprint = str(
            final_eval_handoff.get("final_config_fingerprint") or ""
        )
        actual_fingerprint = str(
            getattr(materialized, "final_config_fingerprint", "") or ""
        )
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(
                "comparator final-eval config fingerprint mismatch"
            )
        return {
            "schema_version": (
                "stage2_final_eval_materialization_consistency_v1"
            ),
            "checked_before_forward": True,
            "expected_final_config_fingerprint": expected_fingerprint,
            "final_config_fingerprint": actual_fingerprint,
            "final_config_fingerprint_exact_match": True,
        }

    def _validate_selected_candidate_handoff(
            self,
            selected_candidates,
            final_eval_handoff,
            ):
        if final_eval_handoff is None:
            return
        if not selected_candidates:
            raise ValueError(
                "comparator final-eval produced no selected candidate"
            )
        candidate = selected_candidates[0]
        actual_vector = np.asarray(
            candidate.action_vec, dtype=int,
        ).reshape(-1).tolist()
        expected_vector = final_eval_handoff.get("blb_v3_best_action_vec")
        if actual_vector != expected_vector:
            raise ValueError(
                "comparator final-eval selected candidate full vector mismatch"
            )
        metadata = getattr(candidate, "metadata", None)
        group = metadata.get("group") if isinstance(metadata, Mapping) else None
        if not isinstance(group, Mapping):
            raise ValueError(
                "comparator final-eval selected candidate has no fusion group"
            )
        expected_group = final_eval_handoff.get("blb_v3_best_action_group")
        if not isinstance(expected_group, Mapping):
            raise ValueError(
                "comparator final-eval handoff has no selected action group"
            )
        if group.get("policy_actions") != expected_group.get("policy_actions"):
            raise ValueError(
                "comparator final-eval selected candidate action matrix mismatch"
            )
        if group.get("boosted_overrides") != expected_group.get(
                "boosted_overrides"
        ):
            raise ValueError(
                "comparator final-eval selected candidate boosted overrides mismatch"
            )

    def run(
        self,
        search_best_stage1: Optional[dict],
        search_best_stage2: Optional[dict],
        baseline_stage1_gelu: np.ndarray,
        baseline_stage1_softmax: np.ndarray,
        baseline_noise_tot_c: float,
        limit_loss: float,
        limit_p: float,
        limit_s: float,
    ) -> Dict[str, object]:
        protocol = require_final_evaluation_protocol(
            self.evaluator,
            search_results=(search_best_stage1, search_best_stage2),
            requested_split=FINAL_EVAL_SPLIT,
        )
        self.final_eval_split = protocol["split_name"]
        if self.random_enabled and self.action_ranges:
            raise ValueError("BLB action final_eval random mode cannot be combined with action ranges")

        ev = self.evaluator
        profile = str(getattr(ev, "dataset_key", "default") or "default")
        final_eval_handoff = self._validate_stage2_final_eval_handoff(
            search_best_stage2,
            expected_profile=profile,
        )
        total_layers = int(ev.total_layers)
        (
            self.rescale_bridge,
            self.rescale_backend,
            self.rescale_optimizer_root,
        ) = self._build_rescale_bridge(
            profile,
        )

        stage1_resolver = UnifiedFinalEvaluationModule(
            evaluator=ev,
            config_source=self.config_source,
            config_path=self.config_path,
            manual_stage1_gelu=self.manual_stage1_gelu,
            manual_stage1_softmax=self.manual_stage1_softmax,
            manual_stage2_noise=None,
            random_seed=self.random_seed,
            permutation_trials=0,
            cost_equivalent_trials=0,
            budget_equivalent_trials=0,
            stage1_budget_trials=0,
            stage2_budget_trials=0,
            repeat_n=self.repeat_n,
            results_dir=self.results_dir,
        )
        opt_gelu, opt_softmax, stage1_source = stage1_resolver.resolve_stage1_only(
            search_best_stage1=search_best_stage1,
            total_layers=total_layers,
        )
        opt_gelu = np.asarray(opt_gelu, dtype=int)
        opt_softmax = np.asarray(opt_softmax, dtype=int)
        action_context = load_calibrated_stage2_action_context(
            rescale_optimizer_root=self.rescale_optimizer_root,
            dataset=profile,
            num_layers=total_layers,
            gelu_per_layer=opt_gelu,
            softmax_per_layer=opt_softmax,
            snap_sf_to_noise_table=False,
        )

        base_action = self._resolve_base_action(search_best_stage2)


        selected_candidates = build_action_candidates(
            num_layers=total_layers,
            profile=profile,
            base_action_vec=base_action,
            fixed_specs=self.action_fixed,
            range_specs=self.action_ranges,
            action_config_path=self.action_config_path,
            max_sfs=action_context.max_sfs,
            gelu_degree=opt_gelu,
            attn_degree=opt_softmax,
            isolate_random_seed=(final_eval_handoff is not None),
        )


        fusion_count_action = bool(
            isinstance(search_best_stage2, dict)
            and search_best_stage2.get("blb_v3_fusion_count_action")
        )
        fusion_group = (
            search_best_stage2.get("blb_v3_best_action_group")
            if isinstance(search_best_stage2, dict) else None
        )
        if (fusion_count_action or fusion_group is not None) and base_action is not None:
            patched: List[ActionCandidate] = []
            for cand in selected_candidates:
                try:
                    md = select_fusion_eval_metadata(
                        action_vec=cand.action_vec,
                        base_action=base_action,
                        existing_metadata=cand.metadata,
                        fusion_group=fusion_group,
                        fusion_count_action=fusion_count_action,
                        profile=profile,
                        num_layers=total_layers,
                        gelu=opt_gelu,
                        softmax=opt_softmax,
                    )
                    patched.append(dataclasses.replace(cand, metadata=md))
                except Exception as exc:
                    if final_eval_handoff is not None:
                        raise RuntimeError(
                            "comparator final-eval could not attach the selected "
                            "fusion metadata"
                        ) from exc
                    ev.log(f"  [blb-eval][warning] fusion metadata resolve failed for {cand.name}: {exc}")
                    patched.append(cand)
            selected_candidates = patched

        if final_eval_handoff is not None:
            selected_candidates = [
                dataclasses.replace(
                    candidate,
                    metadata={
                        **dict(candidate.metadata or {}),
                        "isolate_random_seed": True,
                    },
                )
                for candidate in selected_candidates
            ]

        self._validate_selected_candidate_handoff(
            selected_candidates,
            final_eval_handoff,
        )

        prepared_selected_materialized = None
        selected_materialization_consistency = None
        if final_eval_handoff is not None:
            if len(selected_candidates) != 1:
                raise ValueError(
                    "comparator final-eval requires exactly one selected candidate"
                )
            self._stage2_fusion_map = FusionCountMap.load(profile)
            selected_candidate = selected_candidates[0]
            selected_metadata = dict(selected_candidate.metadata or {})
            decoded = self._decode_action_candidate(
                action_vec=selected_candidate.action_vec,
                metadata=selected_metadata,
                max_sfs=action_context.max_sfs,
                num_layers=total_layers,
                gelu=opt_gelu,
                softmax=opt_softmax,
                profile=profile,
            )
            cfgs_dict = decoded.cfgs_dict()
            opt_outputs, opt_signals = self._optimizer_outputs(
                profile, cfgs_dict,
            )
            prepared_selected_materialized = self._materialize_decoded_action(
                profile=profile,
                action_vec=selected_candidate.action_vec,
                decoded=decoded,
                cfgs_dict=cfgs_dict,
                opt_outputs=opt_outputs,
                opt_signals=opt_signals,
            )
            selected_materialization_consistency = (
                self._validate_prepared_materialization(
                    final_eval_handoff,
                    materialized=prepared_selected_materialized,
                )
            )

        os.makedirs(self.results_dir, exist_ok=True)
        metric_names = ev.get_metric_short_names()
        num_metrics = ev.get_num_metrics()
        ev.log("\n" + "=" * 60)
        ev.log("PHASE: BLB ACTION FINAL EVALUATION (validation_full)")
        ev.log(f"CONFIG_SOURCE={self.config_source}  STAGE1_SOURCE={stage1_source}")
        ev.log(
            f"RESCALE_OPTIMIZER={self.rescale_backend} "
            f"root={self.rescale_optimizer_root or '(none)'} "
            f"mode={self.rescale_optimizer_mode}"
        )
        ev.log(
            f"selected_candidates={len(selected_candidates)} "
            f"random_enabled={self.random_enabled} "
            f"cost_match_count={self.cost_match_count} "
            f"repeat={self.repeat_n}"
        )
        if self.action_ranges:
            ev.log(f"action_ranges={list(self.action_ranges)}")
        if self.action_fixed:
            ev.log(f"action_fixed={list(self.action_fixed)}")
        ev.log("=" * 60)

        baseline_result = self._evaluate_clean_baseline(
            baseline_stage1_gelu=baseline_stage1_gelu,
            baseline_stage1_softmax=baseline_stage1_softmax,
        )
        report_constraints = ev.build_constraint_limits_from_metrics(
            baseline_result["loss"],
            baseline_result["p"],
            baseline_result["s"],
        )
        isolated_candidate_rng_state = None
        if any(
            bool((candidate.metadata or {}).get("isolate_random_seed", False))
            for candidate in selected_candidates
        ):
            isolated_candidate_rng_state = (
                self._capture_isolated_candidate_rng_state()
            )


        selected_results: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(selected_candidates, start=1):
            ev.log(
                f"\n--- BLB selected candidate {idx}/{len(selected_candidates)}: {candidate.name} ---"
            )
            result = self._evaluate_candidate_with_seed_lifecycle(
                metadata=candidate.metadata,
                isolated_candidate_rng_state=isolated_candidate_rng_state,
                evaluate=lambda: self._evaluate_action_candidate(
                    name=candidate.name,
                    action_vec=candidate.action_vec,
                    overrides=candidate.overrides,
                    metadata=candidate.metadata,
                    gelu=opt_gelu,
                    softmax=opt_softmax,
                    report_constraints=report_constraints,
                    max_sfs=action_context.max_sfs,
                    prepared_materialized=prepared_selected_materialized,
                    materialization_consistency=(
                        selected_materialization_consistency
                    ),
                ),
            )
            selected_results.append(result)
            ev.log(
                f"  {candidate.name}: Loss={result['loss']:.4f}, "
                f"{metric_names[0]}={result['p']:.4f}"
                + (f", {metric_names[1]}={result['s']:.4f}" if num_metrics > 1 else "")
                + f", avg_k={result['avg_truncation_k']:.2f}, bits={result['total_bits_sum']}, "
                f"fusion={result['total_fusion_count']}"
            )


        cost_match_diagnostics: Optional[CostMatchedSamplingDiagnostics] = None
        random_results: List[Dict[str, Any]] = []
        if self.random_enabled and self.cost_match_count > 0 and selected_results:
            anchor = selected_results[0]
            anchor_action = np.asarray(selected_candidates[0].action_vec, dtype=int)
            target_total_bits = int(anchor["total_bits_sum"])
            target_total_fusion = int(anchor["total_fusion_count"])
            target_sum_k = sum_truncation_k_in_action(anchor_action, total_layers)
            ev.log(
                "\n--- Cost-matched random sampling ---\n"
                f"  anchor: {selected_candidates[0].name} "
                f"(total_bits={target_total_bits}, total_fusion={target_total_fusion}, "
                f"sum_k={target_sum_k}, avg_k={anchor['avg_truncation_k']:.3f})\n"
                f"  target: {self.cost_match_count} matched configs, "
                f"max {self.cost_match_max_attempts} attempts"
            )
            random_candidates, cost_match_diagnostics = (
                build_cost_matched_random_action_candidates(
                    num_layers=total_layers,
                    profile=profile,
                    selected_action_vec=anchor_action,
                    selected_total_bits=target_total_bits,
                    selected_total_fusion=target_total_fusion,
                    selected_sum_k=target_sum_k,
                    bridge=self.rescale_bridge,
                    max_sfs=action_context.max_sfs,
                    gelu_degree=opt_gelu,
                    attn_degree=opt_softmax,
                    seed=self.random_seed,
                    count=self.cost_match_count,
                    max_attempts=self.cost_match_max_attempts,
                    fixed_specs=self.action_fixed,
                    log_fn=ev.log,
                )
            )
            if final_eval_handoff is not None:
                random_candidates = [
                    dataclasses.replace(
                        candidate,
                        metadata={
                            **dict(candidate.metadata or {}),
                            "isolate_random_seed": True,
                        },
                    )
                    for candidate in random_candidates
                ]
            for idx, candidate in enumerate(random_candidates, start=1):
                ev.log(
                    f"\n--- BLB random candidate {idx}/{len(random_candidates)}: "
                    f"{candidate.name} ---"
                )
                result = self._evaluate_candidate_with_seed_lifecycle(
                    metadata=candidate.metadata,
                    isolated_candidate_rng_state=(
                        isolated_candidate_rng_state
                    ),
                    evaluate=lambda: self._evaluate_action_candidate(
                        name=candidate.name,
                        action_vec=candidate.action_vec,
                        overrides=candidate.overrides,
                        metadata=candidate.metadata,
                        gelu=opt_gelu,
                        softmax=opt_softmax,
                        report_constraints=report_constraints,
                        max_sfs=action_context.max_sfs,
                    ),
                )
                random_results.append(result)
                ev.log(
                    f"  {candidate.name}: Loss={result['loss']:.4f}, "
                    f"{metric_names[0]}={result['p']:.4f}"
                    + (f", {metric_names[1]}={result['s']:.4f}" if num_metrics > 1 else "")
                    + f", avg_k={result['avg_truncation_k']:.2f}, bits={result['total_bits_sum']}, "
                    f"fusion={result['total_fusion_count']}"
                )

        results = selected_results + random_results
        self._attach_relative_metrics(baseline_result, results)
        comparison_summary = self._summarize_selected_vs_random(
            selected_results=selected_results,
            random_results=random_results,
            num_metrics=num_metrics,
        )
        cost_match_payload = self._cost_match_diagnostics_to_dict(cost_match_diagnostics)
        summary_path = self._save_results_json(
            selected_source=f"blb_action(stage1={stage1_source})",
            baseline_stage1_gelu=baseline_stage1_gelu,
            baseline_stage1_softmax=baseline_stage1_softmax,
            opt_gelu=opt_gelu,
            opt_softmax=opt_softmax,
            baseline_result=baseline_result,
            candidate_results=results,
            selection_constraints={
                "limit_loss": float(limit_loss),
                "limit_primary_metric": float(limit_p),
                "limit_secondary_metric": float(limit_s),
            },
            comparison_summary=comparison_summary,
            cost_match_diagnostics=cost_match_payload,
            action_context_provenance=action_context.provenance,
            final_eval_handoff=final_eval_handoff,
        )
        selected_result = selected_results[0] if selected_results else None
        selected_candidate = (
            selected_candidates[0] if selected_candidates else None
        )
        text_path = self._save_results_markdown(
            json_path=summary_path,
            selected_source=f"blb_action(stage1={stage1_source})",
            baseline_result=baseline_result,
            candidate_results=results,
            comparison_summary=comparison_summary,
            cost_match_diagnostics=cost_match_payload,
        )
        plot_path = None
        scatter_path = None
        if self._render_plots_enabled():
            plot_path = self._save_results_plot(candidate_results=results)
            scatter_path = self._save_scatter_plot(
                selected_results=selected_results,
                random_results=random_results,
            )
        else:
            ev.log(
                "BLB action final-eval plots deferred; set "
                "RFR_PAEAN_RENDER_PLOTS=1 to enable."
            )
        ev.log(f"BLB action final-eval summary saved to: {summary_path}")
        ev.log(f"BLB action final-eval text report saved to: {text_path}")
        if plot_path:
            ev.log(f"BLB action final-eval plot saved to: {plot_path}")
        if scatter_path:
            ev.log(f"BLB action scatter plot saved to: {scatter_path}")

        ev.apply_configuration(opt_gelu, opt_softmax)
        self._clear_all_noise()
        best = selected_result
        return {
            "final_eval_split": self.final_eval_split,
            "dataset_protocol_hash": protocol["dataset_protocol_hash"],
            "validation_example_count": protocol["example_count"],
            "selected_source": f"blb_action(stage1={stage1_source})",
            "opt_gelu": opt_gelu,
            "opt_softmax": opt_softmax,
            "opt_noise_config": {},
            "baseline_result": baseline_result,
            "optimized_result": best,
            "candidate_results": results,
            "selected_results": (
                [] if best is None else [best]
            ),
            "random_results": random_results,
            "random_summary": comparison_summary or {},
            "cost_match_diagnostics": cost_match_payload,
            "calibrated_action_context": to_jsonable(action_context.provenance),
            "summary_path": summary_path,
            "text_report_path": text_path,
            "plot_path": plot_path,
            "scatter_path": scatter_path,
            "stage2_final_eval_handoff": final_eval_handoff,
            "variance_plot_path": None,
        }

    def _resolve_base_action(self, search_best_stage2):
        if isinstance(search_best_stage2, dict):
            for key in ("blb_v3_best_action_vec", "best_action_vec", "best_action"):
                raw = search_best_stage2.get(key)
                if raw is None:
                    continue
                arr = np.asarray(raw)
                if arr.size > 0:
                    return arr
        return None

    def _evaluate_clean_baseline(self, *, baseline_stage1_gelu, baseline_stage1_softmax):
        repeats = max(1, int(getattr(self, "repeat_n", 1)))
        if repeats <= 1:
            single = self._run_single_clean_baseline_eval(
                baseline_stage1_gelu=baseline_stage1_gelu,
                baseline_stage1_softmax=baseline_stage1_softmax,
            )
            single["loss_std"] = 0.0
            single["p_std"] = 0.0
            single["s_std"] = 0.0
            single["evaluation_n"] = 1
            single["evaluation_protocol"] = "single_validation_full"
            return single

        trials = self._run_clean_baseline_trials(
            baseline_stage1_gelu=baseline_stage1_gelu,
            baseline_stage1_softmax=baseline_stage1_softmax,
            repeats=repeats,
        )
        repeat = pack_repeat_evaluation(
            trials,
            evaluation_mode="clean_baseline_repeated_validation_full",
        )
        stats = repeat["stats"]
        result = {
            "name": "Baseline (Stage-1 Exact)",
            "family": "Baseline",
            "loss": float(stats["loss_mean"]),
            "p": float(stats["p_mean"]),
            "s": float(stats["s_mean"]),
            "time_ms": float(stats["time_mean_ms"]),
            "loss_std": float(stats["loss_std"]),
            "p_std": float(stats["p_std"]),
            "s_std": float(stats["s_std"]),
            "evaluation_n": int(stats["n"]),
            "evaluation_protocol": f"repeated_mean_n={int(stats['n'])}",
            "repeat_evaluation": repeat,
        }
        return result

    def _run_single_clean_baseline_eval(self, *, baseline_stage1_gelu, baseline_stage1_softmax):
        return self._run_clean_baseline_trials(
            baseline_stage1_gelu=baseline_stage1_gelu,
            baseline_stage1_softmax=baseline_stage1_softmax,
            repeats=1,
        )[0]

    def _run_clean_baseline_trials(
        self,
        *,
        baseline_stage1_gelu,
        baseline_stage1_softmax,
        repeats: int,
    ):
        ev = self.evaluator
        ev.apply_configuration(
            np.asarray(baseline_stage1_gelu, dtype=int),
            np.asarray(baseline_stage1_softmax, dtype=int),
        )
        self._clear_all_noise()
        split_name = ev._resolve_eval_split(
            use_train=False,
            split=getattr(self, "final_eval_split", FINAL_EVAL_SPLIT),
        )
        trials = []
        for _idx in range(max(1, int(repeats))):
            loss, p, s, t = ev._run_evaluation(
                ev.dataloaders[split_name],
                use_train=False,
                split_name=split_name,
            )
            trials.append({
                "name": "Baseline (Stage-1 Exact)",
                "family": "Baseline",
                "loss": float(loss),
                "p": float(p),
                "s": float(s),
                "time_ms": float(t),
            })
        return trials

    def _evaluate_action_candidate(
            self,
            *,
            name,
            action_vec,
            overrides,
            gelu,
            softmax,
            report_constraints,
            max_sfs,
            metadata=None,
            prepared_materialized=None,
            materialization_consistency=None,
            ):
        ev = self.evaluator
        total_layers = int(ev.total_layers)
        profile = str(getattr(ev, "dataset_key", "default") or "default")
        metadata = dict(metadata or {})
        if prepared_materialized is None:
            decoded = self._decode_action_candidate(
                action_vec=action_vec,
                metadata=metadata,
                max_sfs=max_sfs,
                num_layers=total_layers,
                gelu=gelu,
                softmax=softmax,
                profile=profile,
            )
            cfgs_dict = decoded.cfgs_dict()
            opt_outputs, opt_signals = self._optimizer_outputs(
                profile, cfgs_dict,
            )
            materialized = self._materialize_decoded_action(
                profile=profile,
                action_vec=action_vec,
                decoded=decoded,
                cfgs_dict=cfgs_dict,
                opt_outputs=opt_outputs,
                opt_signals=opt_signals,
            )
        else:
            materialized = prepared_materialized
            actual_action = [
                int(value)
                for value in np.asarray(action_vec, dtype=int).reshape(-1)
            ]
            if list(materialized.action_indices) != actual_action:
                raise ValueError(
                    "prepared Stage-2 materialization action mismatch"
                )
            decoded = materialized.decoded
            opt_outputs = materialized.outputs
            opt_signals = materialized.signals
        decoded = materialized.decoded
        replan_application = materialized.replan_application
        skip_reason = str(materialized.failure_reason or "")
        if skip_reason not in {
                "",
                "optimizer_invalid_chain",
                "optimizer_output_set_mismatch",
                "replan_config_not_fully_applied",
        }:
            raise RuntimeError(f"unexpected action materialization failure: {skip_reason}")
        skipped_forward = bool(skip_reason)
        if skipped_forward:
            single = {
                "loss": float("inf"),
                "p": 0.0,
                "s": 0.0,
                "time_ms": 0.0,
                "install_verification": {
                    "model_will_use_selected_cfg": False,
                    "skipped": True,
                    "skip_reason": skip_reason,
                },
            }
            repeat = None
        else:
            single, repeat = self._run_blb_eval(decoded, gelu=gelu, softmax=softmax)
        if repeat is not None:
            stats = repeat["stats"]
            loss = float(stats["loss_mean"])
            p = float(stats["p_mean"])
            s = float(stats["s_mean"])
            time_ms = float(stats["time_mean_ms"])
        else:
            loss = float(single["loss"])
            p = float(single["p"])
            s = float(single["s"])
            time_ms = float(single["time_ms"])

        stage1_tot, g_c, s_c = ev.get_simulated_cost(gelu, softmax)
        result = {
            "name": str(name),
            "family": "BLBActionRandom" if str(name).startswith("ActionRandom_") else "BLBAction",
            "loss": loss,
            "p": p,
            "s": s,
            "time_ms": time_ms,
            "stage1_tot_c": float(stage1_tot),
            "stage1_g_c": float(g_c),
            "stage1_s_c": float(s_c),
            "total_bits_sum": int(opt_signals.total_bits_sum),
            "total_fusion_count": int(opt_signals.total_fusion_count),
            "invalid_block_count": int(opt_signals.invalid_block_count),
            "valid_block_count": int(opt_signals.valid_block_count),
            "any_invalid": bool(opt_signals.any_invalid),
            "action_metadata": to_jsonable(metadata),
            "fusion_group_diagnostics": self._fusion_group_diagnostics(
                metadata=metadata or {},
                opt_signals=opt_signals,
            ),
            "avg_truncation_k": float(avg_truncation_k_in_action(action_vec, total_layers)),
            "action_overrides": dict(overrides or {}),
            "action_vec": np.asarray(action_vec, dtype=int).copy(),
            "config_details": self._config_details(decoded, action_vec, overrides, opt_outputs),
            "replan_application": replan_application,
            "final_config_fingerprint": materialized.final_config_fingerprint,
            "materialization_consistency": to_jsonable(
                materialization_consistency or {}
            ),
            "rescale_optimizer": {
                "invoker_kind": str(getattr(self, "rescale_backend", "unknown")),
                "root": str(getattr(self, "rescale_optimizer_root", "") or ""),
                "mode": str(getattr(self, "rescale_optimizer_mode", "cfg_derived")),
                "request_count": int(len(opt_outputs)),
                "valid_count": int(sum(1 for o in opt_outputs.values() if o.valid)),
                "invalid_count": int(sum(1 for o in opt_outputs.values() if not o.valid)),
                "t_new_sources": sorted({
                    str((o.raw or {}).get("_t_new_source", "unknown"))
                    for o in opt_outputs.values()
                }),
            },
            "install_verification": single.get("install_verification", {}),
            "skipped_forward": bool(skipped_forward),
        }
        if skip_reason:
            result["forward_skipped_reason"] = skip_reason
        if repeat is not None:
            stats = repeat["stats"]
            result.update(
                {
                    "evaluation_n": int(stats["n"]),
                    "loss_std": float(stats["loss_std"]),
                    "p_std": float(stats["p_std"]),
                    "s_std": float(stats["s_std"]),
                    "evaluation_protocol": f"repeated_mean_n={int(stats['n'])}",
                    "repeat_evaluation": repeat,
                }
            )
        else:
            result["loss_std"] = 0.0
            result["p_std"] = 0.0
            result["s_std"] = 0.0
            result["evaluation_n"] = 0 if skipped_forward else 1
            result["evaluation_protocol"] = (
                f"skipped_forward:{skip_reason}"
                if skipped_forward
                else "single_validation_full"
            )
        feasibility = self._build_feasibility_report(
            result=result,
            report_constraints=report_constraints,
            optimizer_valid=not bool(materialized.optimizer_invalid),
            decode_ok=True,
            apply_ok=bool(replan_application.get("model_uses_replan_config", False)),
            eval_ok=not bool(skipped_forward),
        )
        result["final_eval_feasibility"] = feasibility
        result["feasible"] = feasibility.get("feasible")
        result["diagnostic_feasible"] = feasibility.get("diagnostic_feasible")
        result["strict_feasible"] = feasibility.get("strict_feasible")
        return result

    def _build_feasibility_report(self, *, result, report_constraints, optimizer_valid, decode_ok, apply_ok, eval_ok):
        ev = self.evaluator
        threshold_source = str(
            getattr(ev, "blb_final_eval_threshold_source", "")
            or getattr(ev, "final_eval_threshold_source", "")
            or "baseline-derived"
        )
        acc_std_limit = (
            report_constraints.get("metric1_std")
            if isinstance(report_constraints, dict) else None
        )
        f1_std_limit = (
            report_constraints.get("metric2_std")
            if isinstance(report_constraints, dict) else None
        )
        acc_std_limit = getattr(ev, "blb_final_eval_metric1_std_limit", acc_std_limit)
        f1_std_limit = getattr(ev, "blb_final_eval_metric2_std_limit", f1_std_limit)
        strict_z = getattr(ev, "blb_final_eval_strict_z", 1.0)
        return build_final_eval_feasibility(
            optimizer_valid=bool(optimizer_valid),
            decode_ok=bool(decode_ok),
            apply_ok=bool(apply_ok),
            eval_ok=bool(eval_ok),
            acc_mean=result.get("p"),
            f1_mean=result.get("s"),
            acc_std=result.get("p_std", 0.0),
            f1_std=result.get("s_std", 0.0),
            acc_limit=report_constraints.get("metric1") if isinstance(report_constraints, dict) else None,
            f1_limit=report_constraints.get("metric2") if isinstance(report_constraints, dict) else None,
            acc_std_limit=acc_std_limit,
            f1_std_limit=f1_std_limit,
            loss_mean=result.get("loss"),
            loss_std=result.get("loss_std", 0.0),
            threshold_source=threshold_source,
            strict_z=float(strict_z),
        )

    def _optimizer_outputs(self, profile: str, cfgs_dict):
        bridge = getattr(self, "rescale_bridge", None)
        if bridge is None:
            bridge, kind, root = self._build_rescale_bridge(profile)
            self.rescale_bridge = bridge
            self.rescale_backend = kind
            self.rescale_optimizer_root = root
        requests = build_optimizer_requests(profile, cfgs_dict)
        outputs = bridge.evaluate_blocks(requests)
        return outputs, aggregate_optimizer_signals(outputs)

    def _decode_action_candidate(
            self,
            *,
            action_vec,
            metadata: Mapping[str, Any],
            max_sfs,
            num_layers: int,
            gelu,
            softmax,
            profile: str,
            ):
        if str(metadata.get("schema_version", "")) == "fusion_count_fixed_action_v1":
            return self._decode_fusion_count_fixed_action(
                action_vec=action_vec,
                metadata=metadata,
                max_sfs=max_sfs,
                num_layers=int(num_layers),
                gelu=gelu,
                softmax=softmax,
                profile=str(profile),
            )
        return action_vector_to_cfgs(
            action_vec=action_vec,
            max_sfs=max_sfs,
            num_layers=int(num_layers),
            gelu_degree=np.asarray(gelu, dtype=int),
            attn_degree=np.asarray(softmax, dtype=int),
        )

    def _decode_fusion_count_fixed_action(
            self,
            *,
            action_vec,
            metadata: Mapping[str, Any],
            max_sfs,
            num_layers: int,
            gelu,
            softmax,
            profile: str,
            ) -> ActionDecodeResult:
        group = metadata.get("group")
        if not isinstance(group, Mapping):
            raise ValueError("fusion_count_fixed_action_v1 requires group metadata")
        raw_option_by_graph = group.get("option_by_graph")
        raw_option_by_step = group.get("option_by_step")
        if not isinstance(raw_option_by_graph, Mapping) and not isinstance(raw_option_by_step, Mapping):
            raise ValueError(
                "fusion_count_fixed_action_v1 requires group.option_by_step or "
                "group.option_by_graph metadata"
            )
        raw_boosted_overrides = group.get("boosted_overrides", ())
        if raw_boosted_overrides is None:
            raw_boosted_overrides = ()
        if not isinstance(raw_boosted_overrides, (list, tuple)):
            raise ValueError("group.boosted_overrides must be a sequence of rows")
        boosted_overrides = {}
        for row in raw_boosted_overrides:
            if not isinstance(row, Mapping):
                raise ValueError("group.boosted_overrides rows must be mappings")
            field_values = row.get("field_values")
            if not isinstance(field_values, Mapping):
                raise ValueError(
                    "group.boosted_overrides field_values must be a mapping"
                )
            key = (int(row["block_idx"]), int(row["layer_idx"]))
            if key in boosted_overrides:
                raise ValueError(
                    "group.boosted_overrides contains duplicate "
                    f"block/layer row {key}"
                )
            boosted_overrides[key] = {
                str(name): int(value) for name, value in field_values.items()
            }

        base_raw = None
        for key in ("legacy_action_vec", "base", "action_vec"):
            value = metadata.get(key)
            if isinstance(value, (list, tuple, np.ndarray)):
                base_raw = value
                break
        if base_raw is None:
            base_raw = action_vec
        base_arr = validate_action_vector(base_raw, int(num_layers))
        gelu_arr = np.asarray(gelu, dtype=int).reshape(-1)
        softmax_arr = np.asarray(softmax, dtype=int).reshape(-1)

        decoded = action_vector_to_cfgs(
            action_vec=base_arr,
            max_sfs=max_sfs,
            num_layers=int(num_layers),
            gelu_degree=gelu_arr,
            attn_degree=softmax_arr,
        )

        try:
            fusion_map = getattr(self, "_stage2_fusion_map", None)
            if fusion_map is None:
                from rfr.preparation.fusion.count_map import (
                    FusionCountMap as RuntimeFusionCountMap,
                )

                fusion_map = RuntimeFusionCountMap.load(str(profile))
        except Exception as exc:
            raise RuntimeError(
                f"failed to load fusion-count map for profile={profile!r}: {exc}"
            ) from exc

        option_by_graph = (
            {str(k): int(v) for k, v in raw_option_by_graph.items()}
            if isinstance(raw_option_by_graph, Mapping)
            else {}
        )
        option_by_step = (
            {str(k): int(v) for k, v in raw_option_by_step.items()}
            if isinstance(raw_option_by_step, Mapping)
            else {}
        )
        schedule = step_schedule(
            int(num_layers),
            profile=str(profile),
            attn_degree_per_layer=softmax_arr.tolist(),
            gelu_degree_per_layer=gelu_arr.tolist(),
        )
        for step in schedule:
            graph_key = str(step.graph_key_suffix)
            step_key = str(int(step.step_idx))
            if step_key in option_by_step:
                option_id = int(option_by_step[step_key])
            elif graph_key in option_by_graph:
                option_id = int(option_by_graph[graph_key])
            else:
                continue
            graph = fusion_map.graphs.get(graph_key)
            if graph is None:
                raise KeyError(f"fusion map missing graph {graph_key!r}")
            option = None
            for candidate in graph.options:
                if int(candidate.option_id) == option_id:
                    option = candidate
                    break
            if option is None:
                raise KeyError(f"fusion map graph {graph_key!r} has no option {option_id}")

            block_offsets = step.full_vec_offsets
            action_slice = np.take(base_arr, block_offsets)
            layer_idx = int(step.layer_idx)
            block_idx = int(step.block_idx)
            gelu_degree = int(gelu_arr[layer_idx] if gelu_arr.size > 1 else gelu_arr[0])
            softmax_degree = int(softmax_arr[layer_idx] if softmax_arr.size > 1 else softmax_arr[0])
            field_values = _decode_block_field_values(
                layer_idx,
                block_idx,
                action_slice,
                max_sfs,
                attn_degree=softmax_degree,
                gelu_degree=gelu_degree,
            )
            k_field_name = None
            selected_k_value = None
            graph_meta = fusion_map.graphs.get(graph_key)
            if graph_meta is not None:
                try:
                    k_field_name = str(
                        step.slot_field_names[int(graph_meta.k_slot_index)]
                    )
                    selected_k_value = field_values.get(k_field_name)
                except Exception:
                    k_field_name = None
                    selected_k_value = None
            persisted_fields = boosted_overrides.get((block_idx, layer_idx))
            if persisted_fields is not None:
                option_fields = persisted_fields
            else:
                option_fields = (
                    option.explicit_field_values
                    if bool(getattr(option, "boosted", False)) and option.explicit_field_values
                    else option.slots
                )
            for field_name, value in option_fields.items():
                field_values[str(field_name)] = int(value)
            if (
                    k_field_name is not None
                    and selected_k_value is not None
                    and (
                        persisted_fields is None
                        or k_field_name not in persisted_fields
                    )
            ):
                field_values[str(k_field_name)] = int(selected_k_value)
            if decoded.per_layer_field_values and 0 <= layer_idx < len(decoded.per_layer_field_values):
                layer_values = decoded.per_layer_field_values[layer_idx]
                if isinstance(layer_values, dict):
                    layer_values[f"block{block_idx}"] = dict(field_values)

            block_cfg = build_block_cfg_from_field_values(
                block_idx,
                layer_idx,
                field_values,
                N=int(_block_default_N(
                    block_idx,
                    gelu_degree=gelu_degree,
                    attn_degree=softmax_degree,
                )),
                gelu_degree=gelu_degree,
                attn_degree=softmax_degree,
            )
            getattr(decoded, f"block{block_idx}_cfgs")[layer_idx] = block_cfg
        return decoded

    def _materialize_decoded_action(
            self,
            *,
            profile: str,
            action_vec,
            decoded,
            cfgs_dict,
            opt_outputs,
            opt_signals,
            truncation_backend=None,
            truncation_ring_bits=None,
            truncation_source_fractional_bits=None,
            rotation_name_map_provider=None,
            ):
        """Apply Rescale_optimizer/replan results to cfgs before model forward.

        The executable Stage-2 cfg is the optimizer's ``new_compact_config``
        mirrored back into action-decoded cfg objects.  Delegate to the shared
        helper used by RL training and fixed-action experiments so every path
        installs the same cfg before inference.
        """
        bridge = getattr(self, "rescale_bridge", None)
        invoker = getattr(bridge, "invoker", None)
        invoker_baselines: Mapping[str, Any] = getattr(invoker, "baselines", {}) or {}
        evaluator_backend = getattr(
            self.evaluator, "blb_v3_truncation_backend", "binary",
        )
        evaluator_ring_bits = getattr(
            self.evaluator, "blb_v3_truncation_ring_bits", 43,
        )
        evaluator_source_bits = getattr(
            self.evaluator,
            "blb_v3_truncation_source_fractional_bits",
            24,
        )
        backend = str(
            truncation_backend
            if truncation_backend is not None
            else (
                "binary" if evaluator_backend is None else evaluator_backend
            )
        )
        ring_bits = int(
            truncation_ring_bits
            if truncation_ring_bits is not None
            else (43 if evaluator_ring_bits is None else evaluator_ring_bits)
        )
        source_fractional_bits = int(
            truncation_source_fractional_bits
            if truncation_source_fractional_bits is not None
            else (24 if evaluator_source_bits is None else evaluator_source_bits)
        )


        return materialize_decoded_action(
            action_indices=np.asarray(action_vec, dtype=int).reshape(-1).tolist(),
            decoded=decoded,
            profile=str(profile),
            cfgs_dict=cfgs_dict,
            outputs=opt_outputs,
            signals=opt_signals,
            invoker_baselines=invoker_baselines,
            rotation_name_map_provider=(
                rotation_name_map_provider or self._rotation_name_map_for
            ),
            truncation_backend=backend,
            truncation_ring_bits=ring_bits,
            truncation_source_fractional_bits=source_fractional_bits,
        )

    def _rotation_name_map_for(self, block_idx: int, profile: str) -> Mapping[str, str]:
        raw = (
            getattr(self.evaluator, "blb_v3_rotation_name_map", None)
            or getattr(self.evaluator, "rotation_name_map", None)
            or {}
        )
        if not isinstance(raw, Mapping):
            return {}
        direct = raw.get((int(block_idx), str(profile)))
        if isinstance(direct, Mapping):
            return direct
        nested = raw.get(int(block_idx)) or raw.get(str(block_idx))
        if isinstance(nested, Mapping):
            profiled = nested.get(str(profile))
            if isinstance(profiled, Mapping):
                return profiled
            if all(isinstance(k, str) for k in nested.keys()):
                return nested
        return {}

    @staticmethod
    def _fusion_group_diagnostics(*, metadata: Mapping[str, Any], opt_signals) -> Dict[str, Any]:
        group = metadata.get("group") if isinstance(metadata, Mapping) else None
        if not isinstance(group, Mapping):
            return {}
        by_graph = group.get("fusion_count_by_graph") or {}
        counts = group.get("occurrence_counts") or {}
        declared_total = 0
        declared_by_graph: Dict[str, int] = {}
        if isinstance(by_graph, Mapping):
            for graph_key, fusion_count in by_graph.items():
                occurrences = 1
                if isinstance(counts, Mapping):
                    occurrences = int(counts.get(graph_key, 1))
                value = int(fusion_count) * int(occurrences)
                declared_by_graph[str(graph_key)] = value
                declared_total += value
        realized_total = int(getattr(opt_signals, "total_fusion_count", 0))
        return {
            "group_name": str(group.get("name", "")),
            "declared_total_fusion_count": int(declared_total),
            "realized_total_fusion_count": int(realized_total),
            "declared_by_graph": declared_by_graph,
            "matches_realized_total": bool(int(declared_total) == int(realized_total)),
        }

    def _build_rescale_bridge(
            self,
            profile: str,
            ) -> Tuple[RescaleOptimizerBridge, str, str]:
        root = self._resolve_rescale_optimizer_root()
        try:
            invoker = build_rescale_invoker(root=root, profile=str(profile))
        except Exception as exc:
            raise RuntimeError(
                f"BLB final_eval failed to initialize in-process Rescale "
                f"for profile={profile!r}, root={root!r}: {exc}"
            ) from exc
        bridge = RescaleOptimizerBridge(
            invoker=invoker,
            **self._rescale_bridge_options(),
        )
        return bridge, "in_process", root

    def _load_rescale_optimizer_mode(self) -> str:
        if not self.action_config_path:
            return "cfg_derived"
        try:
            payload = read_json_file(self.action_config_path, encoding="utf-8-sig")
        except Exception:
            return "cfg_derived"
        mode = str(
            payload.get("rescale_optimizer_mode")
            or payload.get("optimizer_mode")
            or ""
        ).strip().lower().replace("-", "_")
        if mode in ("baseline", "optimizer_baseline", "rescale_baseline"):
            return "baseline"
        return "cfg_derived"

    def _rescale_bridge_options(self) -> Dict[str, Any]:
        if self.rescale_optimizer_mode != "baseline":
            return {}

        def no_delta(_cfg):
            return {}

        return {
            "cfg_to_delta_overrides": {
                "block1": no_delta,
                "block2": no_delta,
                "block3": no_delta,
                "block4": no_delta,
                "block5": no_delta,
            },
            "auto_t_new_from_cfg": False,
        }

    def _resolve_rescale_optimizer_root(self) -> str:
        ev = self.evaluator
        raw = (
            getattr(ev, "blb_v3_inproc_rescale_optimizer_root", None)
            or "configs/preparation/rescale"
        )
        path = Path(str(raw))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        return str(path)

    def _verify_model_installation(self, bridge: BLBNoiseRLBridge, decoded) -> Dict[str, Any]:
        ev = self.evaluator
        total_layers = int(ev.total_layers)
        expected_all = set(range(total_layers))
        expected = {


            "block1": expected_all,
            "block2": expected_all,
            "block3": expected_all,
            "block4": expected_all,
            "block5": expected_all,
            "first_input": set(),
        }
        active = {}
        getter = getattr(ev.reversible_handler, "get_active_blb_noise_layers", None)
        if callable(getter):
            active = getter()
        bridge_installed = bridge.installed_layers()
        active_json = {k: sorted(int(i) for i in v) for k, v in (active or {}).items()}
        expected_json = {k: sorted(int(i) for i in v) for k, v in expected.items()}
        handler_match = all(set(active.get(k, set())) == v for k, v in expected.items())
        bridge_match = all(
            all(k in bridge_installed.get(i, set()) for i in v)
            for k, v in expected.items()
        )
        identity_match = self._handler_cfg_identity_match(decoded, expected)
        return {
            "checked_before_forward": True,
            "handler_active_layers": active_json,
            "expected_active_layers": expected_json,
            "handler_active_layers_match_expected": bool(handler_match),
            "bridge_installed_layers_match_expected": bool(bridge_match),
            "handler_cfg_objects_match_decoded_cfgs": bool(identity_match),
            "model_will_use_selected_cfg": bool(handler_match and bridge_match and identity_match),
        }

    def _handler_cfg_identity_match(self, decoded, expected_layers_by_block: Optional[Mapping[str, set]] = None) -> bool:
        handler = self.evaluator.reversible_handler
        for block_name in ("block1", "block2", "block3", "block4", "block5"):
            expected = getattr(decoded, f"{block_name}_cfgs")
            installed = getattr(handler, f"{block_name}_cfg_per_layer", {})
            expected_layers = None
            if expected_layers_by_block is not None:
                expected_layers = set(expected_layers_by_block.get(block_name, set()))
            for layer_idx, cfg in expected.items():
                if expected_layers is not None and int(layer_idx) not in expected_layers:
                    continue
                if installed.get(layer_idx) is not cfg:
                    return False
        return True

    def _config_details(self, decoded, action_vec, overrides, opt_outputs) -> Dict[str, Any]:
        return {
            "base_action": (
                "BLB RL baseline action: model-side non-truncation fields use highest "
                "selectable action-space SF; Rescale_optimizer mode="
                f"{self.rescale_optimizer_mode}."
            ),
            "truncation": self._truncation_summary(decoded),
            "non_truncation_unique_scaling_factors": self._non_truncation_sf_summary(decoded),
            "full_noise_config": self._full_noise_config(decoded),
            "action_overrides": dict(overrides or {}),
            "action_vector_length": int(np.asarray(action_vec).size),
            "optimizer_request_names": sorted(str(k) for k in opt_outputs.keys()),
        }

    @staticmethod
    def _truncation_summary(decoded) -> Dict[str, Any]:
        per_block = {}
        effective_count = 0
        skipped = []
        for block_name in ("block1", "block2", "block3", "block4", "block5"):
            cfgs = getattr(decoded, f"{block_name}_cfgs")
            vals = {}
            for layer_idx, cfg in cfgs.items():
                k = getattr(cfg, "output_truncation_k", None)
                vals[int(layer_idx)] = k if k is None else int(k)
                if k is None:
                    skipped.append({"block": block_name, "layer": int(layer_idx)})
                else:
                    effective_count += 1
            unique = sorted({v for v in vals.values() if v is not None})
            per_block[block_name] = {
                "unique_effective_k": unique,
                "per_layer": vals,
            }
        return {
            "per_block": per_block,
            "effective_position_count": int(effective_count),
            "skipped_positions": skipped,
        }

    @staticmethod
    def _non_truncation_sf_summary(decoded) -> Dict[str, Any]:
        def gather_cfg(cfg):
            out = {}
            for name, value in vars(cfg).items():
                if name.startswith("output_truncation"):
                    continue
                sf = getattr(value, "scaling_factor", None)
                if sf is not None:
                    out.setdefault(name, set()).add(int(sf))
                elif isinstance(value, tuple):
                    vals = []
                    for item in value:
                        item_sf = getattr(item, "scaling_factor", None)
                        if item_sf is not None:
                            vals.append(int(item_sf))
                    if vals:
                        out.setdefault(name, set()).update(vals)
            return out

        summary = {}
        for block_name in ("block1", "block2", "block3", "block4", "block5"):
            merged = {}
            for cfg in getattr(decoded, f"{block_name}_cfgs").values():
                for name, vals in gather_cfg(cfg).items():
                    merged.setdefault(name, set()).update(vals)
            summary[block_name] = {
                name: sorted(int(v) for v in vals)
                for name, vals in sorted(merged.items())
            }
        return summary

    @staticmethod
    def _full_noise_config(decoded) -> Dict[str, Any]:
        entries = []

        for block_name in ("block1", "block2", "block3", "block4", "block5"):
            cfgs = getattr(decoded, f"{block_name}_cfgs")
            for layer_idx, cfg in sorted(cfgs.items()):
                entries.extend(
                    BLBActionFinalEvaluationModule._cfg_noise_entries(
                        layer_idx=int(layer_idx),
                        block_name=block_name,
                        cfg=cfg,
                    )
                )

        return {
            "entry_count": int(len(entries)),
            "entries": entries,
        }

    @staticmethod
    def _cfg_noise_entries(*, layer_idx: int, block_name: str, cfg) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        truncation_mode = str(getattr(cfg, "output_truncation_mode", ""))
        for attr, value in vars(cfg).items():
            base_path = f"layer{layer_idx}.{block_name}.{attr}"
            if attr == "output_truncation_mode":
                continue
            if attr == "output_truncation_k":
                entries.append({
                    "path": base_path,
                    "type": "truncation",
                    "layer": int(layer_idx),
                    "block": str(block_name),
                    "point": str(attr),
                    "distribution": None,
                    "N": None,
                    "scaling_factor": None,
                    "truncation_k": (None if value is None else int(value)),
                    "truncation_mode": truncation_mode,
                    "value": (None if value is None else int(value)),
                    "active": value is not None,
                })
                continue
            if attr.startswith("rotation_after"):
                continue
            if hasattr(value, "scaling_factor"):
                entries.append(
                    BLBActionFinalEvaluationModule._noise_point_entry(
                        path=base_path,
                        layer_idx=layer_idx,
                        block_name=block_name,
                        point=attr,
                        noise_point=value,
                    )
                )
                continue
            if value is None and "rescale" in attr:
                entries.append({
                    "path": base_path,
                    "type": "scaling_factor",
                    "layer": int(layer_idx),
                    "block": str(block_name),
                    "point": str(attr),
                    "distribution": None,
                    "N": None,
                    "scaling_factor": None,
                    "truncation_k": None,
                    "value": None,
                    "active": False,
                })
                continue
            if isinstance(value, tuple):
                if not value:
                    entries.append({
                        "path": base_path,
                        "type": "scaling_factor_tuple",
                        "layer": int(layer_idx),
                        "block": str(block_name),
                        "point": str(attr),
                        "distribution": None,
                        "N": None,
                        "scaling_factor": None,
                        "truncation_k": None,
                        "value": [],
                        "active": False,
                    })
                    continue
                for item_idx, item in enumerate(value):
                    item_path = f"{base_path}[{item_idx}]"
                    if hasattr(item, "scaling_factor"):
                        entries.append(
                            BLBActionFinalEvaluationModule._noise_point_entry(
                                path=item_path,
                                layer_idx=layer_idx,
                                block_name=block_name,
                                point=f"{attr}[{item_idx}]",
                                noise_point=item,
                            )
                        )
                    else:
                        entries.append({
                            "path": item_path,
                            "type": "scaling_factor",
                            "layer": int(layer_idx),
                            "block": str(block_name),
                            "point": f"{attr}[{item_idx}]",
                            "distribution": None,
                            "N": None,
                            "scaling_factor": None,
                            "truncation_k": None,
                            "value": None,
                            "active": False,
                        })
                continue
            if attr in ("degree", "gelu_degree"):
                entries.append({
                    "path": base_path,
                    "type": "parameter",
                    "layer": int(layer_idx),
                    "block": str(block_name),
                    "point": str(attr),
                    "distribution": None,
                    "N": None,
                    "scaling_factor": None,
                    "truncation_k": None,
                    "value": int(value),
                    "active": True,
                })
        return entries

    @staticmethod
    def _noise_point_entry(*, path: str, layer_idx: int, block_name: str, point: str, noise_point) -> Dict[str, Any]:
        return {
            "path": str(path),
            "type": "scaling_factor",
            "layer": int(layer_idx),
            "block": str(block_name),
            "point": str(point),
            "distribution": str(getattr(noise_point, "distribution", "")),
            "N": int(getattr(noise_point, "N")),
            "scaling_factor": int(getattr(noise_point, "scaling_factor")),
            "truncation_k": None,
            "value": int(getattr(noise_point, "scaling_factor")),
            "active": True,
        }

    def _run_blb_eval(self, decoded, *, gelu, softmax):
        repeats = self.repeat_n
        if repeats <= 1:
            return self._run_single_blb_eval(decoded, gelu=gelu, softmax=softmax), None
        trials = self._run_blb_eval_trials(
            decoded,
            gelu=gelu,
            softmax=softmax,
            repeats=repeats,
        )
        repeat = pack_repeat_evaluation(
            trials,
            evaluation_mode="blb_action_repeated_validation_full",
        )
        return {
            "loss": float(repeat["stats"]["loss_mean"]),
            "p": float(repeat["stats"]["p_mean"]),
            "s": float(repeat["stats"]["s_mean"]),
            "time_ms": float(repeat["stats"]["time_mean_ms"]),
            "install_verification": trials[0].get("install_verification", {}) if trials else {},
        }, repeat

    def _run_single_blb_eval(self, decoded, *, gelu, softmax):
        return self._run_blb_eval_trials(
            decoded,
            gelu=gelu,
            softmax=softmax,
            repeats=1,
        )[0]

    def _run_blb_eval_trials(self, decoded, *, gelu, softmax, repeats: int):
        ev = self.evaluator
        bridge = BLBNoiseRLBridge(
            ev.reversible_handler,
            layers_attribute="model." + ev.layers_attribute,
        )
        ev.apply_configuration(gelu, softmax)
        self._clear_legacy_noise()
        try:


            bridge.apply(
                block1_cfgs=decoded.block1_cfgs,
                block2_cfgs=decoded.block2_cfgs,
                block3_cfgs=decoded.block3_cfgs,
                block4_cfgs=decoded.block4_cfgs,
                block5_cfgs=decoded.block5_cfgs,
            )
            install_verification = self._verify_model_installation(bridge, decoded)
            if install_verification.get("model_will_use_selected_cfg") is not True:
                raise RuntimeError(
                    "selected configuration installation verification failed"
                )
            split_name = ev._resolve_eval_split(
                use_train=False,
                split=getattr(self, "final_eval_split", FINAL_EVAL_SPLIT),
            )
            trials = []
            for _idx in range(max(1, int(repeats))):
                loss, p, s, time_ms = ev._run_evaluation(
                    ev.dataloaders[split_name],
                    use_train=False,
                    split_name=split_name,
                )
                trials.append({
                    "loss": float(loss),
                    "p": float(p),
                    "s": float(s),
                    "time_ms": float(time_ms),
                    "install_verification": install_verification,
                })
            return trials
        finally:
            bridge.clear()
            self._clear_all_noise()

    def _clear_legacy_noise(self):
        ev = self.evaluator
        handler = ev.reversible_handler
        layer_indices = list(range(ev.total_layers))
        try:
            handler.restore_layer_input_noise(layer_indices=layer_indices)
        except Exception:
            pass
        for restore_name in (
            "restore_layer_query_noise",
            "restore_layer_key_noise",
            "restore_layer_value_noise",
            "restore_layer_wo_noise",
            "restore_layer_ffn1_noise",
            "restore_layer_ffn2_noise",
            "restore_layer_softmax_value_noise",
        ):
            method = getattr(handler, restore_name, None)
            if method is None:
                continue
            try:
                method(layer_indices=layer_indices)
            except Exception:
                pass

    def _clear_all_noise(self):
        self._clear_legacy_noise()
        ev = self.evaluator
        layer_indices = list(range(ev.total_layers))
        layer_name = "model." + ev.layers_attribute
        for restore_name in (
            "restore_layer_block5_noise",
            "restore_layer_block4_noise",
            "restore_layer_block3_noise",
            "restore_layer_block2_noise",
            "restore_layer_block1_noise",
        ):
            method = getattr(ev.reversible_handler, restore_name, None)
            if method is None:
                continue
            try:
                method(
                    layer_indices=layer_indices,
                    layer_name=layer_name,
                )
            except Exception:
                pass

    def _save_results_markdown(
        self,
        *,
        json_path: str,
        selected_source: str,
        baseline_result,
        candidate_results,
        comparison_summary: Optional[Dict[str, Any]] = None,
        cost_match_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> str:
        metric_names = self.evaluator.get_metric_short_names()
        primary = metric_names[0] if metric_names else "metric1"
        secondary = metric_names[1] if len(metric_names) > 1 else "metric2"
        path = os.path.join(
            self.results_dir,
            f"blb_action_final_eval_report_{self.evaluator.dataset_key}.md",
        )
        lines = [
            "# BLB Action Final Evaluation Report",
            "",
            f"- dataset: `{self.evaluator.dataset_key}`",
            f"- split: `validation_full`",
            f"- selected_source: `{selected_source}`",
            f"- repeat_n: `{self.repeat_n}`",
            f"- rescale_optimizer: `{getattr(self, 'rescale_backend', 'unknown')}`",
            f"- rescale_optimizer_root: `{getattr(self, 'rescale_optimizer_root', '') or '(none)'}`",
            f"- json: `{json_path}`",
            "",
            "## Baseline",
            "",
            f"- clean baseline loss: `{baseline_result['loss']:.6f}`",
            f"- clean baseline {primary}: `{baseline_result['p']:.6f}`",
            f"- clean baseline {secondary}: `{baseline_result['s']:.6f}`",
            f"- clean baseline protocol: `{baseline_result.get('evaluation_protocol', 'single_validation_full')}`",
            f"- clean baseline loss std: `{float(baseline_result.get('loss_std', 0.0)):.6f}`",
            f"- clean baseline {primary} std: `{float(baseline_result.get('p_std', 0.0)):.6f}`",
            f"- clean baseline {secondary} std: `{float(baseline_result.get('s_std', 0.0)):.6f}`",
            "",
        ]
        if cost_match_diagnostics:
            lines.extend([
                "## Cost-Matched Random Sampling",
                "",
                f"- target total_bits_sum: `{cost_match_diagnostics.get('target_total_bits')}`",
                f"- target total_fusion_count: `{cost_match_diagnostics.get('target_total_fusion')}`",
                f"- target sum_truncation_k: `{cost_match_diagnostics.get('target_sum_k')}`",
                f"- requested: `{cost_match_diagnostics.get('requested_count')}` configs",
                f"- accepted: `{cost_match_diagnostics.get('accepted')}` configs in "
                f"`{cost_match_diagnostics.get('attempts')}`/`{cost_match_diagnostics.get('max_attempts')}` attempts",
                f"- rejection breakdown: invalid=`{cost_match_diagnostics.get('invalid')}`, "
                f"cost_mismatch=`{cost_match_diagnostics.get('cost_mismatch')}`, "
                f"avg_k_prefilter=`{cost_match_diagnostics.get('avg_k_prefilter_skipped')}`",
                "",
            ])
        if comparison_summary:
            lines.extend(self._comparison_summary_markdown(comparison_summary, primary, secondary))
        lines.extend([
            "## Group Comparison",
            "",
            "| group | truncation k | effective K positions | loss mean | loss std | "
            f"{primary} mean | {primary} std | {secondary} mean | {secondary} std | "
            "time mean ms | total bits | fusion | replan applied | model cfg verified |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ])
        for result in candidate_results:
            trunc = result.get("config_details", {}).get("truncation", {})
            unique_k = self._unique_truncation_label(trunc)
            verify = result.get("install_verification", {}).get("model_will_use_selected_cfg", False)
            replan_ok = result.get("replan_application", {}).get("model_uses_replan_config", False)
            lines.append(
                f"| `{result['name']}` | {unique_k} | "
                f"{int(trunc.get('effective_position_count', 0))} | "
                f"{float(result['loss']):.6f} | {float(result.get('loss_std', 0.0)):.6f} | "
                f"{float(result['p']):.6f} | {float(result.get('p_std', 0.0)):.6f} | "
                f"{float(result['s']):.6f} | {float(result.get('s_std', 0.0)):.6f} | "
                f"{float(result['time_ms']):.3f} | {int(result['total_bits_sum'])} | "
                f"{int(result['total_fusion_count'])} | {replan_ok} | {verify} |"
            )

        lines.extend(["", "## Configuration Details", ""])
        for result in candidate_results:
            details = result.get("config_details", {})
            trunc = details.get("truncation", {})
            verify = result.get("install_verification", {})
            replan_application = result.get("replan_application", {})
            fusion_group = result.get("fusion_group_diagnostics", {})
            lines.extend([
                f"### {result['name']}",
                "",
                f"- action overrides: `{result.get('action_overrides', {})}`",
                f"- base action: {details.get('base_action', '')}",
                f"- truncation summary: `{self._unique_truncation_label(trunc)}`; "
                f"effective positions = `{trunc.get('effective_position_count', 0)}`; "
                f"skipped = `{trunc.get('skipped_positions', [])}`",
                f"- model cfg verified before forward: `{verify.get('model_will_use_selected_cfg', False)}`",
                f"- replan cfg applied before forward: `{replan_application.get('model_uses_replan_config', False)}`",
                f"- replan application summary: `{ {k: v for k, v in replan_application.items() if k != 'per_config'} }`",
                f"- fusion group diagnostics: `{fusion_group}`",
                f"- handler active layers match expected: `{verify.get('handler_active_layers_match_expected', False)}`",
                f"- handler cfg object identity match: `{verify.get('handler_cfg_objects_match_decoded_cfgs', False)}`",
                f"- rescale optimizer: `{result.get('rescale_optimizer', {})}`",
                "",
                "Non-truncation unique scaling factors:",
                "",
                "```json",
                json.dumps(
                    details.get("non_truncation_unique_scaling_factors", {}),
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
                "",
                "Full noise and truncation configuration:",
                "",
            ])
            lines.extend(self._full_noise_config_markdown_table(details.get("full_noise_config", {})))
            lines.append("")

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    @staticmethod
    def _full_noise_config_markdown_table(full_config) -> List[str]:
        entries = (full_config or {}).get("entries", ()) or ()
        lines = [
            "| path | type | distribution | N | scaling_factor | truncation_k | value | active |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for entry in entries:
            lines.append(
                "| "
                + " | ".join(
                    BLBActionFinalEvaluationModule._md_cell(value)
                    for value in (
                        entry.get("path"),
                        entry.get("type"),
                        entry.get("distribution"),
                        entry.get("N"),
                        entry.get("scaling_factor"),
                        entry.get("truncation_k"),
                        entry.get("value"),
                        entry.get("active"),
                    )
                )
                + " |"
            )
        return lines

    @staticmethod
    def _md_cell(value) -> str:
        if value is None:
            return ""
        text = str(value)
        return text.replace("|", "\\|")

    def _save_results_plot(self, *, candidate_results) -> Optional[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.evaluator.log(f"  [plot][warning] matplotlib unavailable: {exc}")
            return None

        labels = []
        loss_values = []
        loss_std_values = []
        p_values = []
        p_std_values = []
        bits_values = []
        time_ms_values = []
        for result in candidate_results:
            labels.append(self._unique_truncation_label(result.get("config_details", {}).get("truncation", {})))
            loss_values.append(float(result["loss"]))
            loss_std_values.append(float(result.get("loss_std", 0.0)))
            p_values.append(float(result["p"]))
            p_std_values.append(float(result.get("p_std", 0.0)))
            bits_values.append(float(result["total_bits_sum"]))
            time_ms_values.append(float(result["time_ms"]))

        x = np.arange(len(labels))
        loss = np.asarray(loss_values, dtype=float)
        loss_std = np.asarray(loss_std_values, dtype=float)
        p = np.asarray(p_values, dtype=float)
        p_std = np.asarray(p_std_values, dtype=float)
        bits = np.asarray(bits_values, dtype=float)
        time_ms = np.asarray(time_ms_values, dtype=float)

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes = axes.reshape(-1)
        axes[0].bar(x, loss, yerr=loss_std, capsize=4, color="#4c78a8")
        axes[0].set_title("Loss mean +/- std")
        axes[1].bar(x, p, yerr=p_std, capsize=4, color="#59a14f")
        axes[1].set_title(f"{self.evaluator.get_metric_short_names()[0]} mean +/- std")
        axes[2].bar(x, bits, color="#f28e2b")
        axes[2].set_title("Rescale optimizer total_bits")
        axes[3].bar(x, time_ms, color="#e15759")
        axes[3].set_title("Time mean ms")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = os.path.join(
            self.results_dir,
            f"blb_action_final_eval_plot_{self.evaluator.dataset_key}.png",
        )
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    @staticmethod
    def _unique_truncation_label(truncation_summary) -> str:
        vals = []
        for block in truncation_summary.get("per_block", {}).values():
            vals.extend(block.get("unique_effective_k", []))
        unique = sorted({int(v) for v in vals})
        if len(unique) == 1:
            return str(unique[0])
        return ",".join(str(v) for v in unique) if unique else "none"

    def _save_results_json(
        self,
        *,
        selected_source,
        baseline_stage1_gelu,
        baseline_stage1_softmax,
        opt_gelu,
        opt_softmax,
        baseline_result,
        candidate_results,
        selection_constraints,
        comparison_summary: Optional[Dict[str, Any]] = None,
        cost_match_diagnostics: Optional[Dict[str, Any]] = None,
        action_context_provenance: Optional[Mapping[str, Any]] = None,
        final_eval_handoff: Optional[Mapping[str, Any]] = None,  # noqa: UP045
    ):
        handoff = to_jsonable(final_eval_handoff)
        output = {
            "schema_version": "paean_blb_action_final_eval_result_v1",
            "status": "complete",
            "dataset": self.evaluator.dataset_key,
            "final_eval_split": getattr(
                self, "final_eval_split", FINAL_EVAL_SPLIT
            ),
            "dataset_protocol_hash": getattr(
                self.evaluator, "dataset_protocol_hash", None
            ),
            "selected_source": selected_source,
            "stage2_final_eval_handoff": handoff,
            "baseline_stage1": {
                "gelu": np.asarray(baseline_stage1_gelu, dtype=int).tolist(),
                "softmax": np.asarray(baseline_stage1_softmax, dtype=int).tolist(),
            },
            "selected_stage1": {
                "gelu": np.asarray(opt_gelu, dtype=int).tolist(),
                "softmax": np.asarray(opt_softmax, dtype=int).tolist(),
            },
            "constraints": {"selection": selection_constraints},
            "baseline": to_jsonable(baseline_result),
            "candidate_results": [to_jsonable(r) for r in candidate_results],
            "comparison_summary": to_jsonable(comparison_summary or {}),
            "cost_match_diagnostics": to_jsonable(cost_match_diagnostics or {}),
            "calibrated_action_context": to_jsonable(
                action_context_provenance or {}
            ),
            "evaluation_protocol": {
                "version": 2,
                "mode": "blb_action_grid_cost_matched",
                "candidate_count": int(len(candidate_results)),
                "random_groups": ("enabled" if len(candidate_results) > 1 else "disabled"),
                "cost_match_count": int(self.cost_match_count),
                "cost_match_max_attempts": int(self.cost_match_max_attempts),
                "action_ranges": self.action_ranges,
                "action_fixed": self.action_fixed,
                "repeat_n": int(self.repeat_n),
                "random_seed": int(self.random_seed),
            },
        }
        output_path = os.path.join(
            self.results_dir,
            f"blb_action_final_eval_results_{self.evaluator.dataset_key}.json",
        )
        _atomic_json(output_path, output)
        return output_path


    def _cost_match_diagnostics_to_dict(
            self, diag: Optional[CostMatchedSamplingDiagnostics]
            ) -> Dict[str, Any]:
        if diag is None:
            return {}
        return {
            "target_total_bits": int(diag.target_total_bits),
            "target_total_fusion": int(diag.target_total_fusion),
            "target_sum_k": int(diag.target_sum_k),
            "accepted": int(diag.accepted),
            "attempts": int(diag.attempts),
            "invalid": int(diag.invalid),
            "cost_mismatch": int(diag.cost_mismatch),
            "avg_k_prefilter_skipped": int(diag.avg_k_prefilter_skipped),
            "max_attempts": int(diag.max_attempts),
            "requested_count": int(diag.requested_count),
        }

    def _summarize_selected_vs_random(
            self,
            *,
            selected_results: List[Dict[str, Any]],
            random_results: List[Dict[str, Any]],
            num_metrics: int,
            ) -> Dict[str, Any]:
        return summarize_selected_vs_random_results(
            selected_results,
            random_results,
            num_metrics=num_metrics,
        )

    def _comparison_summary_markdown(
            self,
            summary: Dict[str, Any],
            primary: str,
            secondary: str,
            ) -> List[str]:
        lines: List[str] = []
        anchor = summary.get("selected_anchor") or {}
        stats = summary.get("random_stats") or {}
        if not anchor and not stats:
            return lines
        lines.extend(["## Selected vs Cost-Matched Random Comparison", ""])
        if anchor:
            lines.append(
                f"- selected (`{anchor.get('name')}`): "
                f"loss={anchor.get('loss_mean', 0.0):.6f} ± {anchor.get('loss_std', 0.0):.6f}, "
                f"{primary}={anchor.get('metric1_mean', 0.0):.6f} ± {anchor.get('metric1_std', 0.0):.6f}, "
                f"{secondary}={anchor.get('metric2_mean', 0.0):.6f} ± {anchor.get('metric2_std', 0.0):.6f}, "
                f"total_bits={anchor.get('total_bits_sum')}, "
                f"fusion={anchor.get('total_fusion_count')}, "
                f"avg_k={anchor.get('avg_truncation_k', 0.0):.3f}"
            )
        if stats:
            lines.extend([
                "",
                "| stat | loss mean | loss std | "
                f"{primary} mean | {primary} std | {secondary} mean | {secondary} std |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            n_random = int(summary.get("random_count", 0))
            lines.append(
                f"| random (n={n_random}) mean | "
                f"{stats.get('loss_mean', {}).get('mean', 0.0):.6f} | "
                f"{stats.get('loss_std', {}).get('mean', 0.0):.6f} | "
                f"{stats.get('metric1_mean', {}).get('mean', 0.0):.6f} | "
                f"{stats.get('metric1_std', {}).get('mean', 0.0):.6f} | "
                f"{stats.get('metric2_mean', {}).get('mean', 0.0):.6f} | "
                f"{stats.get('metric2_std', {}).get('mean', 0.0):.6f} |"
            )
            lines.append(
                f"| random std (across cfgs) | "
                f"{stats.get('loss_mean', {}).get('std', 0.0):.6f} | "
                f"{stats.get('loss_std', {}).get('std', 0.0):.6f} | "
                f"{stats.get('metric1_mean', {}).get('std', 0.0):.6f} | "
                f"{stats.get('metric1_std', {}).get('std', 0.0):.6f} | "
                f"{stats.get('metric2_mean', {}).get('std', 0.0):.6f} | "
                f"{stats.get('metric2_std', {}).get('std', 0.0):.6f} |"
            )
            lines.append(
                f"| random min | "
                f"{stats.get('loss_mean', {}).get('min', 0.0):.6f} | "
                f"{stats.get('loss_std', {}).get('min', 0.0):.6f} | "
                f"{stats.get('metric1_mean', {}).get('min', 0.0):.6f} | "
                f"{stats.get('metric1_std', {}).get('min', 0.0):.6f} | "
                f"{stats.get('metric2_mean', {}).get('min', 0.0):.6f} | "
                f"{stats.get('metric2_std', {}).get('min', 0.0):.6f} |"
            )
            lines.append(
                f"| random max | "
                f"{stats.get('loss_mean', {}).get('max', 0.0):.6f} | "
                f"{stats.get('loss_std', {}).get('max', 0.0):.6f} | "
                f"{stats.get('metric1_mean', {}).get('max', 0.0):.6f} | "
                f"{stats.get('metric1_std', {}).get('max', 0.0):.6f} | "
                f"{stats.get('metric2_mean', {}).get('max', 0.0):.6f} | "
                f"{stats.get('metric2_std', {}).get('max', 0.0):.6f} |"
            )
        rank = summary.get("anchor_rank_vs_random") or {}
        if rank:
            lines.extend(["", "### Anchor rank vs random group", ""])
            for key, item in rank.items():
                if not item:
                    continue
                lines.append(
                    f"- `{key}`: selected is better than "
                    f"`{item.get('rank_better_than_selected')}` / `{item.get('out_of')}` random configs "
                    f"(percentile=`{item.get('percentile'):.3f}`)"
                )
        lines.append("")
        return lines

    def _save_scatter_plot(
            self,
            *,
            selected_results: List[Dict[str, Any]],
            random_results: List[Dict[str, Any]],
            ) -> Optional[str]:
        if not selected_results and not random_results:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.evaluator.log(f"  [scatter][warning] matplotlib unavailable: {exc}")
            return None

        metric_names = self.evaluator.get_metric_short_names()
        primary = metric_names[0] if metric_names else "metric1"
        num_metrics = self.evaluator.get_num_metrics()

        def _scatter_columns(rows: List[Dict[str, Any]]):
            p_x = []
            p_y = []
            s_x = []
            s_y = []
            for row in rows:
                p_x.append(float(row.get("p", 0.0)))
                p_y.append(float(row.get("p_std", 0.0)))
                if num_metrics > 1:
                    s_x.append(float(row.get("s", 0.0)))
                    s_y.append(float(row.get("s_std", 0.0)))
            return p_x, p_y, s_x, s_y

        sel_x, sel_y, sel2_x, sel2_y = _scatter_columns(selected_results)
        rnd_x, rnd_y, rnd2_x, rnd2_y = _scatter_columns(random_results)

        ncols = 2 if num_metrics > 1 else 1
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
        if ncols == 1:
            axes = [axes]

        ax = axes[0]
        if rnd_x:
            ax.scatter(rnd_x, rnd_y, c="#888888", alpha=0.6, label=f"random (n={len(rnd_x)})", marker="o")
        if sel_x:
            ax.scatter(sel_x, sel_y, c="#e15759", s=120, label="selected", marker="*", edgecolors="black")
        ax.set_xlabel(f"{primary} mean")
        ax.set_ylabel(f"{primary} std (across repeat_n)")
        ax.set_title(f"{primary}: mean × std")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

        if ncols > 1:
            secondary = metric_names[1]
            ax2 = axes[1]
            if rnd2_x:
                ax2.scatter(rnd2_x, rnd2_y, c="#888888", alpha=0.6, label=f"random (n={len(rnd2_x)})", marker="o")
            if sel2_x:
                ax2.scatter(sel2_x, sel2_y, c="#e15759", s=120, label="selected", marker="*", edgecolors="black")
            ax2.set_xlabel(f"{secondary} mean")
            ax2.set_ylabel(f"{secondary} std (across repeat_n)")
            ax2.set_title(f"{secondary}: mean × std")
            ax2.legend(loc="best")
            ax2.grid(alpha=0.25)

        fig.tight_layout()
        path = os.path.join(
            self.results_dir,
            f"blb_action_final_eval_scatter_{self.evaluator.dataset_key}.png",
        )
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path


    def _attach_relative_metrics(baseline, results):
        for result in results:
            result["delta_loss_vs_baseline"] = float(result["loss"] - baseline["loss"])
            result["delta_p_vs_baseline"] = float(result["p"] - baseline["p"])
            result["delta_s_vs_baseline"] = float(result["s"] - baseline["s"])

    @staticmethod
    def _is_feasible(loss, p, s, constraints):
        if p < constraints["metric1"]:
            return False
        if s < constraints["metric2"]:
            return False
        return True

    @staticmethod
    def _dominant_degree(degrees, default=4) -> int:
        arr = np.asarray(degrees, dtype=int).reshape(-1)
        if arr.size == 0:
            return int(default)
        vals, counts = np.unique(arr, return_counts=True)
        return int(vals[np.argmax(counts)])
