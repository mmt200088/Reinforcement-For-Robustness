from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from blb_rl_bridge import BLBNoiseRLBridge
from blb_stage2_rl.action_space import (
    action_vector_to_cfgs,
    avg_truncation_k_in_action,
    build_optimizer_requests,
    load_max_sfs,
)
from blb_stage2_rl.default_invoker import HeuristicStubInvoker
from final_evaluation_module import UnifiedFinalEvaluationModule
from rescale_optimizer_bridge import RescaleOptimizerBridge, aggregate_optimizer_signals

from .action_grid import build_action_candidates, build_random_action_candidates, coerce_spec_list


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
        default_results_dir = getattr(
            evaluator, "final_eval_dir", os.path.join("rl_results", "final_eval")
        )
        self.results_dir = results_dir or default_results_dir
        self.action_config_path = str(action_config_path or "").strip()
        self.action_ranges = coerce_spec_list(action_ranges)
        self.action_fixed = coerce_spec_list(action_fixed)

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
        if self.random_enabled and self.action_ranges:
            raise ValueError("BLB action final_eval random mode cannot be combined with action ranges")

        os.makedirs(self.results_dir, exist_ok=True)
        ev = self.evaluator
        total_layers = int(ev.total_layers)
        profile = str(getattr(ev, "dataset_key", "default") or "default")

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

        base_action = self._resolve_base_action(search_best_stage2)
        if self.random_enabled:
            candidates = build_random_action_candidates(
                num_layers=total_layers,
                count=self.random_count,
                seed=self.random_seed,
                base_action_vec=base_action,
                fixed_specs=self.action_fixed,
                profile=profile,
            )
        else:
            candidates = build_action_candidates(
                num_layers=total_layers,
                profile=profile,
                base_action_vec=base_action,
                fixed_specs=self.action_fixed,
                range_specs=self.action_ranges,
                action_config_path=self.action_config_path,
            )

        metric_names = ev.get_metric_short_names()
        num_metrics = ev.get_num_metrics()
        ev.log("\n" + "=" * 60)
        ev.log("PHASE: BLB ACTION FINAL EVALUATION (validation_full)")
        ev.log(f"CONFIG_SOURCE={self.config_source}  STAGE1_SOURCE={stage1_source}")
        ev.log(
            f"action_candidates={len(candidates)} random_enabled={self.random_enabled} "
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

        results = []
        for idx, candidate in enumerate(candidates, start=1):
            ev.log(f"\n--- BLB action candidate {idx}/{len(candidates)}: {candidate.name} ---")
            result = self._evaluate_action_candidate(
                name=candidate.name,
                action_vec=candidate.action_vec,
                overrides=candidate.overrides,
                gelu=opt_gelu,
                softmax=opt_softmax,
                report_constraints=report_constraints,
            )
            results.append(result)
            ev.log(
                f"  {candidate.name}: Loss={result['loss']:.4f}, "
                f"{metric_names[0]}={result['p']:.4f}"
                + (f", {metric_names[1]}={result['s']:.4f}" if num_metrics > 1 else "")
                + f", avg_k={result['avg_truncation_k']:.2f}, bits={result['total_bits_sum']}"
            )

        self._attach_relative_metrics(baseline_result, results)
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
        )
        ev.log(f"BLB action final-eval summary saved to: {summary_path}")

        ev.apply_configuration(opt_gelu, opt_softmax)
        self._clear_all_noise()
        best = results[0] if results else None
        return {
            "selected_source": f"blb_action(stage1={stage1_source})",
            "opt_gelu": opt_gelu,
            "opt_softmax": opt_softmax,
            "opt_noise_config": {},
            "baseline_result": baseline_result,
            "optimized_result": best,
            "candidate_results": results,
            "random_results": results[1:] if self.random_enabled else [],
            "random_summary": {},
            "summary_path": summary_path,
            "plot_path": None,
            "variance_plot_path": None,
        }

    def _resolve_base_action(self, search_best_stage2):
        if isinstance(search_best_stage2, dict):
            raw = (
                search_best_stage2.get("blb_v3_best_action_vec")
                or search_best_stage2.get("best_action_vec")
                or search_best_stage2.get("best_action")
            )
            if raw is not None:
                return np.asarray(raw, dtype=int)
        return None

    def _evaluate_clean_baseline(self, *, baseline_stage1_gelu, baseline_stage1_softmax):
        ev = self.evaluator
        loss, p, s, t = ev.evaluate_model(
            np.asarray(baseline_stage1_gelu, dtype=int),
            np.asarray(baseline_stage1_softmax, dtype=int),
            use_train=False,
            split="validation_full",
        )
        return {
            "name": "Baseline (Stage-1 Exact)",
            "family": "Baseline",
            "loss": float(loss),
            "p": float(p),
            "s": float(s),
            "time_ms": float(t),
        }

    def _evaluate_action_candidate(self, *, name, action_vec, overrides, gelu, softmax, report_constraints):
        ev = self.evaluator
        total_layers = int(ev.total_layers)
        profile = str(getattr(ev, "dataset_key", "default") or "default")
        max_sfs = load_max_sfs(profile)
        decoded = action_vector_to_cfgs(
            action_vec=np.asarray(action_vec, dtype=int),
            max_sfs=max_sfs,
            num_layers=total_layers,
            gelu_degree=self._dominant_degree(gelu, default=4),
            attn_degree=self._dominant_degree(softmax, default=4),
        )

        cfgs_dict = decoded.cfgs_dict()
        opt_signals = self._optimizer_signals(profile, cfgs_dict)
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
            "avg_truncation_k": float(avg_truncation_k_in_action(action_vec, total_layers)),
            "action_overrides": dict(overrides or {}),
            "action_vec": np.asarray(action_vec, dtype=int).copy(),
            "feasible": self._is_feasible(loss, p, s, report_constraints),
        }
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
            result["evaluation_protocol"] = "single_validation_full"
        return result

    def _optimizer_signals(self, profile: str, cfgs_dict):
        heuristic = HeuristicStubInvoker()
        bridge = RescaleOptimizerBridge(invoker=heuristic)
        requests = build_optimizer_requests(profile, cfgs_dict)
        for config_name, (_block_name, cfg) in requests.items():
            heuristic.register_cfg(config_name, cfg)
        try:
            outputs = bridge.evaluate_blocks(requests)
        finally:
            heuristic.clear_cfg_registry()
        return aggregate_optimizer_signals(outputs)

    def _run_blb_eval(self, decoded, *, gelu, softmax):
        ev = self.evaluator
        repeats = self.repeat_n
        if repeats <= 1:
            return self._run_single_blb_eval(decoded, gelu=gelu, softmax=softmax), None
        trials = []
        for _idx in range(repeats):
            trials.append(self._run_single_blb_eval(decoded, gelu=gelu, softmax=softmax))
        losses = np.asarray([t["loss"] for t in trials], dtype=float)
        ps = np.asarray([t["p"] for t in trials], dtype=float)
        ss = np.asarray([t["s"] for t in trials], dtype=float)
        times = np.asarray([t["time_ms"] for t in trials], dtype=float)
        repeat = {
            "trials": [
                {
                    "trial": i + 1,
                    "loss": float(t["loss"]),
                    "p": float(t["p"]),
                    "s": float(t["s"]),
                    "time_ms": float(t["time_ms"]),
                }
                for i, t in enumerate(trials)
            ],
            "stats": {
                "n": int(repeats),
                "loss_mean": float(losses.mean()),
                "loss_std": float(losses.std(ddof=0)),
                "p_mean": float(ps.mean()),
                "p_std": float(ps.std(ddof=0)),
                "s_mean": float(ss.mean()),
                "s_std": float(ss.std(ddof=0)),
                "time_mean_ms": float(times.mean()),
                "time_std_ms": float(times.std(ddof=0)),
                "evaluation_mode": "blb_action_repeated_validation_full",
            },
        }
        return {
            "loss": float(repeat["stats"]["loss_mean"]),
            "p": float(repeat["stats"]["p_mean"]),
            "s": float(repeat["stats"]["s_mean"]),
            "time_ms": float(repeat["stats"]["time_mean_ms"]),
        }, repeat

    def _run_single_blb_eval(self, decoded, *, gelu, softmax):
        ev = self.evaluator
        bridge = BLBNoiseRLBridge(
            ev.reversible_handler,
            layers_attribute="model." + ev.layers_attribute,
        )
        ev.apply_configuration(gelu, softmax)
        self._clear_legacy_noise()
        try:
            bridge.apply(
                first_input_sf=int(decoded.first_input_sf),
                first_input_N=16384,
                block1_cfgs=decoded.block1_cfgs,
                block2_cfgs=decoded.block2_cfgs,
                block3_cfgs=decoded.block3_cfgs,
                block4_cfgs=decoded.block4_cfgs,
                block5_cfgs=decoded.block5_cfgs,
            )
            split_name = ev._resolve_eval_split(use_train=False, split="validation_full")
            loss, p, s, time_ms = ev._run_evaluation(
                ev.dataloaders[split_name],
                use_train=False,
                split_name=split_name,
            )
            return {
                "loss": float(loss),
                "p": float(p),
                "s": float(s),
                "time_ms": float(time_ms),
            }
        finally:
            bridge.clear()
            self._clear_all_noise()

    def _clear_legacy_noise(self):
        ev = self.evaluator
        handler = ev.reversible_handler
        try:
            handler.restore_layer_input_noise(layer_indices=list(range(ev.total_layers)))
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
                method(layer_indices=list(range(ev.total_layers)))
            except Exception:
                pass

    def _clear_all_noise(self):
        self._clear_legacy_noise()
        ev = self.evaluator
        for restore_name in (
            "restore_layer_block5_noise",
            "restore_layer_block4_noise",
            "restore_layer_block3_noise",
            "restore_layer_block2_noise",
            "restore_layer_block1_noise",
            "restore_blb_first_input_noise",
        ):
            method = getattr(ev.reversible_handler, restore_name, None)
            if method is None:
                continue
            try:
                method(
                    layer_indices=list(range(ev.total_layers)),
                    layer_name="model." + ev.layers_attribute,
                )
            except Exception:
                pass

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
    ):
        output = {
            "dataset": self.evaluator.dataset_key,
            "selected_source": selected_source,
            "baseline_stage1": {
                "gelu": np.asarray(baseline_stage1_gelu, dtype=int).tolist(),
                "softmax": np.asarray(baseline_stage1_softmax, dtype=int).tolist(),
            },
            "selected_stage1": {
                "gelu": np.asarray(opt_gelu, dtype=int).tolist(),
                "softmax": np.asarray(opt_softmax, dtype=int).tolist(),
            },
            "constraints": {"selection": selection_constraints},
            "baseline": self._json_ready(baseline_result),
            "candidate_results": [self._json_ready(r) for r in candidate_results],
            "evaluation_protocol": {
                "version": 1,
                "mode": "blb_action_grid",
                "candidate_count": int(len(candidate_results)),
                "random_groups": "enabled" if self.random_enabled else "disabled",
                "action_ranges": list(self.action_ranges),
                "action_fixed": list(self.action_fixed),
                "repeat_n": int(self.repeat_n),
            },
        }
        output_path = os.path.join(
            self.results_dir,
            f"blb_action_final_eval_results_{self.evaluator.dataset_key}.json",
        )
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        return output_path

    @staticmethod
    def _attach_relative_metrics(baseline, results):
        for result in results:
            result["delta_loss_vs_baseline"] = float(result["loss"] - baseline["loss"])
            result["delta_p_vs_baseline"] = float(result["p"] - baseline["p"])
            result["delta_s_vs_baseline"] = float(result["s"] - baseline["s"])

    @staticmethod
    def _is_feasible(loss, p, s, constraints):
        if loss > constraints["loss"]:
            return False
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

    @staticmethod
    def _json_ready(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): BLBActionFinalEvaluationModule._json_ready(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BLBActionFinalEvaluationModule._json_ready(v) for v in obj]
        return obj
