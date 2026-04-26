import os
from typing import Dict, Optional, Tuple

import numpy as np

from final_evaluation_module import NOISE_SCALING_FACTOR_KEYS, UnifiedFinalEvaluationModule
from function_handler import (
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
    WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
)
from genetic_search_module import (
    STAGE1_GA_CONSTRAINT_RATIO,
    build_stage1_context,
    build_stage2_context,
    _clone_noise_config,
    _compute_rl_aligned_noise_search_score,
    _json_dump,
    _noise_signature,
    _series_plot,
    _to_serializable,
)
from noise_rl_module_v2 import (
    NOISE_STAGE_DYNAMIC_LIMIT_TOLERANCE,
    NOISE_STAGE_STATUS_NO_STABLE_FEASIBLE,
    NOISE_STAGE_STATUS_OK,
)


GREEDY_STAGE1_RESULT_FILENAME = "greedy_search_results.json"
GREEDY_STAGE2_RESULT_FILENAME = "noise_greedy_search_results.json"
GREEDY_STAGE1_DEFAULT_MAX_ITERATIONS = 200
GREEDY_STAGE2_DEFAULT_MAX_ITERATIONS = 200
GREEDY_STAGE1_ALLOWED_GELU_DEGREES = (4, 2, 1)


class GreedyUnifiedFinalEvaluationModule(UnifiedFinalEvaluationModule):
    """Greedy-labeled variant of the unified final-eval module."""

    def _log_optimized_label(self):
        if self.config_source == "search":
            return "Optimized (Greedy)"
        return super()._log_optimized_label() if hasattr(super(), "_log_optimized_label") else "Optimized"


def _resolve_greedy_iteration_budget(evaluator, *, attr_name: str, fallback: int) -> int:
    explicit = getattr(evaluator, attr_name, None)
    if explicit is None:
        return int(fallback)
    value = int(explicit)
    if value <= 0:
        raise ValueError(f"{attr_name} must be a positive integer, got {explicit!r}.")
    return value


def _stage1_score(context, cost: float) -> float:
    return float(context.base_tot_c - float(cost))


def _candidate_sort_key(candidate: Dict[str, object]) -> Tuple[float, float, float, float, float]:
    return (
        float(candidate.get("cost_saving", 0.0)),
        float(candidate.get("score", 0.0)),
        -float(candidate.get("loss", float("inf"))),
        float(candidate.get("metric1", 0.0)),
        -float(candidate.get("new_cost", float("inf"))),
    )


class Stage1GreedySearcher:
    def __init__(self, evaluator, max_iterations: Optional[int] = None):
        self.evaluator = evaluator
        self.max_iterations = int(
            max_iterations
            if max_iterations is not None
            else _resolve_greedy_iteration_budget(
                evaluator,
                attr_name="stage1_greedy_max_iterations",
                fallback=getattr(evaluator, "stage1_ga_generations", None)
                or GREEDY_STAGE1_DEFAULT_MAX_ITERATIONS,
            )
        )
        if self.max_iterations <= 0:
            raise ValueError("Stage-1 greedy max_iterations must be positive.")
        stage1_dir = os.path.dirname(evaluator.step_info_file)
        self.result_path = os.path.join(stage1_dir, GREEDY_STAGE1_RESULT_FILENAME)
        self.plot_path = os.path.join(stage1_dir, "greedy_search_curve.png")
        self.context = None
        self._cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Dict[str, object]] = {}

    def _log(self, message: str):
        self.evaluator.log(message)

    @staticmethod
    def _sanitize_stage1_candidate(gelu, softmax):
        gelu_arr = np.asarray(gelu, dtype=int).copy()
        softmax_arr = np.asarray(softmax, dtype=int).copy()
        had_degree0 = bool(np.any(gelu_arr == 0))
        if had_degree0:
            gelu_arr[gelu_arr == 0] = 1
        return gelu_arr, softmax_arr, had_degree0

    def _candidate_key(self, gelu, softmax):
        gelu_arr, softmax_arr, _ = self._sanitize_stage1_candidate(gelu, softmax)
        return tuple(gelu_arr.tolist()), tuple(softmax_arr.tolist())

    def _evaluate(self, gelu, softmax) -> Dict[str, object]:
        key = self._candidate_key(gelu, softmax)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        gelu_arr, softmax_arr, had_degree0 = self._sanitize_stage1_candidate(gelu, softmax)
        if had_degree0:
            self._log(
                "[Stage1][Greedy] Detected GELU degree 0 in a candidate; "
                "it was normalized to degree 1 because degree 0 is disabled."
            )
        loss, p, s, _ = self.evaluator.stage1_evaluate(
            gelu_arr,
            softmax_arr,
            split=self.context.reward_reference_split,
        )
        cost, _, _ = self.evaluator.get_simulated_cost(gelu_arr, softmax_arr)
        feasible = bool(
            self.evaluator._candidate_meets_constraints(
                loss,
                p,
                s,
                self.context.limit_loss,
                self.context.limit_p,
                self.context.limit_s,
            )
        )
        result = {
            "gelu": gelu_arr.copy(),
            "softmax": softmax_arr.copy(),
            "loss": float(loss),
            "metric1": float(p),
            "metric2": float(s),
            "cost": float(cost),
            "score": _stage1_score(self.context, float(cost)),
            "feasible": feasible,
        }
        self._cache[key] = result
        return result

    def _next_gelu_degree(self, current: int) -> Optional[int]:
        if current == 4:
            return 2
        if current == 2:
            return 1
        return None

    @staticmethod
    def _next_softmax_degree(current: int) -> Optional[int]:
        if current > 2:
            return int(current) - 1
        return None

    def run(self) -> Dict[str, object]:
        self._cache.clear()
        self.context = build_stage1_context(
            self.evaluator,
            log_fn=self._log,
            include_distribution=False,
            constraint_ratio=getattr(self.evaluator, "error_threshold", STAGE1_GA_CONSTRAINT_RATIO),
        )
        self._log("")
        self._log(
            "Stage-1 Greedy Search runs as an independent comparison algorithm: "
            "best-first single-step cost reduction over GELU/Softmax, without PPO "
            "and without GELU degree 0."
        )

        current_gelu = self.context.base_gelu.copy()
        current_softmax = self.context.base_softmax.copy()
        current = self._evaluate(current_gelu, current_softmax)
        best = current
        history = []

        for iteration in range(1, self.max_iterations + 1):
            candidates = []
            current_cost = float(current["cost"])

            for layer_idx in range(self.evaluator.total_layers):
                cur_deg = int(current_gelu[layer_idx])
                next_deg = self._next_gelu_degree(cur_deg)
                if next_deg is None:
                    continue
                test_gelu = current_gelu.copy()
                test_gelu[layer_idx] = next_deg
                result = self._evaluate(test_gelu, current_softmax)
                if result["feasible"]:
                    candidates.append(
                        {
                            **result,
                            "type": "GELU",
                            "layer": int(layer_idx),
                            "old_deg": int(cur_deg),
                            "new_deg": int(next_deg),
                            "new_cost": float(result["cost"]),
                            "cost_saving": float(current_cost - float(result["cost"])),
                            "test_gelu": test_gelu,
                            "test_softmax": current_softmax.copy(),
                        }
                    )

            for layer_idx in range(self.evaluator.total_layers):
                cur_deg = int(current_softmax[layer_idx])
                next_deg = self._next_softmax_degree(cur_deg)
                if next_deg is None:
                    continue
                test_softmax = current_softmax.copy()
                test_softmax[layer_idx] = next_deg
                result = self._evaluate(current_gelu, test_softmax)
                if result["feasible"]:
                    candidates.append(
                        {
                            **result,
                            "type": "Softmax",
                            "layer": int(layer_idx),
                            "old_deg": int(cur_deg),
                            "new_deg": int(next_deg),
                            "new_cost": float(result["cost"]),
                            "cost_saving": float(current_cost - float(result["cost"])),
                            "test_gelu": current_gelu.copy(),
                            "test_softmax": test_softmax,
                        }
                    )

            if not candidates:
                self._log(f"[Stage1][Greedy][Iter {iteration:04d}] no feasible single-step reduction; stop.")
                break

            selected = max(candidates, key=_candidate_sort_key)
            if float(selected["cost_saving"]) <= 1e-12:
                self._log(f"[Stage1][Greedy][Iter {iteration:04d}] no positive cost saving; stop.")
                break

            current_gelu = np.asarray(selected.pop("test_gelu"), dtype=int)
            current_softmax = np.asarray(selected.pop("test_softmax"), dtype=int)
            current = self._evaluate(current_gelu, current_softmax)
            if current["feasible"] and float(current["cost"]) <= float(best["cost"]) + 1e-12:
                best = current

            history.append(
                {
                    "generation": int(iteration),
                    "iteration": int(iteration),
                    "best_score": float(best["score"]),
                    "best_cost": float(best["cost"]),
                    "generation_best_score": float(current["score"]),
                    "generation_best_cost": float(current["cost"]),
                    "mean_score": float(current["score"]),
                    "feasible_ratio": 1.0,
                    "accepted_type": selected["type"],
                    "accepted_layer": int(selected["layer"]),
                    "old_deg": int(selected["old_deg"]),
                    "new_deg": int(selected["new_deg"]),
                    "cost_saving": float(selected["cost_saving"]),
                    "improved": True,
                }
            )
            self._log(
                f"[Stage1][Greedy][Iter {iteration:04d}/{self.max_iterations}] "
                f"{selected['type']} layer={selected['layer']} "
                f"{selected['old_deg']}->{selected['new_deg']}  "
                f"cost={current['cost']:.2f}  "
                f"metrics={self.evaluator._fmt_metrics(current['loss'], current['metric1'], current['metric2'])}"
            )

        payload = {
            "status": "ok",
            "algorithm": "greedy_stage1",
            "best_config": {key: value for key, value in best.items()},
            "max_iterations": int(self.max_iterations),
            "iterations": int(len(history)),
            "allowed_gelu_degrees": list(GREEDY_STAGE1_ALLOWED_GELU_DEGREES),
            "history": history,
            "context": self.context,
            "cache_size": int(len(self._cache)),
            "result_path": self.result_path,
            "plot_path": self.plot_path,
            "log_path": self.evaluator.log_file,
        }
        _json_dump(self.result_path, _to_serializable(payload))
        _series_plot(
            self.plot_path,
            "Stage-1 Greedy Search",
            history,
            score_key="best_score",
            cost_key="best_cost",
        )
        self._log("")
        self._log("Stage-1 Greedy Search completed.")
        self._log(f"  best cost: {best['cost']:.2f}")
        self._log(f"  GELU: {np.asarray(best['gelu'], dtype=int).tolist()}")
        self._log(f"  Softmax: {np.asarray(best['softmax'], dtype=int).tolist()}")
        self._log(f"  result: {self.result_path}")
        return payload


class Stage2NoiseGreedySearcher:
    def __init__(
        self,
        evaluator,
        fixed_gelu,
        fixed_softmax,
        fixed_label,
        fixed_source,
        max_iterations: Optional[int] = None,
    ):
        self.evaluator = evaluator
        self.fixed_gelu = np.asarray(fixed_gelu, dtype=int).copy()
        self.fixed_softmax = np.asarray(fixed_softmax, dtype=int).copy()
        self.fixed_label = str(fixed_label)
        self.fixed_source = str(fixed_source)
        self.max_iterations = int(
            max_iterations
            if max_iterations is not None
            else _resolve_greedy_iteration_budget(
                evaluator,
                attr_name="stage2_greedy_max_iterations",
                fallback=getattr(evaluator, "stage2_ga_generations", None)
                or GREEDY_STAGE2_DEFAULT_MAX_ITERATIONS,
            )
        )
        if self.max_iterations <= 0:
            raise ValueError("Stage-2 greedy max_iterations must be positive.")
        stage2_dir = os.path.dirname(evaluator.noise_step_info_file)
        self.result_path = os.path.join(stage2_dir, GREEDY_STAGE2_RESULT_FILENAME)
        self.plot_path = os.path.join(stage2_dir, "noise_greedy_search_curve.png")
        self.context = None
        self._cache: Dict[Tuple[int, Tuple[Tuple[int, ...], ...]], Dict[str, object]] = {}
        self._allowed_values = {
            "input_noise_scaling_factors": tuple(INPUT_NOISE_ALLOWED_SCALING_FACTORS),
            "wq_noise_scaling_factors": tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
            "wk_noise_scaling_factors": tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
            "wv_noise_scaling_factors": tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
            "wo_noise_scaling_factors": tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
            "wffn1_noise_scaling_factors": tuple(WFFN1_NOISE_ALLOWED_SCALING_FACTORS),
            "wffn2_noise_scaling_factors": tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
        }

    def _log(self, message: str):
        self.evaluator.log(message)

    def _evaluate(self, noise_cfg: Dict[str, np.ndarray], segments: Optional[int] = None) -> Dict[str, object]:
        segments = int(self.context.train_segments if segments is None else segments)
        signature = _noise_signature(noise_cfg)
        cache_key = (segments, signature)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        eval_noise_cfg = _clone_noise_config(noise_cfg)
        stats = self.evaluator.evaluate_model_with_attention_noise_segmented(
            self.fixed_gelu,
            self.fixed_softmax,
            segments=segments,
            split=self.context.reward_reference_split,
            **eval_noise_cfg,
        )
        cost, breakdown = self.evaluator.get_noise_simulated_cost(**eval_noise_cfg)
        score_components = _compute_rl_aligned_noise_search_score(
            stats=stats,
            cost=float(cost),
            baseline_reference_stats=self.context.baseline_reference_stats,
            search_limits=self.context.search_limits,
            dynamic_loss_std_cap=self.context.dynamic_loss_std_cap,
            dynamic_m1_std_cap=self.context.dynamic_m1_std_cap,
            dynamic_m2_std_cap=self.context.dynamic_m2_std_cap,
            cost_lower_bound=self.context.cost_lower_bound,
            cost_upper_bound=self.context.cost_upper_bound,
            num_metrics=self.evaluator.get_num_metrics(),
        )
        constraint_pass = bool(
            self.evaluator._candidate_meets_constraints(
                stats["loss_mean"],
                stats["p_mean"],
                stats["s_mean"],
                self.context.search_limits["loss"],
                self.context.search_limits["metric1"],
                self.context.search_limits["metric2"],
            )
        )
        std_pass = bool(
            stats["loss_std"] <= self.context.dynamic_loss_std_cap + 1e-8
            and stats["p_std"] <= self.context.dynamic_m1_std_cap + 1e-8
        )
        if self.evaluator.get_num_metrics() > 1:
            std_pass = std_pass and (
                stats["s_std"] <= self.context.dynamic_m2_std_cap + 1e-8
            )

        result = {
            **eval_noise_cfg,
            "segments": int(segments),
            "loss_mean": float(stats["loss_mean"]),
            "loss_std": float(stats["loss_std"]),
            "p_mean": float(stats["p_mean"]),
            "p_std": float(stats["p_std"]),
            "s_mean": float(stats["s_mean"]),
            "s_std": float(stats["s_std"]),
            "time_mean_ms": float(stats["time_mean_ms"]),
            "cost": float(cost),
            "breakdown": breakdown,
            "margin_loss": float(score_components["margin_loss"]),
            "margin_m1": float(score_components["margin_m1"]),
            "margin_m2": float(score_components["margin_m2"]),
            "perf_score": float(score_components["perf_score"]),
            "violation_penalty": float(score_components["violation_penalty"]),
            "cost_score": float(score_components["cost_score"]),
            "stability_reward_penalty": float(score_components["stability_reward_penalty"]),
            "raw_score": float(score_components["raw_score"]),
            "search_score_mean": float(score_components["score"]),
            "score": float(score_components["score"]),
            "score_mode": "rl_aligned_margin_v1",
            "constraint_pass": constraint_pass,
            "std_pass": std_pass,
            "qualification_passed": bool(constraint_pass and std_pass),
            "stats": {key: value for key, value in stats.items()},
        }
        self._cache[cache_key] = result
        return result

    def _previous_lower_value(self, key: str, current: int) -> Optional[int]:
        allowed = self._allowed_values[key]
        current_idx = allowed.index(int(current))
        if current_idx <= 0:
            return None
        return int(allowed[current_idx - 1])

    def _enumerate_single_step_reductions(self, current_cfg: Dict[str, np.ndarray]):
        for key in NOISE_SCALING_FACTOR_KEYS:
            arr = np.asarray(current_cfg[key], dtype=int)
            for layer_idx in range(self.evaluator.total_layers):
                next_value = self._previous_lower_value(key, int(arr[layer_idx]))
                if next_value is None:
                    continue
                candidate = _clone_noise_config(current_cfg)
                candidate[key][layer_idx] = int(next_value)
                yield key, int(layer_idx), int(arr[layer_idx]), int(next_value), candidate

    def run(self) -> Dict[str, object]:
        self._cache.clear()
        self.context = build_stage2_context(
            self.evaluator,
            self.fixed_gelu,
            self.fixed_softmax,
            log_fn=self._log,
            limit_tolerance=getattr(self.evaluator, "stage2_limit_tolerance", None),
            stability_tolerance=getattr(self.evaluator, "stage2_stability_tolerance", None),
        )
        self._log("")
        self._log(
            "Stage-2 Noise Greedy Search runs as an independent comparison algorithm: "
            "best-first single scaling-factor cost reduction under the same dynamic "
            "performance and stability constraints used by GA/RL final evaluation."
        )

        current_cfg = _clone_noise_config(self.context.baseline_low_risk_config)
        current = self._evaluate(current_cfg, segments=self.context.best_test_segments)
        incumbent = current
        history = []

        for iteration in range(1, self.max_iterations + 1):
            current_cost = float(current["cost"])
            candidates = []
            for key, layer_idx, old_value, new_value, candidate_cfg in self._enumerate_single_step_reductions(current_cfg):
                result = self._evaluate(candidate_cfg)
                if not result["qualification_passed"]:
                    continue
                cost_saving = current_cost - float(result["cost"])
                if cost_saving <= 1e-12:
                    continue
                candidates.append(
                    {
                        **result,
                        "key": key,
                        "layer": layer_idx,
                        "old_value": old_value,
                        "new_value": new_value,
                        "cost_saving": float(cost_saving),
                        "candidate_cfg": candidate_cfg,
                    }
                )

            if not candidates:
                self._log(f"[Stage2][Greedy][Iter {iteration:04d}] no qualified single-step reduction; stop.")
                break

            ranked_candidates = sorted(
                candidates,
                key=lambda item: (
                    float(item["cost_saving"]),
                    float(item["score"]),
                    -float(item["loss_mean"]),
                    float(item["p_mean"]),
                    -float(item["cost"]),
                ),
                reverse=True,
            )
            selected = None
            confirmed_current = None
            for candidate in ranked_candidates:
                candidate_cfg = _clone_noise_config(candidate["candidate_cfg"])
                confirmed = self._evaluate(candidate_cfg, segments=self.context.best_test_segments)
                if confirmed["qualification_passed"] and float(confirmed["cost"]) < current_cost - 1e-12:
                    selected = candidate
                    confirmed_current = confirmed
                    break
            if selected is None or confirmed_current is None:
                self._log(
                    f"[Stage2][Greedy][Iter {iteration:04d}] train-qualified reductions "
                    "failed best-test confirmation; stop."
                )
                break

            current_cfg = _clone_noise_config(selected.pop("candidate_cfg"))
            current = confirmed_current
            if float(current["cost"]) <= float(incumbent["cost"]) + 1e-12:
                incumbent = current

            history.append(
                {
                    "generation": int(iteration),
                    "iteration": int(iteration),
                    "best_score": float(incumbent["score"]),
                    "best_cost": float(incumbent["cost"]),
                    "generation_best_score": float(current["score"]),
                    "generation_best_cost": float(current["cost"]),
                    "mean_score": float(current["score"]),
                    "qualified_ratio": 1.0,
                    "accepted_key": selected["key"],
                    "accepted_layer": int(selected["layer"]),
                    "old_value": int(selected["old_value"]),
                    "new_value": int(selected["new_value"]),
                    "cost_saving": float(selected["cost_saving"]),
                    "improved": True,
                }
            )
            self._log(
                f"[Stage2][Greedy][Iter {iteration:04d}/{self.max_iterations}] "
                f"{selected['key']} layer={selected['layer']} "
                f"{selected['old_value']}->{selected['new_value']}  "
                f"cost={current['cost']:.2f}  score={current['score']:.6f}  "
                f"constraint={current['constraint_pass']}  std={current['std_pass']}"
            )

        selected_config = incumbent if incumbent.get("qualification_passed", False) else None
        status = NOISE_STAGE_STATUS_OK if selected_config is not None else NOISE_STAGE_STATUS_NO_STABLE_FEASIBLE
        payload = {
            "algorithm": "greedy_stage2_noise",
            "fixed_gelu": self.fixed_gelu.copy(),
            "fixed_softmax": self.fixed_softmax.copy(),
            "fixed_label": self.fixed_label,
            "fixed_source": self.fixed_source,
            "baseline_noise_config": _clone_noise_config(self.context.cost_reference_noise_config),
            "baseline_tot_c": float(self.context.cost_reference_tot_c),
            "cost_lower_bound": float(self.context.cost_lower_bound),
            "cost_upper_bound": float(self.context.cost_upper_bound),
            "cost_reference_noise_config": _clone_noise_config(self.context.cost_reference_noise_config),
            "cost_reference_source": "max_noise_configuration",
            "performance_baseline_gelu": self.fixed_gelu.copy(),
            "performance_baseline_softmax": self.fixed_softmax.copy(),
            "performance_baseline_source": "stage1_fixed_low_risk_noise",
            "baseline_segments": int(self.context.baseline_segments),
            "search_baseline_stats": {key: value for key, value in self.context.baseline_reference_stats.items()},
            "worst_reference_stats": {key: value for key, value in self.context.worst_reference_stats.items()},
            "worst_case_noise_config": _clone_noise_config(self.context.worst_case_noise_config),
            "limit_computation_method": "baseline_tolerance_percentage",
            "limit_tolerance": float(getattr(self.evaluator, "stage2_limit_tolerance", NOISE_STAGE_DYNAMIC_LIMIT_TOLERANCE)),
            "stability_tolerance": float(getattr(self.evaluator, "stage2_stability_tolerance", NOISE_STAGE_DYNAMIC_LIMIT_TOLERANCE)),
            "search_limits": {key: float(value) for key, value in self.context.search_limits.items()},
            "status": status,
            "best_noise_config": ({key: value for key, value in selected_config.items()} if selected_config is not None else None),
            "best_config": ({key: value for key, value in selected_config.items()} if selected_config is not None else None),
            "stable_search_best_noise_config": ({key: value for key, value in selected_config.items()} if selected_config is not None else None),
            "stable_joint_best_noise_config": ({key: value for key, value in selected_config.items()} if selected_config is not None else None),
            "selection_diagnostics": {
                "selection_mode": "greedy_best_first_single_step",
                "mc_train_samples": int(self.context.train_segments),
                "mc_confirm_segments": int(self.context.confirm_segments),
                "dynamic_loss_std_cap": float(self.context.dynamic_loss_std_cap),
                "dynamic_m1_std_cap": float(self.context.dynamic_m1_std_cap),
                "dynamic_m2_std_cap": float(self.context.dynamic_m2_std_cap),
                "final_incumbent": ({key: value for key, value in incumbent.items()} if incumbent is not None else None),
            },
            "shortlist_diagnostics": {},
            "limit_loss": float(self.context.search_limits["loss"]),
            "limit_p": float(self.context.search_limits["metric1"]),
            "limit_s": float(self.context.search_limits["metric2"]),
            "proxy_limit_loss": float(self.context.search_limits["loss"]),
            "proxy_limit_p": float(self.context.search_limits["metric1"]),
            "proxy_limit_s": float(self.context.search_limits["metric2"]),
            "proxy_base_loss": float(self.context.baseline_reference_stats["loss_mean"]),
            "proxy_base_p": float(self.context.baseline_reference_stats["p_mean"]),
            "proxy_base_s": float(self.context.baseline_reference_stats["s_mean"]),
            "training_eval_split": str(self.context.reward_reference_split),
            "training_hparams": {
                "max_iterations": int(self.max_iterations),
                "train_segments": int(self.context.train_segments),
                "confirm_segments": int(self.context.confirm_segments),
                "best_test_segments": int(self.context.best_test_segments),
            },
            "greedy_history": history,
            "result_path": self.result_path,
            "plot_path": self.plot_path,
            "log_path": self.evaluator.noise_log_file,
            "best_generation": int(len(history)),
        }
        _json_dump(self.result_path, _to_serializable(payload))
        _series_plot(
            self.plot_path,
            "Stage-2 Noise Greedy Search",
            history,
            score_key="best_score",
            cost_key="best_cost",
        )
        self._log("")
        self._log(f"Stage-2 Noise Greedy Search completed with status={status}.")
        self._log(f"  incumbent cost: {incumbent['cost']:.2f}")
        self._log(f"  result: {self.result_path}")
        return payload


def run_stage1_greedy_search(evaluator, random_seed=42, resume_checkpoint_path=None, max_iterations=None):
    del random_seed, resume_checkpoint_path
    return Stage1GreedySearcher(evaluator, max_iterations=max_iterations).run()


def run_stage2_noise_greedy_search(
    evaluator,
    fixed_gelu,
    fixed_softmax,
    fixed_label,
    fixed_source,
    random_seed=42,
    resume_checkpoint_path=None,
    max_iterations=None,
):
    del random_seed, resume_checkpoint_path
    return Stage2NoiseGreedySearcher(
        evaluator=evaluator,
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        fixed_label=fixed_label,
        fixed_source=fixed_source,
        max_iterations=max_iterations,
    ).run()
