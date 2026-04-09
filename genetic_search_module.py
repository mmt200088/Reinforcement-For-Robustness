import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from final_evaluation_module import FinalEvaluationModule
from layer_importance_evaluator import USE_VALIDATION_FOR_REWARD
from function_handler import (
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
    WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
)
from noise_final_evaluation_module import (
    NOISE_SCALING_FACTOR_KEYS,
    NoiseFinalEvaluationModule,
)
from noise_rl_module_v2 import (
    NOISE_STAGE_BASELINE_SEGMENTS,
    NOISE_STAGE_BEST_TEST_SEGMENTS,
    NOISE_STAGE_BEST_TEST_TRIGGER_MARGIN,
    NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE,
    NOISE_STAGE_MC_CONFIRM_SEGMENTS,
    NOISE_STAGE_MC_TRAIN_SAMPLES,
    NOISE_STAGE_STATUS_NO_STABLE_FEASIBLE,
    NOISE_STAGE_STATUS_OK,
    _auto_adjust_segments,
    _compute_dynamic_limits,
    _compute_dynamic_std_upper_bound,
    _get_low_risk_noise_configuration,
    _get_worst_case_noise_configuration,
)


EPS = 1e-8
SCORE_EXP_BASE = 4.0
STAGE1_DEFAULT_POPULATION = 32
STAGE2_DEFAULT_POPULATION = 16
STAGNATION_TOLERANCE = 5


@dataclass
class Stage1Context:
    base_gelu: np.ndarray
    base_softmax: np.ndarray
    base_tot_c: float
    base_g_c: float
    base_s_c: float
    base_loss: float
    base_p: float
    base_s: float
    limit_loss: float
    limit_p: float
    limit_s: float
    reward_reference_split: str
    gelu_degree0_eligible: np.ndarray


@dataclass
class Stage2Context:
    reward_reference_split: str
    baseline_low_risk_config: Dict[str, np.ndarray]
    worst_case_noise_config: Dict[str, np.ndarray]
    cost_reference_noise_config: Dict[str, np.ndarray]
    cost_reference_tot_c: float
    baseline_reference_stats: Dict[str, object]
    worst_reference_stats: Dict[str, object]
    search_limits: Dict[str, float]
    dynamic_loss_std_cap: float
    dynamic_m1_std_cap: float
    dynamic_m2_std_cap: float
    baseline_segments: int
    train_segments: int
    confirm_segments: int
    best_test_segments: int


class GeneticFinalEvaluationModule(FinalEvaluationModule):
    def _resolve_selected_config(self, search_best_config, total_layers: int):
        gelu, softmax, label, source = super()._resolve_selected_config(
            search_best_config=search_best_config,
            total_layers=total_layers,
        )
        if self.config_source == "search":
            return gelu, softmax, "Optimized (Genetic)", source
        return gelu, softmax, label, source


class GeneticNoiseFinalEvaluationModule(NoiseFinalEvaluationModule):
    def _resolve_selected_config(self, search_best_noise_config, total_layers):
        cfg, label, source = super()._resolve_selected_config(
            search_best_noise_config=search_best_noise_config,
            total_layers=total_layers,
        )
        if self.config_source == "search":
            return cfg, "Optimized (Noise Genetic)", source
        return cfg, label, source


def build_stage1_context(evaluator, log_fn=None, include_distribution=True) -> Stage1Context:
    base_gelu = np.full(evaluator.total_layers, 4, dtype=int)
    base_softmax = np.full(evaluator.total_layers, 6, dtype=int)
    base_tot_c, base_g_c, base_s_c = evaluator.get_simulated_cost(base_gelu, base_softmax)

    reward_reference_split = evaluator.get_reward_reference_split_name()
    if USE_VALIDATION_FOR_REWARD:
        base_loss, base_p, base_s, _ = evaluator.stage1_evaluate(
            base_gelu,
            base_softmax,
            split=reward_reference_split,
        )
    else:
        base_loss, base_p, base_s, _ = evaluator.stage1_evaluate(
            base_gelu,
            base_softmax,
            use_train=True,
        )

    limits = evaluator.build_constraint_limits_from_metrics(base_loss, base_p, base_s)

    gelu_degree0_eligible = np.zeros(evaluator.total_layers, dtype=bool)
    if include_distribution:
        gelu_degree0_eligible, gelu_interval_counts = evaluator.analyze_gelu_distribution()
        if log_fn is not None:
            log_fn("")
            log_fn("Stage-1 GELU distribution screening for degree-0 eligibility:")
            for layer_idx in range(evaluator.total_layers):
                counts = gelu_interval_counts[layer_idx]
                total = float(np.sum(counts))
                if total <= 0:
                    continue
                pcts = counts / total * 100.0
                status = "eligible" if gelu_degree0_eligible[layer_idx] else "not_eligible"
                log_fn(
                    f"  layer={layer_idx:02d}  <-2.7={pcts[0]:6.2f}%  "
                    f"[-2.7,0)={pcts[1]:6.2f}%  [0,2.7]={pcts[2]:6.2f}%  "
                    f">2.7={pcts[3]:6.2f}%  status={status}"
                )

    if log_fn is not None:
        log_fn("")
        log_fn("Stage-1 baseline and constraints:")
        log_fn(
            f"  baseline metrics on {reward_reference_split}: "
            f"{evaluator._fmt_metrics(base_loss, base_p, base_s)}"
        )
        log_fn(
            f"  simulated cost: total={base_tot_c:.2f}, gelu={base_g_c:.2f}, softmax={base_s_c:.2f}"
        )
        log_fn(
            "  constraints: "
            + evaluator._fmt_constraints(
                limits["loss"],
                limits["metric1"],
                limits["metric2"],
            )
        )

    return Stage1Context(
        base_gelu=base_gelu,
        base_softmax=base_softmax,
        base_tot_c=float(base_tot_c),
        base_g_c=float(base_g_c),
        base_s_c=float(base_s_c),
        base_loss=float(base_loss),
        base_p=float(base_p),
        base_s=float(base_s),
        limit_loss=float(limits["loss"]),
        limit_p=float(limits["metric1"]),
        limit_s=float(limits["metric2"]),
        reward_reference_split=reward_reference_split,
        gelu_degree0_eligible=gelu_degree0_eligible,
    )


def build_stage2_context(evaluator, fixed_gelu, fixed_softmax, log_fn=None) -> Stage2Context:
    fixed_gelu = np.asarray(fixed_gelu, dtype=int)
    fixed_softmax = np.asarray(fixed_softmax, dtype=int)

    if evaluator.has_dataset_split("validation_full"):
        reward_reference_split = "validation_full"
    else:
        reward_reference_split = evaluator.get_reward_reference_split_name()

    baseline_low_risk_config = _get_low_risk_noise_configuration(evaluator)
    worst_case_noise_config = _get_worst_case_noise_configuration(evaluator)
    cost_reference_noise_config = evaluator._get_max_noise_configuration()

    dataset = evaluator.dataset_splits.get(reward_reference_split)
    dataset_size = len(dataset) if dataset is not None else 0
    baseline_segments = _auto_adjust_segments(
        dataset_size,
        NOISE_STAGE_BASELINE_SEGMENTS,
        log_fn=log_fn,
    )
    train_segments = _auto_adjust_segments(
        dataset_size,
        NOISE_STAGE_MC_TRAIN_SAMPLES,
        log_fn=log_fn,
    )
    confirm_segments = _auto_adjust_segments(
        dataset_size,
        NOISE_STAGE_MC_CONFIRM_SEGMENTS,
        log_fn=log_fn,
    )
    best_test_segments = _auto_adjust_segments(
        dataset_size,
        NOISE_STAGE_BEST_TEST_SEGMENTS,
        log_fn=log_fn,
    )

    baseline_reference_stats = evaluator.evaluate_model_with_attention_noise_segmented(
        fixed_gelu,
        fixed_softmax,
        segments=baseline_segments,
        split=reward_reference_split,
        **baseline_low_risk_config,
    )
    worst_reference_stats = evaluator.evaluate_model_with_attention_noise_segmented(
        fixed_gelu,
        fixed_softmax,
        segments=baseline_segments,
        split=reward_reference_split,
        **worst_case_noise_config,
    )
    cost_reference_tot_c, _ = evaluator.get_noise_simulated_cost(**cost_reference_noise_config)

    search_limits = _compute_dynamic_limits(
        baseline_reference_stats["loss_mean"],
        baseline_reference_stats["p_mean"],
        baseline_reference_stats["s_mean"],
        worst_reference_stats["loss_mean"],
        worst_reference_stats["p_mean"],
        worst_reference_stats["s_mean"],
    )
    dynamic_loss_std_cap = _compute_dynamic_std_upper_bound(
        baseline_reference_stats["loss_std"],
        worst_reference_stats["loss_std"],
    )
    dynamic_m1_std_cap = _compute_dynamic_std_upper_bound(
        baseline_reference_stats["p_std"],
        worst_reference_stats["p_std"],
    )
    dynamic_m2_std_cap = _compute_dynamic_std_upper_bound(
        baseline_reference_stats["s_std"],
        worst_reference_stats["s_std"],
    )

    if log_fn is not None:
        log_fn("")
        log_fn("Stage-2 baseline and dynamic constraints:")
        log_fn(f"  split={reward_reference_split}  quartile={NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE}")
        log_fn(
            "  low-risk baseline: "
            + evaluator._fmt_metrics(
                baseline_reference_stats["loss_mean"],
                baseline_reference_stats["p_mean"],
                baseline_reference_stats["s_mean"],
            )
        )
        log_fn(
            "  worst-case baseline: "
            + evaluator._fmt_metrics(
                worst_reference_stats["loss_mean"],
                worst_reference_stats["p_mean"],
                worst_reference_stats["s_mean"],
            )
        )
        log_fn(
            "  dynamic constraints: "
            + evaluator._fmt_constraints(
                search_limits["loss"],
                search_limits["metric1"],
                search_limits["metric2"],
            )
        )
        log_fn(
            f"  std caps: loss={dynamic_loss_std_cap:.6f}, "
            f"m1={dynamic_m1_std_cap:.6f}, m2={dynamic_m2_std_cap:.6f}"
        )
        log_fn(
            f"  cost reference (max-sf configuration): total_noise_cost={cost_reference_tot_c:.2f}"
        )

    return Stage2Context(
        reward_reference_split=reward_reference_split,
        baseline_low_risk_config=baseline_low_risk_config,
        worst_case_noise_config=worst_case_noise_config,
        cost_reference_noise_config=cost_reference_noise_config,
        cost_reference_tot_c=float(cost_reference_tot_c),
        baseline_reference_stats=baseline_reference_stats,
        worst_reference_stats=worst_reference_stats,
        search_limits={k: float(v) for k, v in search_limits.items()},
        dynamic_loss_std_cap=float(dynamic_loss_std_cap),
        dynamic_m1_std_cap=float(dynamic_m1_std_cap),
        dynamic_m2_std_cap=float(dynamic_m2_std_cap),
        baseline_segments=int(baseline_segments),
        train_segments=int(train_segments),
        confirm_segments=int(confirm_segments),
        best_test_segments=int(best_test_segments),
    )


def build_stage2_final_eval_context_without_search(evaluator):
    baseline_noise_config = evaluator._get_max_noise_configuration()
    baseline_tot_c, _ = evaluator.get_noise_simulated_cost(**baseline_noise_config)
    exact_baseline_gelu, exact_baseline_softmax = evaluator.get_stage1_exact_baseline_configuration()
    baseline_summary = evaluator.evaluate_model_repeated(
        exact_baseline_gelu,
        exact_baseline_softmax,
        repeats=evaluator.noise_eval_repeat_n,
        split=evaluator.get_reward_reference_split_name(),
    )
    limits = evaluator.build_constraint_limits_from_metrics(
        baseline_summary["loss_mean"],
        baseline_summary["p_mean"],
        baseline_summary["s_mean"],
    )
    return {
        "baseline_noise_config": baseline_noise_config,
        "baseline_tot_c": float(baseline_tot_c),
        "limit_loss": float(limits["loss"]),
        "limit_p": float(limits["metric1"]),
        "limit_s": float(limits["metric2"]),
    }


def resolve_stage1_selected_config(
    evaluator,
    search_best_config=None,
    config_source="search",
    config_path="glue_configs_best_ppo.json",
    manual_gelu=None,
    manual_softmax=None,
):
    module = GeneticFinalEvaluationModule(
        evaluator=evaluator,
        config_source=config_source,
        config_path=config_path,
        manual_gelu=manual_gelu,
        manual_softmax=manual_softmax,
        random_seed=evaluator.final_eval_random_seed,
        permutation_trials=evaluator.final_eval_permutation_trials,
        cost_equivalent_trials=evaluator.final_eval_cost_equivalent_trials,
        budget_equivalent_trials=evaluator.final_eval_budget_equivalent_trials,
        results_dir=evaluator.stage1_final_eval_dir,
    )
    return module._resolve_selected_config(search_best_config, evaluator.total_layers)


def _json_dump(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _series_plot(path, title, history, score_key="best_score", cost_key="best_cost"):
    if not history:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    generations = [item["generation"] for item in history]
    best_scores = [item.get(score_key, 0.0) for item in history]
    mean_scores = [item.get("mean_score", 0.0) for item in history]
    best_costs = [item.get(cost_key, 0.0) for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(generations, best_scores, label="Best score", color="#1f77b4")
    axes[0].plot(generations, mean_scores, label="Mean score", color="#6baed6", alpha=0.8)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Genetic Search Score")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(generations, best_costs, label="Best cost", color="#d62728")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Cost")
    axes[1].set_title("Best Candidate Cost")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _weighted_choice_indices(rng, scores: np.ndarray, n: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.min(scores)
    weights = shifted + EPS
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        probs = np.full(len(scores), 1.0 / max(len(scores), 1), dtype=float)
    else:
        probs = weights / total
    return rng.choice(len(scores), size=n, replace=True, p=probs)


def _penalty_lower_is_better(value: float, reference: float, limit: float) -> float:
    reference = max(abs(reference), EPS)
    linear = max(float(value) / reference, EPS)
    if value <= limit:
        return linear
    gap = (float(value) - float(limit)) / max(abs(float(limit) - float(reference)), EPS)
    return linear + (SCORE_EXP_BASE ** max(0.0, gap))


def _penalty_higher_is_better(value: float, reference: float, limit: float) -> float:
    if value <= EPS:
        return 1e6
    linear = max(float(reference) / max(float(value), EPS), EPS)
    if value >= limit:
        return linear
    gap = (float(limit) - float(value)) / max(abs(float(reference) - float(limit)), EPS)
    return linear + (SCORE_EXP_BASE ** max(0.0, gap))


def _penalty_std_upper_bound(value: float, reference: float, cap: float) -> float:
    reference = max(abs(reference), EPS)
    linear = max(float(value) / reference, 1.0)
    if value <= cap:
        return linear
    gap = (float(value) - float(cap)) / max(abs(float(cap) - float(reference)), EPS)
    return linear + (SCORE_EXP_BASE ** max(0.0, gap))


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _to_serializable(getattr(value, key))
            for key in value.__dataclass_fields__.keys()
        }
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _clone_noise_config(noise_cfg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        key: np.asarray(noise_cfg[key], dtype=int).copy()
        for key in NOISE_SCALING_FACTOR_KEYS
    }


def _noise_signature(noise_cfg: Dict[str, np.ndarray]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(
        tuple(np.asarray(noise_cfg[key], dtype=int).tolist())
        for key in NOISE_SCALING_FACTOR_KEYS
    )


class Stage1GeneticSearcher:
    def __init__(self, evaluator, random_seed=42):
        self.evaluator = evaluator
        self.rng = np.random.default_rng(int(random_seed))
        self.population_size = max(STAGE1_DEFAULT_POPULATION, evaluator.total_layers * 2)
        self.max_generations = max(1, math.ceil(evaluator.stage1_rl_episodes / self.population_size))
        stage1_dir = os.path.dirname(evaluator.step_info_file)
        self.log_path = os.path.join(stage1_dir, "ga_search_log.txt")
        self.result_path = os.path.join(stage1_dir, "ga_search_results.json")
        self.plot_path = os.path.join(stage1_dir, "ga_search_curve.png")
        self._cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Dict[str, object]] = {}
        self._valid_states: List[List[Tuple[int, int]]] = []
        self._neighbor_map: List[Dict[Tuple[int, int], List[Tuple[int, int]]]] = []
        self.context: Optional[Stage1Context] = None

    def _log(self, message: str):
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        self.evaluator.log(message)

    def _allowed_gelu_degrees(self, layer_idx: int) -> Tuple[int, ...]:
        choices = [4, 2, 1]
        if self.context.gelu_degree0_eligible[layer_idx]:
            choices.append(0)
        return tuple(choices)

    @staticmethod
    def _softmax_degrees() -> Tuple[int, ...]:
        return (6, 5, 4, 3, 2)

    def _candidate_key(self, gelu, softmax):
        return tuple(np.asarray(gelu, dtype=int).tolist()), tuple(np.asarray(softmax, dtype=int).tolist())

    def _evaluate(self, gelu, softmax) -> Dict[str, object]:
        key = self._candidate_key(gelu, softmax)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        gelu_arr = np.asarray(gelu, dtype=int)
        softmax_arr = np.asarray(softmax, dtype=int)
        loss, p, s, _ = self.evaluator.stage1_evaluate(
            gelu_arr,
            softmax_arr,
            split=self.context.reward_reference_split,
        )
        tot_c, _, _ = self.evaluator.get_simulated_cost(gelu_arr, softmax_arr)

        penalties = [
            _penalty_lower_is_better(loss, self.context.base_loss, self.context.limit_loss),
            _penalty_higher_is_better(p, self.context.base_p, self.context.limit_p),
        ]
        if self.evaluator.get_num_metrics() > 1:
            penalties.append(
                _penalty_higher_is_better(s, self.context.base_s, self.context.limit_s)
            )
        penalty = float(np.mean(penalties))
        score = float((self.context.base_tot_c / max(float(tot_c), EPS)) / max(penalty, EPS))
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
            "cost": float(tot_c),
            "score": float(score),
            "penalty": float(penalty),
            "feasible": feasible,
        }
        self._cache[key] = result
        return result

    def _enumerate_valid_states(self):
        self._valid_states = []
        self._neighbor_map = []
        baseline_gelu = self.context.base_gelu.copy()
        baseline_softmax = self.context.base_softmax.copy()

        for layer_idx in range(self.evaluator.total_layers):
            layer_states = []
            for gelu_degree in self._allowed_gelu_degrees(layer_idx):
                for softmax_degree in self._softmax_degrees():
                    test_gelu = baseline_gelu.copy()
                    test_softmax = baseline_softmax.copy()
                    test_gelu[layer_idx] = int(gelu_degree)
                    test_softmax[layer_idx] = int(softmax_degree)
                    if self._evaluate(test_gelu, test_softmax)["feasible"]:
                        layer_states.append((int(gelu_degree), int(softmax_degree)))

            if (4, 6) not in layer_states:
                layer_states.append((4, 6))

            gelu_order = {degree: idx for idx, degree in enumerate(self._allowed_gelu_degrees(layer_idx))}
            softmax_order = {degree: idx for idx, degree in enumerate(self._softmax_degrees())}
            layer_states = sorted(
                list(dict.fromkeys(layer_states)),
                key=lambda item: (gelu_order[item[0]], softmax_order[item[1]]),
            )
            self._valid_states.append(layer_states)

            neighbors = {}
            for gelu_degree, softmax_degree in layer_states:
                current = (gelu_degree, softmax_degree)
                adjacent = []
                for other in layer_states:
                    if other == current:
                        continue
                    dg = abs(gelu_order[other[0]] - gelu_order[gelu_degree])
                    ds = abs(softmax_order[other[1]] - softmax_order[softmax_degree])
                    if dg + ds == 1:
                        adjacent.append(other)
                if not adjacent:
                    adjacent = [other for other in layer_states if other != current]
                neighbors[current] = adjacent
            self._neighbor_map.append(neighbors)

    def _mutate(self, gelu, softmax):
        gelu_arr = np.asarray(gelu, dtype=int).copy()
        softmax_arr = np.asarray(softmax, dtype=int).copy()
        mutation_count = int(
            self.rng.integers(
                1,
                max(2, math.ceil(self.evaluator.total_layers / 4)) + 1,
            )
        )
        mutation_layers = self.rng.choice(
            self.evaluator.total_layers,
            size=min(mutation_count, self.evaluator.total_layers),
            replace=False,
        )
        for layer_idx in mutation_layers:
            current = (int(gelu_arr[layer_idx]), int(softmax_arr[layer_idx]))
            neighbors = list(
                self._neighbor_map[layer_idx].get(current, self._valid_states[layer_idx])
            )
            if not neighbors:
                continue
            next_state = neighbors[int(self.rng.integers(len(neighbors)))]
            gelu_arr[layer_idx] = int(next_state[0])
            softmax_arr[layer_idx] = int(next_state[1])
        return gelu_arr, softmax_arr

    def _make_initial_population(self):
        population: List[Tuple[np.ndarray, np.ndarray]] = []
        seen = set()

        def _add_candidate(gelu_arr, softmax_arr):
            key = self._candidate_key(gelu_arr, softmax_arr)
            if key in seen:
                return
            population.append(
                (
                    np.asarray(gelu_arr, dtype=int).copy(),
                    np.asarray(softmax_arr, dtype=int).copy(),
                )
            )
            seen.add(key)

        _add_candidate(self.context.base_gelu, self.context.base_softmax)

        attempts = 0
        max_attempts = max(200, self.population_size * 30)
        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1
            gelu_arr = self.context.base_gelu.copy()
            softmax_arr = self.context.base_softmax.copy()
            rounds = int(
                self.rng.integers(
                    1,
                    max(2, math.ceil(self.evaluator.total_layers / 3)) + 1,
                )
            )
            for _ in range(rounds):
                gelu_arr, softmax_arr = self._mutate(gelu_arr, softmax_arr)
            _add_candidate(gelu_arr, softmax_arr)

        return population

    def run(self) -> Dict[str, object]:
        self._cache.clear()
        with open(self.log_path, "w", encoding="utf-8") as handle:
            handle.write("")

        self.context = build_stage1_context(self.evaluator, log_fn=self._log)
        self._log("")
        self._log(
            "Stage-1 Genetic Search follows COINN: score-guided random selection and "
            "adjacent-mesh mutation, without PPO and without crossover."
        )

        self._enumerate_valid_states()
        self._log("Per-layer valid mesh states:")
        for layer_idx, layer_states in enumerate(self._valid_states):
            self._log(
                f"  layer={layer_idx:02d}  valid_states={len(layer_states)}  "
                f"states={layer_states}"
            )

        population = self._make_initial_population()
        self._log(
            f"Initial population size={len(population)}  "
            f"max_generations={self.max_generations}"
        )

        best_candidate = self._evaluate(
            self.context.base_gelu,
            self.context.base_softmax,
        )
        best_generation = 0
        stagnation = 0
        history = []

        for generation in range(1, self.max_generations + 1):
            results = [self._evaluate(gelu_arr, softmax_arr) for gelu_arr, softmax_arr in population]
            scores = np.asarray([max(float(item["score"]), EPS) for item in results], dtype=float)
            mean_score = float(np.mean(scores)) if len(scores) else 0.0
            feasible_results = [item for item in results if item["feasible"]]
            generation_best = (
                max(feasible_results, key=lambda item: item["score"])
                if feasible_results
                else max(results, key=lambda item: item["score"])
            )
            raw_best = max(results, key=lambda item: item["score"])

            improved = False
            if generation_best["feasible"] and (
                generation_best["score"] > best_candidate["score"] + 1e-12
            ):
                best_candidate = self._evaluate(
                    generation_best["gelu"],
                    generation_best["softmax"],
                )
                best_generation = generation
                stagnation = 0
                improved = True
            else:
                stagnation += 1

            feasible_ratio = (
                float(sum(1 for item in results if item["feasible"])) / float(len(results))
                if results
                else 0.0
            )
            history.append(
                {
                    "generation": generation,
                    "best_score": float(best_candidate["score"]),
                    "best_cost": float(best_candidate["cost"]),
                    "generation_best_score": float(raw_best["score"]),
                    "generation_best_cost": float(raw_best["cost"]),
                    "mean_score": mean_score,
                    "feasible_ratio": feasible_ratio,
                    "improved": improved,
                }
            )
            self._log(
                f"[Stage1][Gen {generation:04d}] "
                f"gen_best_score={raw_best['score']:.6f}  "
                f"gen_best_cost={raw_best['cost']:.2f}  "
                f"global_best_score={best_candidate['score']:.6f}  "
                f"global_best_cost={best_candidate['cost']:.2f}  "
                f"feasible_ratio={feasible_ratio:.2%}  "
                f"stagnation={stagnation}/{STAGNATION_TOLERANCE}"
            )
            if stagnation > STAGNATION_TOLERANCE:
                self._log(
                    f"Early stop: best feasible score has not improved for more than "
                    f"{STAGNATION_TOLERANCE} generations."
                )
                break

            ranked_indices = sorted(
                range(len(results)),
                key=lambda idx: (int(results[idx]["feasible"]), results[idx]["score"]),
                reverse=True,
            )
            next_population: List[Tuple[np.ndarray, np.ndarray]] = []
            next_seen = set()

            def _add_next(gelu_arr, softmax_arr):
                key = self._candidate_key(gelu_arr, softmax_arr)
                if key in next_seen:
                    return
                next_population.append(
                    (
                        np.asarray(gelu_arr, dtype=int).copy(),
                        np.asarray(softmax_arr, dtype=int).copy(),
                    )
                )
                next_seen.add(key)

            _add_next(best_candidate["gelu"], best_candidate["softmax"])
            elite_count = min(max(2, self.population_size // 8), len(results))
            for idx in ranked_indices[:elite_count]:
                _add_next(results[idx]["gelu"], results[idx]["softmax"])

            parent_indices = _weighted_choice_indices(
                self.rng,
                scores,
                max(self.population_size * 2, 1),
            )
            for parent_idx in parent_indices:
                parent = results[int(parent_idx)]
                child_gelu, child_softmax = self._mutate(
                    parent["gelu"],
                    parent["softmax"],
                )
                _add_next(child_gelu, child_softmax)
                if len(next_population) >= self.population_size:
                    break

            refill_attempts = 0
            refill_cap = max(200, self.population_size * 20)
            while len(next_population) < self.population_size and refill_attempts < refill_cap:
                refill_attempts += 1
                immigrant_gelu, immigrant_softmax = self._mutate(
                    self.context.base_gelu,
                    self.context.base_softmax,
                )
                _add_next(immigrant_gelu, immigrant_softmax)

            population = next_population[: self.population_size]

        payload = {
            "status": "ok",
            "algorithm": "genetic_coinn_style_stage1",
            "best_config": {
                key: value
                for key, value in best_candidate.items()
            },
            "best_generation": int(best_generation),
            "population_size": int(self.population_size),
            "max_generations": int(self.max_generations),
            "history": history,
            "context": self.context,
            "cache_size": int(len(self._cache)),
            "result_path": self.result_path,
            "plot_path": self.plot_path,
            "log_path": self.log_path,
        }
        _json_dump(self.result_path, _to_serializable(payload))
        _series_plot(
            self.plot_path,
            "Stage-1 Genetic Search",
            history,
            score_key="best_score",
            cost_key="best_cost",
        )
        self._log(
            f"Stage-1 genetic search finished. "
            f"Best cost={best_candidate['cost']:.2f}, score={best_candidate['score']:.6f}, "
            f"saved to {self.result_path}"
        )
        return payload


class Stage2NoiseGeneticSearcher:
    def __init__(
        self,
        evaluator,
        fixed_gelu,
        fixed_softmax,
        fixed_label,
        fixed_source,
        random_seed=42,
    ):
        self.evaluator = evaluator
        self.fixed_gelu = np.asarray(fixed_gelu, dtype=int).copy()
        self.fixed_softmax = np.asarray(fixed_softmax, dtype=int).copy()
        self.fixed_label = str(fixed_label)
        self.fixed_source = str(fixed_source)
        self.rng = np.random.default_rng(int(random_seed))
        self.population_size = max(STAGE2_DEFAULT_POPULATION, evaluator.total_layers)
        self.max_generations = max(1, math.ceil(evaluator.stage2_rl_episodes / self.population_size))
        stage2_dir = os.path.dirname(evaluator.noise_step_info_file)
        self.log_path = os.path.join(stage2_dir, "noise_ga_search_log.txt")
        self.result_path = os.path.join(stage2_dir, "noise_ga_search_results.json")
        self.plot_path = os.path.join(stage2_dir, "noise_ga_search_curve.png")
        self.context: Optional[Stage2Context] = None
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
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
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

        metric_penalties = [
            _penalty_lower_is_better(
                stats["loss_mean"],
                self.context.baseline_reference_stats["loss_mean"],
                self.context.search_limits["loss"],
            ),
            _penalty_higher_is_better(
                stats["p_mean"],
                self.context.baseline_reference_stats["p_mean"],
                self.context.search_limits["metric1"],
            ),
        ]
        if self.evaluator.get_num_metrics() > 1:
            metric_penalties.append(
                _penalty_higher_is_better(
                    stats["s_mean"],
                    self.context.baseline_reference_stats["s_mean"],
                    self.context.search_limits["metric2"],
                )
            )

        stability_penalties = [
            _penalty_std_upper_bound(
                stats["loss_std"],
                self.context.baseline_reference_stats["loss_std"],
                self.context.dynamic_loss_std_cap,
            ),
            _penalty_std_upper_bound(
                stats["p_std"],
                self.context.baseline_reference_stats["p_std"],
                self.context.dynamic_m1_std_cap,
            ),
        ]
        if self.evaluator.get_num_metrics() > 1:
            stability_penalties.append(
                _penalty_std_upper_bound(
                    stats["s_std"],
                    self.context.baseline_reference_stats["s_std"],
                    self.context.dynamic_m2_std_cap,
                )
            )

        metric_penalty = float(np.mean(metric_penalties))
        stability_penalty = float(np.mean(stability_penalties))
        score = float(
            (self.context.cost_reference_tot_c / max(float(cost), EPS))
            / max(metric_penalty * stability_penalty, EPS)
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
            **_clone_noise_config(eval_noise_cfg),
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
            "metric_penalty": float(metric_penalty),
            "stability_penalty": float(stability_penalty),
            "search_score_mean": float(score),
            "score": float(score),
            "constraint_pass": constraint_pass,
            "std_pass": std_pass,
            "qualification_passed": bool(constraint_pass and std_pass),
            "stats": {
                key: value
                for key, value in stats.items()
            },
        }
        self._cache[cache_key] = result
        return result

    def _mutate(self, noise_cfg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        mutated = _clone_noise_config(noise_cfg)
        mutation_count = int(
            self.rng.integers(
                1,
                max(3, self.evaluator.total_layers // 2) + 1,
            )
        )
        for _ in range(mutation_count):
            key = NOISE_SCALING_FACTOR_KEYS[int(self.rng.integers(len(NOISE_SCALING_FACTOR_KEYS)))]
            arr = mutated[key]
            layer_idx = int(self.rng.integers(self.evaluator.total_layers))
            allowed = self._allowed_values[key]
            current = int(arr[layer_idx])
            current_idx = allowed.index(current)
            adjacent = []
            if current_idx > 0:
                adjacent.append(allowed[current_idx - 1])
            if current_idx + 1 < len(allowed):
                adjacent.append(allowed[current_idx + 1])
            if not adjacent:
                continue
            arr[layer_idx] = int(adjacent[int(self.rng.integers(len(adjacent)))])
        return mutated

    def _make_initial_population(self):
        population: List[Dict[str, np.ndarray]] = []
        seen = set()

        def _add_candidate(noise_cfg):
            signature = _noise_signature(noise_cfg)
            if signature in seen:
                return
            population.append(_clone_noise_config(noise_cfg))
            seen.add(signature)

        _add_candidate(self.context.baseline_low_risk_config)

        attempts = 0
        max_attempts = max(200, self.population_size * 40)
        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1
            candidate = _clone_noise_config(self.context.baseline_low_risk_config)
            rounds = int(
                self.rng.integers(
                    1,
                    max(2, self.evaluator.total_layers // 3) + 1,
                )
            )
            for _ in range(rounds):
                candidate = self._mutate(candidate)
            _add_candidate(candidate)

        return population

    def run(self) -> Dict[str, object]:
        self._cache.clear()
        with open(self.log_path, "w", encoding="utf-8") as handle:
            handle.write("")

        self.context = build_stage2_context(
            self.evaluator,
            self.fixed_gelu,
            self.fixed_softmax,
            log_fn=self._log,
        )
        self._log("")
        self._log(
            "Stage-2 Noise Genetic Search follows COINN: fitness-weighted selection plus "
            "adjacent scaling-factor mutation, while keeping the current dynamic constraints "
            "and challenger-confirmation protocol."
        )

        incumbent = self._evaluate(
            self.context.baseline_low_risk_config,
            segments=self.context.best_test_segments,
        )
        incumbent_history = [
            {
                "source": "initial_incumbent",
                "generation": 0,
                "mean_score": float(incumbent["score"]),
                "cost": float(incumbent["cost"]),
            }
        ]
        population = self._make_initial_population()
        history = []
        best_generation = 0
        stagnation = 0

        self._log(
            f"Initial population size={len(population)}  "
            f"max_generations={self.max_generations}  "
            f"initial_incumbent_cost={incumbent['cost']:.2f}  "
            f"initial_incumbent_score={incumbent['score']:.6f}"
        )

        for generation in range(1, self.max_generations + 1):
            results = [self._evaluate(candidate) for candidate in population]
            scores = np.asarray([max(float(item["score"]), EPS) for item in results], dtype=float)
            mean_score = float(np.mean(scores)) if len(scores) else 0.0
            raw_best = max(results, key=lambda item: item["score"])
            qualified_ratio = (
                float(sum(1 for item in results if item["qualification_passed"])) / float(len(results))
                if results
                else 0.0
            )

            improved = False
            ranked_candidates = sorted(results, key=lambda item: item["score"], reverse=True)
            for candidate in ranked_candidates[: min(4, len(ranked_candidates))]:
                if _noise_signature(candidate) == _noise_signature(incumbent):
                    continue
                if candidate["score"] <= incumbent["score"] + float(NOISE_STAGE_BEST_TEST_TRIGGER_MARGIN):
                    break
                confirm_candidate = self._evaluate(
                    candidate,
                    segments=self.context.confirm_segments,
                )
                self._log(
                    f"[Stage2][Gen {generation:04d}] challenger confirm "
                    f"score(train)={candidate['score']:.6f}  "
                    f"score(confirm)={confirm_candidate['score']:.6f}  "
                    f"constraint={confirm_candidate['constraint_pass']}  "
                    f"std={confirm_candidate['std_pass']}"
                )
                if confirm_candidate["qualification_passed"] and (
                    confirm_candidate["score"] > incumbent["score"] + 1e-12
                ):
                    incumbent = confirm_candidate
                    incumbent_history.append(
                        {
                            "source": "confirmed_challenger",
                            "generation": generation,
                            "mean_score": float(confirm_candidate["score"]),
                            "cost": float(confirm_candidate["cost"]),
                        }
                    )
                    best_generation = generation
                    stagnation = 0
                    improved = True
                    break

            if not improved:
                stagnation += 1

            history.append(
                {
                    "generation": generation,
                    "best_score": float(incumbent["score"]),
                    "best_cost": float(incumbent["cost"]),
                    "generation_best_score": float(raw_best["score"]),
                    "generation_best_cost": float(raw_best["cost"]),
                    "mean_score": mean_score,
                    "qualified_ratio": qualified_ratio,
                    "improved": improved,
                }
            )
            self._log(
                f"[Stage2][Gen {generation:04d}] "
                f"gen_best_score={raw_best['score']:.6f}  "
                f"gen_best_cost={raw_best['cost']:.2f}  "
                f"incumbent_score={incumbent['score']:.6f}  "
                f"incumbent_cost={incumbent['cost']:.2f}  "
                f"qualified_ratio={qualified_ratio:.2%}  "
                f"stagnation={stagnation}/{STAGNATION_TOLERANCE}"
            )
            if stagnation > STAGNATION_TOLERANCE:
                self._log(
                    f"Early stop: confirmed incumbent has not improved for more than "
                    f"{STAGNATION_TOLERANCE} generations."
                )
                break

            next_population: List[Dict[str, np.ndarray]] = []
            next_seen = set()

            def _add_next(noise_cfg):
                signature = _noise_signature(noise_cfg)
                if signature in next_seen:
                    return
                next_population.append(_clone_noise_config(noise_cfg))
                next_seen.add(signature)

            _add_next(incumbent)
            elite_count = min(max(2, self.population_size // 6), len(ranked_candidates))
            for candidate in ranked_candidates[:elite_count]:
                _add_next(candidate)

            parent_indices = _weighted_choice_indices(
                self.rng,
                scores,
                max(self.population_size * 3, 1),
            )
            for parent_idx in parent_indices:
                parent = results[int(parent_idx)]
                child = self._mutate(parent)
                _add_next(child)
                if len(next_population) >= self.population_size:
                    break

            refill_attempts = 0
            refill_cap = max(200, self.population_size * 20)
            while len(next_population) < self.population_size and refill_attempts < refill_cap:
                refill_attempts += 1
                immigrant = self._mutate(self.context.baseline_low_risk_config)
                _add_next(immigrant)

            population = next_population[: self.population_size]

        selected_config = incumbent if incumbent.get("qualification_passed", False) else None
        status = (
            NOISE_STAGE_STATUS_OK
            if selected_config is not None
            else NOISE_STAGE_STATUS_NO_STABLE_FEASIBLE
        )
        payload = {
            "algorithm": "genetic_coinn_style_stage2",
            "fixed_gelu": self.fixed_gelu.copy(),
            "fixed_softmax": self.fixed_softmax.copy(),
            "fixed_label": self.fixed_label,
            "fixed_source": self.fixed_source,
            "baseline_noise_config": _clone_noise_config(self.context.cost_reference_noise_config),
            "baseline_tot_c": float(self.context.cost_reference_tot_c),
            "cost_reference_noise_config": _clone_noise_config(self.context.cost_reference_noise_config),
            "cost_reference_source": "max_noise_configuration",
            "performance_baseline_gelu": self.fixed_gelu.copy(),
            "performance_baseline_softmax": self.fixed_softmax.copy(),
            "performance_baseline_source": "stage1_fixed_low_risk_noise",
            "baseline_segments": int(self.context.baseline_segments),
            "search_baseline_stats": {
                key: value for key, value in self.context.baseline_reference_stats.items()
            },
            "worst_reference_stats": {
                key: value for key, value in self.context.worst_reference_stats.items()
            },
            "worst_case_noise_config": _clone_noise_config(self.context.worst_case_noise_config),
            "limit_computation_method": "dynamic_quartile",
            "limit_quartile": float(NOISE_STAGE_DYNAMIC_LIMIT_QUARTILE),
            "search_limits": {k: float(v) for k, v in self.context.search_limits.items()},
            "status": status,
            "best_noise_config": (
                {key: value for key, value in selected_config.items()}
                if selected_config is not None
                else None
            ),
            "stable_search_best_noise_config": (
                {key: value for key, value in selected_config.items()}
                if selected_config is not None
                else None
            ),
            "stable_joint_best_noise_config": (
                {key: value for key, value in selected_config.items()}
                if selected_config is not None
                else None
            ),
            "selection_diagnostics": {
                "selection_mode": "genetic_incumbent_confirmation",
                "mc_train_samples": int(self.context.train_segments),
                "mc_confirm_segments": int(self.context.confirm_segments),
                "trigger_margin": float(NOISE_STAGE_BEST_TEST_TRIGGER_MARGIN),
                "dynamic_loss_std_cap": float(self.context.dynamic_loss_std_cap),
                "dynamic_m1_std_cap": float(self.context.dynamic_m1_std_cap),
                "dynamic_m2_std_cap": float(self.context.dynamic_m2_std_cap),
                "incumbent_history": [dict(item) for item in incumbent_history],
                "final_incumbent": (
                    {key: value for key, value in incumbent.items()}
                    if incumbent is not None
                    else None
                ),
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
                "population_size": int(self.population_size),
                "max_generations": int(self.max_generations),
                "stagnation_tolerance": int(STAGNATION_TOLERANCE),
                "train_segments": int(self.context.train_segments),
                "confirm_segments": int(self.context.confirm_segments),
                "best_test_segments": int(self.context.best_test_segments),
            },
            "reward_diagnostics": {
                "terminal_reward_mode": "coinn_style_score",
                "mean_generation_score": float(np.mean([item["mean_score"] for item in history]))
                if history
                else None,
                "best_confirmed_score": float(incumbent["score"]) if incumbent is not None else None,
            },
            "ga_history": history,
            "result_path": self.result_path,
            "plot_path": self.plot_path,
            "log_path": self.log_path,
            "best_generation": int(best_generation),
        }
        _json_dump(self.result_path, _to_serializable(payload))
        _series_plot(
            self.plot_path,
            "Stage-2 Noise Genetic Search",
            history,
            score_key="best_score",
            cost_key="best_cost",
        )
        self._log(
            f"Stage-2 noise genetic search finished. "
            f"Status={status}, incumbent_cost={incumbent['cost']:.2f}, "
            f"incumbent_score={incumbent['score']:.6f}, saved to {self.result_path}"
        )
        return payload


def run_stage1_genetic_search(evaluator, random_seed=42):
    return Stage1GeneticSearcher(evaluator, random_seed=random_seed).run()


def run_stage2_noise_genetic_search(
    evaluator,
    fixed_gelu,
    fixed_softmax,
    fixed_label,
    fixed_source,
    random_seed=42,
):
    return Stage2NoiseGeneticSearcher(
        evaluator=evaluator,
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        fixed_label=fixed_label,
        fixed_source=fixed_source,
        random_seed=random_seed,
    ).run()
