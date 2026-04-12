import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from final_evaluation_module import FinalEvaluationModule
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
    NOISE_STAGE_STOP_FLAG_FILENAME,
    _log_rounded_box,
    _auto_adjust_segments,
    _compute_dynamic_limits,
    _compute_dynamic_std_upper_bound,
    _get_low_risk_noise_configuration,
    _get_worst_case_noise_configuration,
    install_graceful_stop_handler,
    uninstall_graceful_stop_handler,
    reset_graceful_stop_state,
    is_graceful_stop_requested,
    consume_stop_flag_file,
)

import torch

import time as _time

EPS = 1e-8
SCORE_EXP_BASE = 4.0
MAX_PENALTY_EXP_ARG = 700.0
STAGE1_DEFAULT_POPULATION = 32
STAGE2_DEFAULT_POPULATION = 32
STAGNATION_TOLERANCE_BASE = 10
GA_PROGRESS_BOX_INTERVAL = 5
STAGE1_GA_CONSTRAINT_RATIO = 0.005
GA_STAGE1_ALLOWED_GELU_DEGREES = (4, 2, 1)


def _progress_bar(current, total, width=30):
    """Render a fixed-width unicode progress bar."""
    ratio = min(current / max(total, 1), 1.0)
    filled = int(round(ratio * width))
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"[{bar}] {ratio:6.1%}"


def _fmt_elapsed(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _log_ga_stage_header(log_fn, stage_label, pop_size, max_gen, stag_tol):
    """Log a visually distinct stage start header."""
    log_fn("")
    log_fn("╔══════════════════════════════════════════════════════════════════╗")
    log_fn(f"║  {stage_label:^64s}║")
    log_fn("╠══════════════════════════════════════════════════════════════════╣")
    log_fn(f"║  种群规模（Population）    : {pop_size:<38d}║")
    log_fn(f"║  最大代数（Max Generations）: {max_gen:<38d}║")
    log_fn(f"║  停滞容忍（Stagnation Tol）: {stag_tol:<38d}║")
    log_fn("╚══════════════════════════════════════════════════════════════════╝")
    log_fn("")


def _log_ga_finish_summary(log_fn, stage_label, status, gen_count, max_gen,
                            best_score, best_cost, elapsed, extra_lines=None):
    """Log a visually distinct completion summary."""
    lines = [
        f"{stage_label} — 搜索完成（Search Completed）",
        f"退出原因: {status}",
        f"已搜索代数: {gen_count} / {max_gen}",
        f"最优得分（Best Score）: {best_score:.6f}",
        f"最优成本（Best Cost） : {best_cost:.2f}",
        f"总耗时: {_fmt_elapsed(elapsed)}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    _log_rounded_box(log_fn, lines, indent="")

GA_STAGE1_CHECKPOINT_FILENAME = "ga_stage1_checkpoint.pt"
GA_STAGE2_CHECKPOINT_FILENAME = "ga_stage2_checkpoint.pt"


def _save_ga_checkpoint(path, data):
    """Save GA checkpoint using torch.save for consistency with RL checkpoints."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(data, path)


def _load_ga_checkpoint(path):
    """Load GA checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


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


def _resolve_ga_reward_reference_split(evaluator) -> str:
    if evaluator.has_dataset_split("validation_full"):
        return "validation_full"
    return evaluator.get_reward_reference_split_name()


def _resolve_ga_generation_budget(
    evaluator,
    *,
    attr_name: str,
    fallback_episode_budget: int,
    population_size: int,
) -> int:
    explicit_generations = getattr(evaluator, attr_name, None)
    if explicit_generations is not None:
        try:
            explicit_generations = int(explicit_generations)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{attr_name} must be a positive integer, got {explicit_generations!r}."
            ) from exc
        if explicit_generations <= 0:
            raise ValueError(
                f"{attr_name} must be a positive integer, got {explicit_generations!r}."
            )
        return explicit_generations
    return max(1, math.ceil(int(fallback_episode_budget) / max(int(population_size), 1)))


def build_stage1_context(evaluator, log_fn=None, include_distribution=True) -> Stage1Context:
    base_gelu = np.full(evaluator.total_layers, 4, dtype=int)
    base_softmax = np.full(evaluator.total_layers, 6, dtype=int)
    base_tot_c, base_g_c, base_s_c = evaluator.get_simulated_cost(base_gelu, base_softmax)

    reward_reference_split = _resolve_ga_reward_reference_split(evaluator)
    base_loss, base_p, base_s, _ = evaluator.stage1_evaluate(
        base_gelu,
        base_softmax,
        split=reward_reference_split,
    )

    limits = {
        "loss": float(base_loss * (1.0 + STAGE1_GA_CONSTRAINT_RATIO)),
        "metric1": float(base_p * (1.0 - STAGE1_GA_CONSTRAINT_RATIO)),
        "metric2": float(base_s * (1.0 - STAGE1_GA_CONSTRAINT_RATIO)),
    }

    gelu_degree0_eligible = np.zeros(evaluator.total_layers, dtype=bool)
    if include_distribution and log_fn is not None:
        log_fn("")
        log_fn(
            "Stage-1 GELU degree 0 is globally disabled for GA; "
            "distribution eligibility screening is skipped."
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
        log_fn(f"  constraint ratio: {STAGE1_GA_CONSTRAINT_RATIO:.2%}")
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

    reward_reference_split = _resolve_ga_reward_reference_split(evaluator)

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
        split=_resolve_ga_reward_reference_split(evaluator),
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
    _ensure_parent_dir(path)
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
    _ensure_parent_dir(path)
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
    return linear + _safe_exp_penalty(gap)


def _penalty_higher_is_better(value: float, reference: float, limit: float) -> float:
    if value <= EPS:
        return 1e6
    linear = max(float(reference) / max(float(value), EPS), EPS)
    if value >= limit:
        return linear
    gap = (float(limit) - float(value)) / max(abs(float(reference) - float(limit)), EPS)
    return linear + _safe_exp_penalty(gap)


def _penalty_std_upper_bound(value: float, reference: float, cap: float) -> float:
    reference = max(abs(reference), EPS)
    linear = max(float(value) / reference, 1.0)
    if value <= cap:
        return linear
    gap = (float(value) - float(cap)) / max(abs(float(cap) - float(reference)), EPS)
    return linear + _safe_exp_penalty(gap)


def _safe_exp_penalty(gap: float) -> float:
    if not np.isfinite(gap):
        return math.exp(MAX_PENALTY_EXP_ARG)
    positive_gap = max(float(gap), 0.0)
    exp_arg = min(positive_gap * math.log(SCORE_EXP_BASE), MAX_PENALTY_EXP_ARG)
    return math.exp(exp_arg)


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
    def __init__(self, evaluator, random_seed=42, resume_checkpoint_path=None):
        self.evaluator = evaluator
        self.rng = np.random.default_rng(int(random_seed))
        self.population_size = max(STAGE1_DEFAULT_POPULATION, evaluator.total_layers * 2)
        self.max_generations = _resolve_ga_generation_budget(
            evaluator,
            attr_name="stage1_ga_generations",
            fallback_episode_budget=evaluator.stage1_rl_episodes,
            population_size=self.population_size,
        )
        stage1_dir = os.path.dirname(evaluator.step_info_file)
        self.result_path = os.path.join(stage1_dir, "ga_search_results.json")
        self.plot_path = os.path.join(stage1_dir, "ga_search_curve.png")
        self.checkpoint_path = os.path.join(stage1_dir, GA_STAGE1_CHECKPOINT_FILENAME)
        self.stop_flag_path = os.path.join(stage1_dir, NOISE_STAGE_STOP_FLAG_FILENAME)
        self.resume_checkpoint_path = resume_checkpoint_path
        self._cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Dict[str, object]] = {}
        self._valid_states: List[List[Tuple[int, int]]] = []
        self._neighbor_map: List[Dict[Tuple[int, int], List[Tuple[int, int]]]] = []
        self.context: Optional[Stage1Context] = None

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

    def _allowed_gelu_degrees(self, layer_idx: int) -> Tuple[int, ...]:
        del layer_idx
        return GA_STAGE1_ALLOWED_GELU_DEGREES

    @staticmethod
    def _softmax_degrees() -> Tuple[int, ...]:
        return (6, 5, 4, 3, 2)

    def _candidate_key(self, gelu, softmax):
        gelu_arr, softmax_arr, _ = self._sanitize_stage1_candidate(gelu, softmax)
        return tuple(gelu_arr.tolist()), tuple(softmax_arr.tolist())

    def _evaluate(self, gelu, softmax) -> Dict[str, object]:
        key = self._candidate_key(gelu, softmax)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        gelu_arr, softmax_arr, had_degree0 = self._sanitize_stage1_candidate(
            gelu,
            softmax,
        )
        if had_degree0:
            self._log(
                "[Stage1] Detected GELU degree 0 in a candidate; "
                "it was normalized to degree 1 because degree 0 is disabled."
            )
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

    def _save_checkpoint(self, generation, best_candidate, best_generation, stagnation,
                         history, population):
        """Save GA Stage-1 checkpoint at current generation boundary."""
        pop_serialized = [
            (g.tolist(), s.tolist()) for g, s in population
        ]
        data = {
            "version": 1,
            "algorithm": "genetic_coinn_style_stage1",
            "completed_generation": generation,
            "best_candidate": _to_serializable(best_candidate),
            "best_generation": best_generation,
            "stagnation": stagnation,
            "history": history,
            "population": pop_serialized,
            "rng_state": self.rng.bit_generator.state,
        }
        _save_ga_checkpoint(self.checkpoint_path, data)
        self._log(f"[Stage1] Checkpoint saved at generation {generation} → {self.checkpoint_path}")

    def run(self) -> Dict[str, object]:
        self._cache.clear()

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

        # Dynamic stagnation tolerance: ensure at least 1/3 of budget is used
        # before early stop. COINN does not use aggressive early stopping.
        effective_stagnation_tolerance = max(
            STAGNATION_TOLERANCE_BASE,
            self.max_generations // 3,
        )
        # Diversity injection threshold: inject random immigrants at halfway
        diversity_injection_threshold = max(
            STAGNATION_TOLERANCE_BASE // 2,
            effective_stagnation_tolerance // 2,
        )

        _log_ga_stage_header(
            self._log,
            "Stage-1 遗传算法搜索（Genetic Algorithm — GELU/Softmax）",
            len(population),
            self.max_generations,
            effective_stagnation_tolerance,
        )
        self._log(
            f"  多样性注入阈值（Diversity Injection）: {diversity_injection_threshold}  "
            f"种群规模（Population）: {len(population)}"
        )

        best_candidate = self._evaluate(
            self.context.base_gelu,
            self.context.base_softmax,
        )
        best_generation = 0
        stagnation = 0
        history = []
        resume_start_generation = 1

        # ---- Resume from checkpoint ----
        if self.resume_checkpoint_path and os.path.isfile(self.resume_checkpoint_path):
            ckpt = _load_ga_checkpoint(self.resume_checkpoint_path)
            resume_start_generation = int(ckpt["completed_generation"]) + 1
            best_candidate = ckpt["best_candidate"]
            # Restore numpy arrays in best_candidate
            for k in ("gelu", "softmax"):
                if k in best_candidate and not isinstance(best_candidate[k], np.ndarray):
                    best_candidate[k] = np.asarray(best_candidate[k], dtype=int)
            best_candidate = self._evaluate(
                best_candidate["gelu"],
                best_candidate["softmax"],
            )
            best_generation = int(ckpt["best_generation"])
            stagnation = int(ckpt["stagnation"])
            history = list(ckpt["history"])
            pop_raw = ckpt["population"]
            population = [
                self._sanitize_stage1_candidate(g, s)[:2]
                for g, s in pop_raw
            ]
            if "rng_state" in ckpt:
                self.rng.bit_generator.state = ckpt["rng_state"]
            self._log(
                f"[Stage1] Resumed from checkpoint: generation={resume_start_generation - 1}, "
                f"best_score={best_candidate['score']:.6f}, stagnation={stagnation}"
            )

        # ---- Install graceful stop handler ----
        reset_graceful_stop_state()
        consume_stop_flag_file(self.stop_flag_path)
        install_graceful_stop_handler(log_fn=self._log)

        _t0 = _time.time()
        graceful_stopped = False
        for generation in range(resume_start_generation, self.max_generations + 1):
            _gen_t0 = _time.time()
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
            _gen_elapsed = _time.time() - _gen_t0
            _star = " ★" if improved else ""
            self._log(
                f"[Stage1][Gen {generation:04d}/{self.max_generations}] "
                f"本代最优={raw_best['score']:.6f}(cost={raw_best['cost']:.2f})  "
                f"全局最优={best_candidate['score']:.6f}(cost={best_candidate['cost']:.2f})  "
                f"可行比={feasible_ratio:.0%}  "
                f"停滞={stagnation}/{effective_stagnation_tolerance}  "
                f"耗时={_gen_elapsed:.1f}s{_star}"
            )

            if generation % GA_PROGRESS_BOX_INTERVAL == 0 or generation == self.max_generations:
                _elapsed_total = _time.time() - _t0
                _avg_gen_time = _elapsed_total / max(generation - resume_start_generation + 1, 1)
                _remaining_gen = self.max_generations - generation
                _eta = _avg_gen_time * _remaining_gen
                _log_rounded_box(
                    self._log,
                    [
                        f"Stage-1 GA 进度 · 第 {generation} / {self.max_generations} 代",
                        _progress_bar(generation, self.max_generations),
                        f"全局最优得分: {best_candidate['score']:.6f}  "
                        f"最优成本: {best_candidate['cost']:.2f}  "
                        f"(第 {best_generation} 代发现)",
                        f"已用时: {_fmt_elapsed(_elapsed_total)}  "
                        f"预计剩余: {_fmt_elapsed(_eta)}  "
                        f"缓存命中: {len(self._cache)} 条",
                    ],
                    indent="  ",
                )

            if stagnation > effective_stagnation_tolerance:
                self._log(
                    f"Early stop: best feasible score has not improved for more than "
                    f"{effective_stagnation_tolerance} generations."
                )
                break

            # ---- Graceful stop check ----
            if is_graceful_stop_requested(self.stop_flag_path):
                self._save_checkpoint(generation, best_candidate, best_generation,
                                      stagnation, history, population)
                consume_stop_flag_file(self.stop_flag_path)
                uninstall_graceful_stop_handler()
                self._log(f"[Stage1] Graceful stop at generation {generation}. "
                          f"Resume with --resume-from to continue.")
                graceful_stopped = True
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

            # Diversity injection: when stagnating, replace bottom portion of
            # population with fresh random immigrants to escape local optima.
            # This follows the COINN principle of maintaining population
            # diversity through random immigrant injection.
            inject_diversity = (
                stagnation > 0
                and stagnation % diversity_injection_threshold == 0
            )

            if inject_diversity:
                # Keep only top elite, fill rest with random immigrants
                elite_count = min(max(2, self.population_size // 4), len(results))
                for idx in ranked_indices[:elite_count]:
                    _add_next(results[idx]["gelu"], results[idx]["softmax"])
                inject_target = self.population_size - len(next_population)
                inject_attempts = 0
                inject_cap = max(200, inject_target * 30)
                while len(next_population) < self.population_size and inject_attempts < inject_cap:
                    inject_attempts += 1
                    immigrant_gelu = self.context.base_gelu.copy()
                    immigrant_softmax = self.context.base_softmax.copy()
                    rounds = int(
                        self.rng.integers(
                            1,
                            max(2, math.ceil(self.evaluator.total_layers / 2)) + 1,
                        )
                    )
                    for _ in range(rounds):
                        immigrant_gelu, immigrant_softmax = self._mutate(
                            immigrant_gelu, immigrant_softmax,
                        )
                    _add_next(immigrant_gelu, immigrant_softmax)
                self._log(
                    f"[Stage1][Gen {generation:04d}] diversity injection: "
                    f"injected {len(next_population) - elite_count - 1} random immigrants"
                )
            else:
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

        uninstall_graceful_stop_handler()

        if graceful_stopped:
            raise SystemExit(0)

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
            "log_path": self.evaluator.log_file,
        }
        _json_dump(self.result_path, _to_serializable(payload))
        _series_plot(
            self.plot_path,
            "Stage-1 Genetic Search",
            history,
            score_key="best_score",
            cost_key="best_cost",
        )
        _total_elapsed = _time.time() - _t0
        _exit_reason = "达到最大代数" if not history or not history[-1].get("improved", False) and stagnation <= effective_stagnation_tolerance else (
            "提前停止（停滞收敛）" if stagnation > effective_stagnation_tolerance else "正常完成"
        )
        _log_ga_finish_summary(
            self._log,
            "Stage-1 GA（GELU/Softmax）",
            _exit_reason,
            len(history),
            self.max_generations,
            best_candidate["score"],
            best_candidate["cost"],
            _total_elapsed,
            extra_lines=[
                f"最优配置发现于第 {best_generation} 代",
                f"GELU : {best_candidate.get('gelu', 'N/A')}",
                f"Softmax: {best_candidate.get('softmax', 'N/A')}",
                f"结果已保存 → {self.result_path}",
            ],
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
        resume_checkpoint_path=None,
    ):
        self.evaluator = evaluator
        self.fixed_gelu = np.asarray(fixed_gelu, dtype=int).copy()
        self.fixed_softmax = np.asarray(fixed_softmax, dtype=int).copy()
        self.fixed_label = str(fixed_label)
        self.fixed_source = str(fixed_source)
        self.rng = np.random.default_rng(int(random_seed))
        self.population_size = max(STAGE2_DEFAULT_POPULATION, evaluator.total_layers)
        self.max_generations = _resolve_ga_generation_budget(
            evaluator,
            attr_name="stage2_ga_generations",
            fallback_episode_budget=evaluator.stage2_rl_episodes,
            population_size=self.population_size,
        )
        stage2_dir = os.path.dirname(evaluator.noise_step_info_file)
        self.result_path = os.path.join(stage2_dir, "noise_ga_search_results.json")
        self.plot_path = os.path.join(stage2_dir, "noise_ga_search_curve.png")
        self.checkpoint_path = os.path.join(stage2_dir, GA_STAGE2_CHECKPOINT_FILENAME)
        self.stop_flag_path = os.path.join(stage2_dir, NOISE_STAGE_STOP_FLAG_FILENAME)
        self.resume_checkpoint_path = resume_checkpoint_path
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

    def _save_checkpoint(self, generation, incumbent, incumbent_history,
                         best_generation, stagnation, history, population):
        """Save GA Stage-2 checkpoint at current generation boundary."""
        pop_serialized = [_to_serializable(cfg) for cfg in population]
        data = {
            "version": 1,
            "algorithm": "genetic_coinn_style_stage2",
            "completed_generation": generation,
            "incumbent": _to_serializable(incumbent),
            "incumbent_history": [dict(item) for item in incumbent_history],
            "best_generation": best_generation,
            "stagnation": stagnation,
            "history": history,
            "population": pop_serialized,
            "rng_state": self.rng.bit_generator.state,
        }
        _save_ga_checkpoint(self.checkpoint_path, data)
        self._log(f"[Stage2] Checkpoint saved at generation {generation} → {self.checkpoint_path}")

    def run(self) -> Dict[str, object]:
        self._cache.clear()

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
        resume_start_generation = 1

        # Dynamic stagnation tolerance: ensure at least 1/3 of budget is used
        # before early stop. COINN does not use aggressive early stopping.
        effective_stagnation_tolerance = max(
            STAGNATION_TOLERANCE_BASE,
            self.max_generations // 3,
        )
        diversity_injection_threshold = max(
            STAGNATION_TOLERANCE_BASE // 2,
            effective_stagnation_tolerance // 2,
        )

        _log_ga_stage_header(
            self._log,
            "Stage-2 遗传算法搜索（Genetic Algorithm — Noise Scaling）",
            len(population),
            self.max_generations,
            effective_stagnation_tolerance,
        )
        self._log(
            f"  多样性注入阈值: {diversity_injection_threshold}  "
            f"初始incumbent: score={incumbent['score']:.6f}, cost={incumbent['cost']:.2f}"
        )

        # ---- Resume from checkpoint ----
        if self.resume_checkpoint_path and os.path.isfile(self.resume_checkpoint_path):
            ckpt = _load_ga_checkpoint(self.resume_checkpoint_path)
            resume_start_generation = int(ckpt["completed_generation"]) + 1
            incumbent = ckpt["incumbent"]
            # Restore numpy arrays in noise config keys
            for key in NOISE_SCALING_FACTOR_KEYS:
                if key in incumbent and not isinstance(incumbent[key], np.ndarray):
                    incumbent[key] = np.asarray(incumbent[key])
            incumbent_history = list(ckpt["incumbent_history"])
            best_generation = int(ckpt["best_generation"])
            stagnation = int(ckpt["stagnation"])
            history = list(ckpt["history"])
            pop_raw = ckpt["population"]
            population = []
            for cfg in pop_raw:
                restored = {}
                for k, v in cfg.items():
                    if k in NOISE_SCALING_FACTOR_KEYS and not isinstance(v, np.ndarray):
                        restored[k] = np.asarray(v)
                    else:
                        restored[k] = v
                population.append(restored)
            if "rng_state" in ckpt:
                self.rng.bit_generator.state = ckpt["rng_state"]
            self._log(
                f"[Stage2] Resumed from checkpoint: generation={resume_start_generation - 1}, "
                f"incumbent_score={incumbent['score']:.6f}, stagnation={stagnation}"
            )

        # ---- Install graceful stop handler ----
        reset_graceful_stop_state()
        consume_stop_flag_file(self.stop_flag_path)
        install_graceful_stop_handler(log_fn=self._log)

        _t0 = _time.time()
        graceful_stopped = False
        for generation in range(resume_start_generation, self.max_generations + 1):
            _gen_t0 = _time.time()
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
            _gen_elapsed = _time.time() - _gen_t0
            _star = " ★" if improved else ""
            self._log(
                f"[Stage2][Gen {generation:04d}/{self.max_generations}] "
                f"本代最优={raw_best['score']:.6f}(cost={raw_best['cost']:.2f})  "
                f"incumbent={incumbent['score']:.6f}(cost={incumbent['cost']:.2f})  "
                f"合格比={qualified_ratio:.0%}  "
                f"停滞={stagnation}/{effective_stagnation_tolerance}  "
                f"耗时={_gen_elapsed:.1f}s{_star}"
            )

            if generation % GA_PROGRESS_BOX_INTERVAL == 0 or generation == self.max_generations:
                _elapsed_total = _time.time() - _t0
                _avg_gen_time = _elapsed_total / max(generation - resume_start_generation + 1, 1)
                _remaining_gen = self.max_generations - generation
                _eta = _avg_gen_time * _remaining_gen
                _log_rounded_box(
                    self._log,
                    [
                        f"Stage-2 GA 进度 · 第 {generation} / {self.max_generations} 代",
                        _progress_bar(generation, self.max_generations),
                        f"Incumbent 得分: {incumbent['score']:.6f}  "
                        f"成本: {incumbent['cost']:.2f}  "
                        f"(第 {best_generation} 代确认)",
                        f"已用时: {_fmt_elapsed(_elapsed_total)}  "
                        f"预计剩余: {_fmt_elapsed(_eta)}  "
                        f"缓存命中: {len(self._cache)} 条",
                    ],
                    indent="  ",
                )

            if stagnation > effective_stagnation_tolerance:
                self._log(
                    f"Early stop: confirmed incumbent has not improved for more than "
                    f"{effective_stagnation_tolerance} generations."
                )
                break

            # ---- Graceful stop check ----
            if is_graceful_stop_requested(self.stop_flag_path):
                self._save_checkpoint(generation, incumbent, incumbent_history,
                                      best_generation, stagnation, history, population)
                consume_stop_flag_file(self.stop_flag_path)
                uninstall_graceful_stop_handler()
                self._log(f"[Stage2] Graceful stop at generation {generation}. "
                          f"Resume with --resume-from to continue.")
                graceful_stopped = True
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

            # Diversity injection: when stagnating, replace bottom portion of
            # population with fresh random immigrants to escape local optima.
            inject_diversity = (
                stagnation > 0
                and stagnation % diversity_injection_threshold == 0
            )

            if inject_diversity:
                elite_count = min(max(2, self.population_size // 4), len(ranked_candidates))
                for candidate in ranked_candidates[:elite_count]:
                    _add_next(candidate)
                inject_attempts = 0
                inject_cap = max(200, self.population_size * 30)
                while len(next_population) < self.population_size and inject_attempts < inject_cap:
                    inject_attempts += 1
                    immigrant = _clone_noise_config(self.context.baseline_low_risk_config)
                    rounds = int(
                        self.rng.integers(
                            1,
                            max(2, self.evaluator.total_layers // 2) + 1,
                        )
                    )
                    for _ in range(rounds):
                        immigrant = self._mutate(immigrant)
                    _add_next(immigrant)
                self._log(
                    f"[Stage2][Gen {generation:04d}] diversity injection: "
                    f"injected {len(next_population) - elite_count - 1} random immigrants"
                )
            else:
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

        uninstall_graceful_stop_handler()

        if graceful_stopped:
            raise SystemExit(0)

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
                "stagnation_tolerance": int(effective_stagnation_tolerance),
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
            "log_path": self.evaluator.noise_log_file,
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
        _total_elapsed = _time.time() - _t0
        _exit_reason = "达到最大代数" if not history or not history[-1].get("improved", False) and stagnation <= effective_stagnation_tolerance else (
            "提前停止（停滞收敛）" if stagnation > effective_stagnation_tolerance else "正常完成"
        )
        _log_ga_finish_summary(
            self._log,
            "Stage-2 GA（Noise Scaling）",
            f"{_exit_reason} (status={status})",
            len(history),
            self.max_generations,
            incumbent["score"],
            incumbent["cost"],
            _total_elapsed,
            extra_lines=[
                f"最优配置发现于第 {best_generation} 代（确认次数: {len(incumbent_history)}）",
                f"结果已保存 → {self.result_path}",
            ],
        )
        return payload


def run_stage1_genetic_search(evaluator, random_seed=42, resume_checkpoint_path=None):
    return Stage1GeneticSearcher(
        evaluator,
        random_seed=random_seed,
        resume_checkpoint_path=resume_checkpoint_path,
    ).run()


def run_stage2_noise_genetic_search(
    evaluator,
    fixed_gelu,
    fixed_softmax,
    fixed_label,
    fixed_source,
    random_seed=42,
    resume_checkpoint_path=None,
):
    return Stage2NoiseGeneticSearcher(
        evaluator=evaluator,
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        fixed_label=fixed_label,
        fixed_source=fixed_source,
        random_seed=random_seed,
        resume_checkpoint_path=resume_checkpoint_path,
    ).run()
