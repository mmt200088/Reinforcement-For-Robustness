"""Evaluate selected two-stage configurations on full validation data.

The module also constructs the fixed cost-matched and permutation controls used
by the final report. Search state is never updated from these evaluations.
"""

import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from rfr.search.common.eval_metrics import pack_repeat_evaluation
from rfr.preparation.data.protocol import FINAL_EVAL_SPLIT, TRAIN_PROBE_SPLIT
from rfr.common.json_utils import read_json_file, to_jsonable

NOISE_SCALING_FACTOR_KEYS = (
    "input_noise_scaling_factors",
    "wq_noise_scaling_factors",
    "wk_noise_scaling_factors",
    "wv_noise_scaling_factors",
    "wo_noise_scaling_factors",
    "wffn1_noise_scaling_factors",
    "wffn2_noise_scaling_factors",
)

SHORT_KEY_TO_FULL = {
    "x": "input_noise_scaling_factors",
    "wq": "wq_noise_scaling_factors",
    "wk": "wk_noise_scaling_factors",
    "wv": "wv_noise_scaling_factors",
    "wo": "wo_noise_scaling_factors",
    "wffn1": "wffn1_noise_scaling_factors",
    "wffn2": "wffn2_noise_scaling_factors",
}

BREAKDOWN_KEYS = ("x", "wq", "wk", "wv", "wo", "wffn1", "wffn2")
MAX_CONFIG_SOURCES = {"max", "stage2-max", "stage2_max", "blb-max", "blb_max"}
_LIST_OR_TUPLE_TYPES = (list, tuple)
_FAMILY_COLOR_MAP = {
    "Stage1Budget": "#72B7B2",
    "Stage2Budget": "#EECA3B",
    "Perm": "#4C78A8",
    "Equiv": "#F58518",
    "Budget": "#54A24B",
    "Optimized": "#E45756",
    "Random": "#4C78A8",
    "Stage1FixedMaxSF": "#B279A2",
}
_FAMILY_COLOR_ORDER = tuple(_FAMILY_COLOR_MAP)


def _protocol_hashes(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        hashes = set()
        own_hash = value.get("dataset_protocol_hash")
        if own_hash not in (None, ""):
            hashes.add(str(own_hash))
        for nested in value.values():
            hashes.update(_protocol_hashes(nested))
        return hashes
    if isinstance(value, (list, tuple)):
        hashes = set()
        for nested in value:
            hashes.update(_protocol_hashes(nested))
        return hashes
    return set()


def require_final_evaluation_protocol(
    evaluator: Any,
    *,
    search_results: Sequence[Any],
    requested_split: str = FINAL_EVAL_SPLIT,
) -> dict[str, Any]:
    if str(requested_split) != FINAL_EVAL_SPLIT:
        raise RuntimeError(
            f"final evaluation requires {FINAL_EVAL_SPLIT}, got "
            f"{requested_split!r}"
        )

    protocol_hash = str(
        getattr(evaluator, "dataset_protocol_hash", "") or ""
    )
    if not protocol_hash:
        raise RuntimeError("final evaluation dataset protocol hash is missing")
    protocol_path = Path(
        str(getattr(evaluator, "dataset_protocol_path", "") or "")
    )
    if not protocol_path.is_file():
        raise RuntimeError("final evaluation dataset_protocol.json is missing")
    protocol_payload = read_json_file(protocol_path)
    if (
        not isinstance(protocol_payload, Mapping)
        or protocol_payload.get("dataset_protocol_hash") != protocol_hash
        or protocol_payload.get("final_eval_split") != FINAL_EVAL_SPLIT
    ):
        raise RuntimeError("final evaluation persisted protocol hash mismatch")

    provided_results = [result for result in search_results if result is not None]
    if not provided_results:
        raise RuntimeError("final evaluation requires a persisted search result")
    for result in provided_results:
        result_hashes = _protocol_hashes(result)
        if result_hashes != {protocol_hash}:
            raise RuntimeError(
                "final evaluation search-result protocol hash mismatch"
            )

    dataset_splits = getattr(evaluator, "dataset_splits", None)
    dataloaders = getattr(evaluator, "dataloaders", None)
    if not isinstance(dataset_splits, Mapping) or not isinstance(
        dataloaders, Mapping
    ):
        raise RuntimeError("final evaluation dataset registry is unavailable")
    dataset = dataset_splits.get(FINAL_EVAL_SPLIT)
    dataloader = dataloaders.get(FINAL_EVAL_SPLIT)
    if dataset is None or dataloader is None:
        raise RuntimeError(
            f"final evaluation requires the complete {FINAL_EVAL_SPLIT} split"
        )
    if dataset is dataset_splits.get(TRAIN_PROBE_SPLIT):
        raise RuntimeError("final evaluation cannot alias train_probe")
    example_count = len(dataset)
    if example_count <= 0:
        raise RuntimeError("final evaluation validation_full is empty")
    return {
        "split_name": FINAL_EVAL_SPLIT,
        "dataset": dataset,
        "dataloader": dataloader,
        "example_count": int(example_count),
        "dataset_protocol_hash": protocol_hash,
    }


class UnifiedFinalEvaluationModule:
    """统一 final-eval：一次评估覆盖 stage1 + stage2 所有组别。"""

    def __init__(
        self,
        evaluator,
        config_source: str = "search",
        config_path: str = "configs/reference/rl.json",
        manual_stage1_gelu: Optional[Sequence[int]] = None,
        manual_stage1_softmax: Optional[Sequence[int]] = None,
        manual_stage2_noise: Optional[Dict[str, Sequence[int]]] = None,
        random_seed: int = 42,
        permutation_trials: int = 10,
        cost_equivalent_trials: int = 10,
        budget_equivalent_trials: int = 10,
        stage1_budget_trials: int = 10,
        stage2_budget_trials: int = 10,
        repeat_n: int = 1,
        results_dir: Optional[str] = None,
    ):
        self.evaluator = evaluator
        self.config_source = (config_source or "search").lower()
        self.config_path = config_path or "configs/reference/rl.json"
        self.manual_stage1_gelu = manual_stage1_gelu
        self.manual_stage1_softmax = manual_stage1_softmax
        self.manual_stage2_noise = manual_stage2_noise
        self.random_seed = int(random_seed)
        self.final_eval_only = bool(getattr(evaluator, "final_eval_only", False))
        self.random_group_seed = None
        self.permutation_trials = max(0, int(permutation_trials))
        self.cost_equivalent_trials = max(0, int(cost_equivalent_trials))
        self.budget_equivalent_trials = max(0, int(budget_equivalent_trials))
        self.stage1_budget_trials = max(0, int(stage1_budget_trials))
        self.stage2_budget_trials = max(0, int(stage2_budget_trials))
        self.repeat_n = max(1, int(repeat_n))

        default_results_dir = getattr(
            evaluator, "final_eval_dir", os.path.join("rl_results", "final_eval")
        )
        self.results_dir = results_dir or default_results_dir

        self.allowed_gelu_selected = [0, 1, 2, 4]
        self.allowed_gelu_random = [1, 2, 4]
        self.allowed_softmax = [2, 3, 4, 5, 6]

        from rfr.search.runtime.model_handler import (
            INPUT_NOISE_ALLOWED_SCALING_FACTORS,
            WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
            WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
        )

        self.input_noise_allowed = list(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
        self.weight_noise_allowed = list(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wffn1_noise_allowed = list(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)
        self.include_random_groups = any(
            trial_count > 0
            for trial_count in (
                self.permutation_trials,
                self.cost_equivalent_trials,
                self.budget_equivalent_trials,
                self.stage1_budget_trials,
                self.stage2_budget_trials,
            )
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
        self.final_eval_protocol = protocol
        self._ensure_results_dir()
        ev = self.evaluator
        num_metrics = ev.get_num_metrics()
        metric_short_names = ev.get_metric_short_names()
        total_layers = int(ev.total_layers)

        selection_constraints = {
            "loss": float(limit_loss),
            "metric1": float(limit_p),
            "metric2": float(limit_s),
        }


        (
            opt_gelu,
            opt_softmax,
            opt_noise_cfg,
            selected_source,
        ) = self._resolve_selected_config(
            search_best_stage1=search_best_stage1,
            search_best_stage2=search_best_stage2,
            total_layers=total_layers,
        )


        opt_stage1_tot_c, opt_g_c, opt_s_c = ev.get_simulated_cost(opt_gelu, opt_softmax)
        opt_stage2_tot_c, opt_breakdown = ev.get_noise_simulated_cost(**opt_noise_cfg)
        opt_stage2_tot_c = self._stage2_cost_key(opt_stage2_tot_c) / 40.0

        ev.log("\n" + "=" * 60)
        ev.log("PHASE: UNIFIED FINAL EVALUATION (validation_full)")
        ev.log(f"CONFIG_SOURCE={self.config_source}  REPEAT_N={self.repeat_n}")
        ev.log(
            f"Optimized stage1 cost={opt_stage1_tot_c:.2f} "
            f"(gelu={opt_g_c:.2f}, softmax={opt_s_c:.2f})"
        )
        ev.log(
            f"Optimized stage2 cost={opt_stage2_tot_c:.2f} "
            f"breakdown={ {k: float(opt_breakdown[k]) for k in BREAKDOWN_KEYS} }"
        )
        ev.log(
            "Selection constraints: "
            f"Loss<={limit_loss:.4f}, {metric_short_names[0]}>={limit_p:.4f}"
            + (f", {metric_short_names[1]}>={limit_s:.4f}" if num_metrics > 1 else "")
        )
        ev.log("=" * 60)


        baseline_repeat = None
        ev.log("\n--- Baseline (Stage-1 Exact) : single deterministic evaluation ---")
        baseline_single = ev.evaluate_model(
            baseline_stage1_gelu,
            baseline_stage1_softmax,
            use_train=False,
            split=self.final_eval_split,
        )
        report_constraints = ev.build_constraint_limits_from_metrics(
            baseline_single[0],
            baseline_single[1],
            baseline_single[2],
        )

        baseline_result = self._build_clean_result(
            "Baseline (Stage-1 Exact)",
            "Baseline",
            baseline_stage1_gelu,
            baseline_stage1_softmax,
            report_constraints,
            repeat_results=baseline_repeat,
            single_result=baseline_single,
        )


        eval_cache: Dict = {}
        repeat_cache: Dict = {}
        variance_cache: Dict = {}

        def _pack_noise_summary(summary):
            repeat = pack_repeat_evaluation(summary["trials"])
            repeat["stats"].update({
                k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                for k, v in summary.items()
                if k != "trials"
            })
            return repeat

        def _log_noise_variance_stats(label, variance_eval):
            stats = variance_eval["stats"]
            mode = stats.get("evaluation_mode", "repeated_validation_full")
            probe_suffix = (
                f", probe_size={int(stats['probe_size'])}"
                if stats.get("probe_size") is not None
                else ""
            )
            ev.log(
                f"  Variance[{label}] {mode}: N={int(stats['n'])}{probe_suffix}, "
                f"VarLoss={float(stats['loss_std']) ** 2:.2e}, "
                f"Var{metric_short_names[0]}={float(stats['p_std']) ** 2:.2e}"
                + (
                    f", Var{metric_short_names[1]}={float(stats['s_std']) ** 2:.2e}"
                    if num_metrics > 1
                    else ""
                )
            )

        def _noise_eval(gelu, softmax, noise_cfg, label, want_repeat):
            sig = self._full_signature(gelu, softmax, noise_cfg)
            if sig in eval_cache:
                return eval_cache[sig], repeat_cache.get(sig), variance_cache.get(sig)

            repeat = None
            variance_repeat = None
            if want_repeat and self.repeat_n > 1:
                ev.log(f"\n--- {label} : N={self.repeat_n} 次重复评估 ---")
                summary = ev.evaluate_model_with_attention_noise_repeated(
                    gelu,
                    softmax,
                    repeats=self.repeat_n,
                    use_train=False,
                    split=self.final_eval_split,
                    random_noise=True,
                    **noise_cfg,
                )
                repeat = _pack_noise_summary(summary)
                variance_repeat = repeat
                stats = repeat["stats"]
                ev.log(
                    f"  统计: Loss={stats['loss_mean']:.4f}±{stats['loss_std']:.6f} "
                    f"{metric_short_names[0]}={stats['p_mean']:.4f}±{stats['p_std']:.6f}"
                    + (
                        f" {metric_short_names[1]}={stats['s_mean']:.4f}±{stats['s_std']:.6f}"
                        if num_metrics > 1
                        else ""
                    )
                )
                cached = {
                    "loss": float(stats["loss_mean"]),
                    "p": float(stats["p_mean"]),
                    "s": float(stats["s_mean"]),
                    "time_ms": float(stats.get("time_mean_ms", 0.0)),
                }
                eval_cache[sig] = cached
                repeat_cache[sig] = repeat
                variance_cache[sig] = variance_repeat
                return cached, repeat, variance_repeat
            loss, p, s, t = ev.evaluate_model_with_attention_noise(
                gelu, softmax, use_train=False,
                split=self.final_eval_split, **noise_cfg
            )
            cached = {"loss": float(loss), "p": float(p), "s": float(s), "time_ms": float(t)}
            eval_cache[sig] = cached
            if want_repeat:
                variance_n = self._variance_repeat_count()
                ev.log(
                    f"\n--- {label} : variance probe N={variance_n} "
                    "(single final metric is kept) ---"
                )
                if hasattr(ev, "evaluate_model_with_attention_noise_segmented"):
                    summary = ev.evaluate_model_with_attention_noise_segmented(
                        gelu,
                        softmax,
                        segments=variance_n,
                        use_train=False,
                        split=self.final_eval_split,
                        random_noise=True,
                        **noise_cfg,
                    )
                else:
                    summary = ev.evaluate_model_with_attention_noise_repeated(
                        gelu,
                        softmax,
                        repeats=variance_n,
                        use_train=False,
                        split=self.final_eval_split,
                        random_noise=True,
                        **noise_cfg,
                    )
                variance_repeat = _pack_noise_summary(summary)
                variance_cache[sig] = variance_repeat
                _log_noise_variance_stats(label, variance_repeat)
            return cached, repeat, variance_repeat

        def _build_noise_result(name, family, gelu, softmax, noise_cfg, want_repeat=False):
            single, repeat, variance_repeat = _noise_eval(
                gelu, softmax, noise_cfg, name, want_repeat
            )
            stage1_tot, g_c, s_c = ev.get_simulated_cost(gelu, softmax)
            stage2_tot, breakdown = ev.get_noise_simulated_cost(**noise_cfg)
            stage2_tot = self._stage2_cost_key(stage2_tot) / 40.0

            if repeat is not None:
                stats = repeat["stats"]
                loss = float(stats["loss_mean"])
                p = float(stats["p_mean"])
                s = float(stats["s_mean"])
                time_ms = float(stats.get("time_mean_ms", single["time_ms"]))
            else:
                loss = single["loss"]
                p = single["p"]
                s = single["s"]
                time_ms = single["time_ms"]

            result = {
                "name": name,
                "family": family,
                "loss": float(loss),
                "p": float(p),
                "s": float(s),
                "time_ms": float(time_ms),
                "stage1_tot_c": float(stage1_tot),
                "stage1_g_c": float(g_c),
                "stage1_s_c": float(s_c),
                "stage2_tot_c": float(stage2_tot),
                "stage2_tot_spd": float(baseline_noise_tot_c / (stage2_tot + 1e-6)),
                "stage2_breakdown": {k: float(v) for k, v in breakdown.items()},
                "gelu": np.asarray(gelu, dtype=int).copy(),
                "softmax": np.asarray(softmax, dtype=int).copy(),
                "noise_config": {
                    k: np.asarray(noise_cfg[k], dtype=int).copy()
                    for k in NOISE_SCALING_FACTOR_KEYS
                },
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
                    }
                )
            if variance_repeat is not None:
                stats = variance_repeat["stats"]
                if repeat is None:
                    result.update(
                        {
                            "variance_evaluation_n": int(stats["n"]),
                            "loss_std": float(stats["loss_std"]),
                            "p_std": float(stats["p_std"]),
                            "s_std": float(stats["s_std"]),
                            "evaluation_protocol": "single_validation_full",
                            "variance_protocol": (
                                f"{stats.get('evaluation_mode', 'repeated_validation_full')}"
                                f"_n={int(stats['n'])}"
                            ),
                        }
                    )
                else:
                    result["variance_protocol"] = (
                        f"repeated_validation_full_n={int(stats['n'])}"
                    )
                result["variance_evaluation"] = variance_repeat
            return result, repeat


        optimized_result, optimized_repeat = _build_noise_result(
            "Optimized", "Optimized", opt_gelu, opt_softmax, opt_noise_cfg, want_repeat=True
        )


        stage1_fixed_max_noise_cfg = self._build_max_noise_config(total_layers)
        stage1_fixed_max_noise_result, stage1_fixed_max_noise_repeat = _build_noise_result(
            "Stage1Fixed+MaxSF",
            "Stage1FixedMaxSF",
            opt_gelu,
            opt_softmax,
            stage1_fixed_max_noise_cfg,
            want_repeat=True,
        )


        if self.include_random_groups:
            random_results = self._generate_random_results(
                opt_gelu=opt_gelu,
                opt_softmax=opt_softmax,
                opt_noise_cfg=opt_noise_cfg,
                opt_stage1_tot_c=opt_stage1_tot_c,
                opt_stage2_tot_c=opt_stage2_tot_c,
                opt_breakdown=opt_breakdown,
                total_layers=total_layers,
                build_result=_build_noise_result,
            )
        else:
            ev.log("Random comparison groups are disabled for this final-eval run.")
            random_results = []

        all_results = itertools.chain(
            (baseline_result, optimized_result, stage1_fixed_max_noise_result),
            random_results,
        )
        self._attach_relative_metrics(baseline_result, all_results, num_metrics)


        summary = self._summarize_random_results(optimized_result, random_results, num_metrics)
        self._log_performance_table(
            metric_short_names,
            num_metrics,
            baseline_result,
            optimized_result,
            stage1_fixed_max_noise_result,
            random_results,
        )
        self._log_random_summary(metric_short_names, summary, optimized_result, num_metrics)

        summary_path = self._save_results_json(
            selected_source=selected_source,
            baseline_stage1_gelu=baseline_stage1_gelu,
            baseline_stage1_softmax=baseline_stage1_softmax,
            opt_gelu=opt_gelu,
            opt_softmax=opt_softmax,
            opt_noise_cfg=opt_noise_cfg,
            baseline_result=baseline_result,
            baseline_repeat=baseline_repeat,
            optimized_result=optimized_result,
            optimized_repeat=optimized_repeat,
            stage1_fixed_max_noise_result=stage1_fixed_max_noise_result,
            stage1_fixed_max_noise_repeat=stage1_fixed_max_noise_repeat,
            random_results=random_results,
            summary=summary,
            selection_constraints=selection_constraints,
            report_constraints=report_constraints,
        )
        plot_path = self._plot_results(
            metric_short_names,
            num_metrics,
            baseline_result,
            optimized_result,
            stage1_fixed_max_noise_result,
            random_results,
            summary,
        )
        variance_plot_path = None
        if type(self)._plot_results is UnifiedFinalEvaluationModule._plot_results:
            variance_plot_path = self._plot_variance_results(
                metric_short_names,
                num_metrics,
                baseline_result,
                optimized_result,
                stage1_fixed_max_noise_result,
                random_results,
            )

        ev.apply_configuration(opt_gelu, opt_softmax)
        ev.clear_input_noise_configuration()
        ev.clear_weight_noise_configuration()

        return {
            "final_eval_split": self.final_eval_split,
            "dataset_protocol_hash": protocol["dataset_protocol_hash"],
            "validation_example_count": protocol["example_count"],
            "selected_source": selected_source,
            "opt_gelu": opt_gelu,
            "opt_softmax": opt_softmax,
            "opt_noise_config": opt_noise_cfg,
            "baseline_result": baseline_result,
            "optimized_result": optimized_result,
            "stage1_fixed_max_noise_result": stage1_fixed_max_noise_result,
            "random_results": random_results,
            "random_summary": summary,
            "baseline_repeat": baseline_repeat,
            "optimized_repeat": optimized_repeat,
            "stage1_fixed_max_noise_repeat": stage1_fixed_max_noise_repeat,
            "summary_path": summary_path,
            "plot_path": plot_path,
            "variance_plot_path": variance_plot_path,
        }


    def resolve_stage1_only(self, search_best_stage1, total_layers):
        """解析 stage-1 (gelu/softmax) 配置，用于 stage-2 RL 把 stage-1 固定下来。

        返回 ``(gelu, softmax, source)``。与 ``_resolve_selected_config`` 不同的是
        不要求 stage-2 search 结果存在。
        """
        if self.config_source == "search":
            if search_best_stage1 is not None:
                gelu, softmax = self._resolve_stage1_from_search(
                    search_best_stage1, total_layers
                )
                return gelu, softmax, "search"
            gelu, softmax, source = self._resolve_stage1_fallback(total_layers)
            return gelu, softmax, source

        if self.config_source == "json":
            gelu, softmax = self._resolve_stage1_from_json(total_layers)
            return gelu, softmax, "json"

        if self.config_source == "manual":
            if self.manual_stage1_gelu is None or self.manual_stage1_softmax is None:
                raise ValueError(
                    "config_source='manual' requires manual_stage1_gelu and manual_stage1_softmax."
                )
            gelu, softmax = self._resolve_stage1_from_manual(total_layers)
            return gelu, softmax, "manual"

        if self.config_source in MAX_CONFIG_SOURCES:
            if search_best_stage1 is not None:
                gelu, softmax = self._resolve_stage1_from_search(
                    search_best_stage1, total_layers
                )
                return gelu, softmax, "search"
            gelu, softmax, source = self._resolve_stage1_fallback(total_layers)
            return gelu, softmax, source

        raise ValueError(
            f"Unsupported config_source '{self.config_source}'. "
            "Use: search / json / manual / max."
        )

    def _resolve_selected_config(self, search_best_stage1, search_best_stage2, total_layers):
        if self.config_source == "search":
            stage1_from_search = search_best_stage1 is not None
            stage2_from_search = search_best_stage2 is not None
            if not stage1_from_search and not stage2_from_search:
                raise ValueError(
                    "config_source='search' requires at least one search result. "
                    "Both stage1 and stage2 search results are missing."
                )

            if stage1_from_search:
                gelu, softmax = self._resolve_stage1_from_search(
                    search_best_stage1, total_layers
                )
                stage1_source = "search"
            else:
                gelu, softmax, stage1_source = self._resolve_stage1_fallback(total_layers)

            if stage2_from_search:
                noise_cfg = self._resolve_stage2_from_search(
                    search_best_stage2, total_layers
                )
                stage2_source = "search"
            else:
                noise_cfg, stage2_source = self._resolve_stage2_fallback(total_layers)

            if stage1_source == "search" and stage2_source == "search":
                selected_source = "search"
            else:
                selected_source = f"search(stage1={stage1_source},stage2={stage2_source})"
            return gelu, softmax, noise_cfg, selected_source

        if self.config_source == "json":
            gelu, softmax = self._resolve_stage1_from_json(total_layers)
            noise_cfg = self._resolve_stage2_from_json(total_layers)
            return gelu, softmax, noise_cfg, "json"

        if self.config_source == "manual":
            if (
                self.manual_stage1_gelu is None
                or self.manual_stage1_softmax is None
                or self.manual_stage2_noise is None
            ):
                raise ValueError(
                    "config_source='manual' requires manual_stage1_gelu, manual_stage1_softmax, "
                    "and manual_stage2_noise."
                )
            gelu, softmax = self._resolve_stage1_from_manual(total_layers)
            noise_cfg = self._resolve_stage2_from_manual(total_layers)
            return gelu, softmax, noise_cfg, "manual"

        if self.config_source in MAX_CONFIG_SOURCES:
            if search_best_stage1 is not None:
                gelu, softmax = self._resolve_stage1_from_search(
                    search_best_stage1, total_layers
                )
                stage1_source = "search"
            else:
                gelu, softmax, stage1_source = self._resolve_stage1_fallback(total_layers)
            noise_cfg = self._build_max_noise_config(total_layers)
            return (
                gelu,
                softmax,
                noise_cfg,
                f"{self.config_source}(stage1={stage1_source},stage2=max)",
            )

        raise ValueError(
            f"Unsupported config_source '{self.config_source}'. "
            "Use: search / json / manual / max."
        )

    def _resolve_stage1_from_search(self, search_best_stage1, total_layers):
        gelu = self._normalize_config_array(
            search_best_stage1["gelu"], total_layers, 4, self.allowed_gelu_selected, "search_gelu"
        )
        softmax = self._normalize_config_array(
            search_best_stage1["softmax"], total_layers, 6, self.allowed_softmax, "search_softmax"
        )
        return gelu, softmax

    def _resolve_stage2_from_search(self, search_best_stage2, total_layers):
        return {
            key: self._normalize_noise_array(search_best_stage2[key], total_layers, key)
            for key in NOISE_SCALING_FACTOR_KEYS
        }

    def _resolve_stage1_from_json(self, total_layers):
        s1, _ = self._load_dataset_config_from_json(required_sections=("stage1",))
        gelu = self._normalize_config_array(
            s1["gelu"], total_layers, 4, self.allowed_gelu_selected, "json_gelu"
        )
        softmax = self._normalize_config_array(
            s1["softmax"], total_layers, 6, self.allowed_softmax, "json_softmax"
        )
        return gelu, softmax

    def _resolve_stage2_from_json(self, total_layers):
        _, s2 = self._load_dataset_config_from_json(required_sections=("stage2",))
        noise_cfg = {}
        for key in NOISE_SCALING_FACTOR_KEYS:
            short = self._full_to_short(key)
            raw = s2.get(key) or s2.get(short)
            if raw is None:
                raise KeyError(f"JSON config missing stage2 key '{key}' / '{short}'.")
            noise_cfg[key] = self._normalize_noise_array(raw, total_layers, key)
        return noise_cfg

    def _resolve_stage1_from_manual(self, total_layers):
        gelu = self._normalize_config_array(
            self.manual_stage1_gelu, total_layers, 4, self.allowed_gelu_selected, "manual_gelu"
        )
        softmax = self._normalize_config_array(
            self.manual_stage1_softmax, total_layers, 6, self.allowed_softmax, "manual_softmax"
        )
        return gelu, softmax

    def _resolve_stage2_from_manual(self, total_layers):
        noise_cfg = {}
        for key in NOISE_SCALING_FACTOR_KEYS:
            short = self._full_to_short(key)
            raw = self.manual_stage2_noise.get(key) or self.manual_stage2_noise.get(short)
            if raw is None:
                raise KeyError(f"manual_stage2_noise missing '{key}' / '{short}'.")
            noise_cfg[key] = self._normalize_noise_array(raw, total_layers, key)
        return noise_cfg

    def _resolve_stage1_fallback(self, total_layers):
        if self.manual_stage1_gelu is not None or self.manual_stage1_softmax is not None:
            if self.manual_stage1_gelu is None or self.manual_stage1_softmax is None:
                raise ValueError(
                    "Stage-1 fallback requires both manual_stage1_gelu and manual_stage1_softmax."
                )
            gelu, softmax = self._resolve_stage1_from_manual(total_layers)
            return gelu, softmax, "manual"

        try:
            gelu, softmax = self._resolve_stage1_from_json(total_layers)
            return gelu, softmax, "json"
        except Exception as exc:
            raise ValueError(
                "Stage-1 search result is unavailable, and fallback resolution failed. "
                "Provide --final-eval-config with valid stage1 content, or provide "
                "both --manual-stage1-gelu and --manual-stage1-softmax."
            ) from exc

    def _resolve_stage2_fallback(self, total_layers):
        if self.manual_stage2_noise is not None:
            noise_cfg = self._resolve_stage2_from_manual(total_layers)
            return noise_cfg, "manual"

        try:
            noise_cfg = self._resolve_stage2_from_json(total_layers)
            return noise_cfg, "json"
        except Exception as exc:
            raise ValueError(
                "Stage-2 search result is unavailable, and fallback resolution failed. "
                "Provide --final-eval-config with valid stage2 content, or provide "
                "--manual-stage2-noise."
            ) from exc

    def _load_dataset_config_from_json(self, required_sections=("stage1", "stage2")):
        config_path = str(self.config_path)
        if getattr(self, "_config_json_cache_path", None) == config_path:
            config_map = self._config_json_cache
        else:
            with open(config_path, encoding="utf-8") as fh:
                config_map = json.load(fh)
            config_map.pop("_comment", None)
            self._config_json_cache_path = config_path
            self._config_json_cache = config_map

        total_layers = int(getattr(self.evaluator, "total_layers", 12) or 12)
        explicit = getattr(self.evaluator, "model_type", None)
        if explicit in ("bert-base", "bert-large"):
            variant = explicit
        else:
            if total_layers >= 24:
                variant = "bert-large"
            else:
                variant = "bert-base"
        if variant not in config_map:
            raise KeyError(
                f"Model variant '{variant}' (total_layers={total_layers}) "
                f"not found in final-eval config '{self.config_path}'."
            )
        section = config_map[variant]
        ds = self.evaluator.dataset_key
        if ds not in section:
            raise KeyError(f"Dataset '{ds}' missing under '{variant}' in '{self.config_path}'.")
        entry = section[ds]
        missing = [name for name in required_sections if name not in entry]
        if missing:
            raise KeyError(
                f"Entry '{variant}/{ds}' in '{self.config_path}' is missing sections: {missing}."
            )
        return entry.get("stage1"), entry.get("stage2")


    def _generate_random_results(
        self,
        opt_gelu,
        opt_softmax,
        opt_noise_cfg,
        opt_stage1_tot_c,
        opt_stage2_tot_c,
        opt_breakdown,
        total_layers,
        build_result,
    ):
        ev = self.evaluator


        if self.final_eval_only:
            self.random_group_seed = (
                int.from_bytes(os.urandom(8), "little") & 0x7FFFFFFFFFFFFFFF
            )
            ev.log(
                "[final_eval_only] Random comparison seed "
                f"(OS entropy) = {self.random_group_seed}"
            )
            rng = np.random.default_rng(self.random_group_seed)
        else:
            rng = np.random.default_rng()
        results: List[dict] = []

        gelu_solution_map = None
        softmax_solution_map = None
        opt_g_cost = None
        opt_s_cost = None

        def stage1_solution_maps():
            nonlocal gelu_solution_map, softmax_solution_map
            if gelu_solution_map is None:
                gelu_solution_map = self._enumerate_cost_solutions(
                    self.allowed_gelu_random, ev.GELU_COST_MAP, total_layers
                )
                softmax_solution_map = self._enumerate_cost_solutions(
                    self.allowed_softmax, ev.SOFTMAX_COST_MAP, total_layers
                )
            return gelu_solution_map, softmax_solution_map

        def stage1_exact_costs():
            nonlocal opt_g_cost, opt_s_cost
            if opt_g_cost is None:
                opt_g_cost = float(np.sum([ev.GELU_COST_MAP[int(d)] for d in opt_gelu]))
                opt_s_cost = float(
                    np.sum([ev.SOFTMAX_COST_MAP[int(d)] for d in opt_softmax])
                )
            return opt_g_cost, opt_s_cost

        seen = {self._full_signature(opt_gelu, opt_softmax, opt_noise_cfg)}

        def register(name, family, gelu, softmax, noise_cfg):
            sig = self._full_signature(gelu, softmax, noise_cfg)
            if sig in seen:
                return False
            seen.add(sig)
            res, repeat = build_result(name, family, gelu, softmax, noise_cfg, want_repeat=True)
            if repeat is not None:
                res["repeat_evaluation"] = repeat
            results.append(res)
            return True


        if self.stage1_budget_trials > 0 and not np.any(opt_gelu == 0):
            ev.log(f"Generating {self.stage1_budget_trials} Stage1Budget configs...")
            gelu_solution_map, softmax_solution_map = stage1_solution_maps()
            for idx in range(self.stage1_budget_trials):
                pair = self._sample_stage1_total_cost(
                    rng, gelu_solution_map, softmax_solution_map, opt_stage1_tot_c
                )
                if pair is None:
                    continue
                g, sm = pair
                register(f"Stage1Budget_{idx + 1}", "Stage1Budget", g, sm, opt_noise_cfg)


        if self.stage2_budget_trials > 0:
            ev.log(f"Generating {self.stage2_budget_trials} Stage2Budget configs...")
            for idx in range(self.stage2_budget_trials):
                cfg = self._sample_stage2_total_cost(rng, opt_stage2_tot_c, total_layers)
                if cfg is None:
                    continue
                register(f"Stage2Budget_{idx + 1}", "Stage2Budget", opt_gelu, opt_softmax, cfg)


        if self.permutation_trials > 0 and not np.any(opt_gelu == 0):
            ev.log(f"Generating {self.permutation_trials} Perm configs...")
            for idx in range(self.permutation_trials):
                for _ in range(100):
                    g = rng.permutation(opt_gelu)
                    sm = rng.permutation(opt_softmax)
                    cfg = {
                        key: rng.permutation(opt_noise_cfg[key])
                        for key in NOISE_SCALING_FACTOR_KEYS
                    }
                    if register(f"Perm_{idx + 1}", "Perm", g, sm, cfg):
                        break


        if self.cost_equivalent_trials > 0 and not np.any(opt_gelu == 0):
            ev.log(f"Generating {self.cost_equivalent_trials} Equiv configs...")
            gelu_solution_map, softmax_solution_map = stage1_solution_maps()
            opt_g_cost, opt_s_cost = stage1_exact_costs()
            for idx in range(self.cost_equivalent_trials):
                pair = self._sample_stage1_equiv(
                    rng,
                    gelu_solution_map,
                    softmax_solution_map,
                    opt_g_cost,
                    opt_s_cost,
                )
                cfg = self._sample_stage2_equiv(rng, opt_breakdown, total_layers)
                if pair is None or cfg is None:
                    continue
                g, sm = pair
                register(f"Equiv_{idx + 1}", "Equiv", g, sm, cfg)


        if self.budget_equivalent_trials > 0 and not np.any(opt_gelu == 0):
            ev.log(f"Generating {self.budget_equivalent_trials} Budget configs...")
            gelu_solution_map, softmax_solution_map = stage1_solution_maps()
            for idx in range(self.budget_equivalent_trials):
                pair = self._sample_stage1_total_cost(
                    rng, gelu_solution_map, softmax_solution_map, opt_stage1_tot_c
                )
                cfg = self._sample_stage2_total_cost(rng, opt_stage2_tot_c, total_layers)
                if pair is None or cfg is None:
                    continue
                g, sm = pair
                register(f"Budget_{idx + 1}", "Budget", g, sm, cfg)

        return results


    def _sample_stage1_total_cost(
        self, rng, gelu_solution_map, softmax_solution_map, target_total_cost
    ):
        target_key = self._cost_key(target_total_cost)
        feasible_pairs = self._stage1_total_cost_pairs(
            gelu_solution_map,
            softmax_solution_map,
            target_key,
        )
        if not feasible_pairs:
            return None
        for _ in range(200):
            g_key, s_key = feasible_pairs[rng.integers(0, len(feasible_pairs))]
            g_counts = gelu_solution_map[g_key]
            s_counts = softmax_solution_map[s_key]
            g_choice = g_counts[rng.integers(0, len(g_counts))]
            s_choice = s_counts[rng.integers(0, len(s_counts))]
            return (
                self._counts_to_shuffled_config(self.allowed_gelu_random, g_choice, rng),
                self._counts_to_shuffled_config(self.allowed_softmax, s_choice, rng),
            )
        return None

    def _stage1_total_cost_pairs(self, gelu_solution_map, softmax_solution_map, target_key):
        cache_key = (
            id(gelu_solution_map),
            len(gelu_solution_map),
            id(softmax_solution_map),
            len(softmax_solution_map),
            int(target_key),
        )
        cache = getattr(self, "_stage1_total_cost_pair_cache", {})
        if cache_key in cache:
            return cache[cache_key]

        feasible_pairs = []
        for g_key in gelu_solution_map.keys():
            s_key = target_key - g_key
            if s_key in softmax_solution_map:
                feasible_pairs.append((g_key, s_key))
        cache[cache_key] = tuple(feasible_pairs)
        self._stage1_total_cost_pair_cache = cache
        return cache[cache_key]

    def _sample_stage1_equiv(
        self, rng, gelu_solution_map, softmax_solution_map, target_g_cost, target_s_cost
    ):
        g_key = self._cost_key(target_g_cost)
        s_key = self._cost_key(target_s_cost)
        g_candidates = gelu_solution_map.get(g_key)
        s_candidates = softmax_solution_map.get(s_key)
        if not g_candidates or not s_candidates:
            return None
        g_choice = g_candidates[rng.integers(0, len(g_candidates))]
        s_choice = s_candidates[rng.integers(0, len(s_candidates))]
        return (
            self._counts_to_shuffled_config(self.allowed_gelu_random, g_choice, rng),
            self._counts_to_shuffled_config(self.allowed_softmax, s_choice, rng),
        )

    def _enumerate_cost_solutions(self, allowed_degrees, cost_map, total_layers):
        solution_map: Dict[int, list] = {}
        counts = [0] * len(allowed_degrees)
        cost_keys = [self._cost_key(cost_map[d]) for d in allowed_degrees]

        def backtrack(index, remaining, accumulated):
            if index == len(allowed_degrees) - 1:
                counts[index] = remaining
                final_cost = accumulated + remaining * cost_keys[index]
                solution_map.setdefault(final_cost, []).append(tuple(counts))
                return
            for c in range(remaining + 1):
                counts[index] = c
                backtrack(index + 1, remaining - c, accumulated + c * cost_keys[index])

        backtrack(0, total_layers, 0)
        return solution_map

    @staticmethod
    def _counts_to_shuffled_config(allowed_degrees, counts, rng):
        values: List[int] = []
        for d, c in zip(allowed_degrees, counts):
            values.extend([d] * int(c))
        arr = np.array(values, dtype=int)
        rng.shuffle(arr)
        return arr


    def _sample_stage2_equiv(self, rng, breakdown, total_layers):
        cfg = {}
        specs, solution_maps = self._stage2_total_cost_solution_maps(total_layers)
        for idx, (short, full, allowed) in enumerate(specs):
            solution_map = solution_maps[idx]
            target = breakdown[short]
            target_key = self._stage2_cost_key(target)
            candidates = solution_map.get(target_key)
            if candidates:
                counts = candidates[int(rng.integers(0, len(candidates)))]
                cfg[full] = self._counts_to_shuffled_config(allowed, counts, rng)
            else:
                cost_map = self._stage2_cost_map(short)
                arr = self._stage2_cost_matched_array(rng, target, cost_map, allowed, total_layers)
                if arr is None:
                    return None
                cfg[full] = arr
        return cfg

    def _sample_stage2_total_cost(self, rng, target_total, total_layers):
        target_key = self._stage2_cost_key(target_total)
        specs, solution_maps = self._stage2_total_cost_solution_maps(total_layers)

        combo_plan = self._stage2_count_combo_plan(solution_maps)
        count_choices = self._sample_stage2_count_combo(
            rng,
            solution_maps,
            target_key,
            combo_plan,
        )
        if count_choices is None:
            return None

        cfg = {}
        for counts, (_, full, allowed) in zip(count_choices, specs):
            cfg[full] = self._counts_to_shuffled_config(allowed, counts, rng)
        if self._stage2_config_cost_key(cfg) != target_key:
            return None
        return cfg

    def _stage2_total_cost_solution_maps(self, total_layers):
        specs = []
        cache_parts = []
        for short in BREAKDOWN_KEYS:
            allowed = tuple(int(v) for v in self._stage2_allowed(short))
            cost_map = self._stage2_cost_map(short)
            cost_keys = tuple(
                (int(value), self._stage2_cost_key(cost_map[int(value)]))
                for value in allowed
            )
            specs.append((short, SHORT_KEY_TO_FULL[short], allowed))
            cache_parts.append((short, allowed, cost_keys))

        cache_key = (int(total_layers), tuple(cache_parts))
        cache = getattr(self, "_stage2_total_cost_solution_cache", {})
        if cache_key in cache:
            return cache[cache_key]

        solution_maps = [
            self._enumerate_stage2_count_solutions(
                allowed,
                self._stage2_cost_map(short),
                total_layers,
            )
            for short, _full, allowed in specs
        ]
        cached = (specs, solution_maps)
        cache[cache_key] = cached
        self._stage2_total_cost_solution_cache = cache
        return cached

    def _enumerate_stage2_count_solutions(self, allowed_degrees, cost_map, total_layers):
        solution_map: Dict[int, list] = {}
        counts = [0] * len(allowed_degrees)
        cost_keys = [self._stage2_cost_key(cost_map[d]) for d in allowed_degrees]

        def backtrack(index, remaining, accumulated):
            if index == len(allowed_degrees) - 1:
                counts[index] = remaining
                final_cost = accumulated + remaining * cost_keys[index]
                solution_map.setdefault(final_cost, []).append(tuple(counts))
                return
            for c in range(remaining + 1):
                counts[index] = c
                backtrack(index + 1, remaining - c, accumulated + c * cost_keys[index])

        backtrack(0, total_layers, 0)
        return solution_map

    def _stage2_count_combo_plan(self, solution_maps):
        cache_key = tuple((id(solution_map), len(solution_map)) for solution_map in solution_maps)
        cache = getattr(self, "_stage2_count_combo_plan_cache", {})
        if cache_key in cache:
            return cache[cache_key]

        plan = self._build_stage2_count_combo_plan(solution_maps)
        cache[cache_key] = plan
        self._stage2_count_combo_plan_cache = cache
        return plan

    @staticmethod
    def _build_stage2_count_combo_plan(solution_maps):
        key_options = tuple(tuple(solution_map.keys()) for solution_map in solution_maps)
        suffix_possible = [set() for _ in range(len(solution_maps) + 1)]
        suffix_possible[-1].add(0)
        for idx in range(len(key_options) - 1, -1, -1):
            for key in key_options[idx]:
                for rest in suffix_possible[idx + 1]:
                    suffix_possible[idx].add(key + rest)
        return key_options, tuple(frozenset(values) for values in suffix_possible), {}

    @staticmethod
    def _sample_stage2_count_combo(rng, solution_maps, target_key, combo_plan=None):
        if combo_plan is None:
            combo_plan = UnifiedFinalEvaluationModule._build_stage2_count_combo_plan(
                solution_maps
            )
        if len(combo_plan) == 2:
            key_options, suffix_possible = combo_plan
            feasible_key_cache = {}
        else:
            key_options, suffix_possible, feasible_key_cache = combo_plan
        if target_key not in suffix_possible[0]:
            return None

        remaining = target_key
        choices = []
        for idx, solution_map in enumerate(solution_maps):
            cache_key = (int(idx), int(remaining))
            feasible_keys = feasible_key_cache.get(cache_key)
            if feasible_keys is None:
                feasible_keys = tuple(
                    key
                    for key in key_options[idx]
                    if remaining - key in suffix_possible[idx + 1]
                )
                feasible_key_cache[cache_key] = feasible_keys
            if not feasible_keys:
                return None
            key = feasible_keys[int(rng.integers(0, len(feasible_keys)))]
            candidates = solution_map[key]
            choices.append(candidates[int(rng.integers(0, len(candidates)))])
            remaining -= key
        return choices if remaining == 0 else None

    def _stage2_config_cost_key(self, cfg):
        total = 0
        for short in BREAKDOWN_KEYS:
            full = SHORT_KEY_TO_FULL[short]
            cost_map = self._stage2_cost_map(short)
            total += sum(
                self._stage2_cost_key(cost_map[int(v)])
                for v in np.asarray(cfg[full], dtype=int)
            )
        return int(total)

    @staticmethod
    def _stage2_cost_matched_array(rng, target_cost, cost_map, allowed, length):
        values = tuple(int(v) for v in allowed)
        cost_by_value = {value: cost_map[value] for value in values}
        for _ in range(2000):
            cfg = np.array(rng.choice(values, size=length), dtype=int)
            curr_cost = sum(cost_by_value[int(d)] for d in cfg)
            for _ in range(500):
                diff = curr_cost - target_cost
                if abs(diff) < 1e-6:
                    return cfg
                idx = int(rng.integers(0, length))
                old_v = int(cfg[idx])
                old_cost = cost_by_value[old_v]
                moves = []
                for value in values:
                    candidate_cost = curr_cost - old_cost + cost_by_value[value]
                    if abs(candidate_cost - target_cost) < abs(diff):
                        moves.append(value)
                new_v = int(rng.choice(moves if moves else values))
                cfg[idx] = new_v
                curr_cost = curr_cost - old_cost + cost_by_value[new_v]
        return None

    def _stage2_allowed(self, short_key):
        if short_key == "x":
            return self.input_noise_allowed
        if short_key == "wffn1":
            return self.wffn1_noise_allowed
        return self.weight_noise_allowed

    def _stage2_cost_map(self, short_key):
        ev = self.evaluator
        if short_key == "x":
            return ev.INPUT_NOISE_COST_MAP
        if short_key == "wffn1":
            return ev.WFFN1_NOISE_COST_MAP
        return ev.WEIGHT_NOISE_COST_MAP


    def _build_clean_result(
        self,
        name,
        family,
        gelu,
        softmax,
        constraints,
        repeat_results=None,
        single_result=None,
    ):
        ev = self.evaluator
        if repeat_results is None and single_result is None:
            loss, p, s, t = ev.evaluate_model(
                gelu,
                softmax,
                use_train=False,
                split=getattr(self, "final_eval_split", FINAL_EVAL_SPLIT),
            )
        elif single_result is not None:
            loss, p, s, t = single_result
        else:
            stats = repeat_results["stats"]
            loss = stats["loss_mean"]
            p = stats["p_mean"]
            s = stats["s_mean"]
            t = stats.get("time_mean_ms", 0.0)
        stage1_tot, g_c, s_c = ev.get_simulated_cost(gelu, softmax)
        result = {
            "name": name,
            "family": family,
            "loss": float(loss),
            "p": float(p),
            "s": float(s),
            "time_ms": float(t),
            "stage1_tot_c": float(stage1_tot),
            "stage1_g_c": float(g_c),
            "stage1_s_c": float(s_c),
            "stage2_tot_c": 0.0,
            "stage2_tot_spd": None,
            "stage2_breakdown": None,
            "gelu": np.asarray(gelu, dtype=int).copy(),
            "softmax": np.asarray(softmax, dtype=int).copy(),
            "noise_config": None,
            "feasible": self._is_feasible(float(loss), float(p), float(s), constraints),
            "show_cost_as_na": True,
        }
        if repeat_results is not None:
            stats = repeat_results["stats"]
            result.update(
                {
                    "evaluation_n": int(stats["n"]),
                    "loss_std": float(stats["loss_std"]),
                    "p_std": float(stats["p_std"]),
                    "s_std": float(stats["s_std"]),
                    "evaluation_protocol": f"repeated_mean_n={int(stats['n'])}",
                }
            )
        else:
            result.update(
                {
                    "evaluation_n": 1,
                    "loss_std": 0.0,
                    "p_std": 0.0,
                    "s_std": 0.0,
                    "evaluation_protocol": "single_clean_deterministic",
                    "variance_protocol": "deterministic_no_noise",
                }
            )
        return result


    def _attach_relative_metrics(self, baseline, results, num_metrics):
        base_loss = float(baseline["loss"])
        base_p = float(baseline["p"])
        base_s = float(baseline["s"])
        for result in results:
            result["total_cost"] = self._combined_total_cost(result)
            result["loss_delta_vs_baseline"] = float(result["loss"] - base_loss)
            result["p_delta_vs_baseline"] = float(result["p"] - base_p)
            result["loss_delta_pct_vs_baseline"] = self._relative_delta_percent(
                result["loss"], base_loss
            )
            result["p_delta_pct_vs_baseline"] = self._relative_delta_percent(
                result["p"], base_p
            )
            if num_metrics > 1:
                result["s_delta_vs_baseline"] = float(result["s"] - base_s)
                result["s_delta_pct_vs_baseline"] = self._relative_delta_percent(
                    result["s"], base_s
                )
            for metric_key in ("loss", "p", "s"):
                std_key = f"{metric_key}_std"
                if std_key in result:
                    std_value = float(result[std_key])
                    result[f"{metric_key}_var"] = float(std_value * std_value)

    @staticmethod
    def _combined_total_cost(result):
        if result.get("show_cost_as_na"):
            return None
        stage1 = result.get("stage1_tot_c")
        stage2 = result.get("stage2_tot_c")
        if stage1 is None or stage2 is None:
            return None
        return float(stage1) + float(stage2)

    @staticmethod
    def _format_fixed(value, width=8, precision=4):
        if value is None:
            return f"{'N/A':<{width}}"
        try:
            if not np.isfinite(float(value)):
                return f"{'N/A':<{width}}"
            return f"{float(value):<{width}.{precision}f}"
        except Exception:
            return f"{'N/A':<{width}}"

    @staticmethod
    def _format_sci(value, width=10):
        if value is None:
            return f"{'N/A':<{width}}"
        try:
            if not np.isfinite(float(value)):
                return f"{'N/A':<{width}}"
            return f"{float(value):<{width}.2e}"
        except Exception:
            return f"{'N/A':<{width}}"

    @staticmethod
    def _relative_delta_percent(value, baseline):
        try:
            baseline = float(baseline)
            value = float(value)
            if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) <= 1e-12:
                return None
            return float((value - baseline) / baseline * 100.0)
        except Exception:
            return None

    @staticmethod
    def _format_percent(value, width=8, precision=2):
        if value is None:
            return f"{'N/A':<{width}}"
        try:
            if not np.isfinite(float(value)):
                return f"{'N/A':<{width}}"
            text = f"{float(value):.{precision}f}%"
            return f"{text:<{width}}"
        except Exception:
            return f"{'N/A':<{width}}"

    @staticmethod
    def _mean_float_or_none(values):
        mean, _std = UnifiedFinalEvaluationModule._finite_float_stats(values)
        return mean

    @staticmethod
    def _std_float_or_none(values):
        _mean, std = UnifiedFinalEvaluationModule._finite_float_stats(values)
        return std

    @staticmethod
    def _finite_float_stats(values):
        count = 0
        total = 0.0
        total_sq = 0.0
        for value in values:
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                continue
            count += 1
            total += value
            total_sq += value * value
        if count == 0:
            return None, None

        mean = total / count
        variance = total_sq / count - mean * mean
        if np.isfinite(variance) and variance < 0.0:
            variance = 0.0
        return float(mean), float(variance ** 0.5)

    def _summarize_random_results(self, selected, random_results, num_metrics):
        summary: Dict = {"overall": {}, "by_family": {}}
        if not random_results:
            return summary

        selected_loss = selected["loss"]
        selected_p = selected["p"]
        selected_s = selected["s"] if num_metrics > 1 else 0.0
        selected_stage1 = selected["stage1_tot_c"]
        selected_stage2 = selected["stage2_tot_c"]

        class _RunningStats:
            __slots__ = ("count", "total", "total_sq")

            def __init__(self):
                self.count = 0
                self.total = 0.0
                self.total_sq = 0.0

            def add(self, value):
                value = float(value)
                self.count += 1
                self.total += value
                self.total_sq += value * value

            def add_optional_finite(self, value):
                if value is None:
                    return
                value = float(value)
                if np.isfinite(value):
                    self.add(value)

            def mean(self, default=None):
                if not self.count:
                    return default
                return float(self.total / self.count)

            def std(self, default=None):
                if not self.count:
                    return default
                mean = self.total / self.count
                variance = self.total_sq / self.count - mean * mean
                if np.isfinite(variance) and variance < 0.0 and variance > -1e-12:
                    variance = 0.0
                return float(variance ** 0.5)

        def new_accumulator():
            return {
                "count": 0,
                "feasible": 0,
                "loss_win": 0,
                "primary_win": 0,
                "secondary_win": 0,
                "dominance": 0,
                "loss": _RunningStats(),
                "p": _RunningStats(),
                "s": _RunningStats(),
                "loss_delta_default": _RunningStats(),
                "p_delta_default": _RunningStats(),
                "s_delta_default": _RunningStats(),
                "loss_delta_optional": _RunningStats(),
                "p_delta_optional": _RunningStats(),
                "s_delta_optional": _RunningStats(),
                "total_cost": _RunningStats(),
                "stage1_cost": _RunningStats(),
                "stage2_cost": _RunningStats(),
                "var": {
                    "loss": _RunningStats(),
                    "p": _RunningStats(),
                    "s": _RunningStats(),
                },
            }

        grouped: Dict[str, dict] = {}
        overall = new_accumulator()

        for it in random_results:
            family_acc = grouped.setdefault(it["family"], new_accumulator())
            for acc in (family_acc, overall):
                acc["count"] += 1
                acc["feasible"] += 1 if it["feasible"] else 0
                acc["loss_win"] += 1 if selected_loss <= it["loss"] else 0
                acc["primary_win"] += 1 if selected_p >= it["p"] else 0
                if num_metrics > 1:
                    acc["secondary_win"] += 1 if selected_s >= it["s"] else 0
                dominates = (
                    selected_stage1 <= it["stage1_tot_c"]
                    and selected_stage2 <= it["stage2_tot_c"]
                    and selected_loss <= it["loss"]
                    and selected_p >= it["p"]
                    and (num_metrics <= 1 or selected_s >= it["s"])
                    and (
                        selected_stage1 < it["stage1_tot_c"]
                        or selected_stage2 < it["stage2_tot_c"]
                        or selected_loss < it["loss"]
                        or selected_p > it["p"]
                        or (num_metrics > 1 and selected_s > it["s"])
                    )
                )
                acc["dominance"] += 1 if dominates else 0
                acc["loss"].add(it["loss"])
                acc["p"].add(it["p"])
                if num_metrics > 1:
                    acc["s"].add(it["s"])
                acc["stage1_cost"].add(it["stage1_tot_c"])
                acc["stage2_cost"].add(it["stage2_tot_c"])
                acc["total_cost"].add_optional_finite(it.get("total_cost"))
                for metric_key in ("loss", "p", "s"):
                    if f"{metric_key}_var" in it:
                        acc["var"][metric_key].add(it[f"{metric_key}_var"])

            family_acc["loss_delta_default"].add(it.get("loss_delta_vs_baseline", 0.0))
            family_acc["p_delta_default"].add(it.get("p_delta_vs_baseline", 0.0))
            if num_metrics > 1:
                family_acc["s_delta_default"].add(it.get("s_delta_vs_baseline", 0.0))
            overall["loss_delta_optional"].add_optional_finite(
                it.get("loss_delta_vs_baseline")
            )
            overall["p_delta_optional"].add_optional_finite(
                it.get("p_delta_vs_baseline")
            )
            if num_metrics > 1:
                overall["s_delta_optional"].add_optional_finite(
                    it.get("s_delta_vs_baseline")
                )

        for family, acc in grouped.items():
            count = int(acc["count"])
            family_summary = {
                "count": count,
                "feasible_rate": float(acc["feasible"] / count),
                "loss_win_rate": float(acc["loss_win"] / count),
                "primary_win_rate": float(acc["primary_win"] / count),
                "secondary_win_rate": (
                    float(acc["secondary_win"] / count) if num_metrics > 1 else None
                ),
                "dominance_rate": float(acc["dominance"] / count),
                "loss_mean": acc["loss"].mean(),
                "loss_std": acc["loss"].std(),
                "primary_metric_mean": acc["p"].mean(),
                "primary_metric_std": acc["p"].std(),
                "secondary_metric_mean": (
                    acc["s"].mean() if num_metrics > 1 else None
                ),
                "secondary_metric_std": (
                    acc["s"].std() if num_metrics > 1 else None
                ),
                "loss_delta_mean": acc["loss_delta_default"].mean(),
                "primary_metric_delta_mean": acc["p_delta_default"].mean(),
                "secondary_metric_delta_mean": (
                    acc["s_delta_default"].mean() if num_metrics > 1 else None
                ),
                "total_cost_mean": acc["total_cost"].mean(),
                "total_cost_std": acc["total_cost"].std(),
                "stage1_cost_mean": acc["stage1_cost"].mean(),
                "stage2_cost_mean": acc["stage2_cost"].mean(),
            }
            for metric_key in ("loss", "p", "s"):
                stat = acc["var"][metric_key]
                if stat.count:
                    family_summary[f"{metric_key}_eval_variance_mean"] = stat.mean()
            summary["by_family"][family] = family_summary

        summary["overall"] = {
            "count": len(random_results),
            "feasible_rate": (
                float(overall["feasible"] / overall["count"]) if overall["count"] else 0.0
            ),
            "loss_win_rate": (
                float(overall["loss_win"] / overall["count"]) if overall["count"] else 0.0
            ),
            "primary_win_rate": (
                float(overall["primary_win"] / overall["count"]) if overall["count"] else 0.0
            ),
            "secondary_win_rate": (
                float(overall["secondary_win"] / overall["count"])
                if num_metrics > 1 and overall["count"]
                else None
            ),
            "dominance_rate": (
                float(overall["dominance"] / overall["count"]) if overall["count"] else 0.0
            ),
            "loss_delta_mean": overall["loss_delta_optional"].mean(),
            "primary_metric_delta_mean": overall["p_delta_optional"].mean(),
            "secondary_metric_delta_mean": (
                overall["s_delta_optional"].mean()
                if num_metrics > 1
                else None
            ),
            "total_cost_mean": overall["total_cost"].mean(),
        }
        for metric_key in ("loss", "p", "s"):
            stat = overall["var"][metric_key]
            if stat.count:
                summary["overall"][f"{metric_key}_eval_variance_mean"] = stat.mean()
        return summary

    def _log_performance_table(
        self,
        metric_short_names,
        num_metrics,
        baseline,
        optimized,
        stage1_fixed_max,
        random_results,
    ):
        ev = self.evaluator
        ev.log("\nUnified Performance Comparison Table:")
        if num_metrics == 1:
            header = (
                f"{'Method':<25} | {'OK':<3} | "
                f"{'Loss':<8} {metric_short_names[0]:<8} | "
                f"{'dLoss%':<8} {('d' + metric_short_names[0] + '%'):<8} | "
                f"{'VarLoss':<10} {('Var' + metric_short_names[0]):<10} | "
                f"{'TotalC':<8}"
            )
        else:
            header = (
                f"{'Method':<25} | {'OK':<3} | "
                f"{'Loss':<8} {metric_short_names[0]:<8} {metric_short_names[1]:<8} | "
                f"{'dLoss%':<8} {('d' + metric_short_names[0] + '%'):<8} {('d' + metric_short_names[1] + '%'):<8} | "
                f"{'VarLoss':<10} {('Var' + metric_short_names[0]):<10} {('Var' + metric_short_names[1]):<10} | "
                f"{'TotalC':<8}"
            )
        ev.log("-" * len(header))
        ev.log(header)
        ev.log("-" * len(header))
        ev.log(self._format_row(baseline, num_metrics))
        ev.log(self._format_row(optimized, num_metrics))
        ev.log(self._format_row(stage1_fixed_max, num_metrics))
        ev.log("-" * len(header))
        for res in random_results:
            ev.log(self._format_row(res, num_metrics))
        ev.log("-" * len(header))

    def _format_row(self, result, num_metrics):
        ok = "Y" if result["feasible"] else "N"
        total_cost = self._format_fixed(result.get("total_cost"), width=8, precision=2)
        loss_delta = self._format_percent(
            result.get("loss_delta_pct_vs_baseline"), width=8, precision=2
        )
        p_delta = self._format_percent(
            result.get("p_delta_pct_vs_baseline"), width=8, precision=2
        )
        loss_var = self._format_sci(result.get("loss_var"), width=10)
        p_var = self._format_sci(result.get("p_var"), width=10)
        if num_metrics == 1:
            return (
                f"{result['name']:<25} | {ok:<3} | "
                f"{result['loss']:<8.4f} {result['p']:<8.4f} | "
                f"{loss_delta} {p_delta} | "
                f"{loss_var} {p_var} | "
                f"{total_cost}"
            )
        s_delta = self._format_percent(
            result.get("s_delta_pct_vs_baseline"), width=8, precision=2
        )
        s_var = self._format_sci(result.get("s_var"), width=10)
        return (
            f"{result['name']:<25} | {ok:<3} | "
            f"{result['loss']:<8.4f} {result['p']:<8.4f} {result['s']:<8.4f} | "
            f"{loss_delta} {p_delta} {s_delta} | "
            f"{loss_var} {p_var} {s_var} | "
            f"{total_cost}"
        )

    def _log_random_summary(self, metric_short_names, summary, selected, num_metrics):
        ev = self.evaluator
        ev.log("\nRandom Baseline Summary:")
        overall = summary.get("overall", {})
        if not overall:
            ev.log("  No random baselines generated.")
            return
        ev.log(
            "  Overall: "
            f"samples={overall['count']}, "
            f"constraint_ok={overall['feasible_rate']:.2%}, "
            f"selected_better_loss={overall['loss_win_rate']:.2%}, "
            f"selected_better_{metric_short_names[0]}={overall['primary_win_rate']:.2%}, "
            f"selected_dominates={overall['dominance_rate']:.2%}, "
            f"mean_dLoss={self._format_fixed(overall.get('loss_delta_mean'), width=1).strip()}, "
            f"mean_d{metric_short_names[0]}="
            f"{self._format_fixed(overall.get('primary_metric_delta_mean'), width=1).strip()}, "
            f"mean_totalC="
            f"{self._format_fixed(overall.get('total_cost_mean'), width=1, precision=2).strip()}"
        )
        if overall.get("secondary_win_rate") is not None:
            ev.log(
                f"  Overall selected_better_{metric_short_names[1]}="
                f"{overall['secondary_win_rate']:.2%}, "
                f"mean_d{metric_short_names[1]}="
                f"{self._format_fixed(overall.get('secondary_metric_delta_mean'), width=1).strip()}"
            )
        overall_var_parts = [
            f"varLoss={self._format_sci(overall.get('loss_eval_variance_mean'), width=1).strip()}",
            f"var{metric_short_names[0]}="
            f"{self._format_sci(overall.get('p_eval_variance_mean'), width=1).strip()}",
        ]
        if num_metrics > 1:
            overall_var_parts.append(
                f"var{metric_short_names[1]}="
                f"{self._format_sci(overall.get('s_eval_variance_mean'), width=1).strip()}"
            )
        ev.log("  Overall eval variance: " + " ".join(overall_var_parts))
        for family, fs in summary.get("by_family", {}).items():
            msg = (
                f"  {family:<14} samples={fs['count']:<3} "
                f"ok={fs['feasible_rate']:.2%} "
                f"loss_win={fs['loss_win_rate']:.2%} "
                f"{metric_short_names[0]}_win={fs['primary_win_rate']:.2%} "
                f"dominates={fs['dominance_rate']:.2%} "
                f"dLoss={self._format_fixed(fs.get('loss_delta_mean'), width=1).strip()} "
                f"d{metric_short_names[0]}="
                f"{self._format_fixed(fs.get('primary_metric_delta_mean'), width=1).strip()} "
                f"totalC={self._format_fixed(fs.get('total_cost_mean'), width=1, precision=2).strip()}"
            )
            if fs.get("secondary_win_rate") is not None:
                msg += (
                    f" {metric_short_names[1]}_win={fs['secondary_win_rate']:.2%}"
                    f" d{metric_short_names[1]}="
                    f"{self._format_fixed(fs.get('secondary_metric_delta_mean'), width=1).strip()}"
                )
            eval_var_parts = [
                f"varLoss={self._format_sci(fs.get('loss_eval_variance_mean'), width=1).strip()}",
                f"var{metric_short_names[0]}="
                f"{self._format_sci(fs.get('p_eval_variance_mean'), width=1).strip()}",
            ]
            if num_metrics > 1:
                eval_var_parts.append(
                    f"var{metric_short_names[1]}="
                    f"{self._format_sci(fs.get('s_eval_variance_mean'), width=1).strip()}"
                )
            msg += " " + " ".join(eval_var_parts)
            ev.log(msg)

    def _plot_results(
        self,
        metric_short_names,
        num_metrics,
        baseline,
        optimized,
        stage1_fixed_max,
        random_results,
        summary,
    ):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
            fig.suptitle(
                f"Unified Final Evaluation ({self.evaluator.dataset_key.upper()})",
                fontsize=14,
                fontweight="bold",
            )
            family_colors = _FAMILY_COLOR_MAP
            grouped: Dict[str, list] = {}
            for r in random_results:
                grouped.setdefault(r["family"], []).append(r)
            ordered_families = self._ordered_families(grouped)

            metric_panels = [
                ("Loss", "loss"),
                (metric_short_names[0], "p"),
                (
                    metric_short_names[1] if num_metrics > 1 else "Time (ms)",
                    "s" if num_metrics > 1 else "time_ms",
                ),
            ]

            for ax, (label, key) in zip(itertools.islice(axes.flat, 3), metric_panels):
                panel_xs = []
                for fam in ordered_families:
                    items = grouped[fam]
                    xs = []
                    ys = []
                    for it in items:
                        cost = it.get("total_cost")
                        if cost is None:
                            continue
                        xs.append(cost)
                        ys.append(it[key])
                    if xs:
                        panel_xs.extend(xs)
                        ax.scatter(
                            xs,
                            ys,
                            s=40,
                            alpha=0.75,
                            label=fam,
                            color=family_colors.get(fam, "#999999"),
                        )
                if optimized.get("total_cost") is not None:
                    panel_xs.append(optimized["total_cost"])
                    ax.scatter(
                        optimized["total_cost"],
                        optimized[key],
                        marker="*",
                        s=230,
                        color="#E45756",
                        label="Optimized",
                        zorder=5,
                    )
                if stage1_fixed_max.get("total_cost") is not None:
                    panel_xs.append(stage1_fixed_max["total_cost"])
                    ax.scatter(
                        stage1_fixed_max["total_cost"],
                        stage1_fixed_max[key],
                        marker="D",
                        s=120,
                        color="#B279A2",
                        label="Stage1Fixed+MaxSF",
                        zorder=4,
                    )
                if key in baseline:
                    ax.axhline(
                        baseline[key],
                        color="#666666",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.75,
                        label="Baseline",
                    )
                ax.set_title(f"{label} vs Total Cost")
                ax.set_xlabel("Total Cost (Stage-1 + Stage-2)")
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
                self._set_numeric_axis_limits(ax, panel_xs)
                ax.margins(x=0.08, y=0.08)
                ax.legend(loc="best", fontsize=8)


            ax = axes[1, 1]
            families = []
            feasible = []
            dominance = []
            for family, family_summary in summary.get("by_family", {}).items():
                families.append(family)
                feasible.append(family_summary["feasible_rate"])
                dominance.append(family_summary["dominance_rate"])
            if families:
                x = np.arange(len(families))
                width = 0.34
                ax.bar(x - width / 2, feasible, width=width,
                       label="Constraint OK", color="#72B7B2")
                ax.bar(x + width / 2, dominance, width=width,
                       label="Dominated by Optimized", color="#E45756")
                ax.set_xticks(x)
                ax.set_xticklabels(families, rotation=20, ha="right")
                ax.set_ylim(0.0, 1.05)
                ax.set_title("Random Baseline Summary")
                ax.set_ylabel("Rate")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No random results", ha="center", va="center")
                ax.set_title("Random Baseline Summary")

            plot_path = os.path.join(
                self.results_dir,
                f"final_eval_comparison_{self.evaluator.dataset_key}.png",
            )
            plt.savefig(plot_path, dpi=180)
            plt.close(fig)
            self.evaluator.log(f"Unified final-eval plot saved to: {plot_path}")
            return plot_path
        except Exception as exc:
            self.evaluator.log(f"[Warning] Failed to plot unified final-eval: {exc}")
            return None

    @staticmethod
    def _family_colors():
        return dict(_FAMILY_COLOR_MAP)

    def _ordered_families(self, grouped):
        ordered = [family for family in _FAMILY_COLOR_ORDER if family in grouped]
        ordered.extend(sorted(family for family in grouped.keys() if family not in _FAMILY_COLOR_MAP))
        return ordered

    @staticmethod
    def _set_numeric_axis_limits(ax, values):
        lo = None
        hi = None
        for value in values:
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                continue
            if lo is None:
                lo = value
                hi = value
            else:
                lo = min(lo, value)
                hi = max(hi, value)
        if lo is None:
            return
        if lo == hi:
            pad = max(0.5, abs(lo) * 0.01)
            ax.set_xlim(lo - pad, hi + pad)
        else:
            pad = (hi - lo) * 0.08
            ax.set_xlim(lo - pad, hi + pad)

    def _plot_variance_results(
        self,
        metric_short_names,
        num_metrics,
        baseline,
        optimized,
        stage1_fixed_max,
        random_results,
    ):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            family_colors = _FAMILY_COLOR_MAP
            random_grouped: Dict[str, list] = {}
            for result in random_results:
                random_grouped.setdefault(result["family"], []).append(result)
            random_families = self._ordered_families(random_grouped)

            special_markers = [
                ("Optimized", "Optimized", optimized, "*", 230, "#E45756"),
                (
                    "Stage1FixedMaxSF",
                    "Stage1Fixed+MaxSF",
                    stage1_fixed_max,
                    "D",
                    120,
                    "#B279A2",
                ),
            ]

            fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
            fig.suptitle(
                f"Final Evaluation Test Variance Comparison ({self.evaluator.dataset_key.upper()})",
                fontsize=14,
                fontweight="bold",
            )

            variance_panels = [
                ("Loss Variance vs Total Cost", "loss_var", "VarLoss"),
                (
                    f"{metric_short_names[0]} Variance vs Total Cost",
                    "p_var",
                    f"Var{metric_short_names[0]}",
                ),
            ]
            if num_metrics > 1:
                variance_panels.append(
                    (
                        f"{metric_short_names[1]} Variance vs Total Cost",
                        "s_var",
                        f"Var{metric_short_names[1]}",
                    )
                )
            else:
                variance_panels.append(("Time Variance vs Total Cost", "time_var", "VarTime"))

            for ax, (title, key, ylabel) in zip(itertools.islice(axes.flat, 3), variance_panels):
                panel_xs = []
                has_data = False
                for family in random_families:
                    items = random_grouped[family]
                    xs = []
                    ys = []
                    for item in items:
                        cost = item.get("total_cost")
                        if cost is None or key not in item:
                            continue
                        value = item.get(key)
                        if value is None:
                            continue
                        value = float(value)
                        if not np.isfinite(value):
                            continue
                        xs.append(float(cost))
                        ys.append(value)
                    if xs:
                        panel_xs.extend(xs)
                        has_data = True
                        ax.scatter(
                            xs,
                            ys,
                            s=40,
                            alpha=0.75,
                            label=family,
                            color=family_colors.get(family, "#999999"),
                        )
                for family, label, result, marker, size, color in special_markers:
                    value = result.get(key)
                    cost = result.get("total_cost")
                    if (
                        value is None
                        or cost is None
                        or not np.isfinite(float(value))
                        or not np.isfinite(float(cost))
                    ):
                        continue
                    panel_xs.append(float(cost))
                    has_data = True
                    ax.scatter(
                        float(cost),
                        float(value),
                        marker=marker,
                        s=size,
                        color=color,
                        label=label,
                        zorder=5 if family == "Optimized" else 4,
                    )
                baseline_value = baseline.get(key)
                if baseline_value is not None and np.isfinite(float(baseline_value)):
                    ax.axhline(
                        float(baseline_value),
                        color="#666666",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.75,
                        label="Baseline",
                    )
                if not has_data:
                    ax.text(0.5, 0.5, "No variance data", ha="center", va="center")
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Total Cost (Stage-1 + Stage-2)")
                ax.grid(True, alpha=0.3)
                self._set_numeric_axis_limits(ax, panel_xs)
                ax.margins(x=0.08, y=0.12)
                try:
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
                except Exception:
                    pass
                ax.legend(loc="best", fontsize=8)

            ax = axes[1, 1]
            variance_specs = [
                ("Loss", "loss_var"),
                (metric_short_names[0], "p_var"),
            ]
            if num_metrics > 1:
                variance_specs.append((metric_short_names[1], "s_var"))

            bar_groups = [("Baseline", [baseline]), ("Optimized", [optimized])]
            for family in random_families:
                bar_groups.append((family, random_grouped[family]))
            bar_groups.append(("Stage1Fixed+MaxSF", [stage1_fixed_max]))

            variance_rows = []
            for family, items in bar_groups:
                row = []
                has_any = False
                for _, key in variance_specs:
                    mean_value = self._mean_float_or_none(item.get(key) for item in items)
                    if mean_value is not None:
                        row.append(mean_value)
                        has_any = True
                    else:
                        row.append(0.0)
                if has_any:
                    variance_rows.append((family, row))

            if variance_rows:
                x = np.arange(len(variance_rows))
                width = min(0.25, 0.75 / max(1, len(variance_specs)))
                for idx, (metric_label, _) in enumerate(variance_specs):
                    offset = (idx - (len(variance_specs) - 1) / 2.0) * width
                    ax.bar(
                        x + offset,
                        [row[idx] for _, row in variance_rows],
                        width=width,
                        label=metric_label,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels([family for family, _ in variance_rows], rotation=20, ha="right")
                ax.set_title("Mean Test Variance by Group")
                ax.set_ylabel("Variance")
                ax.grid(True, axis="y", alpha=0.3)
                try:
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
                except Exception:
                    pass
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No variance data", ha="center", va="center")
                ax.set_title("Mean Test Variance by Group")

            plot_path = os.path.join(
                self.results_dir,
                f"final_eval_variance_{self.evaluator.dataset_key}.png",
            )
            plt.savefig(plot_path, dpi=180)
            plt.close(fig)
            self.evaluator.log(f"Unified final-eval variance plot saved to: {plot_path}")
            return plot_path
        except Exception as exc:
            self.evaluator.log(f"[Warning] Failed to plot unified final-eval variance: {exc}")
            return None

    def _save_results_json(
        self,
        selected_source,
        baseline_stage1_gelu,
        baseline_stage1_softmax,
        opt_gelu,
        opt_softmax,
        opt_noise_cfg,
        baseline_result,
        baseline_repeat,
        optimized_result,
        optimized_repeat,
        stage1_fixed_max_noise_result,
        stage1_fixed_max_noise_repeat,
        random_results,
        summary,
        selection_constraints,
        report_constraints,
    ):
        output = {
            "dataset": self.evaluator.dataset_key,
            "final_eval_split": getattr(
                self, "final_eval_split", FINAL_EVAL_SPLIT
            ),
            "dataset_protocol_hash": getattr(
                self.evaluator, "dataset_protocol_hash", None
            ),
            "selected_source": selected_source,
            "baseline_stage1": {
                "gelu": np.asarray(baseline_stage1_gelu, dtype=int).tolist(),
                "softmax": np.asarray(baseline_stage1_softmax, dtype=int).tolist(),
            },
            "optimized_stage1": {
                "gelu": np.asarray(opt_gelu, dtype=int).tolist(),
                "softmax": np.asarray(opt_softmax, dtype=int).tolist(),
            },
            "optimized_stage2": {
                key: np.asarray(opt_noise_cfg[key], dtype=int).tolist()
                for key in NOISE_SCALING_FACTOR_KEYS
            },
            "constraints": {
                "selection": {
                    "limit_loss": float(selection_constraints["loss"]),
                    "limit_primary_metric": float(selection_constraints["metric1"]),
                    "limit_secondary_metric": float(selection_constraints["metric2"]),
                },
                "report": {
                    "limit_loss": float(report_constraints["loss"]),
                    "limit_primary_metric": float(report_constraints["metric1"]),
                    "limit_secondary_metric": float(report_constraints["metric2"]),
                },
            },
            "baseline": to_jsonable(baseline_result),
            "optimized": to_jsonable(optimized_result),
            "stage1_fixed_max_scaling": to_jsonable(stage1_fixed_max_noise_result),
            "random_results": [to_jsonable(r) for r in random_results],
            "random_summary": summary,
            "evaluation_protocol": {
                "version": 4,
                "baseline": "single_clean_validation_full",
                "noisy_groups": "repeated_mean" if self.repeat_n > 1 else "single",
                "noisy_repeat_n": int(self.repeat_n),
                "variance_repeat_n": int(
                    self.repeat_n if self.repeat_n > 1 else self._variance_repeat_count()
                ),
                "variance_source": (
                    "full_validation_repeats"
                    if self.repeat_n > 1
                    else "fixed_probe_noise_trials"
                ),
                "variance_groups": ["optimized", "random", "stage1_fixed_max_scaling"],
                "random_groups": "enabled" if self.include_random_groups else "disabled",
                "random_group_seed": self.random_group_seed,
                "relative_metrics": "delta_vs_baseline",
                "cost_axis": "total_cost_stage1_plus_stage2",
            },
        }
        if baseline_repeat is not None:
            output["baseline_repeat_evaluation"] = baseline_repeat
        if optimized_repeat is not None:
            output["optimized_repeat_evaluation"] = optimized_repeat
        if stage1_fixed_max_noise_repeat is not None:
            output["stage1_fixed_max_scaling_repeat_evaluation"] = (
                stage1_fixed_max_noise_repeat
            )
        output_path = os.path.join(
            self.results_dir, f"final_eval_results_{self.evaluator.dataset_key}.json"
        )
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        self.evaluator.log(f"Unified final-eval summary saved to: {output_path}")
        return output_path


    def _variance_repeat_count(self):
        stage2_k = getattr(self.evaluator, "stage2_k_trials", None)
        try:
            stage2_k = int(stage2_k)
        except (TypeError, ValueError):
            stage2_k = 1
        return max(2, int(self.repeat_n), stage2_k)

    def _build_max_noise_config(self, total_layers):
        return {
            "input_noise_scaling_factors": np.full(
                total_layers, max(self.input_noise_allowed), dtype=int
            ),
            "wq_noise_scaling_factors": np.full(
                total_layers, max(self.weight_noise_allowed), dtype=int
            ),
            "wk_noise_scaling_factors": np.full(
                total_layers, max(self.weight_noise_allowed), dtype=int
            ),
            "wv_noise_scaling_factors": np.full(
                total_layers, max(self.weight_noise_allowed), dtype=int
            ),
            "wo_noise_scaling_factors": np.full(
                total_layers, max(self.weight_noise_allowed), dtype=int
            ),
            "wffn1_noise_scaling_factors": np.full(
                total_layers, max(self.wffn1_noise_allowed), dtype=int
            ),
            "wffn2_noise_scaling_factors": np.full(
                total_layers, max(self.weight_noise_allowed), dtype=int
            ),
        }

    @staticmethod
    def _cost_key(value):
        return int(round(float(value) * 2.0))

    @staticmethod
    def _stage2_cost_key(value):
        return int(round(float(value) * 40.0))

    @staticmethod
    def _int_tuple(values):
        return tuple(int(value) for value in np.asarray(values, dtype=int).reshape(-1))

    @staticmethod
    def _full_signature(gelu, softmax, noise_cfg):
        int_tuple = UnifiedFinalEvaluationModule._int_tuple
        return (
            int_tuple(gelu),
            int_tuple(softmax),
            tuple(int_tuple(noise_cfg[k]) for k in NOISE_SCALING_FACTOR_KEYS),
        )

    @staticmethod
    def _full_to_short(full_key):
        for s, f in SHORT_KEY_TO_FULL.items():
            if f == full_key:
                return s
        return full_key

    @staticmethod
    def _as_int_vector(values):
        if isinstance(values, np.ndarray):
            return np.asarray(values, dtype=int).reshape(-1)
        if isinstance(values, _LIST_OR_TUPLE_TYPES):
            return np.asarray(values, dtype=int).reshape(-1)
        return np.fromiter((int(value) for value in values), dtype=int)

    @staticmethod
    def _unsupported_int_values(values, allowed):
        allowed_set = {int(value) for value in allowed}
        invalid = set()
        for value in np.asarray(values, dtype=int).reshape(-1):
            int_value = int(value)
            if int_value not in allowed_set:
                invalid.add(int_value)
        return sorted(invalid)

    def _normalize_config_array(self, values, total_layers, default_degree, allowed, label):
        arr = self._as_int_vector(values)
        if arr.size < total_layers:
            pad = np.full(total_layers - arr.size, default_degree, dtype=int)
            self.evaluator.log(
                f"[Info] {label} length {arr.size} < total_layers={total_layers}; "
                f"padding with {default_degree}."
            )
            arr = np.concatenate([arr, pad])
        elif arr.size > total_layers:
            self.evaluator.log(
                f"[Info] {label} length {arr.size} > total_layers={total_layers}; truncating."
            )
            arr = arr[:total_layers].copy()
        invalid = self._unsupported_int_values(arr, allowed)
        if invalid:
            raise ValueError(f"{label} contains unsupported degrees: {invalid}")
        return arr

    def _normalize_noise_array(self, values, total_layers, label):
        arr = self._as_int_vector(values)
        short = self._full_to_short(label) if label in SHORT_KEY_TO_FULL.values() else label
        allowed = self._stage2_allowed(short if short in BREAKDOWN_KEYS else "wq")
        default_val = max(allowed)
        if arr.size < total_layers:
            pad = np.full(total_layers - arr.size, default_val, dtype=int)
            self.evaluator.log(
                f"[Info] {label} length {arr.size} < total_layers={total_layers}; "
                f"padding with {default_val}."
            )
            arr = np.concatenate([arr, pad])
        elif arr.size > total_layers:
            self.evaluator.log(
                f"[Info] {label} length {arr.size} > total_layers={total_layers}; truncating."
            )
            arr = arr[:total_layers].copy()
        invalid = self._unsupported_int_values(arr, allowed)
        if invalid:
            raise ValueError(
                f"{label} contains unsupported scaling factors {invalid}. Allowed: {list(allowed)}"
            )
        return arr

    def _is_feasible(self, loss, p, s, constraints):
        if loss > constraints["loss"]:
            return False
        if p < constraints["metric1"]:
            return False
        if self.evaluator.get_num_metrics() > 1 and s < constraints["metric2"]:
            return False
        return True

    def _dominates(self, selected, other):
        stage1_ok = selected["stage1_tot_c"] <= other["stage1_tot_c"]
        stage2_ok = selected["stage2_tot_c"] <= other["stage2_tot_c"]
        better_or_equal = (
            stage1_ok
            and stage2_ok
            and selected["loss"] <= other["loss"]
            and selected["p"] >= other["p"]
        )
        if self.evaluator.get_num_metrics() > 1:
            better_or_equal = better_or_equal and selected["s"] >= other["s"]
        if not better_or_equal:
            return False
        strict = (
            selected["stage1_tot_c"] < other["stage1_tot_c"]
            or selected["stage2_tot_c"] < other["stage2_tot_c"]
            or selected["loss"] < other["loss"]
            or selected["p"] > other["p"]
        )
        if self.evaluator.get_num_metrics() > 1:
            strict = strict or selected["s"] > other["s"]
        return strict

    def _ensure_results_dir(self):
        os.makedirs(self.results_dir, exist_ok=True)
