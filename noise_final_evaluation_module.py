import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

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
WEIGHT_BREAKDOWN_KEYS = ("wq", "wk", "wv", "wo", "wffn1", "wffn2")

# 噪声随机消融实验支持的三种模式
RANDOM_MODE_X_ONLY = "x_only"               # 仅随机 X，W 与非线性层固定
RANDOM_MODE_X_W = "x_w"                     # 随机 X + 所有 W（默认，原始行为）
RANDOM_MODE_X_W_NONLINEAR = "x_w_nonlinear" # 随机 X + 所有 W + 非线性层（GELU/Softmax）
VALID_RANDOM_MODES = (
    RANDOM_MODE_X_ONLY,
    RANDOM_MODE_X_W,
    RANDOM_MODE_X_W_NONLINEAR,
)

# 非线性层（GELU/Softmax）允许的多项式阶数
GELU_DEGREE_CHOICES = (0, 1, 2, 4)
SOFTMAX_DEGREE_CHOICES = (2, 3, 4, 5, 6)


class NoiseFinalEvaluationModule:
    """第二阶段噪声 RL 最终评估独立模块。

    与 FinalEvaluationModule 结构一致，支持：
    - 噪声配置来源选择（search / json / manual）
    - 随机对照实验（Permutation + Cost-Equivalent + Budget-Equivalent）
    - N 次重复评估（测量方差）
    - 结果 JSON 保存与可视化绘图
    """

    def __init__(
        self,
        evaluator,
        config_source: str = "search",
        config_path: str = "glue_noise_configs_best_ppo.json",
        manual_noise_config: Optional[Dict[str, Sequence[int]]] = None,
        random_seed: int = 42,
        permutation_trials: int = 10,
        cost_equivalent_trials: int = 10,
        budget_equivalent_trials: int = 10,
        full_random_trials: int = 50,
        repeat_n: int = 1,
        results_dir: Optional[str] = None,
        random_mode: Optional[str] = None,
    ):
        self.evaluator = evaluator
        self.config_source = (config_source or "search").lower()
        self.config_path = config_path or "glue_noise_configs_best_ppo.json"
        self.manual_noise_config = manual_noise_config
        self.random_seed = int(random_seed)
        self.permutation_trials = max(0, int(permutation_trials))
        self.cost_equivalent_trials = max(0, int(cost_equivalent_trials))
        self.budget_equivalent_trials = max(0, int(budget_equivalent_trials))
        self.full_random_trials = max(0, int(full_random_trials))
        self.repeat_n = max(1, int(repeat_n))
        default_results_dir = getattr(
            evaluator,
            "noise_final_eval_dir",
            os.path.join("rl_results", "noise_final_evaluation"),
        )
        self.results_dir = results_dir or default_results_dir

        from function_handler import (
            INPUT_NOISE_ALLOWED_SCALING_FACTORS,
            WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
            WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
        )

        self.input_noise_allowed = list(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
        self.weight_noise_allowed = list(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wffn1_noise_allowed = list(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)

        # 噪声随机消融模式：参数 > 环境变量 NOISE_RANDOM_MODE > 默认 x_w
        mode = random_mode if random_mode is not None else os.environ.get(
            "NOISE_RANDOM_MODE", RANDOM_MODE_X_W
        )
        mode = (mode or RANDOM_MODE_X_W).strip().lower()
        if mode not in VALID_RANDOM_MODES:
            raise ValueError(
                f"Unsupported random_mode '{mode}'. "
                f"Valid: {VALID_RANDOM_MODES}"
            )
        self.random_mode = mode

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        search_best_noise_config: Optional[dict],
        search_status: Optional[str],
        fixed_gelu: np.ndarray,
        fixed_softmax: np.ndarray,
        baseline_noise_config: dict,
        baseline_tot_c: float,
        limit_loss: float,
        limit_p: float,
        limit_s: float,
    ) -> Dict[str, object]:
        self._ensure_results_dir()
        ev = self.evaluator
        num_metrics = ev.get_num_metrics()
        metric_short_names = ev.get_metric_short_names()
        eval_cache: Dict = {}
        exact_baseline_gelu, exact_baseline_softmax = (
            ev.get_stage1_exact_baseline_configuration()
        )
        selection_constraints = {
            "loss": float(limit_loss),
            "metric1": float(limit_p),
            "metric2": float(limit_s),
        }

        def _get_noise_eval(noise_cfg):
            # 允许 cfg 内携带非线性层覆盖（仅 nonlinear 消融模式使用）
            gelu_override = noise_cfg.pop("_gelu_override", None) if isinstance(noise_cfg, dict) else None
            softmax_override = noise_cfg.pop("_softmax_override", None) if isinstance(noise_cfg, dict) else None
            use_gelu = fixed_gelu if gelu_override is None else np.asarray(gelu_override, dtype=int)
            use_softmax = fixed_softmax if softmax_override is None else np.asarray(softmax_override, dtype=int)
            sig = self._noise_config_signature(noise_cfg, use_gelu, use_softmax)
            if sig in eval_cache:
                return eval_cache[sig]

            loss, p, s, t = ev.evaluate_model_with_attention_noise(
                use_gelu, use_softmax, use_train=False, **noise_cfg
            )
            tot_c, breakdown = ev.get_noise_simulated_cost(**noise_cfg)
            cached = {
                "loss": float(loss),
                "p": float(p),
                "s": float(s),
                "time_ms": float(t),
                "tot_c": float(tot_c),
                "breakdown": breakdown,
            }
            eval_cache[sig] = cached
            return cached

        def _build_noise_result(name, family, noise_cfg, constraints, repeat_results=None):
            cached = _get_noise_eval(noise_cfg)
            loss = float(
                repeat_results["stats"]["loss_mean"]
                if repeat_results is not None
                else cached["loss"]
            )
            p = float(
                repeat_results["stats"]["p_mean"]
                if repeat_results is not None
                else cached["p"]
            )
            s = float(
                repeat_results["stats"]["s_mean"]
                if repeat_results is not None
                else cached["s"]
            )
            time_ms = float(
                repeat_results["stats"]["time_mean_ms"]
                if repeat_results is not None
                else cached["time_ms"]
            )
            result = {
                "name": name,
                "family": family,
                "loss": loss,
                "p": p,
                "s": s,
                "time_ms": time_ms,
                "tot_c": float(cached["tot_c"]),
                "tot_spd": float(baseline_tot_c / (cached["tot_c"] + 1e-6)),
                "breakdown": cached["breakdown"],
                "noise_config": {
                    k: np.asarray(noise_cfg[k], dtype=int).copy()
                    for k in NOISE_SCALING_FACTOR_KEYS
                },
                "feasible": self._is_feasible(
                    loss,
                    p,
                    s,
                    constraints["loss"],
                    constraints["metric1"],
                    constraints["metric2"],
                ),
            }
            if repeat_results is not None:
                stats = repeat_results["stats"]
                result.update(
                    {
                        "evaluation_n": int(stats["n"]),
                        "loss_std": float(stats["loss_std"]),
                        "loss_min": float(stats["loss_min"]),
                        "loss_max": float(stats["loss_max"]),
                        "loss_range": float(stats["loss_range"]),
                        "p_std": float(stats["p_std"]),
                        "p_min": float(stats["p_min"]),
                        "p_max": float(stats["p_max"]),
                        "p_range": float(stats["p_range"]),
                        "s_std": float(stats["s_std"]),
                        "s_min": float(stats["s_min"]),
                        "s_max": float(stats["s_max"]),
                        "s_range": float(stats["s_range"]),
                        "time_std_ms": float(stats["time_std_ms"]),
                        "evaluation_protocol": f"repeated_mean_n={int(stats['n'])}",
                    }
                )
            return result

        def _run_repeated_clean_evaluation(
            gelu_degrees,
            softmax_degrees,
            label,
            repeats=None,
            log_trials=False,
        ):
            repeats = self.repeat_n if repeats is None else max(1, int(repeats))
            ev.log(f"\n--- {label} : N={repeats} 次重复评估 ---")
            summary = ev.evaluate_model_repeated(
                gelu_degrees,
                softmax_degrees,
                repeats=repeats,
                use_train=False,
                split="validation_full",
            )

            trials: List[dict] = []
            for idx, trial in enumerate(summary["trials"], start=1):
                entry = {
                    "trial": idx,
                    "loss": float(trial["loss"]),
                    "p": float(trial["p"]),
                    "s": float(trial["s"]),
                    "time_ms": float(trial["time_ms"]),
                }
                trials.append(entry)
                if log_trials:
                    msg = (
                        f"  Trial {idx}/{repeats}: "
                        f"Loss={entry['loss']:.4f}, {metric_short_names[0]}={entry['p']:.4f}"
                    )
                    if num_metrics > 1:
                        msg += f", {metric_short_names[1]}={entry['s']:.4f}"
                    msg += f", Time={entry['time_ms']:.1f}ms"
                    ev.log(msg)

            stats = {
                "split_name": summary["split_name"],
                "n": int(summary["n"]),
                "loss_mean": float(summary["loss_mean"]),
                "loss_std": float(summary["loss_std"]),
                "loss_min": float(summary["loss_min"]),
                "loss_max": float(summary["loss_max"]),
                "loss_range": float(summary["loss_range"]),
                "p_mean": float(summary["p_mean"]),
                "p_std": float(summary["p_std"]),
                "p_min": float(summary["p_min"]),
                "p_max": float(summary["p_max"]),
                "p_range": float(summary["p_range"]),
                "s_mean": float(summary["s_mean"]),
                "s_std": float(summary["s_std"]),
                "s_min": float(summary["s_min"]),
                "s_max": float(summary["s_max"]),
                "s_range": float(summary["s_range"]),
                "time_mean_ms": float(summary["time_mean_ms"]),
                "time_std_ms": float(summary["time_std_ms"]),
            }
            ev.log(
                f"  统计: Loss={stats['loss_mean']:.4f}±{stats['loss_std']:.6f} "
                f"[{stats['loss_min']:.4f}, {stats['loss_max']:.4f}]"
            )
            ev.log(
                f"  统计: {metric_short_names[0]}={stats['p_mean']:.4f}±{stats['p_std']:.6f} "
                f"[{stats['p_min']:.4f}, {stats['p_max']:.4f}]"
            )
            if num_metrics > 1:
                ev.log(
                    f"  统计: {metric_short_names[1]}={stats['s_mean']:.4f}±{stats['s_std']:.6f} "
                    f"[{stats['s_min']:.4f}, {stats['s_max']:.4f}]"
                )
            return {"trials": trials, "stats": stats}

        def _build_clean_result(
            name,
            family,
            gelu_degrees,
            softmax_degrees,
            constraints,
            repeat_results=None,
        ):
            if repeat_results is None:
                loss, p, s, time_ms = ev.evaluate_model(
                    gelu_degrees,
                    softmax_degrees,
                    use_train=False,
                    split="validation_full",
                )
            else:
                loss = repeat_results["stats"]["loss_mean"]
                p = repeat_results["stats"]["p_mean"]
                s = repeat_results["stats"]["s_mean"]
                time_ms = repeat_results["stats"]["time_mean_ms"]

            result = {
                "name": name,
                "family": family,
                "loss": float(loss),
                "p": float(p),
                "s": float(s),
                "time_ms": float(time_ms),
                "tot_c": 0.0,
                "tot_spd": None,
                "breakdown": None,
                "noise_config": None,
                "feasible": self._is_feasible(
                    float(loss),
                    float(p),
                    float(s),
                    constraints["loss"],
                    constraints["metric1"],
                    constraints["metric2"],
                ),
                "show_cost_as_na": True,
            }
            if repeat_results is not None:
                stats = repeat_results["stats"]
                result.update(
                    {
                        "evaluation_n": int(stats["n"]),
                        "loss_std": float(stats["loss_std"]),
                        "loss_min": float(stats["loss_min"]),
                        "loss_max": float(stats["loss_max"]),
                        "loss_range": float(stats["loss_range"]),
                        "p_std": float(stats["p_std"]),
                        "p_min": float(stats["p_min"]),
                        "p_max": float(stats["p_max"]),
                        "p_range": float(stats["p_range"]),
                        "s_std": float(stats["s_std"]),
                        "s_min": float(stats["s_min"]),
                        "s_max": float(stats["s_max"]),
                        "s_range": float(stats["s_range"]),
                        "time_std_ms": float(stats["time_std_ms"]),
                        "evaluation_protocol": f"repeated_mean_n={int(stats['n'])}",
                    }
                )
            return result

        ev.log("\n" + "=" * 60)
        ev.log("PHASE 5.5: NOISE RL FINAL EVALUATION (验证集)")
        ev.log(f"NOISE_EVAL_CONFIG_SOURCE={self.config_source}")
        ev.log(f"NOISE_RANDOM_MODE={self.random_mode}")
        if self.repeat_n > 1:
            ev.log(f"NOISE_EVAL_REPEAT_N={self.repeat_n}")
        ev.log(
            "Selection constraints (from stage-2 search protocol): "
            f"Loss<={selection_constraints['loss']:.4f}, "
            f"{metric_short_names[0]}>={selection_constraints['metric1']:.4f}"
            + (
                f", {metric_short_names[1]}>={selection_constraints['metric2']:.4f}"
                if num_metrics > 1
                else ""
            )
        )
        ev.log(
            "Noise cost reference (max-noise, for cost normalization only): "
            f"{baseline_tot_c:.2f}"
        )
        ev.log("=" * 60)

        # 1. Baseline (Stage-1 exact baseline, no noise)
        baseline_repeat_results = None
        if self.repeat_n > 1:
            baseline_repeat_results = _run_repeated_clean_evaluation(
                exact_baseline_gelu,
                exact_baseline_softmax,
                "Baseline (Stage-1 Exact)",
                repeats=self.repeat_n,
                log_trials=False,
            )
        baseline_reference = (
            baseline_repeat_results["stats"]
            if baseline_repeat_results is not None
            else ev.evaluate_model_repeated(
                exact_baseline_gelu,
                exact_baseline_softmax,
                repeats=1,
                use_train=False,
                split="validation_full",
            )
        )
        report_constraints = ev.build_constraint_limits_from_metrics(
            baseline_reference["loss_mean"] if "loss_mean" in baseline_reference else baseline_reference["loss"],
            baseline_reference["p_mean"] if "p_mean" in baseline_reference else baseline_reference["p"],
            baseline_reference["s_mean"] if "s_mean" in baseline_reference else baseline_reference["s"],
        )
        ev.log(
            "Final-eval report constraints (based on repeated baseline on validation_full): "
            f"Loss<={report_constraints['loss']:.4f}, "
            f"{metric_short_names[0]}>={report_constraints['metric1']:.4f}"
            + (
                f", {metric_short_names[1]}>={report_constraints['metric2']:.4f}"
                if num_metrics > 1
                else ""
            )
        )
        baseline_single_result = _build_clean_result(
            "Baseline (Stage-1 Exact)",
            "Baseline",
            exact_baseline_gelu,
            exact_baseline_softmax,
            report_constraints,
            repeat_results=None,
        )
        baseline_result = _build_clean_result(
            "Baseline (Stage-1 Exact)",
            "Baseline",
            exact_baseline_gelu,
            exact_baseline_softmax,
            report_constraints,
            repeat_results=baseline_repeat_results,
        )

        # 2. No-Noise 对照组（仅 GELU/Softmax，不注入噪声）
        ev.log(
            "Evaluating no-noise control for the fixed stage-1 configuration..."
        )
        no_noise_repeat_results = None
        if self.repeat_n > 1:
            no_noise_repeat_results = _run_repeated_clean_evaluation(
                fixed_gelu,
                fixed_softmax,
                "No-Noise (Fixed Stage-1)",
                repeats=self.repeat_n,
                log_trials=False,
            )
        no_noise_result = _build_clean_result(
            "No-Noise (Fixed Stage-1)",
            "Control",
            fixed_gelu,
            fixed_softmax,
            report_constraints,
            repeat_results=no_noise_repeat_results,
        )

        if self.config_source == "search" and search_best_noise_config is None:
            status = search_status or "no_search_config_available"
            message = (
                "Noise final evaluation skipped search-source selected configuration "
                "because stage-2 did not produce a stable feasible noise configuration."
            )
            ev.log(f"[Warning] {message}")
            summary_path = self._save_results_json(
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                selected_source="search",
                baseline_result=baseline_result,
                baseline_single_result=baseline_single_result,
                no_noise_result=no_noise_result,
                selected_result=None,
                selected_single_result=None,
                random_results=[],
                repeat_results=None,
                baseline_repeat_results=baseline_repeat_results,
                summary=None,
                selection_constraints=selection_constraints,
                report_constraints=report_constraints,
                status=status,
                message=message,
            )

            ev.apply_configuration(fixed_gelu, fixed_softmax)
            ev.clear_input_noise_configuration()
            ev.clear_weight_noise_configuration()

            return {
                "status": status,
                "message": message,
                "selected_source": "search",
                "selected_label": None,
                "selected_noise_config": None,
                "baseline_result": baseline_result,
                "no_noise_result": no_noise_result,
                "selected_result": None,
                "baseline_repeat_results": baseline_repeat_results,
                "random_results": [],
                "repeat_results": None,
                "random_summary": None,
                "eval_cache": eval_cache,
                "summary_path": summary_path,
                "plot_path": None,
            }

        # 3. 解析选中配置
        selected_noise_cfg, selected_name, selected_source = (
            self._resolve_selected_config(
                search_best_noise_config=search_best_noise_config,
                total_layers=ev.total_layers,
            )
        )
        selected_single_result = _build_noise_result(
            selected_name, "Selected", selected_noise_cfg, report_constraints
        )

        # 4. N 次重复评估
        repeat_results = None
        if self.repeat_n > 1:
            repeat_results = self._run_repeated_evaluation(
                fixed_gelu,
                fixed_softmax,
                selected_noise_cfg,
                selected_name,
                repeats=self.repeat_n,
                log_trials=True,
            )
        selected_result = _build_noise_result(
            selected_name,
            "Selected",
            selected_noise_cfg,
            report_constraints,
            repeat_results=repeat_results,
        )

        # 5. 随机对照实验
        random_results = self._generate_random_results(
            selected_noise_cfg=selected_noise_cfg,
            selected_result=selected_result,
            baseline_tot_c=baseline_tot_c,
            eval_noise_result=lambda name, family, noise_cfg: _build_noise_result(
                name,
                family,
                noise_cfg,
                report_constraints,
            ),
        )

        # 6. 随机对照统计汇总
        summary = self._summarize_random_results(
            selected_result, random_results, num_metrics
        )

        # 7. 日志输出
        self._log_configuration_details(selected_result, random_results)
        self._log_performance_table(
            metric_short_names, num_metrics,
            baseline_result, no_noise_result, selected_result, random_results,
        )
        if repeat_results is not None:
            self._log_repeat_results(
                selected_name,
                repeat_results,
                metric_short_names,
                num_metrics,
            )
        self._log_random_summary(metric_short_names, summary, selected_result)

        # 8. 保存 JSON
        summary_path = self._save_results_json(
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            selected_source=selected_source,
            baseline_result=baseline_result,
            baseline_single_result=baseline_single_result,
            no_noise_result=no_noise_result,
            selected_result=selected_result,
            selected_single_result=selected_single_result,
            random_results=random_results,
            repeat_results=repeat_results,
            baseline_repeat_results=baseline_repeat_results,
            summary=summary,
            selection_constraints=selection_constraints,
            report_constraints=report_constraints,
            status="ok",
            message=None,
        )

        # 9. 绘图
        plot_path = self._plot_results(
            metric_short_names, num_metrics,
            baseline_result, no_noise_result, selected_result,
            random_results, summary,
        )
        self._plot_full_random_distribution(
            metric_short_names, num_metrics,
            baseline_result, no_noise_result, selected_result,
            random_results,
        )

        # 10. 清理
        ev.apply_configuration(fixed_gelu, fixed_softmax)
        ev.clear_input_noise_configuration()
        ev.clear_weight_noise_configuration()

        return {
            "status": "ok",
            "message": None,
            "selected_source": selected_source,
            "selected_label": selected_name,
            "selected_noise_config": selected_noise_cfg,
            "baseline_result": baseline_result,
            "baseline_single_result": baseline_single_result,
            "baseline_repeat_results": baseline_repeat_results,
            "no_noise_result": no_noise_result,
            "selected_result": selected_result,
            "selected_single_result": selected_single_result,
            "random_results": random_results,
            "repeat_results": repeat_results,
            "random_summary": summary,
            "eval_cache": eval_cache,
            "summary_path": summary_path,
            "plot_path": plot_path,
        }

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    def _resolve_selected_config(self, search_best_noise_config, total_layers):
        if self.config_source == "search":
            if search_best_noise_config is None:
                raise ValueError(
                    "config_source='search' 需要来自噪声 RL 搜索的 best_noise_config。"
                )
            cfg = {}
            for key in NOISE_SCALING_FACTOR_KEYS:
                cfg[key] = self._normalize_noise_array(
                    search_best_noise_config[key], total_layers, key
                )
            return cfg, "Optimized (Noise PPO)", "search"

        if self.config_source == "json":
            json_cfg = self._load_dataset_config_from_json()
            cfg = {}
            for key in NOISE_SCALING_FACTOR_KEYS:
                short_key = self._full_key_to_short(key)
                raw = json_cfg.get(key) or json_cfg.get(short_key)
                if raw is None:
                    raise KeyError(
                        f"JSON 配置文件中缺少 '{key}' 或 '{short_key}'。"
                    )
                cfg[key] = self._normalize_noise_array(raw, total_layers, key)
            return cfg, "Selected (Saved Noise RL)", "json"

        if self.config_source == "manual":
            if self.manual_noise_config is None:
                raise ValueError(
                    "config_source='manual' 需要通过 --manual-noise-config 指定噪声配置。"
                )
            cfg = {}
            for key in NOISE_SCALING_FACTOR_KEYS:
                short_key = self._full_key_to_short(key)
                raw = (
                    self.manual_noise_config.get(key)
                    or self.manual_noise_config.get(short_key)
                )
                if raw is None:
                    raise KeyError(
                        f"manual_noise_config 中缺少 '{key}' 或 '{short_key}'。"
                    )
                cfg[key] = self._normalize_noise_array(raw, total_layers, key)
            return cfg, "Selected (Manual Noise)", "manual"

        raise ValueError(
            f"不支持的 config_source '{self.config_source}'。可选: search, json, manual。"
        )

    # ------------------------------------------------------------------
    # N-times repeated evaluation
    # ------------------------------------------------------------------

    def _run_repeated_evaluation_legacy(
        self,
        fixed_gelu,
        fixed_softmax,
        noise_cfg,
        selected_name,
        repeats=None,
        log_trials=True,
    ):
        ev = self.evaluator
        short_names = ev.get_metric_short_names()
        num_metrics = ev.get_num_metrics()
        ev.log(f"\n--- {selected_name} : N={self.repeat_n} 次重复评估 ---")
        trials: List[dict] = []
        for i in range(self.repeat_n):
            loss, p, s, t = ev.evaluate_model_with_attention_noise(
                fixed_gelu, fixed_softmax, use_train=False, **noise_cfg
            )
            trials.append(
                {
                    "trial": i + 1,
                    "loss": float(loss),
                    "p": float(p),
                    "s": float(s),
                    "time_ms": float(t),
                }
            )
            msg = (
                f"  Trial {i + 1}/{self.repeat_n}: "
                f"Loss={loss:.4f}, {short_names[0]}={p:.4f}"
            )
            if num_metrics > 1:
                msg += f", {short_names[1]}={s:.4f}"
            msg += f", Time={t:.1f}ms"
            ev.log(msg)

        losses = [r["loss"] for r in trials]
        ps = [r["p"] for r in trials]
        ss = [r["s"] for r in trials]
        stats = {
            "n": self.repeat_n,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "loss_min": float(np.min(losses)),
            "loss_max": float(np.max(losses)),
            "p_mean": float(np.mean(ps)),
            "p_std": float(np.std(ps)),
            "p_min": float(np.min(ps)),
            "p_max": float(np.max(ps)),
            "s_mean": float(np.mean(ss)),
            "s_std": float(np.std(ss)),
            "s_min": float(np.min(ss)),
            "s_max": float(np.max(ss)),
        }
        ev.log(
            f"  统计: Loss={stats['loss_mean']:.4f}±{stats['loss_std']:.6f} "
            f"[{stats['loss_min']:.4f}, {stats['loss_max']:.4f}]"
        )
        ev.log(
            f"  统计: {short_names[0]}={stats['p_mean']:.4f}±{stats['p_std']:.6f} "
            f"[{stats['p_min']:.4f}, {stats['p_max']:.4f}]"
        )
        if num_metrics > 1:
            ev.log(
                f"  统计: {short_names[1]}={stats['s_mean']:.4f}±{stats['s_std']:.6f} "
                f"[{stats['s_min']:.4f}, {stats['s_max']:.4f}]"
            )
        return {"trials": trials, "stats": stats}

    def _run_repeated_evaluation(
        self,
        fixed_gelu,
        fixed_softmax,
        noise_cfg,
        selected_name,
        repeats=None,
        log_trials=True,
    ):
        ev = self.evaluator
        short_names = ev.get_metric_short_names()
        num_metrics = ev.get_num_metrics()
        repeats = self.repeat_n if repeats is None else max(1, int(repeats))
        ev.log(f"\n--- {selected_name} : N={repeats} 次重复评估 ---")
        summary = ev.evaluate_model_with_attention_noise_repeated(
            fixed_gelu,
            fixed_softmax,
            repeats=repeats,
            use_train=False,
            **noise_cfg,
        )

        trials: List[dict] = []
        for idx, trial in enumerate(summary["trials"], start=1):
            entry = {
                "trial": idx,
                "loss": float(trial["loss"]),
                "p": float(trial["p"]),
                "s": float(trial["s"]),
                "time_ms": float(trial["time_ms"]),
            }
            trials.append(entry)
            if log_trials:
                msg = (
                    f"  Trial {idx}/{repeats}: "
                    f"Loss={entry['loss']:.4f}, {short_names[0]}={entry['p']:.4f}"
                )
                if num_metrics > 1:
                    msg += f", {short_names[1]}={entry['s']:.4f}"
                msg += f", Time={entry['time_ms']:.1f}ms"
                ev.log(msg)

        stats = {
            "split_name": summary["split_name"],
            "n": int(summary["n"]),
            "loss_mean": float(summary["loss_mean"]),
            "loss_std": float(summary["loss_std"]),
            "loss_min": float(summary["loss_min"]),
            "loss_max": float(summary["loss_max"]),
            "loss_range": float(summary["loss_range"]),
            "p_mean": float(summary["p_mean"]),
            "p_std": float(summary["p_std"]),
            "p_min": float(summary["p_min"]),
            "p_max": float(summary["p_max"]),
            "p_range": float(summary["p_range"]),
            "s_mean": float(summary["s_mean"]),
            "s_std": float(summary["s_std"]),
            "s_min": float(summary["s_min"]),
            "s_max": float(summary["s_max"]),
            "s_range": float(summary["s_range"]),
            "time_mean_ms": float(summary["time_mean_ms"]),
            "time_std_ms": float(summary["time_std_ms"]),
        }
        ev.log(
            f"  统计: Loss={stats['loss_mean']:.4f}±{stats['loss_std']:.6f} "
            f"[{stats['loss_min']:.4f}, {stats['loss_max']:.4f}]"
        )
        ev.log(
            f"  统计: {short_names[0]}={stats['p_mean']:.4f}±{stats['p_std']:.6f} "
            f"[{stats['p_min']:.4f}, {stats['p_max']:.4f}]"
        )
        if num_metrics > 1:
            ev.log(
                f"  统计: {short_names[1]}={stats['s_mean']:.4f}±{stats['s_std']:.6f} "
                f"[{stats['s_min']:.4f}, {stats['s_max']:.4f}]"
            )
        return {"trials": trials, "stats": stats}

    # ------------------------------------------------------------------
    # Random baseline generation
    # ------------------------------------------------------------------

    def _generate_random_results(
        self, selected_noise_cfg, selected_result, baseline_tot_c, eval_noise_result
    ):
        ev = self.evaluator
        rng = np.random.default_rng(self.random_seed)
        random_results: List[dict] = []
        seen = {self._noise_config_signature(selected_noise_cfg)}

        # --- Permutation ---
        ev.log(
            f"Generating {self.permutation_trials} permutation noise random configs..."
        )
        for idx in range(self.permutation_trials):
            perm_cfg = {}
            for key in NOISE_SCALING_FACTOR_KEYS:
                perm_cfg[key] = rng.permutation(selected_noise_cfg[key])
            sig = self._noise_config_signature(perm_cfg)
            if sig in seen:
                continue
            seen.add(sig)
            random_results.append(
                eval_noise_result(f"NoisePerm_{idx + 1}", "Perm", perm_cfg)
            )

        # --- Cost-Equivalent (逐类型精确匹配) ---
        ev.log(
            f"Generating {self.cost_equivalent_trials} cost-equivalent noise random configs..."
        )
        breakdown = selected_result["breakdown"]
        for idx in range(self.cost_equivalent_trials):
            equiv_cfg = self._sample_cost_equivalent(rng, breakdown, ev.total_layers, seen)
            if equiv_cfg is None:
                continue
            sig = self._noise_config_signature(equiv_cfg)
            seen.add(sig)
            random_results.append(
                eval_noise_result(f"NoiseEquiv_{idx + 1}", "Equiv", equiv_cfg)
            )

        # --- Budget-Equivalent (总代价匹配) ---
        ev.log(
            f"Generating {self.budget_equivalent_trials} budget-equivalent noise random configs..."
        )
        target_total = selected_result["tot_c"]
        for idx in range(self.budget_equivalent_trials):
            budget_cfg = self._sample_budget_equivalent(
                rng, target_total, ev.total_layers, seen
            )
            if budget_cfg is None:
                continue
            sig = self._noise_config_signature(budget_cfg)
            seen.add(sig)
            random_results.append(
                eval_noise_result(f"NoiseBudget_{idx + 1}", "Budget", budget_cfg)
            )

        # --- Full Random (按 random_mode 决定随机的范围) ---
        mode_desc = {
            RANDOM_MODE_X_ONLY: "仅 X 随机, W 与非线性层固定",
            RANDOM_MODE_X_W: "X 与所有 W 随机, 非线性层固定",
            RANDOM_MODE_X_W_NONLINEAR: "X + 所有 W + 非线性层(GELU/Softmax) 全部随机",
        }[self.random_mode]
        ev.log(
            f"Generating {self.full_random_trials} fully random noise configs "
            f"[mode={self.random_mode}: {mode_desc}]..."
        )
        for idx in range(self.full_random_trials):
            full_cfg = self._sample_full_random(
                rng, ev.total_layers, seen, selected_noise_cfg
            )
            if full_cfg is None:
                continue
            # 拆出可能存在的非线性层覆盖, 用于 signature
            gelu_ov = full_cfg.get("_gelu_override")
            softmax_ov = full_cfg.get("_softmax_override")
            sig_cfg = {k: full_cfg[k] for k in NOISE_SCALING_FACTOR_KEYS}
            sig = self._noise_config_signature(sig_cfg, gelu_ov, softmax_ov)
            seen.add(sig)
            random_results.append(
                eval_noise_result(f"NoiseFullRand_{idx + 1}", "FullRand", full_cfg)
            )

        return random_results

    def _sample_full_random(self, rng, total_layers, seen, selected_noise_cfg=None):
        """根据 self.random_mode 采样一组随机噪声配置。

        - x_only:        仅随机 X, 其余 W 沿用 selected_noise_cfg, 非线性层固定
        - x_w:           随机 X + 所有 W, 非线性层固定（默认行为）
        - x_w_nonlinear: 随机 X + 所有 W + GELU/Softmax 阶数
        """
        ev = self.evaluator
        for _ in range(20):
            cfg = {}
            for short_key in BREAKDOWN_KEYS:
                full_key = SHORT_KEY_TO_FULL[short_key]
                if self.random_mode == RANDOM_MODE_X_ONLY and short_key != "x":
                    # 沿用 selected 配置中对应 W 的值
                    if selected_noise_cfg is None:
                        return None
                    cfg[full_key] = np.asarray(
                        selected_noise_cfg[full_key], dtype=int
                    ).copy()
                else:
                    allowed = self._get_allowed(short_key)
                    cfg[full_key] = np.array(
                        rng.choice(allowed, size=total_layers), dtype=int
                    )

            gelu_ov = None
            softmax_ov = None
            if self.random_mode == RANDOM_MODE_X_W_NONLINEAR:
                gelu_ov = np.array(
                    rng.choice(GELU_DEGREE_CHOICES, size=total_layers), dtype=int
                )
                softmax_ov = np.array(
                    rng.choice(SOFTMAX_DEGREE_CHOICES, size=total_layers), dtype=int
                )
                cfg["_gelu_override"] = gelu_ov
                cfg["_softmax_override"] = softmax_ov

            sig_cfg = {k: cfg[k] for k in NOISE_SCALING_FACTOR_KEYS}
            sig = self._noise_config_signature(sig_cfg, gelu_ov, softmax_ov)
            if sig not in seen:
                return cfg
        return None

    def _sample_cost_equivalent(self, rng, breakdown, total_layers, seen):
        cfg = {}
        for short_key in BREAKDOWN_KEYS:
            full_key = SHORT_KEY_TO_FULL[short_key]
            target_cost = breakdown[short_key]
            allowed = self._get_allowed(short_key)
            cost_map = self._get_cost_map(short_key)
            arr = self._generate_cost_matched_array(
                rng, target_cost, cost_map, allowed, total_layers
            )
            if arr is None:
                return None
            cfg[full_key] = arr
        sig = self._noise_config_signature(cfg)
        if sig in seen:
            return None
        return cfg

    def _sample_budget_equivalent(self, rng, target_total, total_layers, seen):
        ev = self.evaluator
        for _ in range(200):
            cfg = {}
            total = 0.0
            for short_key in BREAKDOWN_KEYS:
                full_key = SHORT_KEY_TO_FULL[short_key]
                allowed = self._get_allowed(short_key)
                arr = np.array(
                    rng.choice(allowed, size=total_layers), dtype=int
                )
                cfg[full_key] = arr
                cost_map = self._get_cost_map(short_key)
                total += sum(cost_map[int(v)] for v in arr)

            for _ in range(1000):
                diff = total - target_total
                if abs(diff) < 0.5:
                    sig = self._noise_config_signature(cfg)
                    if sig not in seen:
                        return cfg
                    break
                bidx = int(rng.integers(0, 7))
                short_key = BREAKDOWN_KEYS[bidx]
                full_key = SHORT_KEY_TO_FULL[short_key]
                layer_idx = int(rng.integers(0, total_layers))
                allowed = self._get_allowed(short_key)
                cost_map = self._get_cost_map(short_key)
                old_val = int(cfg[full_key][layer_idx])
                old_cost = cost_map[old_val]
                best_val, best_diff = old_val, abs(diff)
                for v in allowed:
                    new_diff = abs(total - old_cost + cost_map[int(v)] - target_total)
                    if new_diff < best_diff:
                        best_diff = new_diff
                        best_val = v
                if best_val != old_val:
                    total = total - old_cost + cost_map[int(best_val)]
                    cfg[full_key][layer_idx] = best_val
        return None

    @staticmethod
    def _generate_cost_matched_array(rng, target_cost, cost_map, allowed, length):
        values = list(allowed)
        for _ in range(2000):
            cfg = np.array(rng.choice(values, size=length), dtype=int)
            for _ in range(500):
                curr = sum(cost_map[int(d)] for d in cfg)
                diff = curr - target_cost
                if abs(diff) < 1e-6:
                    return cfg
                idx = int(rng.integers(0, length))
                old_v = int(cfg[idx])
                moves = [
                    d
                    for d in values
                    if abs(
                        (curr - cost_map[old_v] + cost_map[int(d)]) - target_cost
                    )
                    < abs(diff)
                ]
                cfg[idx] = int(rng.choice(moves if moves else values))
        return np.asarray(rng.choice(values, size=length), dtype=int)

    # ------------------------------------------------------------------
    # Summarise random baselines
    # ------------------------------------------------------------------

    def _summarize_random_results(self, selected_result, random_results, num_metrics):
        summary: Dict = {"overall": {}, "by_family": {}}
        if not random_results:
            return summary

        grouped: Dict[str, list] = {}
        for res in random_results:
            grouped.setdefault(res["family"], []).append(res)

        all_feasible, all_loss_win, all_primary_win = [], [], []
        all_secondary_win, all_dominance = [], []

        for family, items in grouped.items():
            feasible_rate = float(
                np.mean([1.0 if it["feasible"] else 0.0 for it in items])
            )
            loss_win = float(
                np.mean(
                    [
                        1.0 if selected_result["loss"] <= it["loss"] else 0.0
                        for it in items
                    ]
                )
            )
            primary_win = float(
                np.mean(
                    [
                        1.0 if selected_result["p"] >= it["p"] else 0.0
                        for it in items
                    ]
                )
            )
            secondary_win = (
                float(
                    np.mean(
                        [
                            1.0 if selected_result["s"] >= it["s"] else 0.0
                            for it in items
                        ]
                    )
                )
                if num_metrics > 1
                else None
            )
            dominance_rate = float(
                np.mean(
                    [
                        1.0 if self._dominates(selected_result, it) else 0.0
                        for it in items
                    ]
                )
            )

            summary["by_family"][family] = {
                "count": len(items),
                "feasible_rate": feasible_rate,
                "loss_win_rate": loss_win,
                "primary_win_rate": primary_win,
                "secondary_win_rate": secondary_win,
                "dominance_rate": dominance_rate,
                "primary_metric_mean": float(np.mean([it["p"] for it in items])),
                "primary_metric_std": float(np.std([it["p"] for it in items])),
                "loss_mean": float(np.mean([it["loss"] for it in items])),
                "loss_std": float(np.std([it["loss"] for it in items])),
                "cost_mean": float(np.mean([it["tot_c"] for it in items])),
                "cost_std": float(np.std([it["tot_c"] for it in items])),
            }

            all_feasible.extend(
                [1.0 if it["feasible"] else 0.0 for it in items]
            )
            all_loss_win.extend(
                [
                    1.0 if selected_result["loss"] <= it["loss"] else 0.0
                    for it in items
                ]
            )
            all_primary_win.extend(
                [
                    1.0 if selected_result["p"] >= it["p"] else 0.0
                    for it in items
                ]
            )
            all_dominance.extend(
                [
                    1.0 if self._dominates(selected_result, it) else 0.0
                    for it in items
                ]
            )
            if num_metrics > 1:
                all_secondary_win.extend(
                    [
                        1.0 if selected_result["s"] >= it["s"] else 0.0
                        for it in items
                    ]
                )

        summary["overall"] = {
            "count": len(random_results),
            "feasible_rate": float(np.mean(all_feasible)) if all_feasible else 0.0,
            "loss_win_rate": float(np.mean(all_loss_win)) if all_loss_win else 0.0,
            "primary_win_rate": (
                float(np.mean(all_primary_win)) if all_primary_win else 0.0
            ),
            "secondary_win_rate": (
                float(np.mean(all_secondary_win)) if all_secondary_win else None
            ),
            "dominance_rate": (
                float(np.mean(all_dominance)) if all_dominance else 0.0
            ),
        }
        return summary

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_configuration_details(self, selected_result, random_results):
        ev = self.evaluator
        ev.log("\nNoise Final Configurations Details:")
        ncfg = selected_result.get("noise_config")
        if ncfg is not None:
            for key in NOISE_SCALING_FACTOR_KEYS:
                ev.log(
                    f"[{selected_result['name']}] {key}: "
                    f"{ncfg[key].tolist()}"
                )
        ev.log("\nNoise Random Configurations Details:")
        for res in random_results:
            ncfg_r = res.get("noise_config")
            if ncfg_r is not None:
                ev.log(
                    f"[{res['name']}] x: {ncfg_r['input_noise_scaling_factors'].tolist()}"
                )

    def _log_performance_table(
        self,
        metric_short_names,
        num_metrics,
        baseline_result,
        no_noise_result,
        selected_result,
        random_results,
    ):
        ev = self.evaluator
        ev.log("\nNoise Performance Comparison Table:")
        if num_metrics == 1:
            header = (
                f"{'Method':<25} | {'OK':<3} | "
                f"{'Loss':<8} {metric_short_names[0]:<8} | "
                f"{'dLoss%':<9} {'d' + metric_short_names[0] + '%':<11} | "
                f"{'Noise C':<8} {'Speedup':<8}"
            )
        else:
            header = (
                f"{'Method':<25} | {'OK':<3} | "
                f"{'Loss':<8} {metric_short_names[0]:<8} {metric_short_names[1]:<8} | "
                f"{'dLoss%':<9} {'d' + metric_short_names[0] + '%':<11} "
                f"{'d' + metric_short_names[1] + '%':<11} | "
                f"{'Noise C':<8} {'Speedup':<8}"
            )
        ev.log("-" * len(header))
        ev.log(header)
        ev.log("-" * len(header))
        ev.log(self._format_row(baseline_result, baseline_result, num_metrics))
        ev.log(
            self._format_row_no_noise(no_noise_result, baseline_result, num_metrics)
        )
        ev.log(self._format_row(selected_result, baseline_result, num_metrics))
        ev.log("-" * len(header))
        for res in random_results:
            ev.log(self._format_row(res, baseline_result, num_metrics))
        ev.log("-" * len(header))

    def _log_repeat_results_legacy(self, label, repeat_results, metric_short_names, num_metrics):
        ev = self.evaluator
        stats = repeat_results["stats"]
        ev.log(f"\nN={stats['n']} 次重复评估汇总:")
        ev.log(
            f"  Loss : {stats['loss_mean']:.4f} ± {stats['loss_std']:.6f} "
            f"[{stats['loss_min']:.4f}, {stats['loss_max']:.4f}]"
        )
        ev.log(
            f"  {metric_short_names[0]} : {stats['p_mean']:.4f} ± {stats['p_std']:.6f} "
            f"[{stats['p_min']:.4f}, {stats['p_max']:.4f}]"
        )
        if num_metrics > 1:
            ev.log(
                f"  {metric_short_names[1]} : {stats['s_mean']:.4f} ± {stats['s_std']:.6f} "
                f"[{stats['s_min']:.4f}, {stats['s_max']:.4f}]"
            )

    def _log_repeat_results(self, label, repeat_results, metric_short_names, num_metrics):
        ev = self.evaluator
        stats = repeat_results["stats"]
        ev.log(f"\n{label} repeated-eval summary (N={stats['n']}):")
        ev.log(
            f"  Loss : {stats['loss_mean']:.4f} ± {stats['loss_std']:.6f} "
            f"[{stats['loss_min']:.4f}, {stats['loss_max']:.4f}]"
        )
        ev.log(
            f"  {metric_short_names[0]} : {stats['p_mean']:.4f} ± {stats['p_std']:.6f} "
            f"[{stats['p_min']:.4f}, {stats['p_max']:.4f}]"
        )
        if num_metrics > 1:
            ev.log(
                f"  {metric_short_names[1]} : {stats['s_mean']:.4f} ± {stats['s_std']:.6f} "
                f"[{stats['s_min']:.4f}, {stats['s_max']:.4f}]"
            )

    def _log_random_summary(self, metric_short_names, summary, selected_result):
        ev = self.evaluator
        ev.log("\nNoise Random Baseline Summary:")
        overall = summary.get("overall", {})
        if not overall:
            ev.log("  No random baselines were generated.")
            return

        ev.log(
            "  Overall: "
            f"samples={overall['count']}, "
            f"constraint_ok={overall['feasible_rate']:.2%}, "
            f"selected_better_loss={overall['loss_win_rate']:.2%}, "
            f"selected_better_{metric_short_names[0]}={overall['primary_win_rate']:.2%}, "
            f"selected_dominates={overall['dominance_rate']:.2%}"
        )
        if overall.get("secondary_win_rate") is not None:
            ev.log(
                f"  Overall selected_better_{metric_short_names[1]}="
                f"{overall['secondary_win_rate']:.2%}"
            )

        for family, fstats in summary.get("by_family", {}).items():
            msg = (
                f"  {family:<7} samples={fstats['count']:<3} "
                f"constraint_ok={fstats['feasible_rate']:.2%} "
                f"selected_better_loss={fstats['loss_win_rate']:.2%} "
                f"selected_better_{metric_short_names[0]}={fstats['primary_win_rate']:.2%} "
                f"selected_dominates={fstats['dominance_rate']:.2%}"
            )
            if fstats.get("secondary_win_rate") is not None:
                msg += (
                    f" selected_better_{metric_short_names[1]}="
                    f"{fstats['secondary_win_rate']:.2%}"
                )
            ev.log(msg)

        ev.log(
            f"  Selected config: Loss={selected_result['loss']:.4f}, "
            f"{metric_short_names[0]}={selected_result['p']:.4f}"
            + (
                f", {metric_short_names[1]}={selected_result['s']:.4f}"
                if len(metric_short_names) > 1
                else ""
            )
            + f", NoiseCost={selected_result['tot_c']:.2f}"
        )

    # ------------------------------------------------------------------
    # Row formatting
    # ------------------------------------------------------------------

    def _format_row(self, result, base_result, num_metrics):
        loss_dp = (
            ((result["loss"] - base_result["loss"]) / (base_result["loss"] + 1e-8))
            * 100.0
        )
        m1_dp = (
            ((result["p"] - base_result["p"]) / (base_result["p"] + 1e-8)) * 100.0
        )
        ok = "Y" if result["feasible"] else "N"
        show_cost_as_na = (
            result.get("show_cost_as_na", False)
            or result.get("tot_c") is None
            or result.get("tot_spd") is None
        )
        cost_str = f"{'N/A':<8} {'N/A':<8}" if show_cost_as_na else (
            f"{result['tot_c']:<8.2f} {result['tot_spd']:<8.2f}"
        )
        if num_metrics == 1:
            return (
                f"{result['name']:<25} | {ok:<3} | "
                f"{result['loss']:<8.4f} {result['p']:<8.4f} | "
                f"{loss_dp:>8.2f}% {m1_dp:>10.2f}% | "
                f"{cost_str}"
            )
        m2_dp = (
            ((result["s"] - base_result["s"]) / (base_result["s"] + 1e-8)) * 100.0
        )
        return (
            f"{result['name']:<25} | {ok:<3} | "
            f"{result['loss']:<8.4f} {result['p']:<8.4f} {result['s']:<8.4f} | "
            f"{loss_dp:>8.2f}% {m1_dp:>10.2f}% {m2_dp:>10.2f}% | "
            f"{cost_str}"
        )

    def _format_row_no_noise(self, result, base_result, num_metrics):
        loss_dp = (
            ((result["loss"] - base_result["loss"]) / (base_result["loss"] + 1e-8))
            * 100.0
        )
        m1_dp = (
            ((result["p"] - base_result["p"]) / (base_result["p"] + 1e-8)) * 100.0
        )
        ok = "Y" if result["feasible"] else "N"
        if num_metrics == 1:
            return (
                f"{result['name']:<25} | {ok:<3} | "
                f"{result['loss']:<8.4f} {result['p']:<8.4f} | "
                f"{loss_dp:>8.2f}% {m1_dp:>10.2f}% | "
                f"{'N/A':<8} {'N/A':<8}"
            )
        m2_dp = (
            ((result["s"] - base_result["s"]) / (base_result["s"] + 1e-8)) * 100.0
        )
        return (
            f"{result['name']:<25} | {ok:<3} | "
            f"{result['loss']:<8.4f} {result['p']:<8.4f} {result['s']:<8.4f} | "
            f"{loss_dp:>8.2f}% {m1_dp:>10.2f}% {m2_dp:>10.2f}% | "
            f"{'N/A':<8} {'N/A':<8}"
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_results(
        self,
        metric_short_names,
        num_metrics,
        baseline_result,
        no_noise_result,
        selected_result,
        random_results,
        summary,
    ):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                f"Noise Final Evaluation ({self.evaluator.dataset_key.upper()})",
                fontsize=14,
                fontweight="bold",
            )

            family_colors = {
                "Perm": "#4C78A8",
                "Equiv": "#F58518",
                "Budget": "#54A24B",
                "FullRand": "#9D4EDD",
            }

            grouped: Dict[str, list] = {}
            for res in random_results:
                grouped.setdefault(res["family"], []).append(res)

            # --- Loss vs Cost ---
            ax = axes[0, 0]
            for family, items in grouped.items():
                ax.scatter(
                    [it["tot_c"] for it in items],
                    [it["loss"] for it in items],
                    s=42,
                    alpha=0.75,
                    label=family,
                    color=family_colors.get(family, "#999999"),
                )
            ax.scatter(
                baseline_result["tot_c"],
                baseline_result["loss"],
                marker="s",
                s=90,
                color="black",
                label="Baseline",
            )
            ax.scatter(
                selected_result["tot_c"],
                selected_result["loss"],
                marker="*",
                s=180,
                color="#E45756",
                label=selected_result["name"],
            )
            ax.set_title("Loss vs Noise Cost")
            ax.set_xlabel("Noise Cost")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            # --- Primary metric vs Cost ---
            ax = axes[0, 1]
            for family, items in grouped.items():
                ax.scatter(
                    [it["tot_c"] for it in items],
                    [it["p"] for it in items],
                    s=42,
                    alpha=0.75,
                    label=family,
                    color=family_colors.get(family, "#999999"),
                )
            ax.scatter(
                baseline_result["tot_c"],
                baseline_result["p"],
                marker="s",
                s=90,
                color="black",
                label="Baseline",
            )
            ax.scatter(
                selected_result["tot_c"],
                selected_result["p"],
                marker="*",
                s=180,
                color="#E45756",
                label=selected_result["name"],
            )
            ax.set_title(f"{metric_short_names[0]} vs Noise Cost")
            ax.set_xlabel("Noise Cost")
            ax.set_ylabel(metric_short_names[0])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            # --- Secondary metric or Boxplot ---
            if num_metrics > 1:
                ax = axes[1, 0]
                for family, items in grouped.items():
                    ax.scatter(
                        [it["tot_c"] for it in items],
                        [it["s"] for it in items],
                        s=42,
                        alpha=0.75,
                        label=family,
                        color=family_colors.get(family, "#999999"),
                    )
                ax.scatter(
                    baseline_result["tot_c"],
                    baseline_result["s"],
                    marker="s",
                    s=90,
                    color="black",
                    label="Baseline",
                )
                ax.scatter(
                    selected_result["tot_c"],
                    selected_result["s"],
                    marker="*",
                    s=180,
                    color="#E45756",
                    label=selected_result["name"],
                )
                ax.set_title(f"{metric_short_names[1]} vs Noise Cost")
                ax.set_xlabel("Noise Cost")
                ax.set_ylabel(metric_short_names[1])
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")
            else:
                ax = axes[1, 0]
                labels = list(grouped.keys())
                metric_values = [
                    [it["p"] for it in grouped[lbl]] for lbl in labels
                ]
                if metric_values:
                    box = ax.boxplot(
                        metric_values, labels=labels, patch_artist=True
                    )
                    for patch, lbl in zip(box["boxes"], labels):
                        patch.set_facecolor(
                            family_colors.get(lbl, "#BBBBBB")
                        )
                else:
                    ax.text(0.5, 0.5, "No random results", ha="center", va="center")
                ax.axhline(
                    baseline_result["p"],
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label="Baseline",
                )
                ax.axhline(
                    selected_result["p"],
                    color="#E45756",
                    linestyle="-",
                    linewidth=1.4,
                    label=selected_result["name"],
                )
                ax.set_title(
                    f"{metric_short_names[0]} Distribution by Random Family"
                )
                ax.set_ylabel(metric_short_names[0])
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend(loc="best")

            # --- Summary bar chart ---
            ax = axes[1, 1]
            families = list(summary.get("by_family", {}).keys())
            if families:
                x = np.arange(len(families))
                feasible = [
                    summary["by_family"][f]["feasible_rate"] for f in families
                ]
                dominance = [
                    summary["by_family"][f]["dominance_rate"] for f in families
                ]
                width = 0.34
                ax.bar(
                    x - width / 2,
                    feasible,
                    width=width,
                    label="Constraint OK rate",
                    color="#72B7B2",
                )
                ax.bar(
                    x + width / 2,
                    dominance,
                    width=width,
                    label="Dominated by selected",
                    color="#E45756",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(families)
                ax.set_ylim(0.0, 1.05)
                ax.set_title("Noise Random Baseline Summary")
                ax.set_ylabel("Rate")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend(loc="best")
            else:
                ax.text(0.5, 0.5, "No random results", ha="center", va="center")
                ax.set_title("Noise Random Baseline Summary")

            plt.tight_layout()
            plot_path = os.path.join(
                self.results_dir,
                f"noise_final_eval_comparison_{self.evaluator.dataset_key}.png",
            )
            plt.savefig(plot_path, dpi=180)
            plt.close(fig)
            self.evaluator.log(
                f"Noise final evaluation plot saved to: {plot_path}"
            )
            return plot_path
        except Exception as exc:
            self.evaluator.log(
                f"[Warning] Failed to plot noise final evaluation: {exc}"
            )
            return None

    def _plot_full_random_distribution(
        self,
        metric_short_names,
        num_metrics,
        baseline_result,
        no_noise_result,
        selected_result,
        random_results,
    ):
        try:
            full_rand = [r for r in random_results if r["family"] == "FullRand"]
            if not full_rand:
                return None

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            losses = np.array([r["loss"] for r in full_rand], dtype=float)
            ps = np.array([r["p"] for r in full_rand], dtype=float)
            ss = np.array([r["s"] for r in full_rand], dtype=float)
            costs = np.array([r["tot_c"] for r in full_rand], dtype=float)
            n = len(full_rand)

            ncols = 2
            nrows = 2 if num_metrics > 1 else 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9))
            fig.suptitle(
                f"Full-Random Noise Comparison (x & each w randomized, n={n}) "
                f"- {self.evaluator.dataset_key.upper()}",
                fontsize=14,
                fontweight="bold",
            )

            def _hist(ax, data, title, xlabel, sel_val, base_val, no_noise_val):
                ax.hist(data, bins=20, color="#9D4EDD", alpha=0.75,
                        edgecolor="white")
                ax.axvline(sel_val, color="#E45756", linestyle="-",
                           linewidth=2, label=f"Selected ({sel_val:.4f})")
                ax.axvline(base_val, color="black", linestyle="--",
                           linewidth=1.5, label=f"Baseline ({base_val:.4f})")
                if no_noise_val is not None:
                    ax.axvline(no_noise_val, color="#54A24B", linestyle=":",
                               linewidth=1.5,
                               label=f"No-Noise ({no_noise_val:.4f})")
                mean_v = float(np.mean(data))
                ax.axvline(mean_v, color="#4C78A8", linestyle="-.",
                           linewidth=1.2, label=f"Rand mean ({mean_v:.4f})")
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend(loc="best", fontsize=8)

            _hist(axes[0, 0], losses, "Loss distribution", "Loss",
                  selected_result["loss"], baseline_result["loss"],
                  no_noise_result["loss"] if no_noise_result else None)
            _hist(axes[0, 1], ps,
                  f"{metric_short_names[0]} distribution",
                  metric_short_names[0],
                  selected_result["p"], baseline_result["p"],
                  no_noise_result["p"] if no_noise_result else None)

            ax = axes[1, 0]
            ax.scatter(costs, losses, s=42, alpha=0.75, color="#9D4EDD",
                       label=f"FullRand (n={n})")
            ax.scatter(selected_result["tot_c"], selected_result["loss"],
                       marker="*", s=200, color="#E45756",
                       label=selected_result["name"])
            ax.scatter(baseline_result["tot_c"], baseline_result["loss"],
                       marker="s", s=90, color="black", label="Baseline")
            ax.set_title("Loss vs Noise Cost")
            ax.set_xlabel("Noise Cost")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            ax = axes[1, 1]
            if num_metrics > 1:
                _hist(ax, ss,
                      f"{metric_short_names[1]} distribution",
                      metric_short_names[1],
                      selected_result["s"], baseline_result["s"],
                      no_noise_result["s"] if no_noise_result else None)
            else:
                ax.scatter(costs, ps, s=42, alpha=0.75, color="#9D4EDD",
                           label=f"FullRand (n={n})")
                ax.scatter(selected_result["tot_c"], selected_result["p"],
                           marker="*", s=200, color="#E45756",
                           label=selected_result["name"])
                ax.scatter(baseline_result["tot_c"], baseline_result["p"],
                           marker="s", s=90, color="black", label="Baseline")
                ax.set_title(f"{metric_short_names[0]} vs Noise Cost")
                ax.set_xlabel("Noise Cost")
                ax.set_ylabel(metric_short_names[0])
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)

            plt.tight_layout()
            plot_path = os.path.join(
                self.results_dir,
                f"noise_final_eval_full_random_{self.evaluator.dataset_key}.png",
            )
            plt.savefig(plot_path, dpi=180)
            plt.close(fig)
            self.evaluator.log(
                f"Full-random noise distribution plot saved to: {plot_path}"
            )
            return plot_path
        except Exception as exc:
            self.evaluator.log(
                f"[Warning] Failed to plot full-random noise distribution: {exc}"
            )
            return None

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def _save_results_json(
        self,
        fixed_gelu,
        fixed_softmax,
        selected_source,
        baseline_result,
        baseline_single_result,
        no_noise_result,
        selected_result,
        selected_single_result,
        random_results,
        repeat_results,
        baseline_repeat_results,
        summary,
        selection_constraints,
        report_constraints,
        status,
        message,
    ):
        output = {
            "dataset": self.evaluator.dataset_key,
            "status": status,
            "message": message,
            "selected_source": selected_source,
            "fixed_stage1_config": {
                "gelu": np.asarray(fixed_gelu, dtype=int).tolist(),
                "softmax": np.asarray(fixed_softmax, dtype=int).tolist(),
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
                "limit_loss": float(selection_constraints["loss"]),
                "limit_primary_metric": float(selection_constraints["metric1"]),
                "limit_secondary_metric": float(selection_constraints["metric2"]),
            },
            "baseline": self._json_ready(baseline_result),
            "baseline_single": self._json_ready(baseline_single_result),
            "no_noise": self._json_ready(no_noise_result),
            "selected": self._json_ready(selected_result),
            "selected_single": self._json_ready(selected_single_result),
            "random_results": [self._json_ready(r) for r in random_results],
            "random_summary": summary,
        }
        if repeat_results is not None:
            output["repeat_evaluation"] = {
                "stats": repeat_results["stats"],
                "trials": repeat_results["trials"],
            }
        if baseline_repeat_results is not None:
            output["baseline_repeat_evaluation"] = {
                "stats": baseline_repeat_results["stats"],
                "trials": baseline_repeat_results["trials"],
            }

        output_path = os.path.join(
            self.results_dir,
            f"noise_final_eval_results_{self.evaluator.dataset_key}.json",
        )
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        self.evaluator.log(f"Noise final evaluation summary saved to: {output_path}")
        return output_path

    def _json_ready(self, result):
        if result is None:
            return None
        out = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, np.bool_):
                out[key] = bool(value)
            elif isinstance(value, dict):
                out[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                out[key] = value
        return out

    # ------------------------------------------------------------------
    # JSON config loading
    # ------------------------------------------------------------------

    def _load_dataset_config_from_json(self):
        with open(self.config_path, "r", encoding="utf-8") as fh:
            config_map = json.load(fh)
        config_map.pop("_comment", None)

        total_layers = int(getattr(self.evaluator, "total_layers", 12) or 12)
        explicit_variant = getattr(self.evaluator, "model_type", None)
        if explicit_variant in ("bert-base", "bert-large", "gpt-2"):
            variant_key = explicit_variant
        else:
            model = getattr(self.evaluator, "model", None)
            is_gpt2 = bool(
                model is not None
                and getattr(model, "transformer", None) is not None
                and hasattr(model.transformer, "h")
            )
            if is_gpt2:
                variant_key = "gpt-2"
            elif total_layers >= 24:
                variant_key = "bert-large"
            else:
                variant_key = "bert-base"

        if variant_key in config_map and isinstance(config_map[variant_key], dict):
            section = config_map[variant_key]
        elif any(key in config_map for key in ("bert-base", "bert-large", "gpt-2")):
            raise KeyError(
                f"模型变体 '{variant_key}' (total_layers={total_layers}) "
                f"在噪声配置文件 '{self.config_path}' 中未找到。"
            )
        else:
            if variant_key != "bert-base":
                raise KeyError(
                    f"噪声配置文件 '{self.config_path}' 使用的是旧版扁平 schema，"
                    f"仅支持 bert-base；请为 total_layers={total_layers} 添加 'bert-large' 段。"
                )
            section = config_map

        ds_key = self.evaluator.dataset_key
        if ds_key not in section:
            raise KeyError(
                f"数据集 '{ds_key}' 在噪声配置文件 '{self.config_path}' 的 '{variant_key}' 段中未找到。"
            )
        return section[ds_key]

    # ------------------------------------------------------------------
    # Array normalisation
    # ------------------------------------------------------------------

    def _normalize_noise_array(self, values, total_layers, label):
        arr = np.asarray(list(values), dtype=int).flatten()
        allowed = self._get_allowed_for_key(label)
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
                f"[Info] {label} length {arr.size} > total_layers={total_layers}; "
                f"truncating to {total_layers}."
            )
            arr = arr[:total_layers].copy()

        invalid = sorted(set(arr.tolist()) - set(allowed))
        if invalid:
            raise ValueError(
                f"{label} contains unsupported scaling factors {invalid}. "
                f"Allowed values: {list(allowed)}"
            )
        return arr

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _get_allowed(self, short_key):
        if short_key == "x":
            return self.input_noise_allowed
        if short_key == "wffn1":
            return self.wffn1_noise_allowed
        return self.weight_noise_allowed

    def _get_allowed_for_key(self, full_key):
        if "input" in full_key or full_key == "x":
            return self.input_noise_allowed
        if full_key in ("wffn1", "wffn1_noise_scaling_factors"):
            return self.wffn1_noise_allowed
        return self.weight_noise_allowed

    def _get_cost_map(self, short_key):
        ev = self.evaluator
        if short_key == "x":
            return ev.INPUT_NOISE_COST_MAP
        if short_key == "wffn1":
            return ev.WFFN1_NOISE_COST_MAP
        return ev.WEIGHT_NOISE_COST_MAP

    @staticmethod
    def _full_key_to_short(full_key):
        for short, full in SHORT_KEY_TO_FULL.items():
            if full == full_key:
                return short
        return full_key

    @staticmethod
    def _noise_config_signature(noise_cfg, gelu_override=None, softmax_override=None):
        base = tuple(
            tuple(np.asarray(noise_cfg[key], dtype=int).tolist())
            for key in NOISE_SCALING_FACTOR_KEYS
        )
        if gelu_override is None and softmax_override is None:
            return base
        gelu_t = tuple(np.asarray(gelu_override, dtype=int).tolist()) if gelu_override is not None else ()
        softmax_t = tuple(np.asarray(softmax_override, dtype=int).tolist()) if softmax_override is not None else ()
        return (base, gelu_t, softmax_t)

    def _is_feasible(self, loss, p, s, limit_loss, limit_p, limit_s):
        if loss > limit_loss:
            return False
        if p < limit_p:
            return False
        if self.evaluator.get_num_metrics() > 1 and s < limit_s:
            return False
        return True

    def _dominates(self, selected, other):
        better_or_equal = (
            selected["tot_c"] <= other["tot_c"]
            and selected["loss"] <= other["loss"]
            and selected["p"] >= other["p"]
        )
        if self.evaluator.get_num_metrics() > 1:
            better_or_equal = better_or_equal and selected["s"] >= other["s"]
        if not better_or_equal:
            return False
        strict = (
            selected["tot_c"] < other["tot_c"]
            or selected["loss"] < other["loss"]
            or selected["p"] > other["p"]
        )
        if self.evaluator.get_num_metrics() > 1:
            strict = strict or selected["s"] > other["s"]
        return strict

    def _ensure_results_dir(self):
        os.makedirs(self.results_dir, exist_ok=True)
