"""Unified final-evaluation module.

合并了旧版 stage1 (``FinalEvaluationModule``) 与 stage2 (``NoiseFinalEvaluationModule``)。
在一次 ``run()`` 里同时解析并评估两阶段的 optimized 配置，并生成 7 个组别的结果：

1. ``Baseline``      — stage-1 exact baseline (gelu=4, softmax=6)，无噪声。
2. ``Optimized``     — 两阶段各自的 optimized 配置 + 噪声注入。
3. ``Stage1Budget``  — stage1 在与 optimized **stage1 总代价相同** 下随机，
                       stage2 = optimized。
4. ``Stage2Budget``  — stage1 = optimized，stage2 在与 optimized
                       **stage2 总代价相同** 下随机。
5. ``Perm``          — stage1 (gelu/softmax) 与 stage2 (x/wq/wk/wv/wo/wffn1/wffn2)
                       各自按层随机排列 (permutation)，两阶段总代价自动保持不变。
6. ``Equiv``         — 每个类型的总代价分别与 optimized 匹配：
                       stage1 的 gelu、softmax；stage2 的 x/wq/wk/wv/wo/wffn1/wffn2。
7. ``Budget``        — stage1 总代价 = optimized_stage1 总代价，
                       stage2 总代价 = optimized_stage2 总代价（两阶段各自约束，
                       不是两者之和）。

除 Baseline 以外所有组别都使用 ``evaluate_model_with_attention_noise`` 进行评估。
"""

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


class UnifiedFinalEvaluationModule:
    """统一 final-eval：一次评估覆盖 stage1 + stage2 所有组别。"""

    def __init__(
        self,
        evaluator,
        config_source: str = "search",
        config_path: str = "glue_final_configs_best_ppo.json",
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
        self.config_path = config_path or "glue_final_configs_best_ppo.json"
        self.manual_stage1_gelu = manual_stage1_gelu
        self.manual_stage1_softmax = manual_stage1_softmax
        self.manual_stage2_noise = manual_stage2_noise
        self.random_seed = int(random_seed)
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

        from function_handler import (
            INPUT_NOISE_ALLOWED_SCALING_FACTORS,
            WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
            WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
        )

        self.input_noise_allowed = list(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
        self.weight_noise_allowed = list(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wffn1_noise_allowed = list(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Resolve optimized configs (stage1 + stage2 together).
        # ------------------------------------------------------------------
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

        # Cost references for optimized.
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

        # ------------------------------------------------------------------
        # Baseline group: stage-1 exact, NO noise. Clean baseline is
        # deterministic, so evaluate it once even when noisy groups repeat.
        # ------------------------------------------------------------------
        baseline_repeat = None
        ev.log("\n--- Baseline (Stage-1 Exact) : single deterministic evaluation ---")
        baseline_single = ev.evaluate_model(
            baseline_stage1_gelu,
            baseline_stage1_softmax,
            use_train=False,
            split="validation_full",
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

        # ------------------------------------------------------------------
        # Optimized group + random groups — all noise-evaluated.
        # ------------------------------------------------------------------
        eval_cache: Dict = {}

        def _noise_eval(gelu, softmax, noise_cfg, label, want_repeat):
            sig = (
                tuple(np.asarray(gelu, dtype=int).tolist()),
                tuple(np.asarray(softmax, dtype=int).tolist()),
                tuple(
                    tuple(np.asarray(noise_cfg[k], dtype=int).tolist())
                    for k in NOISE_SCALING_FACTOR_KEYS
                ),
            )
            repeat = None
            if want_repeat and self.repeat_n > 1:
                ev.log(f"\n--- {label} : N={self.repeat_n} 次重复评估 ---")
                summary = ev.evaluate_model_with_attention_noise_repeated(
                    gelu, softmax, repeats=self.repeat_n, use_train=False, **noise_cfg
                )
                repeat = {
                    "trials": [
                        {
                            "trial": i + 1,
                            "loss": float(t["loss"]),
                            "p": float(t["p"]),
                            "s": float(t["s"]),
                            "time_ms": float(t["time_ms"]),
                        }
                        for i, t in enumerate(summary["trials"])
                    ],
                    "stats": {
                        k: (float(v) if not isinstance(v, str) else v)
                        for k, v in summary.items()
                        if k != "trials"
                    },
                }
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
                return cached, repeat
            if sig in eval_cache:
                return eval_cache[sig], repeat
            loss, p, s, t = ev.evaluate_model_with_attention_noise(
                gelu, softmax, use_train=False, **noise_cfg
            )
            cached = {"loss": float(loss), "p": float(p), "s": float(s), "time_ms": float(t)}
            eval_cache[sig] = cached
            return cached, repeat

        def _build_noise_result(name, family, gelu, softmax, noise_cfg, want_repeat=False):
            single, repeat = _noise_eval(gelu, softmax, noise_cfg, name, want_repeat)
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
            return result, repeat

        # 2. Optimized
        optimized_result, optimized_repeat = _build_noise_result(
            "Optimized", "Optimized", opt_gelu, opt_softmax, opt_noise_cfg, want_repeat=True
        )

        # 3–7. Random groups.
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

        # Summaries / logs / outputs.
        summary = self._summarize_random_results(optimized_result, random_results, num_metrics)
        self._log_performance_table(
            metric_short_names,
            num_metrics,
            baseline_result,
            optimized_result,
            random_results,
        )
        self._log_random_summary(metric_short_names, summary, optimized_result)

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
            random_results,
            summary,
        )

        ev.apply_configuration(opt_gelu, opt_softmax)
        ev.clear_input_noise_configuration()
        ev.clear_weight_noise_configuration()

        return {
            "selected_source": selected_source,
            "opt_gelu": opt_gelu,
            "opt_softmax": opt_softmax,
            "opt_noise_config": opt_noise_cfg,
            "baseline_result": baseline_result,
            "optimized_result": optimized_result,
            "random_results": random_results,
            "random_summary": summary,
            "baseline_repeat": baseline_repeat,
            "optimized_repeat": optimized_repeat,
            "summary_path": summary_path,
            "plot_path": plot_path,
        }

    # ------------------------------------------------------------------
    # Config resolution (merged JSON: stage1 + stage2)
    # ------------------------------------------------------------------

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

        raise ValueError(
            f"Unsupported config_source '{self.config_source}'. Use: search / json / manual."
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

        raise ValueError(
            f"Unsupported config_source '{self.config_source}'. Use: search / json / manual."
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
        with open(self.config_path, "r", encoding="utf-8") as fh:
            config_map = json.load(fh)
        config_map.pop("_comment", None)

        total_layers = int(getattr(self.evaluator, "total_layers", 12) or 12)
        explicit = getattr(self.evaluator, "model_type", None)
        if explicit in ("bert-base", "bert-large", "gpt-2"):
            variant = explicit
        else:
            model = getattr(self.evaluator, "model", None)
            is_gpt2 = bool(
                model is not None
                and getattr(model, "transformer", None) is not None
                and hasattr(model.transformer, "h")
            )
            if is_gpt2:
                variant = "gpt-2"
            elif total_layers >= 24:
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

    # ------------------------------------------------------------------
    # Random group generation
    # ------------------------------------------------------------------

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
        rng = np.random.default_rng(self.random_seed)
        results: List[dict] = []

        # Pre-enumerate stage1 cost solutions for budget/equiv.
        gelu_solution_map = self._enumerate_cost_solutions(
            self.allowed_gelu_random, ev.GELU_COST_MAP, total_layers
        )
        softmax_solution_map = self._enumerate_cost_solutions(
            self.allowed_softmax, ev.SOFTMAX_COST_MAP, total_layers
        )
        opt_g_cost = float(np.sum([ev.GELU_COST_MAP[int(d)] for d in opt_gelu]))
        opt_s_cost = float(np.sum([ev.SOFTMAX_COST_MAP[int(d)] for d in opt_softmax]))

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

        # --- Stage1Budget: random stage1 @ stage1 total cost, stage2 = optimized ---
        if self.stage1_budget_trials > 0 and not np.any(opt_gelu == 0):
            ev.log(f"Generating {self.stage1_budget_trials} Stage1Budget configs...")
            for idx in range(self.stage1_budget_trials):
                pair = self._sample_stage1_total_cost(
                    rng, gelu_solution_map, softmax_solution_map, opt_stage1_tot_c
                )
                if pair is None:
                    continue
                g, sm = pair
                register(f"Stage1Budget_{idx + 1}", "Stage1Budget", g, sm, opt_noise_cfg)

        # --- Stage2Budget: stage1 = optimized, random stage2 @ stage2 total cost ---
        if self.stage2_budget_trials > 0:
            ev.log(f"Generating {self.stage2_budget_trials} Stage2Budget configs...")
            for idx in range(self.stage2_budget_trials):
                cfg = self._sample_stage2_total_cost(rng, opt_stage2_tot_c, total_layers)
                if cfg is None:
                    continue
                register(f"Stage2Budget_{idx + 1}", "Stage2Budget", opt_gelu, opt_softmax, cfg)

        # --- Perm: permute stage1 + stage2 independently ---
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

        # --- Equiv: per-type cost match on stage1 (gelu,softmax) + stage2 (7 types) ---
        if self.cost_equivalent_trials > 0 and not np.any(opt_gelu == 0):
            ev.log(f"Generating {self.cost_equivalent_trials} Equiv configs...")
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

        # --- Budget: stage1 total match + stage2 total match (separately) ---
        if self.budget_equivalent_trials > 0 and not np.any(opt_gelu == 0):
            ev.log(f"Generating {self.budget_equivalent_trials} Budget configs...")
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

    # ------------------------------------------------------------------
    # Stage-1 cost sampling
    # ------------------------------------------------------------------

    def _sample_stage1_total_cost(
        self, rng, gelu_solution_map, softmax_solution_map, target_total_cost
    ):
        target_key = self._cost_key(target_total_cost)
        feasible_pairs = []
        for g_key in gelu_solution_map.keys():
            s_key = target_key - g_key
            if s_key in softmax_solution_map:
                feasible_pairs.append((g_key, s_key))
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

    # ------------------------------------------------------------------
    # Stage-2 cost sampling
    # ------------------------------------------------------------------

    def _sample_stage2_equiv(self, rng, breakdown, total_layers):
        cfg = {}
        for short in BREAKDOWN_KEYS:
            full = SHORT_KEY_TO_FULL[short]
            target = breakdown[short]
            allowed = self._stage2_allowed(short)
            cost_map = self._stage2_cost_map(short)
            arr = self._stage2_cost_matched_array(rng, target, cost_map, allowed, total_layers)
            if arr is None:
                return None
            cfg[full] = arr
        return cfg

    def _sample_stage2_total_cost(self, rng, target_total, total_layers):
        target_key = self._stage2_cost_key(target_total)
        specs = []
        solution_maps = []
        for short in BREAKDOWN_KEYS:
            allowed = self._stage2_allowed(short)
            solution_map = self._enumerate_stage2_count_solutions(
                allowed,
                self._stage2_cost_map(short),
                total_layers,
            )
            specs.append((short, SHORT_KEY_TO_FULL[short], allowed))
            solution_maps.append(solution_map)

        count_choices = self._sample_stage2_count_combo(rng, solution_maps, target_key)
        if count_choices is None:
            return None

        cfg = {}
        for counts, (_, full, allowed) in zip(count_choices, specs):
            cfg[full] = self._counts_to_shuffled_config(allowed, counts, rng)
        if self._stage2_config_cost_key(cfg) != target_key:
            return None
        return cfg

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

    @staticmethod
    def _sample_stage2_count_combo(rng, solution_maps, target_key):
        suffix_possible = [set() for _ in range(len(solution_maps) + 1)]
        suffix_possible[-1].add(0)
        for idx in range(len(solution_maps) - 1, -1, -1):
            for key in solution_maps[idx].keys():
                for rest in suffix_possible[idx + 1]:
                    suffix_possible[idx].add(key + rest)
        if target_key not in suffix_possible[0]:
            return None

        remaining = target_key
        choices = []
        for idx, solution_map in enumerate(solution_maps):
            feasible_keys = [
                key
                for key in solution_map.keys()
                if remaining - key in suffix_possible[idx + 1]
            ]
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
                    if abs((curr - cost_map[old_v] + cost_map[int(d)]) - target_cost)
                    < abs(diff)
                ]
                cfg[idx] = int(rng.choice(moves if moves else values))
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

    # ------------------------------------------------------------------
    # Baseline (no-noise) repeated evaluation helper
    # ------------------------------------------------------------------

    def _run_clean_repeated(self, gelu, softmax, label):
        ev = self.evaluator
        ev.log(f"\n--- {label} : N={self.repeat_n} 次重复评估 ---")
        summary = ev.evaluate_model_repeated(
            gelu, softmax, repeats=self.repeat_n, use_train=False, split="validation_full"
        )
        trials = [
            {
                "trial": i + 1,
                "loss": float(t["loss"]),
                "p": float(t["p"]),
                "s": float(t["s"]),
                "time_ms": float(t["time_ms"]),
            }
            for i, t in enumerate(summary["trials"])
        ]
        stats = {k: (float(v) if not isinstance(v, str) else v) for k, v in summary.items() if k != "trials"}
        return {"trials": trials, "stats": stats}

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
                gelu, softmax, use_train=False, split="validation_full"
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
        return result

    # ------------------------------------------------------------------
    # Summary / logging / plotting / persistence
    # ------------------------------------------------------------------

    def _summarize_random_results(self, selected, random_results, num_metrics):
        summary: Dict = {"overall": {}, "by_family": {}}
        if not random_results:
            return summary
        grouped: Dict[str, list] = {}
        for res in random_results:
            grouped.setdefault(res["family"], []).append(res)

        all_feasible, all_loss_win, all_primary_win, all_secondary_win, all_dom = (
            [], [], [], [], []
        )
        for family, items in grouped.items():
            feasible = [1.0 if it["feasible"] else 0.0 for it in items]
            loss_win = [1.0 if selected["loss"] <= it["loss"] else 0.0 for it in items]
            p_win = [1.0 if selected["p"] >= it["p"] else 0.0 for it in items]
            s_win = (
                [1.0 if selected["s"] >= it["s"] else 0.0 for it in items]
                if num_metrics > 1
                else []
            )
            dom = [1.0 if self._dominates(selected, it) else 0.0 for it in items]
            summary["by_family"][family] = {
                "count": len(items),
                "feasible_rate": float(np.mean(feasible)),
                "loss_win_rate": float(np.mean(loss_win)),
                "primary_win_rate": float(np.mean(p_win)),
                "secondary_win_rate": float(np.mean(s_win)) if s_win else None,
                "dominance_rate": float(np.mean(dom)),
                "loss_mean": float(np.mean([it["loss"] for it in items])),
                "loss_std": float(np.std([it["loss"] for it in items])),
                "primary_metric_mean": float(np.mean([it["p"] for it in items])),
                "primary_metric_std": float(np.std([it["p"] for it in items])),
                "stage1_cost_mean": float(np.mean([it["stage1_tot_c"] for it in items])),
                "stage2_cost_mean": float(np.mean([it["stage2_tot_c"] for it in items])),
            }
            all_feasible.extend(feasible)
            all_loss_win.extend(loss_win)
            all_primary_win.extend(p_win)
            all_secondary_win.extend(s_win)
            all_dom.extend(dom)

        summary["overall"] = {
            "count": len(random_results),
            "feasible_rate": float(np.mean(all_feasible)) if all_feasible else 0.0,
            "loss_win_rate": float(np.mean(all_loss_win)) if all_loss_win else 0.0,
            "primary_win_rate": float(np.mean(all_primary_win)) if all_primary_win else 0.0,
            "secondary_win_rate": (
                float(np.mean(all_secondary_win)) if all_secondary_win else None
            ),
            "dominance_rate": float(np.mean(all_dom)) if all_dom else 0.0,
        }
        return summary

    def _log_performance_table(
        self, metric_short_names, num_metrics, baseline, optimized, random_results
    ):
        ev = self.evaluator
        ev.log("\nUnified Performance Comparison Table:")
        if num_metrics == 1:
            header = (
                f"{'Method':<25} | {'OK':<3} | "
                f"{'Loss':<8} {metric_short_names[0]:<8} | "
                f"{'S1 C':<7} {'S2 C':<7} {'S2 Spd':<7}"
            )
        else:
            header = (
                f"{'Method':<25} | {'OK':<3} | "
                f"{'Loss':<8} {metric_short_names[0]:<8} {metric_short_names[1]:<8} | "
                f"{'S1 C':<7} {'S2 C':<7} {'S2 Spd':<7}"
            )
        ev.log("-" * len(header))
        ev.log(header)
        ev.log("-" * len(header))
        ev.log(self._format_row(baseline, num_metrics))
        ev.log(self._format_row(optimized, num_metrics))
        ev.log("-" * len(header))
        for res in random_results:
            ev.log(self._format_row(res, num_metrics))
        ev.log("-" * len(header))

    def _format_row(self, result, num_metrics):
        ok = "Y" if result["feasible"] else "N"
        s2_na = result.get("show_cost_as_na") or result.get("stage2_tot_c") in (None, 0.0) and result.get("stage2_breakdown") is None
        s2c = "N/A" if s2_na else f"{result['stage2_tot_c']:<7.2f}"
        s2spd = "N/A" if s2_na or result.get("stage2_tot_spd") is None else f"{result['stage2_tot_spd']:<7.2f}"
        if num_metrics == 1:
            return (
                f"{result['name']:<25} | {ok:<3} | "
                f"{result['loss']:<8.4f} {result['p']:<8.4f} | "
                f"{result['stage1_tot_c']:<7.2f} {s2c} {s2spd}"
            )
        return (
            f"{result['name']:<25} | {ok:<3} | "
            f"{result['loss']:<8.4f} {result['p']:<8.4f} {result['s']:<8.4f} | "
            f"{result['stage1_tot_c']:<7.2f} {s2c} {s2spd}"
        )

    def _log_random_summary(self, metric_short_names, summary, selected):
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
            f"selected_dominates={overall['dominance_rate']:.2%}"
        )
        if overall.get("secondary_win_rate") is not None:
            ev.log(
                f"  Overall selected_better_{metric_short_names[1]}="
                f"{overall['secondary_win_rate']:.2%}"
            )
        for family, fs in summary.get("by_family", {}).items():
            msg = (
                f"  {family:<14} samples={fs['count']:<3} "
                f"ok={fs['feasible_rate']:.2%} "
                f"loss_win={fs['loss_win_rate']:.2%} "
                f"{metric_short_names[0]}_win={fs['primary_win_rate']:.2%} "
                f"dominates={fs['dominance_rate']:.2%}"
            )
            if fs.get("secondary_win_rate") is not None:
                msg += f" {metric_short_names[1]}_win={fs['secondary_win_rate']:.2%}"
            ev.log(msg)

    def _plot_results(
        self,
        metric_short_names,
        num_metrics,
        baseline,
        optimized,
        random_results,
        summary,
    ):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                f"Unified Final Evaluation ({self.evaluator.dataset_key.upper()})",
                fontsize=14,
                fontweight="bold",
            )
            family_colors = {
                "Stage1Budget": "#72B7B2",
                "Stage2Budget": "#EECA3B",
                "Perm": "#4C78A8",
                "Equiv": "#F58518",
                "Budget": "#54A24B",
            }
            grouped: Dict[str, list] = {}
            for r in random_results:
                grouped.setdefault(r["family"], []).append(r)

            # Loss vs stage2 cost
            ax = axes[0, 0]
            for fam, items in grouped.items():
                ax.scatter(
                    [it["stage2_tot_c"] for it in items],
                    [it["loss"] for it in items],
                    s=40, alpha=0.75, label=fam,
                    color=family_colors.get(fam, "#999999"),
                )
            ax.scatter(optimized["stage2_tot_c"], optimized["loss"],
                       marker="*", s=200, color="#E45756", label="Optimized")
            ax.set_title("Loss vs Stage-2 Cost")
            ax.set_xlabel("Stage-2 Noise Cost")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            # Primary metric vs stage2 cost
            ax = axes[0, 1]
            for fam, items in grouped.items():
                ax.scatter(
                    [it["stage2_tot_c"] for it in items],
                    [it["p"] for it in items],
                    s=40, alpha=0.75, label=fam,
                    color=family_colors.get(fam, "#999999"),
                )
            ax.scatter(optimized["stage2_tot_c"], optimized["p"],
                       marker="*", s=200, color="#E45756", label="Optimized")
            ax.set_title(f"{metric_short_names[0]} vs Stage-2 Cost")
            ax.set_xlabel("Stage-2 Noise Cost")
            ax.set_ylabel(metric_short_names[0])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            # Stage1 cost vs loss
            ax = axes[1, 0]
            for fam, items in grouped.items():
                ax.scatter(
                    [it["stage1_tot_c"] for it in items],
                    [it["loss"] for it in items],
                    s=40, alpha=0.75, label=fam,
                    color=family_colors.get(fam, "#999999"),
                )
            ax.scatter(optimized["stage1_tot_c"], optimized["loss"],
                       marker="*", s=200, color="#E45756", label="Optimized")
            ax.set_title("Loss vs Stage-1 Cost")
            ax.set_xlabel("Stage-1 Cost")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            # Summary bar chart
            ax = axes[1, 1]
            families = list(summary.get("by_family", {}).keys())
            if families:
                x = np.arange(len(families))
                feasible = [summary["by_family"][f]["feasible_rate"] for f in families]
                dominance = [summary["by_family"][f]["dominance_rate"] for f in families]
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

            plt.tight_layout()
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
        random_results,
        summary,
        selection_constraints,
        report_constraints,
    ):
        output = {
            "dataset": self.evaluator.dataset_key,
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
            "baseline": self._json_ready(baseline_result),
            "optimized": self._json_ready(optimized_result),
            "random_results": [self._json_ready(r) for r in random_results],
            "random_summary": summary,
            "evaluation_protocol": {
                "version": 2,
                "baseline": "single_clean_validation_full",
                "noisy_groups": "repeated_mean" if self.repeat_n > 1 else "single",
                "noisy_repeat_n": int(self.repeat_n),
            },
        }
        if baseline_repeat is not None:
            output["baseline_repeat_evaluation"] = baseline_repeat
        if optimized_repeat is not None:
            output["optimized_repeat_evaluation"] = optimized_repeat
        output_path = os.path.join(
            self.results_dir, f"final_eval_results_{self.evaluator.dataset_key}.json"
        )
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        self.evaluator.log(f"Unified final-eval summary saved to: {output_path}")
        return output_path

    def _json_ready(self, result):
        if result is None:
            return None
        out = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, np.bool_):
                out[k] = bool(v)
            elif isinstance(v, dict):
                out[k] = {
                    kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                    for kk, vv in v.items()
                }
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cost_key(value):
        return int(round(float(value) * 2.0))

    @staticmethod
    def _stage2_cost_key(value):
        return int(round(float(value) * 40.0))

    @staticmethod
    def _full_signature(gelu, softmax, noise_cfg):
        return (
            tuple(np.asarray(gelu, dtype=int).tolist()),
            tuple(np.asarray(softmax, dtype=int).tolist()),
            tuple(
                tuple(np.asarray(noise_cfg[k], dtype=int).tolist())
                for k in NOISE_SCALING_FACTOR_KEYS
            ),
        )

    @staticmethod
    def _full_to_short(full_key):
        for s, f in SHORT_KEY_TO_FULL.items():
            if f == full_key:
                return s
        return full_key

    def _normalize_config_array(self, values, total_layers, default_degree, allowed, label):
        arr = np.asarray(list(values), dtype=int).flatten()
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
        invalid = sorted(set(arr.tolist()) - set(allowed))
        if invalid:
            raise ValueError(f"{label} contains unsupported degrees: {invalid}")
        return arr

    def _normalize_noise_array(self, values, total_layers, label):
        arr = np.asarray(list(values), dtype=int).flatten()
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
        invalid = sorted(set(arr.tolist()) - set(allowed))
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


