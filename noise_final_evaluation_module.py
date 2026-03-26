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
        repeat_n: int = 1,
        results_dir: Optional[str] = None,
    ):
        self.evaluator = evaluator
        self.config_source = (config_source or "search").lower()
        self.config_path = config_path or "glue_noise_configs_best_ppo.json"
        self.manual_noise_config = manual_noise_config
        self.random_seed = int(random_seed)
        self.permutation_trials = max(0, int(permutation_trials))
        self.cost_equivalent_trials = max(0, int(cost_equivalent_trials))
        self.budget_equivalent_trials = max(0, int(budget_equivalent_trials))
        self.repeat_n = max(1, int(repeat_n))
        self.results_dir = results_dir or os.path.join(
            "experiment_results", "noise_final_evaluation"
        )

        from function_handler import (
            INPUT_NOISE_ALLOWED_SCALING_FACTORS,
            WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
            WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
        )

        self.input_noise_allowed = list(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
        self.weight_noise_allowed = list(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wffn1_noise_allowed = list(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)

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

        def eval_noise_result(name, family, noise_cfg):
            sig = self._noise_config_signature(noise_cfg)
            if sig in eval_cache:
                loss, p, s, t = eval_cache[sig]
            else:
                loss, p, s, t = ev.evaluate_model_with_attention_noise(
                    fixed_gelu, fixed_softmax, use_train=False, **noise_cfg
                )
                eval_cache[sig] = (loss, p, s, t)
            tot_c, breakdown = ev.get_noise_simulated_cost(**noise_cfg)
            return {
                "name": name,
                "family": family,
                "loss": float(loss),
                "p": float(p),
                "s": float(s),
                "time_ms": float(t),
                "tot_c": float(tot_c),
                "tot_spd": float(baseline_tot_c / (tot_c + 1e-6)),
                "breakdown": breakdown,
                "noise_config": {
                    k: np.asarray(noise_cfg[k], dtype=int).copy()
                    for k in NOISE_SCALING_FACTOR_KEYS
                },
                "feasible": self._is_feasible(
                    loss, p, s, limit_loss, limit_p, limit_s
                ),
            }

        ev.log("\n" + "=" * 60)
        ev.log("PHASE 5.5: NOISE RL FINAL EVALUATION (验证集)")
        ev.log(f"NOISE_EVAL_CONFIG_SOURCE={self.config_source}")
        if self.repeat_n > 1:
            ev.log(f"NOISE_EVAL_REPEAT_N={self.repeat_n}")
        ev.log("=" * 60)

        # 1. Baseline (Max Scaling)
        baseline_result = eval_noise_result(
            "Baseline (Max Scaling)", "Baseline", baseline_noise_config
        )

        # 2. No-Noise 对照组（仅 GELU/Softmax，不注入噪声）
        ev.log(
            "Evaluating No-Noise control (GELU/Softmax only, no noise injection)..."
        )
        no_noise_loss, no_noise_p, no_noise_s, no_noise_t = ev.evaluate_model(
            fixed_gelu, fixed_softmax, use_train=False
        )
        no_noise_result = {
            "name": "No-Noise (Exact)",
            "family": "Control",
            "loss": float(no_noise_loss),
            "p": float(no_noise_p),
            "s": float(no_noise_s),
            "time_ms": float(no_noise_t),
            "tot_c": float(baseline_tot_c),
            "tot_spd": 1.0,
            "breakdown": None,
            "noise_config": None,
            "feasible": self._is_feasible(
                no_noise_loss, no_noise_p, no_noise_s,
                limit_loss, limit_p, limit_s,
            ),
        }

        if self.config_source == "search" and search_best_noise_config is None:
            status = search_status or "no_search_config_available"
            message = (
                "Noise final evaluation skipped search-source selected configuration "
                "because stage-2 did not produce a stable feasible noise configuration."
            )
            ev.log(f"[Warning] {message}")
            summary_path = self._save_results_json(
                selected_source="search",
                baseline_result=baseline_result,
                no_noise_result=no_noise_result,
                selected_result=None,
                random_results=[],
                repeat_results=None,
                summary=None,
                limit_loss=limit_loss,
                limit_p=limit_p,
                limit_s=limit_s,
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
        selected_result = eval_noise_result(
            selected_name, "Selected", selected_noise_cfg
        )

        # 4. N 次重复评估
        repeat_results = None
        if self.repeat_n > 1:
            repeat_results = self._run_repeated_evaluation(
                fixed_gelu, fixed_softmax, selected_noise_cfg, selected_name
            )

        # 5. 随机对照实验
        random_results = self._generate_random_results(
            selected_noise_cfg=selected_noise_cfg,
            selected_result=selected_result,
            baseline_tot_c=baseline_tot_c,
            eval_noise_result=eval_noise_result,
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
                repeat_results, metric_short_names, num_metrics
            )
        self._log_random_summary(metric_short_names, summary, selected_result)

        # 8. 保存 JSON
        summary_path = self._save_results_json(
            selected_source=selected_source,
            baseline_result=baseline_result,
            no_noise_result=no_noise_result,
            selected_result=selected_result,
            random_results=random_results,
            repeat_results=repeat_results,
            summary=summary,
            limit_loss=limit_loss,
            limit_p=limit_p,
            limit_s=limit_s,
            status="ok",
            message=None,
        )

        # 9. 绘图
        plot_path = self._plot_results(
            metric_short_names, num_metrics,
            baseline_result, no_noise_result, selected_result,
            random_results, summary,
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
            "no_noise_result": no_noise_result,
            "selected_result": selected_result,
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

    def _run_repeated_evaluation(
        self, fixed_gelu, fixed_softmax, noise_cfg, selected_name
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

        return random_results

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

    def _log_repeat_results(self, repeat_results, metric_short_names, num_metrics):
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
        if num_metrics == 1:
            return (
                f"{result['name']:<25} | {ok:<3} | "
                f"{result['loss']:<8.4f} {result['p']:<8.4f} | "
                f"{loss_dp:>8.2f}% {m1_dp:>10.2f}% | "
                f"{result['tot_c']:<8.2f} {result['tot_spd']:<8.2f}"
            )
        m2_dp = (
            ((result["s"] - base_result["s"]) / (base_result["s"] + 1e-8)) * 100.0
        )
        return (
            f"{result['name']:<25} | {ok:<3} | "
            f"{result['loss']:<8.4f} {result['p']:<8.4f} {result['s']:<8.4f} | "
            f"{loss_dp:>8.2f}% {m1_dp:>10.2f}% {m2_dp:>10.2f}% | "
            f"{result['tot_c']:<8.2f} {result['tot_spd']:<8.2f}"
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

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def _save_results_json(
        self,
        selected_source,
        baseline_result,
        no_noise_result,
        selected_result,
        random_results,
        repeat_results,
        summary,
        limit_loss,
        limit_p,
        limit_s,
        status,
        message,
    ):
        output = {
            "dataset": self.evaluator.dataset_key,
            "status": status,
            "message": message,
            "selected_source": selected_source,
            "constraints": {
                "limit_loss": float(limit_loss),
                "limit_primary_metric": float(limit_p),
                "limit_secondary_metric": float(limit_s),
            },
            "baseline": self._json_ready(baseline_result),
            "no_noise": self._json_ready(no_noise_result),
            "selected": self._json_ready(selected_result),
            "random_results": [self._json_ready(r) for r in random_results],
            "random_summary": summary,
        }
        if repeat_results is not None:
            output["repeat_evaluation"] = {
                "stats": repeat_results["stats"],
                "trials": repeat_results["trials"],
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
        ds_key = self.evaluator.dataset_key
        if ds_key not in config_map:
            raise KeyError(
                f"数据集 '{ds_key}' 在噪声配置文件 '{self.config_path}' 中未找到。"
            )
        return config_map[ds_key]

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
    def _noise_config_signature(noise_cfg):
        return tuple(
            tuple(np.asarray(noise_cfg[key], dtype=int).tolist())
            for key in NOISE_SCALING_FACTOR_KEYS
        )

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
