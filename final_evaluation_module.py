import json
import os
from typing import Dict, Optional, Sequence

import numpy as np


class FinalEvaluationModule:
    """Standalone final-evaluation helper for selected and random configurations."""

    def __init__(
        self,
        evaluator,
        config_source: str = "search",
        config_path: str = "glue_configs_best_ppo.json",
        manual_gelu: Optional[Sequence[int]] = None,
        manual_softmax: Optional[Sequence[int]] = None,
        random_seed: int = 42,
        permutation_trials: int = 10,
        cost_equivalent_trials: int = 10,
        budget_equivalent_trials: int = 10,
        results_dir: Optional[str] = None,
    ):
        self.evaluator = evaluator
        self.config_source = (config_source or "search").lower()
        self.config_path = config_path or "glue_configs_best_ppo.json"
        self.manual_gelu = manual_gelu
        self.manual_softmax = manual_softmax
        self.random_seed = int(random_seed)
        self.permutation_trials = max(0, int(permutation_trials))
        self.cost_equivalent_trials = max(0, int(cost_equivalent_trials))
        self.budget_equivalent_trials = max(0, int(budget_equivalent_trials))
        default_results_dir = getattr(
            evaluator,
            "stage1_final_eval_dir",
            os.path.join("rl_results", "final_evaluation"),
        )
        self.results_dir = results_dir or default_results_dir

        self.allowed_gelu_selected = [0, 1, 2, 4]
        self.allowed_gelu_random = [1, 2, 4]
        self.allowed_softmax = [2, 3, 4, 5, 6]

    def run(
        self,
        search_best_config,
        base_gelu: np.ndarray,
        base_softmax: np.ndarray,
        base_tot_c: float,
        base_g_c: float,
        base_s_c: float,
        limit_loss: float,
        limit_p: float,
        limit_s: float,
    ) -> Dict[str, object]:
        self._ensure_results_dir()

        base_loss, base_p, base_s, base_time = self.evaluator.evaluate_model(
            base_gelu, base_softmax, use_train=False
        )
        eval_cache = {
            (tuple(base_gelu.tolist()), tuple(base_softmax.tolist())): (
                base_loss,
                base_p,
                base_s,
                base_time,
            )
        }

        metric_short_names = self.evaluator.get_metric_short_names()
        num_metrics = self.evaluator.get_num_metrics()

        def get_result_dict(name, family, gelu_arr, softmax_arr):
            sig = (tuple(gelu_arr.tolist()), tuple(softmax_arr.tolist()))
            if sig in eval_cache:
                loss, p, s, t = eval_cache[sig]
            else:
                loss, p, s, t = self.evaluator.evaluate_model(gelu_arr, softmax_arr, use_train=False)
                eval_cache[sig] = (loss, p, s, t)

            tot_c, g_c, s_c = self.evaluator.get_simulated_cost(gelu_arr, softmax_arr)
            feasible = self._is_feasible(loss, p, s, limit_loss, limit_p, limit_s)

            return {
                "name": name,
                "family": family,
                "loss": float(loss),
                "p": float(p),
                "s": float(s),
                "time_ms": float(t),
                "tot_c": float(tot_c),
                "tot_spd": float(base_tot_c / (tot_c + 1e-6)),
                "g_c": float(g_c),
                "g_spd": float(base_g_c / (g_c + 1e-6)),
                "s_c": float(s_c),
                "s_spd": float(base_s_c / (s_c + 1e-6)),
                "gelu": gelu_arr.copy(),
                "softmax": softmax_arr.copy(),
                "feasible": feasible,
            }

        base_res = get_result_dict("Baseline", "Baseline", base_gelu, base_softmax)

        selected_gelu, selected_softmax, selected_name, selected_source = self._resolve_selected_config(
            search_best_config=search_best_config,
            total_layers=len(base_gelu),
        )
        selected_res = get_result_dict(selected_name, "Selected", selected_gelu, selected_softmax)

        no_approx_res = self._evaluate_exact_control(
            selected_gelu=selected_gelu,
            selected_softmax=selected_softmax,
            base_tot_c=base_tot_c,
            base_g_c=base_g_c,
            base_s_c=base_s_c,
            limit_loss=limit_loss,
            limit_p=limit_p,
            limit_s=limit_s,
        )

        random_results = self._generate_random_results(
            selected_res=selected_res,
            selected_gelu=selected_gelu,
            selected_softmax=selected_softmax,
            get_result_dict=get_result_dict,
        )

        summary = self._summarize_random_results(
            selected_res=selected_res,
            random_results=random_results,
            num_metrics=num_metrics,
        )

        self._log_configuration_details(selected_res, random_results)
        self._log_performance_table(
            metric_short_names=metric_short_names,
            num_metrics=num_metrics,
            base_res=base_res,
            no_approx_res=no_approx_res,
            selected_res=selected_res,
            random_results=random_results,
        )
        self._log_random_summary(
            metric_short_names=metric_short_names,
            summary=summary,
            selected_res=selected_res,
        )

        summary_path = self._save_results_json(
            selected_source=selected_source,
            base_res=base_res,
            no_approx_res=no_approx_res,
            selected_res=selected_res,
            random_results=random_results,
            summary=summary,
            limit_loss=limit_loss,
            limit_p=limit_p,
            limit_s=limit_s,
        )
        plot_path = self._plot_results(
            metric_short_names=metric_short_names,
            num_metrics=num_metrics,
            base_res=base_res,
            no_approx_res=no_approx_res,
            selected_res=selected_res,
            random_results=random_results,
            summary=summary,
        )

        return {
            "selected_source": selected_source,
            "selected_label": selected_name,
            "selected_gelu": selected_gelu,
            "selected_softmax": selected_softmax,
            "selected_result": selected_res,
            "base_result": base_res,
            "no_approx_result": no_approx_res,
            "random_results": random_results,
            "random_summary": summary,
            "eval_cache": eval_cache,
            "summary_path": summary_path,
            "plot_path": plot_path,
        }

    def _resolve_selected_config(self, search_best_config, total_layers: int):
        if self.config_source == "search":
            if search_best_config is None:
                raise ValueError("config_source='search' requires a search_best_config result.")
            gelu = self._normalize_config_array(
                search_best_config["gelu"],
                total_layers=total_layers,
                default_degree=4,
                allowed_values=self.allowed_gelu_selected,
                label="search_gelu",
            )
            softmax = self._normalize_config_array(
                search_best_config["softmax"],
                total_layers=total_layers,
                default_degree=6,
                allowed_values=self.allowed_softmax,
                label="search_softmax",
            )
            return gelu, softmax, "Optimized (PPO)", "search"

        if self.config_source == "json":
            json_cfg = self._load_dataset_config_from_json()
            gelu = self._normalize_config_array(
                json_cfg["gelu"],
                total_layers=total_layers,
                default_degree=4,
                allowed_values=self.allowed_gelu_selected,
                label="json_gelu",
            )
            softmax = self._normalize_config_array(
                json_cfg["softmax"],
                total_layers=total_layers,
                default_degree=6,
                allowed_values=self.allowed_softmax,
                label="json_softmax",
            )
            return gelu, softmax, "Selected (Saved RL)", "json"

        if self.config_source == "manual":
            if self.manual_gelu is None or self.manual_softmax is None:
                raise ValueError(
                    "config_source='manual' requires both manual_gelu and manual_softmax."
                )
            gelu = self._normalize_config_array(
                self.manual_gelu,
                total_layers=total_layers,
                default_degree=4,
                allowed_values=self.allowed_gelu_selected,
                label="manual_gelu",
            )
            softmax = self._normalize_config_array(
                self.manual_softmax,
                total_layers=total_layers,
                default_degree=6,
                allowed_values=self.allowed_softmax,
                label="manual_softmax",
            )
            return gelu, softmax, "Selected (Manual)", "manual"

        raise ValueError(
            f"Unsupported config_source '{self.config_source}'. Use one of: search, json, manual."
        )

    def _evaluate_exact_control(
        self,
        selected_gelu: np.ndarray,
        selected_softmax: np.ndarray,
        base_tot_c: float,
        base_g_c: float,
        base_s_c: float,
        limit_loss: float,
        limit_p: float,
        limit_s: float,
    ) -> Dict[str, object]:
        self.evaluator.log("Evaluating No-Approx control (exact GELU + exact softmax)...")
        self.evaluator.reversible_handler.restore_all()
        self.evaluator.model = self.evaluator.reversible_handler.model
        loss, p, s, t = self.evaluator._run_evaluation(self.evaluator.dataloader_test, use_train=False)
        self.evaluator.apply_configuration(selected_gelu, selected_softmax)

        return {
            "name": "No-Approx (Exact)",
            "family": "Control",
            "loss": float(loss),
            "p": float(p),
            "s": float(s),
            "time_ms": float(t),
            "tot_c": float(base_tot_c),
            "tot_spd": 1.0,
            "g_c": float(base_g_c),
            "g_spd": 1.0,
            "s_c": float(base_s_c),
            "s_spd": 1.0,
            "gelu": np.full(self.evaluator.total_layers, 4, dtype=int),
            "softmax": np.full(self.evaluator.total_layers, 6, dtype=int),
            "feasible": self._is_feasible(loss, p, s, limit_loss, limit_p, limit_s),
        }

    def _generate_random_results(self, selected_res, selected_gelu, selected_softmax, get_result_dict):
        rng = np.random.default_rng(self.random_seed)
        random_results = []
        seen_signatures = {
            (tuple(selected_gelu.tolist()), tuple(selected_softmax.tolist()))
        }

        gelu_solution_map = self._enumerate_cost_solutions(
            allowed_degrees=self.allowed_gelu_random,
            cost_map=self.evaluator.GELU_COST_MAP,
            total_layers=self.evaluator.total_layers,
        )
        softmax_solution_map = self._enumerate_cost_solutions(
            allowed_degrees=self.allowed_softmax,
            cost_map=self.evaluator.SOFTMAX_COST_MAP,
            total_layers=self.evaluator.total_layers,
        )

        selected_gelu_counts = self._count_signature(selected_gelu, self.allowed_gelu_random)
        selected_softmax_counts = self._count_signature(selected_softmax, self.allowed_softmax)
        selected_pair = (
            self._cost_key(selected_res["g_c"]),
            self._cost_key(selected_res["s_c"]),
        )

        if np.any(selected_gelu == 0):
            self.evaluator.log(
                "[Info] Selected config contains GELU degree 0, so permutation random trials are skipped "
                "to keep all random baselines degree-0 free."
            )
        else:
            self.evaluator.log(f"Generating {self.permutation_trials} permutation random configs...")
            for idx in range(self.permutation_trials):
                signature = None
                gelu_arr = None
                softmax_arr = None
                for _ in range(200):
                    gelu_arr = rng.permutation(selected_gelu)
                    softmax_arr = rng.permutation(selected_softmax)
                    signature = (tuple(gelu_arr.tolist()), tuple(softmax_arr.tolist()))
                    if signature not in seen_signatures and signature != (
                        tuple(selected_gelu.tolist()),
                        tuple(selected_softmax.tolist()),
                    ):
                        break
                if signature is None or signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                random_results.append(get_result_dict(f"Perm_{idx + 1}", "Perm", gelu_arr, softmax_arr))

        self.evaluator.log(f"Generating {self.cost_equivalent_trials} exact cost-matched random configs...")
        for idx in range(self.cost_equivalent_trials):
            sample = self._sample_exact_cost_pair(
                rng=rng,
                gelu_solution_map=gelu_solution_map,
                softmax_solution_map=softmax_solution_map,
                target_g_cost=selected_res["g_c"],
                target_s_cost=selected_res["s_c"],
                seen_signatures=seen_signatures,
                avoid_g_counts=selected_gelu_counts,
                avoid_s_counts=selected_softmax_counts,
            )
            if sample is None:
                continue
            gelu_arr, softmax_arr = sample
            random_results.append(get_result_dict(f"Equiv_{idx + 1}", "Equiv", gelu_arr, softmax_arr))

        self.evaluator.log(f"Generating {self.budget_equivalent_trials} total-budget random configs...")
        for idx in range(self.budget_equivalent_trials):
            sample = self._sample_budget_matched_pair(
                rng=rng,
                gelu_solution_map=gelu_solution_map,
                softmax_solution_map=softmax_solution_map,
                target_total_cost=selected_res["tot_c"],
                seen_signatures=seen_signatures,
                avoid_pair=selected_pair,
            )
            if sample is None:
                continue
            gelu_arr, softmax_arr = sample
            random_results.append(get_result_dict(f"Budget_{idx + 1}", "Budget", gelu_arr, softmax_arr))

        return random_results

    def _sample_exact_cost_pair(
        self,
        rng,
        gelu_solution_map,
        softmax_solution_map,
        target_g_cost,
        target_s_cost,
        seen_signatures,
        avoid_g_counts,
        avoid_s_counts,
    ):
        gelu_key = self._cost_key(target_g_cost)
        softmax_key = self._cost_key(target_s_cost)

        gelu_counts = self._choose_count_solution(
            rng=rng,
            solution_map=gelu_solution_map,
            target_cost_key=gelu_key,
            avoid_counts=avoid_g_counts,
        )
        softmax_counts = self._choose_count_solution(
            rng=rng,
            solution_map=softmax_solution_map,
            target_cost_key=softmax_key,
            avoid_counts=avoid_s_counts,
        )
        if gelu_counts is None or softmax_counts is None:
            return None

        for _ in range(200):
            gelu_arr = self._counts_to_shuffled_config(self.allowed_gelu_random, gelu_counts, rng)
            softmax_arr = self._counts_to_shuffled_config(self.allowed_softmax, softmax_counts, rng)
            signature = (tuple(gelu_arr.tolist()), tuple(softmax_arr.tolist()))
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                return gelu_arr, softmax_arr
        return None

    def _sample_budget_matched_pair(
        self,
        rng,
        gelu_solution_map,
        softmax_solution_map,
        target_total_cost,
        seen_signatures,
        avoid_pair,
    ):
        total_key = self._cost_key(target_total_cost)
        feasible_pairs = []
        for gelu_key in gelu_solution_map.keys():
            softmax_key = total_key - gelu_key
            if softmax_key in softmax_solution_map:
                feasible_pairs.append((gelu_key, softmax_key))

        if not feasible_pairs:
            return None

        alt_pairs = [pair for pair in feasible_pairs if pair != avoid_pair]
        pairs = alt_pairs if alt_pairs else feasible_pairs

        for _ in range(200):
            gelu_key, softmax_key = pairs[rng.integers(0, len(pairs))]
            gelu_counts = self._choose_count_solution(
                rng=rng,
                solution_map=gelu_solution_map,
                target_cost_key=gelu_key,
            )
            softmax_counts = self._choose_count_solution(
                rng=rng,
                solution_map=softmax_solution_map,
                target_cost_key=softmax_key,
            )
            if gelu_counts is None or softmax_counts is None:
                continue
            gelu_arr = self._counts_to_shuffled_config(self.allowed_gelu_random, gelu_counts, rng)
            softmax_arr = self._counts_to_shuffled_config(self.allowed_softmax, softmax_counts, rng)
            signature = (tuple(gelu_arr.tolist()), tuple(softmax_arr.tolist()))
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                return gelu_arr, softmax_arr
        return None

    def _summarize_random_results(self, selected_res, random_results, num_metrics):
        summary = {"overall": {}, "by_family": {}}
        if not random_results:
            return summary

        grouped = {}
        for res in random_results:
            grouped.setdefault(res["family"], []).append(res)

        all_dominance = []
        all_feasible = []
        all_primary_win = []
        all_loss_win = []
        all_secondary_win = []

        for family, items in grouped.items():
            feasible_rate = float(np.mean([1.0 if item["feasible"] else 0.0 for item in items]))
            loss_win_rate = float(np.mean([1.0 if selected_res["loss"] <= item["loss"] else 0.0 for item in items]))
            primary_win_rate = float(np.mean([1.0 if selected_res["p"] >= item["p"] else 0.0 for item in items]))
            if num_metrics > 1:
                secondary_win_rate = float(
                    np.mean([1.0 if selected_res["s"] >= item["s"] else 0.0 for item in items])
                )
            else:
                secondary_win_rate = None
            dominance_rate = float(np.mean([1.0 if self._dominates(selected_res, item) else 0.0 for item in items]))

            summary["by_family"][family] = {
                "count": len(items),
                "feasible_rate": feasible_rate,
                "loss_win_rate": loss_win_rate,
                "primary_win_rate": primary_win_rate,
                "secondary_win_rate": secondary_win_rate,
                "dominance_rate": dominance_rate,
                "primary_metric_mean": float(np.mean([item["p"] for item in items])),
                "primary_metric_std": float(np.std([item["p"] for item in items])),
                "loss_mean": float(np.mean([item["loss"] for item in items])),
                "loss_std": float(np.std([item["loss"] for item in items])),
                "cost_mean": float(np.mean([item["tot_c"] for item in items])),
                "cost_std": float(np.std([item["tot_c"] for item in items])),
            }

            all_feasible.extend([1.0 if item["feasible"] else 0.0 for item in items])
            all_loss_win.extend([1.0 if selected_res["loss"] <= item["loss"] else 0.0 for item in items])
            all_primary_win.extend([1.0 if selected_res["p"] >= item["p"] else 0.0 for item in items])
            all_dominance.extend([1.0 if self._dominates(selected_res, item) else 0.0 for item in items])
            if num_metrics > 1:
                all_secondary_win.extend([1.0 if selected_res["s"] >= item["s"] else 0.0 for item in items])

        summary["overall"] = {
            "count": len(random_results),
            "feasible_rate": float(np.mean(all_feasible)) if all_feasible else 0.0,
            "loss_win_rate": float(np.mean(all_loss_win)) if all_loss_win else 0.0,
            "primary_win_rate": float(np.mean(all_primary_win)) if all_primary_win else 0.0,
            "secondary_win_rate": float(np.mean(all_secondary_win)) if all_secondary_win else None,
            "dominance_rate": float(np.mean(all_dominance)) if all_dominance else 0.0,
        }
        return summary

    def _plot_results(
        self,
        metric_short_names,
        num_metrics,
        base_res,
        no_approx_res,
        selected_res,
        random_results,
        summary,
    ):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                f"Final Evaluation Comparison ({self.evaluator.dataset_key.upper()})",
                fontsize=14,
                fontweight="bold",
            )

            family_colors = {
                "Perm": "#4C78A8",
                "Equiv": "#F58518",
                "Budget": "#54A24B",
            }

            grouped = {}
            for res in random_results:
                grouped.setdefault(res["family"], []).append(res)

            ax_loss = axes[0, 0]
            for family, items in grouped.items():
                ax_loss.scatter(
                    [item["tot_c"] for item in items],
                    [item["loss"] for item in items],
                    s=42,
                    alpha=0.75,
                    label=family,
                    color=family_colors.get(family, "#999999"),
                )
            ax_loss.scatter(base_res["tot_c"], base_res["loss"], marker="s", s=90, color="black", label="Baseline")
            ax_loss.scatter(
                no_approx_res["tot_c"],
                no_approx_res["loss"],
                marker="^",
                s=90,
                color="#666666",
                label="Exact",
            )
            ax_loss.scatter(
                selected_res["tot_c"],
                selected_res["loss"],
                marker="*",
                s=180,
                color="#E45756",
                label=selected_res["name"],
            )
            ax_loss.set_title("Loss vs Simulated Cost")
            ax_loss.set_xlabel("Simulated Cost")
            ax_loss.set_ylabel("Loss")
            ax_loss.grid(True, alpha=0.3)
            ax_loss.legend(loc="best")

            ax_metric1 = axes[0, 1]
            for family, items in grouped.items():
                ax_metric1.scatter(
                    [item["tot_c"] for item in items],
                    [item["p"] for item in items],
                    s=42,
                    alpha=0.75,
                    label=family,
                    color=family_colors.get(family, "#999999"),
                )
            ax_metric1.scatter(base_res["tot_c"], base_res["p"], marker="s", s=90, color="black", label="Baseline")
            ax_metric1.scatter(
                no_approx_res["tot_c"],
                no_approx_res["p"],
                marker="^",
                s=90,
                color="#666666",
                label="Exact",
            )
            ax_metric1.scatter(
                selected_res["tot_c"],
                selected_res["p"],
                marker="*",
                s=180,
                color="#E45756",
                label=selected_res["name"],
            )
            ax_metric1.set_title(f"{metric_short_names[0]} vs Simulated Cost")
            ax_metric1.set_xlabel("Simulated Cost")
            ax_metric1.set_ylabel(metric_short_names[0])
            ax_metric1.grid(True, alpha=0.3)
            ax_metric1.legend(loc="best")

            if num_metrics > 1:
                ax_metric2 = axes[1, 0]
                for family, items in grouped.items():
                    ax_metric2.scatter(
                        [item["tot_c"] for item in items],
                        [item["s"] for item in items],
                        s=42,
                        alpha=0.75,
                        label=family,
                        color=family_colors.get(family, "#999999"),
                    )
                ax_metric2.scatter(base_res["tot_c"], base_res["s"], marker="s", s=90, color="black", label="Baseline")
                ax_metric2.scatter(
                    no_approx_res["tot_c"],
                    no_approx_res["s"],
                    marker="^",
                    s=90,
                    color="#666666",
                    label="Exact",
                )
                ax_metric2.scatter(
                    selected_res["tot_c"],
                    selected_res["s"],
                    marker="*",
                    s=180,
                    color="#E45756",
                    label=selected_res["name"],
                )
                ax_metric2.set_title(f"{metric_short_names[1]} vs Simulated Cost")
                ax_metric2.set_xlabel("Simulated Cost")
                ax_metric2.set_ylabel(metric_short_names[1])
                ax_metric2.grid(True, alpha=0.3)
                ax_metric2.legend(loc="best")
            else:
                ax_metric2 = axes[1, 0]
                labels = list(grouped.keys())
                metric_values = [[item["p"] for item in grouped[label]] for label in labels]
                if metric_values:
                    box = ax_metric2.boxplot(metric_values, labels=labels, patch_artist=True)
                    for patch, label in zip(box["boxes"], labels):
                        patch.set_facecolor(family_colors.get(label, "#BBBBBB"))
                else:
                    ax_metric2.text(0.5, 0.5, "No random results", ha="center", va="center")
                ax_metric2.axhline(base_res["p"], color="black", linestyle="--", linewidth=1.2, label="Baseline")
                ax_metric2.axhline(
                    selected_res["p"],
                    color="#E45756",
                    linestyle="-",
                    linewidth=1.4,
                    label=selected_res["name"],
                )
                ax_metric2.set_title(f"{metric_short_names[0]} Distribution by Random Family")
                ax_metric2.set_ylabel(metric_short_names[0])
                ax_metric2.grid(True, axis="y", alpha=0.3)
                ax_metric2.legend(loc="best")

            ax_summary = axes[1, 1]
            families = list(summary.get("by_family", {}).keys())
            if families:
                x = np.arange(len(families))
                feasible = [summary["by_family"][family]["feasible_rate"] for family in families]
                dominance = [summary["by_family"][family]["dominance_rate"] for family in families]
                width = 0.34
                ax_summary.bar(x - width / 2, feasible, width=width, label="Constraint OK rate", color="#72B7B2")
                ax_summary.bar(x + width / 2, dominance, width=width, label="Dominated by selected", color="#E45756")
                ax_summary.set_xticks(x)
                ax_summary.set_xticklabels(families)
                ax_summary.set_ylim(0.0, 1.05)
                ax_summary.set_title("Random Baseline Summary")
                ax_summary.set_ylabel("Rate")
                ax_summary.grid(True, axis="y", alpha=0.3)
                ax_summary.legend(loc="best")
            else:
                ax_summary.text(0.5, 0.5, "No random results", ha="center", va="center")
                ax_summary.set_title("Random Baseline Summary")

            plt.tight_layout()
            plot_path = os.path.join(
                self.results_dir,
                f"final_eval_comparison_{self.evaluator.dataset_key}.png",
            )
            plt.savefig(plot_path, dpi=180)
            plt.close(fig)
            self.evaluator.log(f"Final evaluation comparison plot saved to: {plot_path}")
            return plot_path
        except Exception as exc:
            self.evaluator.log(f"[Warning] Failed to plot final evaluation comparison: {exc}")
            return None

    def _log_configuration_details(self, selected_res, random_results):
        self.evaluator.log("\nFinal Configurations Details:")
        self.evaluator.log(f"[{selected_res['name']}] GELU   : {selected_res['gelu'].tolist()}")
        self.evaluator.log(f"[{selected_res['name']}] Softmax: {selected_res['softmax'].tolist()}")
        self.evaluator.log("\nRandom Configurations Details:")
        for res in random_results:
            self.evaluator.log(f"[{res['name']}] GELU   : {res['gelu'].tolist()}")
            self.evaluator.log(f"[{res['name']}] Softmax: {res['softmax'].tolist()}")

    def _log_performance_table(
        self,
        metric_short_names,
        num_metrics,
        base_res,
        no_approx_res,
        selected_res,
        random_results,
    ):
        self.evaluator.log("\nPerformance Comparison Table:")
        if num_metrics == 1:
            header = (
                f"{'Method':<20} | {'OK':<3} | {'Loss':<7} {metric_short_names[0]:<7} | "
                f"{'dLoss%':<8} {'d' + metric_short_names[0] + '%':<10} | "
                f"{'Tot C':<6} {'Tot S':<6} | {'GELU C':<6} {'GELU S':<6} | {'Smax C':<6} {'Smax S':<6}"
            )
        else:
            header = (
                f"{'Method':<20} | {'OK':<3} | {'Loss':<7} {metric_short_names[0]:<7} {metric_short_names[1]:<7} | "
                f"{'dLoss%':<8} {'d' + metric_short_names[0] + '%':<10} {'d' + metric_short_names[1] + '%':<10} | "
                f"{'Tot C':<6} {'Tot S':<6} | {'GELU C':<6} {'GELU S':<6} | {'Smax C':<6} {'Smax S':<6}"
            )
        self.evaluator.log("-" * len(header))
        self.evaluator.log(header)
        self.evaluator.log("-" * len(header))
        self.evaluator.log(self._format_row(base_res, base_res, num_metrics))
        self.evaluator.log(self._format_row(no_approx_res, base_res, num_metrics))
        self.evaluator.log(self._format_row(selected_res, base_res, num_metrics))
        self.evaluator.log("-" * len(header))
        for res in random_results:
            self.evaluator.log(self._format_row(res, base_res, num_metrics))
        self.evaluator.log("-" * len(header))

    def _log_random_summary(self, metric_short_names, summary, selected_res):
        self.evaluator.log("\nRandom Baseline Summary:")
        overall = summary.get("overall", {})
        if not overall:
            self.evaluator.log("  No random baselines were generated.")
            return

        self.evaluator.log(
            "  Overall: "
            f"samples={overall['count']}, "
            f"constraint_ok={overall['feasible_rate']:.2%}, "
            f"selected_better_loss={overall['loss_win_rate']:.2%}, "
            f"selected_better_{metric_short_names[0]}={overall['primary_win_rate']:.2%}, "
            f"selected_dominates={overall['dominance_rate']:.2%}"
        )
        if overall.get("secondary_win_rate") is not None:
            self.evaluator.log(
                f"  Overall selected_better_{metric_short_names[1]}={overall['secondary_win_rate']:.2%}"
            )

        for family, family_stats in summary.get("by_family", {}).items():
            msg = (
                f"  {family:<7} samples={family_stats['count']:<3} "
                f"constraint_ok={family_stats['feasible_rate']:.2%} "
                f"selected_better_loss={family_stats['loss_win_rate']:.2%} "
                f"selected_better_{metric_short_names[0]}={family_stats['primary_win_rate']:.2%} "
                f"selected_dominates={family_stats['dominance_rate']:.2%}"
            )
            if family_stats.get("secondary_win_rate") is not None:
                msg += f" selected_better_{metric_short_names[1]}={family_stats['secondary_win_rate']:.2%}"
            self.evaluator.log(msg)

        self.evaluator.log(
            f"  Selected config actual result: Loss={selected_res['loss']:.4f}, "
            f"{metric_short_names[0]}={selected_res['p']:.4f}"
            + (
                f", {metric_short_names[1]}={selected_res['s']:.4f}"
                if len(metric_short_names) > 1
                else ""
            )
            + f", Cost={selected_res['tot_c']:.2f}"
        )

    def _save_results_json(
        self,
        selected_source,
        base_res,
        no_approx_res,
        selected_res,
        random_results,
        summary,
        limit_loss,
        limit_p,
        limit_s,
    ):
        output = {
            "dataset": self.evaluator.dataset_key,
            "selected_source": selected_source,
            "constraints": {
                "limit_loss": float(limit_loss),
                "limit_primary_metric": float(limit_p),
                "limit_secondary_metric": float(limit_s),
            },
            "baseline": self._json_ready_result(base_res),
            "no_approx": self._json_ready_result(no_approx_res),
            "selected": self._json_ready_result(selected_res),
            "random_results": [self._json_ready_result(res) for res in random_results],
            "random_summary": summary,
        }

        output_path = os.path.join(
            self.results_dir,
            f"final_eval_results_{self.evaluator.dataset_key}.json",
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        self.evaluator.log(f"Final evaluation summary saved to: {output_path}")
        return output_path

    def _json_ready_result(self, result):
        return {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else bool(value)
                if isinstance(value, np.bool_)
                else value
            )
            for key, value in result.items()
        }

    def _format_row(self, result, base_res, num_metrics):
        loss_delta_pct = ((result["loss"] - base_res["loss"]) / (base_res["loss"] + 1e-8)) * 100.0
        metric1_delta_pct = ((result["p"] - base_res["p"]) / (base_res["p"] + 1e-8)) * 100.0
        ok_flag = "Y" if result["feasible"] else "N"
        cost_part = (
            f"{result['tot_c']:<6.1f} {result['tot_spd']:<6.2f} | "
            f"{result['g_c']:<6.1f} {result['g_spd']:<6.2f} | "
            f"{result['s_c']:<6.1f} {result['s_spd']:<6.2f}"
        )
        if num_metrics == 1:
            return (
                f"{result['name']:<20} | {ok_flag:<3} | {result['loss']:<7.4f} {result['p']:<7.4f} | "
                f"{loss_delta_pct:>7.2f}% {metric1_delta_pct:>9.2f}% | {cost_part}"
            )

        metric2_delta_pct = ((result["s"] - base_res["s"]) / (base_res["s"] + 1e-8)) * 100.0
        return (
            f"{result['name']:<20} | {ok_flag:<3} | {result['loss']:<7.4f} {result['p']:<7.4f} {result['s']:<7.4f} | "
            f"{loss_delta_pct:>7.2f}% {metric1_delta_pct:>9.2f}% {metric2_delta_pct:>9.2f}% | {cost_part}"
        )

    def _is_feasible(self, loss, p, s, limit_loss, limit_p, limit_s):
        if loss > limit_loss:
            return False
        if p < limit_p:
            return False
        if self.evaluator.get_num_metrics() > 1 and s < limit_s:
            return False
        return True

    def _dominates(self, selected_res, other_res):
        better_or_equal = (
            selected_res["tot_c"] <= other_res["tot_c"]
            and selected_res["loss"] <= other_res["loss"]
            and selected_res["p"] >= other_res["p"]
        )
        if self.evaluator.get_num_metrics() > 1:
            better_or_equal = better_or_equal and selected_res["s"] >= other_res["s"]
        if not better_or_equal:
            return False

        strict = (
            selected_res["tot_c"] < other_res["tot_c"]
            or selected_res["loss"] < other_res["loss"]
            or selected_res["p"] > other_res["p"]
        )
        if self.evaluator.get_num_metrics() > 1:
            strict = strict or selected_res["s"] > other_res["s"]
        return strict

    def _enumerate_cost_solutions(self, allowed_degrees, cost_map, total_layers):
        solution_map = {}
        counts = [0] * len(allowed_degrees)
        cost_keys = [self._cost_key(cost_map[degree]) for degree in allowed_degrees]

        def backtrack(index, remaining_layers, accumulated_cost):
            if index == len(allowed_degrees) - 1:
                counts[index] = remaining_layers
                final_cost = accumulated_cost + remaining_layers * cost_keys[index]
                solution_map.setdefault(final_cost, []).append(tuple(counts))
                return

            for count in range(remaining_layers + 1):
                counts[index] = count
                backtrack(
                    index + 1,
                    remaining_layers - count,
                    accumulated_cost + count * cost_keys[index],
                )

        backtrack(0, total_layers, 0)
        return solution_map

    def _choose_count_solution(self, rng, solution_map, target_cost_key, avoid_counts=None):
        candidates = solution_map.get(target_cost_key, [])
        if not candidates:
            return None

        filtered = [cand for cand in candidates if cand != avoid_counts]
        pool = filtered if filtered else candidates
        return pool[rng.integers(0, len(pool))]

    def _counts_to_shuffled_config(self, allowed_degrees, counts, rng):
        values = []
        for degree, count in zip(allowed_degrees, counts):
            values.extend([degree] * int(count))
        arr = np.array(values, dtype=int)
        rng.shuffle(arr)
        return arr

    def _count_signature(self, config, allowed_degrees):
        return tuple(int(np.sum(np.asarray(config) == degree)) for degree in allowed_degrees)

    def _normalize_config_array(self, values, total_layers, default_degree, allowed_values, label):
        arr = np.asarray(list(values), dtype=int).flatten()
        if arr.size < total_layers:
            pad = np.full(total_layers - arr.size, default_degree, dtype=int)
            self.evaluator.log(
                f"[Info] {label} length {arr.size} < total_layers={total_layers}; padding with {default_degree}."
            )
            arr = np.concatenate([arr, pad])
        elif arr.size > total_layers:
            self.evaluator.log(
                f"[Info] {label} length {arr.size} > total_layers={total_layers}; truncating to {total_layers}."
            )
            arr = arr[:total_layers].copy()

        invalid = sorted(set(arr.tolist()) - set(allowed_values))
        if invalid:
            raise ValueError(f"{label} contains unsupported degrees: {invalid}")
        return arr

    def _load_dataset_config_from_json(self):
        with open(self.config_path, "r", encoding="utf-8") as handle:
            config_map = json.load(handle)
        config_map.pop("_comment", None)

        total_layers = int(getattr(self.evaluator, "total_layers", 12) or 12)
        variant_key = "bert-large" if total_layers >= 24 else "bert-base"

        # New schema: {"bert-base": {task: {...}}, "bert-large": {task: {...}}}
        if variant_key in config_map and isinstance(config_map[variant_key], dict):
            section = config_map[variant_key]
        elif "bert-base" in config_map or "bert-large" in config_map:
            raise KeyError(
                f"Model variant '{variant_key}' (total_layers={total_layers}) "
                f"not found in config file '{self.config_path}'."
            )
        else:
            # Legacy flat schema — only valid for bert-base.
            if variant_key != "bert-base":
                raise KeyError(
                    f"Config file '{self.config_path}' uses the legacy flat schema "
                    f"which only supports bert-base; add a 'bert-large' section for "
                    f"total_layers={total_layers}."
                )
            section = config_map

        ds_key = self.evaluator.dataset_key
        if ds_key not in section:
            raise KeyError(
                f"Dataset '{ds_key}' not found under '{variant_key}' in '{self.config_path}'."
            )
        return section[ds_key]

    def _cost_key(self, value):
        return int(round(float(value) * 2.0))

    def _ensure_results_dir(self):
        os.makedirs(self.results_dir, exist_ok=True)
