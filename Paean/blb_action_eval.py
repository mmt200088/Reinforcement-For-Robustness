from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from rescale_optimizer_bridge import (
    InProcessInvoker,
    RescaleOptimizerBridge,
    SubprocessInvoker,
    aggregate_optimizer_signals,
    load_baseline_archive,
)

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
        self.rescale_optimizer_mode = self._load_rescale_optimizer_mode()

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
        (
            self.rescale_bridge,
            self.heuristic_invoker,
            self.rescale_invoker_kind,
            self.rescale_optimizer_root,
        ) = self._build_rescale_bridge(profile)

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
            f"RESCALE_OPTIMIZER={self.rescale_invoker_kind} "
            f"root={self.rescale_optimizer_root or '(none)'} "
            f"mode={self.rescale_optimizer_mode}"
        )
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
        text_path = self._save_results_markdown(
            json_path=summary_path,
            selected_source=f"blb_action(stage1={stage1_source})",
            baseline_result=baseline_result,
            candidate_results=results,
        )
        plot_path = self._save_results_plot(candidate_results=results)
        ev.log(f"BLB action final-eval summary saved to: {summary_path}")
        ev.log(f"BLB action final-eval text report saved to: {text_path}")
        if plot_path:
            ev.log(f"BLB action final-eval plot saved to: {plot_path}")

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
            "text_report_path": text_path,
            "plot_path": plot_path,
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
            gelu_degree=np.asarray(gelu, dtype=int),
            attn_degree=np.asarray(softmax, dtype=int),
        )

        cfgs_dict = decoded.cfgs_dict()
        opt_outputs, opt_signals = self._optimizer_outputs(profile, cfgs_dict)
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
            "config_details": self._config_details(decoded, action_vec, overrides, opt_outputs),
            "rescale_optimizer": {
                "invoker_kind": str(getattr(self, "rescale_invoker_kind", "unknown")),
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

    def _optimizer_outputs(self, profile: str, cfgs_dict):
        bridge = getattr(self, "rescale_bridge", None)
        if bridge is None:
            bridge, heuristic, kind, root = self._build_rescale_bridge(profile)
            self.rescale_bridge = bridge
            self.heuristic_invoker = heuristic
            self.rescale_invoker_kind = kind
            self.rescale_optimizer_root = root
        requests = build_optimizer_requests(profile, cfgs_dict)
        outputs = bridge.evaluate_blocks(requests)
        return outputs, aggregate_optimizer_signals(outputs)

    def _build_rescale_bridge(self, profile: str) -> Tuple[RescaleOptimizerBridge, Optional[HeuristicStubInvoker], str, str]:
        ev = self.evaluator
        kind = str(getattr(ev, "blb_v3_rescale_invoker_kind", "heuristic") or "heuristic")
        kind = kind.lower().replace("-", "_")
        require_real = bool(getattr(ev, "final_eval_require_rescale_optimizer", False))
        root = self._resolve_rescale_optimizer_root()

        def fallback(reason: Exception | str):
            if require_real:
                raise RuntimeError(
                    "final_eval requires a real Rescale_optimizer invoker, "
                    f"but initialization failed: {reason}"
                )
            ev.log(f"  [final_eval][warning] Rescale_optimizer unavailable ({reason}); fallback to heuristic.")
            heuristic = HeuristicStubInvoker()
            return RescaleOptimizerBridge(invoker=heuristic, **self._rescale_bridge_options()), heuristic, "heuristic", ""

        if kind == "in_process":
            try:
                invoker = InProcessInvoker.from_profile(
                    rescale_optimizer_root=root,
                    profile=str(profile),
                )
                return RescaleOptimizerBridge(invoker=invoker, **self._rescale_bridge_options()), None, "in_process", root
            except Exception as exc:
                return fallback(exc)

        if kind == "subprocess":
            try:
                cfg_dir = Path(root) / "configs" / str(profile)
                archive = cfg_dir / f"static_skeletons_{profile}.json"
                baselines = load_baseline_archive(str(archive))
                configs = {
                    name: str(cfg_dir / f"{name}.json")
                    for name in baselines
                    if (cfg_dir / f"{name}.json").is_file()
                }
                invoker = SubprocessInvoker(
                    rescale_optimizer_root=root,
                    configs=configs,
                    baseline_archive=str(archive),
                )
                return RescaleOptimizerBridge(invoker=invoker, **self._rescale_bridge_options()), None, "subprocess", root
            except Exception as exc:
                return fallback(exc)

        if kind == "stub":
            canned = getattr(ev, "blb_v3_stub_canned", None)
            if canned:
                from rescale_optimizer_bridge import StubInvoker

                return RescaleOptimizerBridge(invoker=StubInvoker(canned), **self._rescale_bridge_options()), None, "stub", ""
            return fallback("stub invoker requested but no blb_v3_stub_canned was provided")

        if require_real:
            return fallback("rescale invoker kind is heuristic")
        heuristic = HeuristicStubInvoker()
        return RescaleOptimizerBridge(invoker=heuristic, **self._rescale_bridge_options()), heuristic, "heuristic", ""

    def _load_rescale_optimizer_mode(self) -> str:
        if not self.action_config_path:
            return "cfg_derived"
        try:
            payload = json.loads(Path(self.action_config_path).read_text(encoding="utf-8-sig"))
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
            or getattr(ev, "blb_v3_subprocess_optimizer_root", None)
            or "Rescale_optimizer"
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
            "first_input": {0},
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
        identity_match = self._handler_cfg_identity_match(decoded)
        return {
            "checked_before_forward": True,
            "handler_active_layers": active_json,
            "expected_active_layers": expected_json,
            "handler_active_layers_match_expected": bool(handler_match),
            "bridge_installed_layers_match_expected": bool(bridge_match),
            "handler_cfg_objects_match_decoded_cfgs": bool(identity_match),
            "model_will_use_selected_cfg": bool(handler_match and bridge_match and identity_match),
        }

    def _handler_cfg_identity_match(self, decoded) -> bool:
        handler = self.evaluator.reversible_handler
        for block_name in ("block1", "block2", "block3", "block4", "block5"):
            expected = getattr(decoded, f"{block_name}_cfgs")
            installed = getattr(handler, f"{block_name}_cfg_per_layer", {})
            for layer_idx, cfg in expected.items():
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
            "first_input_sf": int(decoded.first_input_sf),
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
        entries = [
            {
                "path": "first_input.fresh",
                "type": "scaling_factor",
                "layer": None,
                "block": "first_input",
                "point": "fresh",
                "distribution": "fresh",
                "N": 16384,
                "scaling_factor": int(decoded.first_input_sf),
                "truncation_k": None,
                "value": None,
                "active": True,
            }
        ]

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
            "install_verification": trials[0].get("install_verification", {}) if trials else {},
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
            install_verification = self._verify_model_installation(bridge, decoded)
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
                "install_verification": install_verification,
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

    def _save_results_markdown(
        self,
        *,
        json_path: str,
        selected_source: str,
        baseline_result,
        candidate_results,
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
            f"- rescale_optimizer: `{getattr(self, 'rescale_invoker_kind', 'unknown')}`",
            f"- rescale_optimizer_root: `{getattr(self, 'rescale_optimizer_root', '') or '(none)'}`",
            f"- json: `{json_path}`",
            "",
            "## Baseline",
            "",
            f"- clean baseline loss: `{baseline_result['loss']:.6f}`",
            f"- clean baseline {primary}: `{baseline_result['p']:.6f}`",
            f"- clean baseline {secondary}: `{baseline_result['s']:.6f}`",
            "",
            "## Group Comparison",
            "",
            "| group | truncation k | effective K positions | loss mean | loss std | "
            f"{primary} mean | {primary} std | {secondary} mean | {secondary} std | "
            "time mean ms | total bits | fusion | model cfg verified |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for result in candidate_results:
            trunc = result.get("config_details", {}).get("truncation", {})
            unique_k = self._unique_truncation_label(trunc)
            verify = result.get("install_verification", {}).get("model_will_use_selected_cfg", False)
            lines.append(
                f"| `{result['name']}` | {unique_k} | "
                f"{int(trunc.get('effective_position_count', 0))} | "
                f"{float(result['loss']):.6f} | {float(result.get('loss_std', 0.0)):.6f} | "
                f"{float(result['p']):.6f} | {float(result.get('p_std', 0.0)):.6f} | "
                f"{float(result['s']):.6f} | {float(result.get('s_std', 0.0)):.6f} | "
                f"{float(result['time_ms']):.3f} | {int(result['total_bits_sum'])} | "
                f"{int(result['total_fusion_count'])} | {verify} |"
            )

        lines.extend(["", "## Configuration Details", ""])
        for result in candidate_results:
            details = result.get("config_details", {})
            trunc = details.get("truncation", {})
            verify = result.get("install_verification", {})
            lines.extend([
                f"### {result['name']}",
                "",
                f"- action overrides: `{result.get('action_overrides', {})}`",
                f"- base action: {details.get('base_action', '')}",
                f"- first_input_sf: `{details.get('first_input_sf')}`",
                f"- truncation summary: `{self._unique_truncation_label(trunc)}`; "
                f"effective positions = `{trunc.get('effective_position_count', 0)}`; "
                f"skipped = `{trunc.get('skipped_positions', [])}`",
                f"- model cfg verified before forward: `{verify.get('model_will_use_selected_cfg', False)}`",
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
        entries = list((full_config or {}).get("entries", []) or [])
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

        labels = [self._unique_truncation_label(r.get("config_details", {}).get("truncation", {})) for r in candidate_results]
        x = np.arange(len(candidate_results))
        loss = np.asarray([float(r["loss"]) for r in candidate_results], dtype=float)
        loss_std = np.asarray([float(r.get("loss_std", 0.0)) for r in candidate_results], dtype=float)
        p = np.asarray([float(r["p"]) for r in candidate_results], dtype=float)
        p_std = np.asarray([float(r.get("p_std", 0.0)) for r in candidate_results], dtype=float)
        bits = np.asarray([float(r["total_bits_sum"]) for r in candidate_results], dtype=float)
        time_ms = np.asarray([float(r["time_ms"]) for r in candidate_results], dtype=float)

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
