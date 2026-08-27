from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from rfr.search.runtime.blb_bridge import BLBNoiseRLBridge
from rfr.search.runtime.model_handler import reseed_noise_rng
from rfr.search.common.action_space import (
    ActionDecodeResult,
    _block_default_N,
    _decode_block_field_values,
    action_vector_to_cfgs,
    avg_truncation_k_in_action,
    block_field_names,
    build_block_cfg_from_field_values,
    build_optimizer_requests,
    validate_action_vector,
)
from rfr.search.common.layerwise_action import (
    fusion_materialization_blocks,
    layerwise_fusion_option_by_step,
    layerwise_schedule,
    materialize_layerwise_counterfactuals,
)
from rfr.preparation.rescale.baseline_bootstrap import (
    load_calibrated_stage2_action_context,
)
from rfr.search.common.feasibility import build_final_eval_feasibility
from rfr.search.common.eval_metrics import (
    pack_repeat_evaluation,
)
from rfr.preparation.fusion.count_map import FusionCountMap
from rfr.preparation.fusion.fixed_action import select_fusion_eval_metadata
from rfr.preparation.rescale.optimizer_cost import materialize_decoded_action
from rfr.evaluation.protocol import require_final_evaluation_protocol
from rfr.preparation.data.protocol import FINAL_EVAL_SPLIT
from rfr.common.json_utils import read_json_file, to_jsonable
from rfr.preparation.rescale.bridge import (
    RescaleOptimizerBridge,
    aggregate_optimizer_signals,
    build_rescale_invoker,
)

def _atomic_json(path: str, payload: Any) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(payload, stringify_unknown=True),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class BLBActionFinalEvaluationModule:
    """Evaluate one selected layerwise search configuration."""

    def __init__(
        self,
        *,
        evaluator,
        random_seed: int = 42,
        repeat_n: int = 1,
        results_dir: Optional[str] = None,
    ):
        self.evaluator = evaluator
        self.random_seed = int(random_seed)
        self.repeat_n = max(1, int(repeat_n))
        default_results_dir = getattr(
            evaluator, "final_eval_dir", os.path.join("outputs", "rl", "evaluation")
        )
        self.results_dir = results_dir or default_results_dir
        self.rescale_optimizer_mode = "cfg_derived"

    @staticmethod
    def _capture_isolated_candidate_rng_state() -> Dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    def _restore_isolated_candidate_rng_state(
        self,
        metadata: Mapping[str, Any],
        state: Optional[Mapping[str, Any]],
    ) -> None:
        if not bool((metadata or {}).get("isolate_random_seed", False)):
            return
        if state is None:
            raise RuntimeError("isolated batch candidate RNG state was not captured")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        reseed_noise_rng(self.random_seed)

    def _evaluate_candidate_with_seed_lifecycle(
        self,
        *,
        metadata: Mapping[str, Any],
        isolated_candidate_rng_state: Optional[Mapping[str, Any]],  # noqa: UP045
        evaluate: Callable[[], Any],
    ) -> Any:
        candidate_metadata = dict(metadata or {})
        isolate_noise_rng = (
            candidate_metadata.get("isolate_random_seed") is True
        )
        self._restore_isolated_candidate_rng_state(
            candidate_metadata,
            isolated_candidate_rng_state,
        )
        try:
            return evaluate()
        finally:
            if isolate_noise_rng:
                reseed_noise_rng(None)

    def _validate_prepared_materialization(
            self,
            selected_config,
            *,
            materialized,
            ):
        if not isinstance(selected_config, Mapping):
            raise ValueError("selected search configuration is missing")
        if not bool(getattr(materialized, "model_ready", False)):
            reason = str(getattr(materialized, "failure_reason", "") or "")
            raise ValueError(
                "selected final-evaluation action is not model-ready"
                + (f": {reason}" if reason else "")
            )
        actual_fingerprint = str(
            getattr(materialized, "final_config_fingerprint", "") or ""
        )
        if len(actual_fingerprint) != 64:
            raise ValueError(
                "selected final-eval config fingerprint is invalid"
            )
        return {
            "schema_version": (
                "stage2_final_eval_materialization_consistency_v1"
            ),
            "checked_before_forward": True,
            "final_config_fingerprint": actual_fingerprint,
            "derived_from_search_best_json": True,
        }

    def run(
        self,
        search_config: Mapping[str, Any],
        baseline_stage1_gelu: np.ndarray,
        baseline_stage1_softmax: np.ndarray,
        limit_loss: float,
        limit_p: float,
        limit_s: float,
    ) -> Dict[str, object]:
        from rfr.search.common.best_config import (
            profile_for,
            validate_search_best_config,
        )

        selected_config = validate_search_best_config(search_config)
        protocol = require_final_evaluation_protocol(
            self.evaluator,
            search_results=(selected_config,),
        )
        self.final_eval_split = protocol["split_name"]
        ev = self.evaluator
        profile = profile_for(
            selected_config["model_type"], selected_config["dataset"],
        )
        total_layers = int(ev.total_layers)
        if total_layers != int(selected_config["num_layers"]):
            raise ValueError(
                "search-best layer count does not match the loaded model"
            )

        opt_gelu = np.asarray(selected_config["stage1"]["gelu"], dtype=int)
        opt_softmax = np.asarray(
            selected_config["stage1"]["softmax"], dtype=int,
        )
        (
            self.rescale_bridge,
            self.rescale_backend,
            self.rescale_optimizer_root,
        ) = self._build_rescale_bridge(profile)
        action_context = load_calibrated_stage2_action_context(
            rescale_optimizer_root=self.rescale_optimizer_root,
            dataset=profile,
            num_layers=total_layers,
            gelu_per_layer=opt_gelu,
            softmax_per_layer=opt_softmax,
            snap_sf_to_noise_table=False,
        )
        fusion_map = FusionCountMap.load(profile)
        schedule = layerwise_schedule(
            total_layers,
            fusion_map,
            profile=profile,
            gelu_degrees=opt_gelu.tolist(),
        )
        action_matrix = selected_config["stage2"]["action_matrix"]
        materialization = materialize_layerwise_counterfactuals(
            action_context.baseline_action_vec,
            action_matrix,
            schedule,
            fusion_map,
        )["joint"]
        boosted_overrides = [
            {
                "block_idx": int(block_idx),
                "layer_idx": int(layer_idx),
                "field_values": {
                    str(name): int(value) for name, value in fields.items()
                },
            }
            for (block_idx, layer_idx), fields in sorted(
                materialization.boosted_overrides.items()
            )
        ]
        fusion_group = {
            "policy_actions": [list(row) for row in action_matrix],
            "option_by_step": dict(layerwise_fusion_option_by_step(
                action_matrix, schedule, fusion_map,
            )),
            "boosted_overrides": boosted_overrides,
        }
        action_vec = np.asarray(materialization.full_vector, dtype=int)
        metadata = {
            "schema_version": "fusion_count_fixed_action_v1",
            "group": fusion_group,
            "isolate_random_seed": True,
        }
        self._stage2_fusion_map = fusion_map
        decoded = self._decode_action_candidate(
            action_vec=action_vec,
            metadata=metadata,
            max_sfs=action_context.max_sfs,
            num_layers=total_layers,
            gelu=opt_gelu,
            softmax=opt_softmax,
            profile=profile,
        )
        cfgs_dict = decoded.cfgs_dict()
        opt_outputs, opt_signals = self._optimizer_outputs(profile, cfgs_dict)
        prepared_materialized = self._materialize_decoded_action(
            profile=profile,
            action_vec=action_vec,
            decoded=decoded,
            cfgs_dict=cfgs_dict,
            opt_outputs=opt_outputs,
            opt_signals=opt_signals,
        )
        final_eval_handoff = {
            "schema_version": "selected_search_config_final_eval_v1",
            "algorithm": selected_config["algorithm"],
            "profile": profile,
            "action_matrix": [list(row) for row in action_matrix],
            "full_vector": action_vec.tolist(),
            "fusion_group": fusion_group,
            "materialization": self._validate_prepared_materialization(
                selected_config,
                materialized=prepared_materialized,
            ),
        }

        os.makedirs(self.results_dir, exist_ok=True)
        ev.log("\n" + "=" * 60)
        ev.log("SELECTED CONFIGURATION FINAL EVALUATION (validation_full)")
        ev.log(
            f"algorithm={selected_config['algorithm']} repeat={self.repeat_n} "
            f"profile={profile}"
        )
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
        isolated_rng_state = self._capture_isolated_candidate_rng_state()
        name = f"{selected_config['algorithm']}_selected"
        result = self._evaluate_candidate_with_seed_lifecycle(
            metadata=metadata,
            isolated_candidate_rng_state=isolated_rng_state,
            evaluate=lambda: self._evaluate_action_candidate(
                name=name,
                action_vec=action_vec,
                overrides={},
                metadata=metadata,
                gelu=opt_gelu,
                softmax=opt_softmax,
                report_constraints=report_constraints,
                max_sfs=action_context.max_sfs,
                prepared_materialized=prepared_materialized,
                materialization_consistency=final_eval_handoff["materialization"],
            ),
        )
        results = [result]
        self._attach_relative_metrics(baseline_result, results)
        summary_path = self._save_results_json(
            selected_source="search_best_config.json",
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
            action_context_provenance=action_context.provenance,
            final_eval_handoff=final_eval_handoff,
        )
        text_path = self._save_results_markdown(
            json_path=summary_path,
            selected_source="search_best_config.json",
            baseline_result=baseline_result,
            candidate_results=results,
        )
        ev.log(f"Final-evaluation summary: {summary_path}")
        ev.log(f"Final-evaluation report: {text_path}")

        ev.apply_configuration(opt_gelu, opt_softmax)
        self._clear_all_noise()
        return {
            "final_eval_split": self.final_eval_split,
            "dataset_protocol_hash": protocol["dataset_protocol_hash"],
            "validation_example_count": protocol["example_count"],
            "selected_source": "search_best_config.json",
            "opt_gelu": opt_gelu,
            "opt_softmax": opt_softmax,
            "baseline_result": baseline_result,
            "optimized_result": result,
            "candidate_results": results,
            "selected_results": results,
            "calibrated_action_context": to_jsonable(action_context.provenance),
            "summary_path": summary_path,
            "text_report_path": text_path,
            "stage2_final_eval_handoff": final_eval_handoff,
        }

    def _evaluate_clean_baseline(self, *, baseline_stage1_gelu, baseline_stage1_softmax):
        repeats = max(1, int(getattr(self, "repeat_n", 1)))
        if repeats <= 1:
            single = self._run_single_clean_baseline_eval(
                baseline_stage1_gelu=baseline_stage1_gelu,
                baseline_stage1_softmax=baseline_stage1_softmax,
            )
            single["loss_std"] = 0.0
            single["p_std"] = 0.0
            single["s_std"] = 0.0
            single["evaluation_n"] = 1
            single["evaluation_protocol"] = "single_validation_full"
            return single

        trials = self._run_clean_baseline_trials(
            baseline_stage1_gelu=baseline_stage1_gelu,
            baseline_stage1_softmax=baseline_stage1_softmax,
            repeats=repeats,
        )
        repeat = pack_repeat_evaluation(
            trials,
            evaluation_mode="clean_baseline_repeated_validation_full",
        )
        stats = repeat["stats"]
        result = {
            "name": "Baseline (Stage-1 Exact)",
            "family": "Baseline",
            "loss": float(stats["loss_mean"]),
            "p": float(stats["p_mean"]),
            "s": float(stats["s_mean"]),
            "time_ms": float(stats["time_mean_ms"]),
            "loss_std": float(stats["loss_std"]),
            "p_std": float(stats["p_std"]),
            "s_std": float(stats["s_std"]),
            "evaluation_n": int(stats["n"]),
            "evaluation_protocol": f"repeated_mean_n={int(stats['n'])}",
            "repeat_evaluation": repeat,
        }
        return result

    def _run_single_clean_baseline_eval(self, *, baseline_stage1_gelu, baseline_stage1_softmax):
        return self._run_clean_baseline_trials(
            baseline_stage1_gelu=baseline_stage1_gelu,
            baseline_stage1_softmax=baseline_stage1_softmax,
            repeats=1,
        )[0]

    def _run_clean_baseline_trials(
        self,
        *,
        baseline_stage1_gelu,
        baseline_stage1_softmax,
        repeats: int,
    ):
        ev = self.evaluator
        ev.apply_configuration(
            np.asarray(baseline_stage1_gelu, dtype=int),
            np.asarray(baseline_stage1_softmax, dtype=int),
        )
        self._clear_all_noise()
        split_name = ev._resolve_eval_split(
            use_train=False,
            split=getattr(self, "final_eval_split", FINAL_EVAL_SPLIT),
        )
        trials = []
        for _idx in range(max(1, int(repeats))):
            loss, p, s, t = ev._run_evaluation(
                ev.dataloaders[split_name],
                use_train=False,
                split_name=split_name,
            )
            trials.append({
                "name": "Baseline (Stage-1 Exact)",
                "family": "Baseline",
                "loss": float(loss),
                "p": float(p),
                "s": float(s),
                "time_ms": float(t),
            })
        return trials

    def _evaluate_action_candidate(
            self,
            *,
            name,
            action_vec,
            overrides,
            gelu,
            softmax,
            report_constraints,
            max_sfs,
            metadata=None,
            prepared_materialized=None,
            materialization_consistency=None,
            ):
        ev = self.evaluator
        total_layers = int(ev.total_layers)
        profile = str(getattr(ev, "dataset_key", "default") or "default")
        metadata = dict(metadata or {})
        if prepared_materialized is None:
            decoded = self._decode_action_candidate(
                action_vec=action_vec,
                metadata=metadata,
                max_sfs=max_sfs,
                num_layers=total_layers,
                gelu=gelu,
                softmax=softmax,
                profile=profile,
            )
            cfgs_dict = decoded.cfgs_dict()
            opt_outputs, opt_signals = self._optimizer_outputs(
                profile, cfgs_dict,
            )
            materialized = self._materialize_decoded_action(
                profile=profile,
                action_vec=action_vec,
                decoded=decoded,
                cfgs_dict=cfgs_dict,
                opt_outputs=opt_outputs,
                opt_signals=opt_signals,
            )
        else:
            materialized = prepared_materialized
            actual_action = [
                int(value)
                for value in np.asarray(action_vec, dtype=int).reshape(-1)
            ]
            if list(materialized.action_indices) != actual_action:
                raise ValueError(
                    "prepared Stage-2 materialization action mismatch"
                )
            decoded = materialized.decoded
            opt_outputs = materialized.outputs
            opt_signals = materialized.signals
        decoded = materialized.decoded
        replan_application = materialized.replan_application
        skip_reason = str(materialized.failure_reason or "")
        if skip_reason not in {
                "",
                "optimizer_invalid_chain",
                "optimizer_output_set_mismatch",
                "replan_config_not_fully_applied",
        }:
            raise RuntimeError(f"unexpected action materialization failure: {skip_reason}")
        skipped_forward = bool(skip_reason)
        if skipped_forward:
            single = {
                "loss": float("inf"),
                "p": 0.0,
                "s": 0.0,
                "time_ms": 0.0,
                "install_verification": {
                    "model_will_use_selected_cfg": False,
                    "skipped": True,
                    "skip_reason": skip_reason,
                },
            }
            repeat = None
        else:
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
            "action_metadata": to_jsonable(metadata),
            "fusion_group_diagnostics": self._fusion_group_diagnostics(
                metadata=metadata or {},
                opt_signals=opt_signals,
            ),
            "avg_truncation_k": float(avg_truncation_k_in_action(action_vec, total_layers)),
            "action_overrides": dict(overrides or {}),
            "action_vec": np.asarray(action_vec, dtype=int).copy(),
            "config_details": self._config_details(decoded, action_vec, overrides, opt_outputs),
            "replan_application": replan_application,
            "final_config_fingerprint": materialized.final_config_fingerprint,
            "materialization_consistency": to_jsonable(
                materialization_consistency or {}
            ),
            "rescale_optimizer": {
                "invoker_kind": str(getattr(self, "rescale_backend", "unknown")),
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
            "skipped_forward": bool(skipped_forward),
        }
        if skip_reason:
            result["forward_skipped_reason"] = skip_reason
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
            result["loss_std"] = 0.0
            result["p_std"] = 0.0
            result["s_std"] = 0.0
            result["evaluation_n"] = 0 if skipped_forward else 1
            result["evaluation_protocol"] = (
                f"skipped_forward:{skip_reason}"
                if skipped_forward
                else "single_validation_full"
            )
        feasibility = self._build_feasibility_report(
            result=result,
            report_constraints=report_constraints,
            optimizer_valid=not bool(materialized.optimizer_invalid),
            decode_ok=True,
            apply_ok=bool(replan_application.get("model_uses_replan_config", False)),
            eval_ok=not bool(skipped_forward),
        )
        result["final_eval_feasibility"] = feasibility
        result["feasible"] = feasibility.get("feasible")
        result["diagnostic_feasible"] = feasibility.get("diagnostic_feasible")
        result["strict_feasible"] = feasibility.get("strict_feasible")
        return result

    def _build_feasibility_report(self, *, result, report_constraints, optimizer_valid, decode_ok, apply_ok, eval_ok):
        ev = self.evaluator
        threshold_source = str(
            getattr(ev, "blb_final_eval_threshold_source", "")
            or getattr(ev, "final_eval_threshold_source", "")
            or "baseline-derived"
        )
        acc_std_limit = (
            report_constraints.get("metric1_std")
            if isinstance(report_constraints, dict) else None
        )
        f1_std_limit = (
            report_constraints.get("metric2_std")
            if isinstance(report_constraints, dict) else None
        )
        acc_std_limit = getattr(ev, "blb_final_eval_metric1_std_limit", acc_std_limit)
        f1_std_limit = getattr(ev, "blb_final_eval_metric2_std_limit", f1_std_limit)
        strict_z = getattr(ev, "blb_final_eval_strict_z", 1.0)
        return build_final_eval_feasibility(
            optimizer_valid=bool(optimizer_valid),
            decode_ok=bool(decode_ok),
            apply_ok=bool(apply_ok),
            eval_ok=bool(eval_ok),
            acc_mean=result.get("p"),
            f1_mean=result.get("s"),
            acc_std=result.get("p_std", 0.0),
            f1_std=result.get("s_std", 0.0),
            acc_limit=report_constraints.get("metric1") if isinstance(report_constraints, dict) else None,
            f1_limit=report_constraints.get("metric2") if isinstance(report_constraints, dict) else None,
            acc_std_limit=acc_std_limit,
            f1_std_limit=f1_std_limit,
            loss_mean=result.get("loss"),
            loss_std=result.get("loss_std", 0.0),
            threshold_source=threshold_source,
            strict_z=float(strict_z),
        )

    def _optimizer_outputs(self, profile: str, cfgs_dict):
        bridge = getattr(self, "rescale_bridge", None)
        if bridge is None:
            bridge, kind, root = self._build_rescale_bridge(profile)
            self.rescale_bridge = bridge
            self.rescale_backend = kind
            self.rescale_optimizer_root = root
        requests = build_optimizer_requests(profile, cfgs_dict)
        outputs = bridge.evaluate_blocks(requests)
        return outputs, aggregate_optimizer_signals(outputs)

    def _decode_action_candidate(
            self,
            *,
            action_vec,
            metadata: Mapping[str, Any],
            max_sfs,
            num_layers: int,
            gelu,
            softmax,
            profile: str,
            ):
        if str(metadata.get("schema_version", "")) == "fusion_count_fixed_action_v1":
            return self._decode_fusion_count_fixed_action(
                action_vec=action_vec,
                metadata=metadata,
                max_sfs=max_sfs,
                num_layers=int(num_layers),
                gelu=gelu,
                softmax=softmax,
                profile=str(profile),
            )
        return action_vector_to_cfgs(
            action_vec=action_vec,
            max_sfs=max_sfs,
            num_layers=int(num_layers),
            gelu_degree=np.asarray(gelu, dtype=int),
            attn_degree=np.asarray(softmax, dtype=int),
        )

    def _decode_fusion_count_fixed_action(
            self,
            *,
            action_vec,
            metadata: Mapping[str, Any],
            max_sfs,
            num_layers: int,
            gelu,
            softmax,
            profile: str,
            ) -> ActionDecodeResult:
        group = metadata.get("group")
        if not isinstance(group, Mapping):
            raise ValueError("fusion_count_fixed_action_v1 requires group metadata")
        raw_option_by_graph = group.get("option_by_graph")
        raw_option_by_step = group.get("option_by_step")
        if not isinstance(raw_option_by_graph, Mapping) and not isinstance(raw_option_by_step, Mapping):
            raise ValueError(
                "fusion_count_fixed_action_v1 requires group.option_by_step or "
                "group.option_by_graph metadata"
            )
        raw_boosted_overrides = group.get("boosted_overrides", ())
        if raw_boosted_overrides is None:
            raw_boosted_overrides = ()
        if not isinstance(raw_boosted_overrides, (list, tuple)):
            raise ValueError("group.boosted_overrides must be a sequence of rows")
        boosted_overrides = {}
        for row in raw_boosted_overrides:
            if not isinstance(row, Mapping):
                raise ValueError("group.boosted_overrides rows must be mappings")
            field_values = row.get("field_values")
            if not isinstance(field_values, Mapping):
                raise ValueError(
                    "group.boosted_overrides field_values must be a mapping"
                )
            key = (int(row["block_idx"]), int(row["layer_idx"]))
            if key in boosted_overrides:
                raise ValueError(
                    "group.boosted_overrides contains duplicate "
                    f"block/layer row {key}"
                )
            boosted_overrides[key] = {
                str(name): int(value) for name, value in field_values.items()
            }

        base_raw = None
        for key in ("legacy_action_vec", "base", "action_vec"):
            value = metadata.get(key)
            if isinstance(value, (list, tuple, np.ndarray)):
                base_raw = value
                break
        if base_raw is None:
            base_raw = action_vec
        base_arr = validate_action_vector(base_raw, int(num_layers))
        gelu_arr = np.asarray(gelu, dtype=int).reshape(-1)
        softmax_arr = np.asarray(softmax, dtype=int).reshape(-1)

        decoded = action_vector_to_cfgs(
            action_vec=base_arr,
            max_sfs=max_sfs,
            num_layers=int(num_layers),
            gelu_degree=gelu_arr,
            attn_degree=softmax_arr,
        )

        try:
            fusion_map = getattr(self, "_stage2_fusion_map", None)
            if fusion_map is None:
                from rfr.preparation.fusion.count_map import (
                    FusionCountMap as RuntimeFusionCountMap,
                )

                fusion_map = RuntimeFusionCountMap.load(str(profile))
        except Exception as exc:
            raise RuntimeError(
                f"failed to load fusion-count map for profile={profile!r}: {exc}"
            ) from exc

        option_by_graph = (
            {str(k): int(v) for k, v in raw_option_by_graph.items()}
            if isinstance(raw_option_by_graph, Mapping)
            else {}
        )
        option_by_step = (
            {str(k): int(v) for k, v in raw_option_by_step.items()}
            if isinstance(raw_option_by_step, Mapping)
            else {}
        )
        blocks = fusion_materialization_blocks(
            int(num_layers),
            profile=str(profile),
            gelu_degrees=gelu_arr.tolist(),
        )
        for block in blocks:
            graph_key = str(block.graph_key)
            step_key = str(int(block.artifact_index))
            if step_key in option_by_step:
                option_id = int(option_by_step[step_key])
            elif graph_key in option_by_graph:
                option_id = int(option_by_graph[graph_key])
            else:
                continue
            graph = fusion_map.graphs.get(graph_key)
            if graph is None:
                raise KeyError(f"fusion map missing graph {graph_key!r}")
            option = None
            for candidate in graph.options:
                if int(candidate.option_id) == option_id:
                    option = candidate
                    break
            if option is None:
                raise KeyError(f"fusion map graph {graph_key!r} has no option {option_id}")

            block_offsets = block.full_vec_offsets
            action_slice = np.take(base_arr, block_offsets)
            layer_idx = int(block.layer_idx)
            block_idx = int(block.block_idx)
            gelu_degree = int(gelu_arr[layer_idx] if gelu_arr.size > 1 else gelu_arr[0])
            softmax_degree = int(softmax_arr[layer_idx] if softmax_arr.size > 1 else softmax_arr[0])
            field_values = _decode_block_field_values(
                layer_idx,
                block_idx,
                action_slice,
                max_sfs,
                attn_degree=softmax_degree,
                gelu_degree=gelu_degree,
            )
            k_field_name = None
            selected_k_value = None
            graph_meta = fusion_map.graphs.get(graph_key)
            if graph_meta is not None:
                try:
                    k_field_name = str(
                        block_field_names(block_idx)[int(graph_meta.k_slot_index)]
                    )
                    selected_k_value = field_values.get(k_field_name)
                except Exception:
                    k_field_name = None
                    selected_k_value = None
            persisted_fields = boosted_overrides.get((block_idx, layer_idx))
            if persisted_fields is not None:
                option_fields = persisted_fields
            else:
                option_fields = (
                    option.explicit_field_values
                    if bool(getattr(option, "boosted", False)) and option.explicit_field_values
                    else option.slots
                )
            for field_name, value in option_fields.items():
                field_values[str(field_name)] = int(value)
            if (
                    k_field_name is not None
                    and selected_k_value is not None
                    and (
                        persisted_fields is None
                        or k_field_name not in persisted_fields
                    )
            ):
                field_values[str(k_field_name)] = int(selected_k_value)
            if decoded.per_layer_field_values and 0 <= layer_idx < len(decoded.per_layer_field_values):
                layer_values = decoded.per_layer_field_values[layer_idx]
                if isinstance(layer_values, dict):
                    layer_values[f"block{block_idx}"] = dict(field_values)

            block_cfg = build_block_cfg_from_field_values(
                block_idx,
                layer_idx,
                field_values,
                N=int(_block_default_N(
                    block_idx,
                    gelu_degree=gelu_degree,
                    attn_degree=softmax_degree,
                )),
                gelu_degree=gelu_degree,
                attn_degree=softmax_degree,
            )
            getattr(decoded, f"block{block_idx}_cfgs")[layer_idx] = block_cfg
        return decoded

    def _materialize_decoded_action(
            self,
            *,
            profile: str,
            action_vec,
            decoded,
            cfgs_dict,
            opt_outputs,
            opt_signals,
            truncation_backend=None,
            truncation_ring_bits=None,
            truncation_source_fractional_bits=None,
            rotation_name_map_provider=None,
            ):
        """Apply Rescale_optimizer/replan results to cfgs before model forward.

        The executable Stage-2 cfg is the optimizer's ``new_compact_config``
        mirrored back into action-decoded cfg objects.  Delegate to the shared
        helper used by RL training and fixed-action experiments so every path
        installs the same cfg before inference.
        """
        bridge = getattr(self, "rescale_bridge", None)
        invoker = getattr(bridge, "invoker", None)
        invoker_baselines: Mapping[str, Any] = getattr(invoker, "baselines", {}) or {}
        evaluator_backend = getattr(
            self.evaluator, "blb_v3_truncation_backend", "binary",
        )
        evaluator_ring_bits = getattr(
            self.evaluator, "blb_v3_truncation_ring_bits", 43,
        )
        evaluator_source_bits = getattr(
            self.evaluator,
            "blb_v3_truncation_source_fractional_bits",
            24,
        )
        backend = str(
            truncation_backend
            if truncation_backend is not None
            else (
                "binary" if evaluator_backend is None else evaluator_backend
            )
        )
        ring_bits = int(
            truncation_ring_bits
            if truncation_ring_bits is not None
            else (43 if evaluator_ring_bits is None else evaluator_ring_bits)
        )
        source_fractional_bits = int(
            truncation_source_fractional_bits
            if truncation_source_fractional_bits is not None
            else (24 if evaluator_source_bits is None else evaluator_source_bits)
        )


        return materialize_decoded_action(
            action_indices=np.asarray(action_vec, dtype=int).reshape(-1).tolist(),
            decoded=decoded,
            profile=str(profile),
            cfgs_dict=cfgs_dict,
            outputs=opt_outputs,
            signals=opt_signals,
            invoker_baselines=invoker_baselines,
            rotation_name_map_provider=(
                rotation_name_map_provider or self._rotation_name_map_for
            ),
            truncation_backend=backend,
            truncation_ring_bits=ring_bits,
            truncation_source_fractional_bits=source_fractional_bits,
        )

    def _rotation_name_map_for(self, block_idx: int, profile: str) -> Mapping[str, str]:
        raw = (
            getattr(self.evaluator, "blb_v3_rotation_name_map", None)
            or getattr(self.evaluator, "rotation_name_map", None)
            or {}
        )
        if not isinstance(raw, Mapping):
            return {}
        direct = raw.get((int(block_idx), str(profile)))
        if isinstance(direct, Mapping):
            return direct
        nested = raw.get(int(block_idx)) or raw.get(str(block_idx))
        if isinstance(nested, Mapping):
            profiled = nested.get(str(profile))
            if isinstance(profiled, Mapping):
                return profiled
            if all(isinstance(k, str) for k in nested.keys()):
                return nested
        return {}

    @staticmethod
    def _fusion_group_diagnostics(*, metadata: Mapping[str, Any], opt_signals) -> Dict[str, Any]:
        group = metadata.get("group") if isinstance(metadata, Mapping) else None
        if not isinstance(group, Mapping):
            return {}
        by_graph = group.get("fusion_count_by_graph") or {}
        counts = group.get("occurrence_counts") or {}
        declared_total = 0
        declared_by_graph: Dict[str, int] = {}
        if isinstance(by_graph, Mapping):
            for graph_key, fusion_count in by_graph.items():
                occurrences = 1
                if isinstance(counts, Mapping):
                    occurrences = int(counts.get(graph_key, 1))
                value = int(fusion_count) * int(occurrences)
                declared_by_graph[str(graph_key)] = value
                declared_total += value
        realized_total = int(getattr(opt_signals, "total_fusion_count", 0))
        return {
            "group_name": str(group.get("name", "")),
            "declared_total_fusion_count": int(declared_total),
            "realized_total_fusion_count": int(realized_total),
            "declared_by_graph": declared_by_graph,
            "matches_realized_total": bool(int(declared_total) == int(realized_total)),
        }

    def _build_rescale_bridge(
            self,
            profile: str,
            ) -> Tuple[RescaleOptimizerBridge, str, str]:
        root = self._resolve_rescale_optimizer_root()
        try:
            invoker = build_rescale_invoker(root=root, profile=str(profile))
        except Exception as exc:
            raise RuntimeError(
                f"BLB final_eval failed to initialize in-process Rescale "
                f"for profile={profile!r}, root={root!r}: {exc}"
            ) from exc
        bridge = RescaleOptimizerBridge(
            invoker=invoker,
            **self._rescale_bridge_options(),
        )
        return bridge, "in_process", root

    def _load_rescale_optimizer_mode(self) -> str:
        if not self.action_config_path:
            return "cfg_derived"
        try:
            payload = read_json_file(self.action_config_path, encoding="utf-8-sig")
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
            or "configs/preparation/rescale"
        )
        path = Path(str(raw))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
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
            "first_input": set(),
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
        identity_match = self._handler_cfg_identity_match(decoded, expected)
        return {
            "checked_before_forward": True,
            "handler_active_layers": active_json,
            "expected_active_layers": expected_json,
            "handler_active_layers_match_expected": bool(handler_match),
            "bridge_installed_layers_match_expected": bool(bridge_match),
            "handler_cfg_objects_match_decoded_cfgs": bool(identity_match),
            "model_will_use_selected_cfg": bool(handler_match and bridge_match and identity_match),
        }

    def _handler_cfg_identity_match(self, decoded, expected_layers_by_block: Optional[Mapping[str, set]] = None) -> bool:
        handler = self.evaluator.reversible_handler
        for block_name in ("block1", "block2", "block3", "block4", "block5"):
            expected = getattr(decoded, f"{block_name}_cfgs")
            installed = getattr(handler, f"{block_name}_cfg_per_layer", {})
            expected_layers = None
            if expected_layers_by_block is not None:
                expected_layers = set(expected_layers_by_block.get(block_name, set()))
            for layer_idx, cfg in expected.items():
                if expected_layers is not None and int(layer_idx) not in expected_layers:
                    continue
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
        entries = []

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
        repeats = self.repeat_n
        if repeats <= 1:
            return self._run_single_blb_eval(decoded, gelu=gelu, softmax=softmax), None
        trials = self._run_blb_eval_trials(
            decoded,
            gelu=gelu,
            softmax=softmax,
            repeats=repeats,
        )
        repeat = pack_repeat_evaluation(
            trials,
            evaluation_mode="blb_action_repeated_validation_full",
        )
        return {
            "loss": float(repeat["stats"]["loss_mean"]),
            "p": float(repeat["stats"]["p_mean"]),
            "s": float(repeat["stats"]["s_mean"]),
            "time_ms": float(repeat["stats"]["time_mean_ms"]),
            "install_verification": trials[0].get("install_verification", {}) if trials else {},
        }, repeat

    def _run_single_blb_eval(self, decoded, *, gelu, softmax):
        return self._run_blb_eval_trials(
            decoded,
            gelu=gelu,
            softmax=softmax,
            repeats=1,
        )[0]

    def _run_blb_eval_trials(self, decoded, *, gelu, softmax, repeats: int):
        ev = self.evaluator
        bridge = BLBNoiseRLBridge(
            ev.reversible_handler,
            layers_attribute="model." + ev.layers_attribute,
        )
        ev.apply_configuration(gelu, softmax)
        self._clear_legacy_noise()
        try:


            bridge.apply(
                block1_cfgs=decoded.block1_cfgs,
                block2_cfgs=decoded.block2_cfgs,
                block3_cfgs=decoded.block3_cfgs,
                block4_cfgs=decoded.block4_cfgs,
                block5_cfgs=decoded.block5_cfgs,
            )
            install_verification = self._verify_model_installation(bridge, decoded)
            if install_verification.get("model_will_use_selected_cfg") is not True:
                raise RuntimeError(
                    "selected configuration installation verification failed"
                )
            split_name = ev._resolve_eval_split(
                use_train=False,
                split=getattr(self, "final_eval_split", FINAL_EVAL_SPLIT),
            )
            trials = []
            for _idx in range(max(1, int(repeats))):
                loss, p, s, time_ms = ev._run_evaluation(
                    ev.dataloaders[split_name],
                    use_train=False,
                    split_name=split_name,
                )
                trials.append({
                    "loss": float(loss),
                    "p": float(p),
                    "s": float(s),
                    "time_ms": float(time_ms),
                    "install_verification": install_verification,
                })
            return trials
        finally:
            bridge.clear()
            self._clear_all_noise()

    def _clear_legacy_noise(self):
        ev = self.evaluator
        handler = ev.reversible_handler
        layer_indices = list(range(ev.total_layers))
        try:
            handler.restore_layer_input_noise(layer_indices=layer_indices)
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
                method(layer_indices=layer_indices)
            except Exception:
                pass

    def _clear_all_noise(self):
        self._clear_legacy_noise()
        ev = self.evaluator
        layer_indices = list(range(ev.total_layers))
        layer_name = "model." + ev.layers_attribute
        for restore_name in (
            "restore_layer_block5_noise",
            "restore_layer_block4_noise",
            "restore_layer_block3_noise",
            "restore_layer_block2_noise",
            "restore_layer_block1_noise",
        ):
            method = getattr(ev.reversible_handler, restore_name, None)
            if method is None:
                continue
            try:
                method(
                    layer_indices=layer_indices,
                    layer_name=layer_name,
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
            f"- rescale_optimizer: `{getattr(self, 'rescale_backend', 'unknown')}`",
            f"- rescale_optimizer_root: `{getattr(self, 'rescale_optimizer_root', '') or '(none)'}`",
            f"- json: `{json_path}`",
            "",
            "## Baseline",
            "",
            f"- clean baseline loss: `{baseline_result['loss']:.6f}`",
            f"- clean baseline {primary}: `{baseline_result['p']:.6f}`",
            f"- clean baseline {secondary}: `{baseline_result['s']:.6f}`",
            f"- clean baseline protocol: `{baseline_result.get('evaluation_protocol', 'single_validation_full')}`",
            f"- clean baseline loss std: `{float(baseline_result.get('loss_std', 0.0)):.6f}`",
            f"- clean baseline {primary} std: `{float(baseline_result.get('p_std', 0.0)):.6f}`",
            f"- clean baseline {secondary} std: `{float(baseline_result.get('s_std', 0.0)):.6f}`",
            "",
        ]
        lines.extend([
            "## Selected Configuration",
            "",
            "| group | truncation k | effective K positions | loss mean | loss std | "
            f"{primary} mean | {primary} std | {secondary} mean | {secondary} std | "
            "time mean ms | total bits | fusion | replan applied | model cfg verified |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ])
        for result in candidate_results:
            trunc = result.get("config_details", {}).get("truncation", {})
            unique_k = self._unique_truncation_label(trunc)
            verify = result.get("install_verification", {}).get("model_will_use_selected_cfg", False)
            replan_ok = result.get("replan_application", {}).get("model_uses_replan_config", False)
            lines.append(
                f"| `{result['name']}` | {unique_k} | "
                f"{int(trunc.get('effective_position_count', 0))} | "
                f"{float(result['loss']):.6f} | {float(result.get('loss_std', 0.0)):.6f} | "
                f"{float(result['p']):.6f} | {float(result.get('p_std', 0.0)):.6f} | "
                f"{float(result['s']):.6f} | {float(result.get('s_std', 0.0)):.6f} | "
                f"{float(result['time_ms']):.3f} | {int(result['total_bits_sum'])} | "
                f"{int(result['total_fusion_count'])} | {replan_ok} | {verify} |"
            )

        lines.extend(["", "## Configuration Details", ""])
        for result in candidate_results:
            details = result.get("config_details", {})
            trunc = details.get("truncation", {})
            verify = result.get("install_verification", {})
            replan_application = result.get("replan_application", {})
            fusion_group = result.get("fusion_group_diagnostics", {})
            lines.extend([
                f"### {result['name']}",
                "",
                f"- action overrides: `{result.get('action_overrides', {})}`",
                f"- base action: {details.get('base_action', '')}",
                f"- truncation summary: `{self._unique_truncation_label(trunc)}`; "
                f"effective positions = `{trunc.get('effective_position_count', 0)}`; "
                f"skipped = `{trunc.get('skipped_positions', [])}`",
                f"- model cfg verified before forward: `{verify.get('model_will_use_selected_cfg', False)}`",
                f"- replan cfg applied before forward: `{replan_application.get('model_uses_replan_config', False)}`",
                f"- replan application summary: `{ {k: v for k, v in replan_application.items() if k != 'per_config'} }`",
                f"- fusion group diagnostics: `{fusion_group}`",
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
        entries = (full_config or {}).get("entries", ()) or ()
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
        action_context_provenance: Optional[Mapping[str, Any]] = None,
        final_eval_handoff: Optional[Mapping[str, Any]] = None,  # noqa: UP045
    ):
        handoff = to_jsonable(final_eval_handoff)
        output = {
            "schema_version": "paean_blb_action_final_eval_result_v1",
            "status": "complete",
            "dataset": self.evaluator.dataset_key,
            "final_eval_split": getattr(
                self, "final_eval_split", FINAL_EVAL_SPLIT
            ),
            "dataset_protocol_hash": getattr(
                self.evaluator, "dataset_protocol_hash", None
            ),
            "selected_source": selected_source,
            "stage2_final_eval_handoff": handoff,
            "baseline_stage1": {
                "gelu": np.asarray(baseline_stage1_gelu, dtype=int).tolist(),
                "softmax": np.asarray(baseline_stage1_softmax, dtype=int).tolist(),
            },
            "selected_stage1": {
                "gelu": np.asarray(opt_gelu, dtype=int).tolist(),
                "softmax": np.asarray(opt_softmax, dtype=int).tolist(),
            },
            "constraints": {"selection": selection_constraints},
            "baseline": to_jsonable(baseline_result),
            "candidate_results": [to_jsonable(r) for r in candidate_results],
            "calibrated_action_context": to_jsonable(
                action_context_provenance or {}
            ),
            "evaluation_protocol": {
                "version": 3,
                "mode": "selected_configuration_repeated_validation_full",
                "candidate_count": 1,
                "repeat_n": int(self.repeat_n),
                "random_seed": int(self.random_seed),
            },
        }
        output_path = os.path.join(
            self.results_dir,
            f"blb_action_final_eval_results_{self.evaluator.dataset_key}.json",
        )
        _atomic_json(output_path, output)
        return output_path

    @staticmethod
    def _attach_relative_metrics(baseline, results):
        for result in results:
            result["delta_loss_vs_baseline"] = float(result["loss"] - baseline["loss"])
            result["delta_p_vs_baseline"] = float(result["p"] - baseline["p"])
            result["delta_s_vs_baseline"] = float(result["s"] - baseline["s"])
