#!/usr/bin/env python3
"""Evaluate fixed fusion-count groups through the Stage-2 RL terminal path.

This diagnostic runner intentionally mirrors the online training install path:

    BLBStage2SequentialEnv.evaluate_step
    -> BLBStage2SequentialEnv.commit_step
    -> BLBStage2Env.step(..., boosted_overrides=...)

It uses the same action-config groups as ``run_fusion_count_action_eval.py`` but
does not call the Paean final-eval decoder.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from json_utils import read_json_file, to_jsonable, write_json_file
from report_format_utils import format_float, html_table, metric_float
from scripts.fusion_count_action_eval_common import (
    load_rlpath_action_configs,
    parse_json_int_list,
    resolve_repo_path,
    rlpath_config_group_key,
    unique_rlpath_action_configs,
)
from scripts.fusion_count_prediction_capture import (
    PREDICTION_ROW_SCHEMA,
    ExampleIdentityCatalog,
    ForwardPredictionRecorder,
    PredictionJsonlWriter,
)


DEFAULT_STAGE1_GELU = [4] * 12
DEFAULT_STAGE1_SOFTMAX = [6] * 12
DEFAULT_STAGE1_GELU_JSON = json.dumps(DEFAULT_STAGE1_GELU)
DEFAULT_STAGE1_SOFTMAX_JSON = json.dumps(DEFAULT_STAGE1_SOFTMAX)

_RUNTIME_DEPS: dict[str, object] | None = None


def _group_seed(base_seed: int, group_index: int, *, shared: bool) -> int:
    return int(base_seed) if shared else int(base_seed) + int(group_index)


def _trial_metric_value(value: Any) -> Any:
    numeric = float(value)
    if math.isfinite(numeric):
        return numeric
    if math.isnan(numeric):
        return {"non_finite": "nan"}
    if numeric > 0:
        return {"non_finite": "positive_infinity"}
    return {"non_finite": "negative_infinity"}


def _trial_metric_payload(
        losses: Sequence[float],
        metric1s: Sequence[float],
        metric2s: Sequence[float],
        ) -> dict:
    return {
        "loss": [_trial_metric_value(value) for value in losses],
        "metric1": [_trial_metric_value(value) for value in metric1s],
        "metric2": [_trial_metric_value(value) for value in metric2s],
    }


class _HtmlPartsWriter:
    def __init__(self, path: Path):
        self._handle = path.open("w", encoding="utf-8")
        self._first = True

    def append(self, value: Any) -> None:
        if not self._first:
            self._handle.write("\n")
        self._handle.write(str(value))
        self._first = False

    def extend(self, values: Iterable[Any]) -> None:
        for value in values:
            self.append(value)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "_HtmlPartsWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _load_runtime_deps() -> dict[str, object]:
    """Import model/RL dependencies only for the actual RL-path evaluation."""
    global _RUNTIME_DEPS
    if _RUNTIME_DEPS is None:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
        )

        from blb_stage2_rl.action_space import K_LEVELS
        from blb_stage2_rl.baseline_bootstrap import (
            load_static_skeletons_baseline,
            static_skeletons_baseline_to_action,
        )
        from blb_stage2_rl.env import BLBStage2Env, BLBStage2EnvConfig, estimate_baseline_cost_stats
        from blb_stage2_rl.fusion_count_map import FusionCountMap
        from blb_stage2_rl.reward import BaselineCostStats, RewardWeights, calibrate_weights_from_baseline
        from blb_stage2_rl.runner import BLBStage2RLRunner, BLBStage2TrainConfig
        from blb_stage2_rl.sequential_env import BLBStage2SequentialEnv, SequentialEnvConfig
        from layer_importance_evaluator import LayerImportanceEvaluator
        from rl_tune import load_glue_dataset_equivalent, seed_everything

        _RUNTIME_DEPS = {
            "torch": torch,
            "AutoConfig": AutoConfig,
            "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
            "AutoTokenizer": AutoTokenizer,
            "DataCollatorWithPadding": DataCollatorWithPadding,
            "load_glue_dataset_equivalent": load_glue_dataset_equivalent,
            "seed_everything": seed_everything,
            "LayerImportanceEvaluator": LayerImportanceEvaluator,
            "K_LEVELS": K_LEVELS,
            "load_static_skeletons_baseline": load_static_skeletons_baseline,
            "static_skeletons_baseline_to_action": static_skeletons_baseline_to_action,
            "BLBStage2Env": BLBStage2Env,
            "BLBStage2EnvConfig": BLBStage2EnvConfig,
            "estimate_baseline_cost_stats": estimate_baseline_cost_stats,
            "FusionCountMap": FusionCountMap,
            "BaselineCostStats": BaselineCostStats,
            "RewardWeights": RewardWeights,
            "calibrate_weights_from_baseline": calibrate_weights_from_baseline,
            "BLBStage2RLRunner": BLBStage2RLRunner,
            "BLBStage2TrainConfig": BLBStage2TrainConfig,
            "BLBStage2SequentialEnv": BLBStage2SequentialEnv,
            "SequentialEnvConfig": SequentialEnvConfig,
        }
    return _RUNTIME_DEPS


def _base_model(model_type: str, dataset: str) -> str:
    if model_type != "bert-base" or dataset != "mrpc":
        raise ValueError("this diagnostic script currently supports bert-base MRPC only")
    return "textattack/bert-base-uncased-MRPC"


def _tokenize_glue(
        data,
        *,
        task: str,
        tokenizer,
        seed: int,
        include_identity_catalog: bool = False,
        ):
    def tokenize_fn(examples):
        if task == "mrpc":
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding=False,
                max_length=128,
                return_tensors=None,
            )
        raise ValueError(f"unsupported task {task!r}")

    train_data = data["train"].shuffle(seed=int(seed)).map(tokenize_fn)
    val_data = data["validation"].shuffle(seed=int(seed)).map(tokenize_fn)
    train_data = train_data.rename_column("label", "labels")
    val_data = val_data.rename_column("label", "labels")
    identity_catalog = None
    if include_identity_catalog:
        identity_catalog = ExampleIdentityCatalog.from_tokenized_rows(val_data)
    cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    train_data.set_format(type="torch", columns=cols)
    val_data.set_format(type="torch", columns=cols)
    if include_identity_catalog:
        return train_data, val_data, identity_catalog
    return train_data, val_data


def _build_evaluator(args, *, stage1_gelu: Sequence[int], stage1_softmax: Sequence[int]):
    deps = _load_runtime_deps()
    torch = deps["torch"]
    AutoTokenizer = deps["AutoTokenizer"]
    AutoConfig = deps["AutoConfig"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    DataCollatorWithPadding = deps["DataCollatorWithPadding"]
    load_glue_dataset_equivalent = deps["load_glue_dataset_equivalent"]
    seed_everything = deps["seed_everything"]
    LayerImportanceEvaluator = deps["LayerImportanceEvaluator"]

    seed_everything(int(args.seed))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    cfg = AutoConfig.from_pretrained(args.base_model)
    _ = cfg
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    data = load_glue_dataset_equivalent(
        args.dataset,
        route_log_dir=str(resolve_repo_path(args.output_json).parent / "logs"),
    )
    tokenized = _tokenize_glue(
        data,
        task=args.dataset,
        tokenizer=tokenizer,
        seed=int(args.seed),
        include_identity_catalog=bool(args.prediction_jsonl),
    )
    if args.prediction_jsonl:
        train_data, val_data, identity_catalog = tokenized
    else:
        train_data, val_data = tokenized
        identity_catalog = None
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )
    ev = LayerImportanceEvaluator(
        model=model,
        train_data=train_data,
        test_data=val_data,
        data_collator=collator,
        batch_size=int(args.batch_size),
        stage1_rl_episodes=51000,
        stage2_rl_episodes=40000,
        stage1_rl_episodes_specified=False,
        stage2_rl_episodes_specified=False,
        run_output_dir=str(resolve_repo_path(args.run_output_dir)),
        final_eval_config_source="json",
        final_eval_config_path=str(resolve_repo_path(args.stage1_config_json)),
        manual_stage1_gelu=[int(v) for v in stage1_gelu],
        manual_stage1_softmax=[int(v) for v in stage1_softmax],
        skip_stage1_rl=True,
        skip_noise_rl=True,
        skip_final_eval=True,
        data_path=args.dataset,
        stage1_accuracy_tolerance=0.001,
        stage2_limit_tolerance=float(args.stage2_limit_tolerance),
        stage2_stability_tolerance=float(args.stage2_stability_tolerance),
        stage2_k_trials=int(args.repeat),
        stage2_probe_size=int(args.probe_size),
        stage2_rl_variant="blb_v3",
        blb_v3_inproc_rescale_optimizer_root=str(resolve_repo_path(args.rescale_optimizer_root)),
        blb_v3_fusion_count_action=True,
        blb_v3_fusion_neighbor_curriculum=False,
        blb_v3_fusion_probe_interval=0,
        blb_v3_fusion_exploration_epsilon=0.0,
        rl_algo="ppo",
    )
    if identity_catalog is not None:
        ev.fixed_eval_identity_catalog = identity_catalog
    return ev


def _build_seq_env(args, ev, *, stage1_gelu: Sequence[int], stage1_softmax: Sequence[int]):
    deps = _load_runtime_deps()
    BLBStage2RLRunner = deps["BLBStage2RLRunner"]
    BLBStage2TrainConfig = deps["BLBStage2TrainConfig"]
    load_static_skeletons_baseline = deps["load_static_skeletons_baseline"]
    static_skeletons_baseline_to_action = deps["static_skeletons_baseline_to_action"]
    BLBStage2Env = deps["BLBStage2Env"]
    BLBStage2EnvConfig = deps["BLBStage2EnvConfig"]
    estimate_baseline_cost_stats = deps["estimate_baseline_cost_stats"]
    BaselineCostStats = deps["BaselineCostStats"]
    RewardWeights = deps["RewardWeights"]
    calibrate_weights_from_baseline = deps["calibrate_weights_from_baseline"]
    FusionCountMap = deps["FusionCountMap"]
    BLBStage2SequentialEnv = deps["BLBStage2SequentialEnv"]
    SequentialEnvConfig = deps["SequentialEnvConfig"]

    class RecordingBLBStage2Env(BLBStage2Env):
        def _aggregate_probe_trials(
                self,
                per_trial_loss,
                per_trial_metric1,
                per_trial_metric2,
                ):
            self.fixed_eval_trial_metrics = _trial_metric_payload(
                per_trial_loss,
                per_trial_metric1,
                per_trial_metric2,
            )
            return super()._aggregate_probe_trials(
                per_trial_loss,
                per_trial_metric1,
                per_trial_metric2,
            )

    runner = BLBStage2RLRunner(ev)
    train_cfg = BLBStage2TrainConfig(
        total_episodes=1,
        rollout_size=1,
        seed=int(args.seed),
        profile=str(args.dataset),
        num_trials_per_step=int(args.repeat),
        probe_batch_count=1,
        calibrate_baseline_samples=1,
        inproc_rescale_optimizer_root=str(resolve_repo_path(args.rescale_optimizer_root)),
        fusion_count_action=True,
        fusion_neighbor_curriculum_enabled=False,
        fusion_probe_interval=0,
        fusion_exploration_epsilon=0.0,
    )

    ev.apply_configuration(np.asarray(stage1_gelu, dtype=int), np.asarray(stage1_softmax, dtype=int))
    try:
        ev.reversible_handler.restore_layer_input_noise(layer_indices=list(range(ev.total_layers)))
    except Exception:
        pass

    probe_batches = runner._build_probe_batches(ev, train_cfg)
    train_cfg.probe_batch_count = max(1, int(len(probe_batches) or train_cfg.probe_batch_count))
    rescale_bridge = runner._build_rescale_bridge(train_cfg, log=lambda m: print(m, flush=True))
    ss_baseline = load_static_skeletons_baseline(
        rescale_optimizer_root=str(resolve_repo_path(args.rescale_optimizer_root)),
        dataset=str(args.dataset),
        num_layers=int(ev.total_layers),
        gelu_per_layer=[int(x) for x in stage1_gelu],
        softmax_per_layer=[int(x) for x in stage1_softmax],
    )
    _baseline_vec, max_sfs, ss_cost_stats, _diag = static_skeletons_baseline_to_action(
        ss_baseline,
        snap_sf_to_noise_table=False,
    )
    base_env = RecordingBLBStage2Env(
        handler=ev.reversible_handler,
        model=ev.model,
        probe_batches=probe_batches,
        rescale_bridge=rescale_bridge,
        baseline=BaselineCostStats(),
        reward_weights=RewardWeights(),
        acc_threshold=0.0,
        stab_threshold=float("inf"),
        max_sfs=max_sfs,
        num_layers=int(ev.total_layers),
        gelu_degree=np.asarray(stage1_gelu, dtype=int),
        attn_degree=np.asarray(stage1_softmax, dtype=int),
        layers_attribute="model." + ev.layers_attribute,
        is_regression=bool(getattr(ev, "is_regression", False)),
        env_cfg=BLBStage2EnvConfig(
            profile=str(args.dataset),
            num_trials_per_step=int(args.repeat),
            probe_batch_count=train_cfg.probe_batch_count,
            borderline_retest_enabled=False,
            borderline_retest_trials_multiplier=1,
        ),
    )
    precomputed = {
        "total_bits_sum": int(ss_cost_stats.total_bits_sum),
        "total_fusion_count": int(ss_cost_stats.total_fusion_count),
        "avg_k": float(ss_cost_stats.avg_k),
    }
    baseline = estimate_baseline_cost_stats(
        base_env,
        sample_count=1,
        precomputed_baseline_signals=precomputed,
    )
    base_env.baseline = baseline
    clean = runner._estimate_baseline_metrics(base_env)
    baseline.loss_mean = float(clean.loss_mean)
    baseline.loss_std = float(clean.loss_std)
    baseline.metric1_mean = float(clean.metric1_mean)
    baseline.metric2_mean = float(clean.metric2_mean)
    baseline.metric1_std = float(getattr(clean, "metric1_std", 0.0) or 0.0)
    baseline.metric2_std = float(getattr(clean, "metric2_std", 0.0) or 0.0)
    base_env.reward_weights = calibrate_weights_from_baseline(baseline)
    base_env.sync_degree_vectors_from_model()

    fusion_map = FusionCountMap.load(str(args.dataset))
    seq_env = BLBStage2SequentialEnv(
        base_env=base_env,
        env_cfg=SequentialEnvConfig(
            invalid_penalty=1.0,
            cost_shaping_coeff=0.0,
            fusion_shaping_coeff=0.0,
            early_terminate_on_invalid=False,
        ),
        fusion_map=fusion_map,
    )
    return seq_env, baseline


def _metric_dict(metrics: Any) -> Dict[str, float]:
    if metrics is None:
        return {}
    if hasattr(metrics, "__dataclass_fields__"):
        raw = asdict(metrics)
    elif isinstance(metrics, Mapping):
        raw = dict(metrics)
    else:
        raw = {}
    out = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            pass
    return out


def _run_group_canonical(seq_env, cfg: Mapping[str, Any], *, seed: int) -> dict:
    K_LEVELS = _load_runtime_deps()["K_LEVELS"]
    group = cfg.get("group") or {}
    option_by_graph = {str(k): int(v) for k, v in dict(group.get("option_by_graph") or {}).items()}
    option_by_step = {str(k): int(v) for k, v in dict(group.get("option_by_step") or {}).items()}
    k_index = int(cfg.get("baseline_k_index", 3))
    if k_index < 0 or k_index >= len(K_LEVELS):
        raise ValueError(f"invalid baseline_k_index={k_index} for {cfg['name']}")

    seq_env.base.fixed_eval_trial_metrics = {}
    seq_env.reset(seed=int(seed))
    seq_env.base.probe_noise_seed = int(seed)
    done = False
    info: Dict[str, Any] = {}
    reward = 0.0
    step_records: List[dict] = []
    while not done:
        spec = seq_env._schedule[seq_env._step_idx]
        graph_key = str(spec.graph_key_suffix)
        map_option_id = int(option_by_step.get(str(spec.step_idx), option_by_graph.get(graph_key, 0)))
        policy_option_index = 0
        eval_info = seq_env.evaluate_step(
            [policy_option_index, k_index],
            map_option_id_override=map_option_id,
        )
        _obs, reward, done, info = seq_env.commit_step(eval_info, defer_terminal_forward=False)
        replan_application = info.get("replan_application") or {}
        step_records.append({
            "step_idx": int(spec.step_idx),
            "layer_idx": int(spec.layer_idx),
            "block_idx": int(spec.block_idx),
            "graph_key": graph_key,
            "option_id": int(map_option_id),
            "map_option_id": int(map_option_id),
            "policy_option_index": int(policy_option_index),
            "k_index": int(k_index),
            "k_value": int(K_LEVELS[k_index]),
            "valid": bool(eval_info.get("valid", False)),
            "fusion_count_replan": int(eval_info.get("fusion_count", 0) or 0),
            "boosted": bool(eval_info.get("boosted_field_values")),
            "replan_application": to_jsonable(
                replan_application,
                stringify_unknown=True,
                preserve_native=True,
            ),
            "model_uses_replan_config": bool(
                replan_application.get("model_uses_replan_config", False)
            ),
        })
        if done:
            break

    terminal_info = dict(info.get("terminal_info") or {})
    metrics = _metric_dict(terminal_info.get("metrics") or info.get("metrics"))
    action_steps = terminal_info.get("fusion_action_steps") or []
    fusion_total = sum(int(x.get("fusion_count", 0) or 0) for x in action_steps if isinstance(x, Mapping))
    fusion_by_block: Dict[str, int] = {}
    k_dist: Dict[str, int] = {}
    block5_graphs: Dict[str, int] = {}
    for x in action_steps:
        if not isinstance(x, Mapping):
            continue
        bi = str(int(x.get("block_idx", -1)))
        fusion_by_block[bi] = fusion_by_block.get(bi, 0) + int(x.get("fusion_count", 0) or 0)
        kv = str(int(x.get("k_value", -1)))
        k_dist[kv] = k_dist.get(kv, 0) + 1
        g = str(x.get("graph_key", ""))
        if g.startswith("block5_"):
            block5_graphs[g.split("_L")[0]] = block5_graphs.get(g.split("_L")[0], 0) + 1
    if not action_steps:
        fusion_total = sum(int(x.get("fusion_count_replan", 0) or 0) for x in step_records)
        for x in step_records:
            bi = str(int(x.get("block_idx", -1)))
            fusion_by_block[bi] = fusion_by_block.get(bi, 0) + int(x.get("fusion_count_replan", 0) or 0)
            kv = str(int(x.get("k_value", -1)))
            k_dist[kv] = k_dist.get(kv, 0) + 1
            g = str(x.get("graph_key", ""))
            if g.startswith("block5_"):
                block5_graphs[g.split("_L")[0]] = block5_graphs.get(g.split("_L")[0], 0) + 1
    return {
        "name": str(cfg["name"]),
        "action_config_path": str(cfg["path"]),
        "fusion_group": group,
        "reward": float(reward),
        "terminal_priority": terminal_info.get("terminal_priority"),
        "metrics": metrics,
        "trial_metrics": to_jsonable(
            seq_env.base.fixed_eval_trial_metrics,
            stringify_unknown=True,
            preserve_native=True,
        ),
        "fusion_total": int(fusion_total),
        "fusion_by_block": fusion_by_block,
        "k_distribution": k_dist,
        "block5_graph_counts": block5_graphs,
        "terminal_probe": to_jsonable(
            terminal_info.get("probe_diagnostics") or {},
            stringify_unknown=True,
            preserve_native=True,
        ),
        "reward_breakdown": to_jsonable(
            terminal_info.get("reward_breakdown") or {},
            stringify_unknown=True,
            preserve_native=True,
        ),
        "step_records": step_records,
        "fusion_action_steps": to_jsonable(
            action_steps,
            stringify_unknown=True,
            preserve_native=True,
        ),
    }


def _run_group(
        seq_env,
        cfg: Mapping[str, Any],
        *,
        seed: int,
        prediction_recorder=None,
        prediction_writer=None,
        ) -> dict:
    if prediction_recorder is None:
        return _run_group_canonical(seq_env, cfg, seed=seed)
    if prediction_writer is None:
        raise ValueError("prediction_writer is required with prediction_recorder")

    prediction_recorder.begin_group(run_seed=seed, group=str(cfg["name"]))
    try:
        result = _run_group_canonical(seq_env, cfg, seed=seed)
        terminal_probe = result["terminal_probe"]
        trial_seeds = terminal_probe["per_worker_trial_seeds"][0]
        prediction_rows = prediction_recorder.finish_group(
            trial_seeds=trial_seeds,
        )
        prediction_writer.write_rows(prediction_rows)
        result["prediction_capture"] = {"row_count": len(prediction_rows)}
        return result
    except Exception:
        prediction_recorder.abort_group()
        raise


def _emit_rendered_html(combined: Mapping[str, Any], parts: Any) -> None:
    rows = []
    for r in combined["group_results"]:
        m = r.get("metrics") or {}
        orig = r.get("original_metrics") or {}
        rows.append([
            r["name"],
            format_float(m.get("loss_mean")),
            format_float(m.get("loss_std")),
            format_float(m.get("metric1_mean")),
            format_float(m.get("metric1_std")),
            format_float(m.get("metric2_mean")),
            format_float(m.get("metric2_std")),
            format_float(orig.get("loss")),
            format_float(orig.get("loss_std")),
            format_float(orig.get("p")),
            format_float(orig.get("p_std")),
            format_float(orig.get("s")),
            format_float(orig.get("s_std")),
            format_float(r.get("delta_loss_mean_vs_original")),
            format_float(r.get("delta_metric1_mean_vs_original")),
            format_float(r.get("delta_metric2_mean_vs_original")),
            r.get("fusion_total", ""),
            json.dumps(r.get("fusion_by_block", {}), ensure_ascii=False),
            json.dumps(r.get("k_distribution", {}), ensure_ascii=False),
            json.dumps(r.get("block5_graph_counts", {}), ensure_ascii=False),
            r.get("terminal_priority", ""),
        ])
    parts.extend([
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>MRPC Fusion Count RL-Path Eval</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:28px;color:#111827;background:#fbfcfd}"
        "table{border-collapse:collapse;width:100%;background:white;margin:14px 0}th,td{border:1px solid #d9e2ec;padding:6px 8px;font-size:12px;text-align:left;vertical-align:top}"
        "th{background:#eef2f7}.note{background:#eef6ff;border-left:4px solid #2563eb;padding:10px 12px;margin:12px 0}</style>",
        "</head><body>",
        "<h1>MRPC Fusion Count Evaluation: RL Training Install Path</h1>",
        f"<p>Generated: {html.escape(str(combined['generated_at_utc']))}</p>",
        "<div class='note'>RL-path = SequentialEnv.evaluate_step/commit_step + BLBStage2Env.step(boosted_overrides). Original = Paean final-eval action-config decoder.</div>",
        "<h2>Context</h2>",
        html_table(["Stage1 GELU", "Stage1 Softmax", "repeat/K trials", "probe size", "groups"],
                    [[json.dumps(combined["stage1_gelu"]), json.dumps(combined["stage1_softmax"]), combined["repeat"], combined["probe_size"], len(combined["group_results"])]]),
        "<h2>RL Path vs Original Paean Path</h2>",
        html_table([
            "group",
            "RL loss", "RL loss std", "RL m1", "RL m1 std", "RL m2", "RL m2 std",
            "orig loss", "orig loss std", "orig acc", "orig acc std", "orig f1", "orig f1 std",
            "Δloss", "Δm1", "Δm2", "fusion", "fusion by block", "K dist", "block5 graphs", "priority",
        ], rows),
        "</body></html>",
    ])


def _render_html(combined: Mapping[str, Any]) -> str:
    parts: List[str] = []
    _emit_rendered_html(combined, parts)
    return "\n".join(parts)


def write_rendered_html(output_html: Path, combined: Mapping[str, Any]) -> None:
    with _HtmlPartsWriter(output_html) as parts:
        _emit_rendered_html(combined, parts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mrpc")
    parser.add_argument("--model-type", default="bert-base")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--action-dir", required=True)
    parser.add_argument("--original-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--run-output-dir", default="experiments/server_command_runs/fusion_count_rlpath_tmp")
    parser.add_argument("--stage1-config-json", default="experiments/server_command_runs/mrpc_stage2_fixed_stage1_rlbest_20260627.json")
    parser.add_argument("--stage1-gelu", default=DEFAULT_STAGE1_GELU_JSON)
    parser.add_argument("--stage1-softmax", default=DEFAULT_STAGE1_SOFTMAX_JSON)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--probe-size", type=int, default=408)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-group-seed", action="store_true")
    parser.add_argument("--stage2-limit-tolerance", type=float, default=0.001)
    parser.add_argument("--stage2-stability-tolerance", type=float, default=3.5)
    parser.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    parser.add_argument(
        "--prediction-jsonl",
        default="",
        help="optional per-example prediction JSONL captured from terminal probe forwards",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()

    args.base_model = args.base_model or _base_model(args.model_type, args.dataset)
    action_dir = resolve_repo_path(args.action_dir)
    original_json = resolve_repo_path(args.original_json)
    output_json = resolve_repo_path(args.output_json)
    output_html = resolve_repo_path(args.output_html)
    prediction_jsonl = (
        resolve_repo_path(args.prediction_jsonl)
        if args.prediction_jsonl
        else None
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    stage1_gelu = parse_json_int_list(args.stage1_gelu, default=DEFAULT_STAGE1_GELU, name="--stage1-gelu")
    stage1_softmax = parse_json_int_list(args.stage1_softmax, default=DEFAULT_STAGE1_SOFTMAX, name="--stage1-softmax")

    configs = load_rlpath_action_configs(action_dir)
    unique = unique_rlpath_action_configs(configs)
    print(f"[info] groups={len(configs)} unique_group_actions={len(unique)}", flush=True)
    ev = _build_evaluator(args, stage1_gelu=stage1_gelu, stage1_softmax=stage1_softmax)
    seq_env, baseline = _build_seq_env(args, ev, stage1_gelu=stage1_gelu, stage1_softmax=stage1_softmax)
    original = read_json_file(original_json)
    original_by_name = {
        str(r.get("name")): r
        for r in (original.get("group_results") or [])
        if isinstance(r, Mapping)
    }

    result_by_key: Dict[str, dict] = {}
    prediction_recorder = None
    prediction_writer = None
    prediction_hook = None
    try:
        if prediction_jsonl is not None:
            prediction_recorder = ForwardPredictionRecorder(
                catalog=ev.fixed_eval_identity_catalog,
                probe_batch_count=int(seq_env.base.env_cfg.probe_batch_count),
            )
            prediction_writer = PredictionJsonlWriter(prediction_jsonl)
            prediction_hook = ev.model.register_forward_hook(
                prediction_recorder.hook,
                with_kwargs=True,
            )

        for idx, cfg in enumerate(unique):
            print(f"[run] {cfg['name']}", flush=True)
            group_seed = _group_seed(
                args.seed,
                idx,
                shared=bool(args.shared_group_seed),
            )
            result_by_key[rlpath_config_group_key(cfg)] = _run_group(
                seq_env,
                cfg,
                seed=group_seed,
                prediction_recorder=prediction_recorder,
                prediction_writer=prediction_writer,
            )
    finally:
        if prediction_hook is not None:
            prediction_hook.remove()
        if prediction_writer is not None:
            prediction_writer.close()

    group_results = []
    for cfg in configs:
        r = dict(result_by_key[rlpath_config_group_key(cfg)])
        r["name"] = str(cfg["name"])
        orig = original_by_name.get(str(cfg["name"])) or {}
        r["original_metrics"] = {
            "loss": orig.get("loss"),
            "loss_std": orig.get("loss_std"),
            "p": orig.get("p"),
            "p_std": orig.get("p_std"),
            "s": orig.get("s"),
            "s_std": orig.get("s_std"),
        }
        metrics = r.get("metrics") or {}
        r["delta_loss_mean_vs_original"] = (
            metric_float(metrics, "loss_mean", default=float("nan"))
            - metric_float(r["original_metrics"], "loss", default=float("nan"))
        )
        r["delta_metric1_mean_vs_original"] = (
            metric_float(metrics, "metric1_mean", default=float("nan"))
            - metric_float(r["original_metrics"], "p", default=float("nan"))
        )
        r["delta_metric2_mean_vs_original"] = (
            metric_float(metrics, "metric2_mean", default=float("nan"))
            - metric_float(r["original_metrics"], "s", default=float("nan"))
        )
        group_results.append(r)

    combined = {
        "schema_version": "fusion_count_action_eval_rlpath_compare_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "install_path": "BLBStage2SequentialEnv.evaluate_step -> commit_step -> BLBStage2Env.step(boosted_overrides)",
        "original_path": str(original_json),
        "action_dir": str(action_dir),
        "seed": int(args.seed),
        "shared_group_seed": bool(args.shared_group_seed),
        "stage1_gelu": [int(v) for v in stage1_gelu],
        "stage1_softmax": [int(v) for v in stage1_softmax],
        "repeat": int(args.repeat),
        "probe_size": int(args.probe_size),
        "baseline": to_jsonable(baseline, stringify_unknown=True, preserve_native=True),
        "group_results": to_jsonable(group_results, stringify_unknown=True, preserve_native=True),
    }
    if prediction_jsonl is not None:
        combined["prediction_artifact"] = {
            "schema_version": PREDICTION_ROW_SCHEMA,
            "path": str(prediction_jsonl),
            "row_count": int(prediction_writer.row_count),
            "dataset_indices": list(ev.fixed_eval_identity_catalog.dataset_indices),
        }
    write_json_file(output_json, combined)
    write_rendered_html(output_html, combined)
    json.dump({"output_json": str(output_json), "output_html": str(output_html)}, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
