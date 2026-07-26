#!/usr/bin/env python3
"""Evaluate BERT-large MRPC Stage-1/Stage-2 handoff through the RL path."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blb_stage2_rl.eval_metrics import pack_repeat_evaluation
from blb_stage2_rl.layerwise_action import K_LEVELS
from json_utils import to_jsonable, write_json_file


MODEL_TYPE = "bert-large"
DATASET = "mrpc"
NUM_LAYERS = 24
VALIDATION_SIZE = 408
STAGE1_RECORD_ID = "bert large mrpc 1 20260725"
STAGE1_BEST_GELU: Tuple[int, ...] = (
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
)
FIXED_SOFTMAX: Tuple[int, ...] = (6,) * NUM_LAYERS
ALL4_GELU: Tuple[int, ...] = (4,) * NUM_LAYERS
ORIGINAL_DEGREES: Tuple[int, ...] = (-1,) * NUM_LAYERS


@dataclass(frozen=True)
class GroupSpec:
    name: str
    label: str
    stage1_config: str
    stage2_enabled: bool
    block4_fusion: Optional[int] = None
    action_matrix: Optional[Tuple[Tuple[int, ...], ...]] = None


@dataclass
class Stage2Runtime:
    base_env: Any
    layerwise_env: Any
    fusion_map: Any
    calibrated_context: Any
    profile: str


def build_group_specs(
        *,
        num_layers: int = NUM_LAYERS,
        k_value: int = 13,
        ) -> Tuple[GroupSpec, ...]:
    """Return the six requested groups with production policy actions."""
    layers = int(num_layers)
    if layers < 1:
        raise ValueError("num_layers must be positive")
    try:
        k_index = tuple(int(value) for value in K_LEVELS).index(int(k_value))
    except ValueError as exc:
        raise ValueError(
            f"K={int(k_value)} is absent from K_LEVELS={tuple(K_LEVELS)}"
        ) from exc

    def action(fusion: int) -> Tuple[Tuple[int, ...], ...]:
        row = (int(fusion), k_index, k_index, k_index, k_index, k_index)
        return tuple(row for _ in range(layers))

    return (
        GroupSpec(
            "original_plaintext",
            "Original plaintext model",
            "original",
            False,
        ),
        GroupSpec(
            "stage1_best_plaintext",
            "Stage-1 best, no Stage-2 noise",
            "stage1_best",
            False,
        ),
        GroupSpec(
            "gelu4_stage2_b4_f0",
            "GELU4 + Stage-2 B2/B5=1, B4=0, K=13",
            "all4",
            True,
            0,
            action(0),
        ),
        GroupSpec(
            "gelu4_stage2_b4_f1",
            "GELU4 + Stage-2 B2/B5=1, B4=1, K=13",
            "all4",
            True,
            1,
            action(1),
        ),
        GroupSpec(
            "stage1_best_stage2_b4_f0",
            "Stage-1 best + Stage-2 B2/B5=1, B4=0, K=13",
            "stage1_best",
            True,
            0,
            action(0),
        ),
        GroupSpec(
            "stage1_best_stage2_b4_f1",
            "Stage-1 best + Stage-2 B2/B5=1, B4=1, K=13",
            "stage1_best",
            True,
            1,
            action(1),
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _map_manifest(profile: str) -> List[dict]:
    map_dir = REPO_ROOT / "blb_stage2_rl" / "fusion_maps" / str(profile)
    rows = []
    for path in sorted(map_dir.glob("block*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(path),
            "graph_key": str(payload["graph_key"]),
            "fusion_counts": sorted({
                int(option["fusion_count"]) for option in payload["options"]
            }),
            "option_count": len(payload["options"]),
        })
    if not rows:
        raise FileNotFoundError(f"no fusion maps under {map_dir}")
    return rows


def _prepare_stage1_record_layout(output_dir: Path, record_id: str) -> Path:
    """Mirror the tracked record into a runtime-only decoupled layout."""
    source = (
        REPO_ROOT / "Parting Chapter" / "stage1" / "record" / str(record_id)
    )
    if not source.is_dir():
        raise FileNotFoundError(f"Stage-1 record is missing: {source}")
    root = output_dir / "resolver_layout"
    target_root = root / "stage1" / "record"
    target = target_root / str(record_id)
    target_root.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    shutil.copytree(source, target)
    return root / "stage2" / "bert large mrpc"


def _stage1_record_provenance(record_id: str) -> dict:
    record_dir = (
        REPO_ROOT / "Parting Chapter" / "stage1" / "record" / str(record_id)
    )
    final_config = json.loads(
        (record_dir / "final_config.json").read_text(encoding="utf-8")
    )
    final_eval = json.loads(
        (record_dir / "final_eval.json").read_text(encoding="utf-8")
    )
    return {
        "selection": dict(final_config["selection"]),
        "independent_validation_evidence": dict(
            final_eval["independent_validation_evidence"]
        ),
    }


def _load_runtime_deps() -> Dict[str, Any]:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    from blb_stage2_rl.baseline_bootstrap import (
        load_calibrated_stage2_action_context,
        resolve_stage2_profile,
        validate_calibrated_stage2_action_context,
    )
    from blb_stage2_rl.env import (
        BLBStage2Env,
        BLBStage2EnvConfig,
        estimate_baseline_cost_stats,
    )
    from blb_stage2_rl.fusion_count_map import FusionCountMap
    from blb_stage2_rl.layerwise_env import BLBStage2LayerwiseEnv
    from blb_stage2_rl.reward import (
        BaselineCostStats,
        RewardWeights,
        calibrate_weights_from_baseline,
    )
    from blb_stage2_rl.runner import BLBStage2RLRunner, BLBStage2TrainConfig
    from layer_importance_evaluator import LayerImportanceEvaluator
    from Paean.run_final_eval import _base_model
    from rl_tune import load_glue_dataset_equivalent, seed_everything

    return {
        "torch": torch,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "load_calibrated_stage2_action_context": load_calibrated_stage2_action_context,
        "resolve_stage2_profile": resolve_stage2_profile,
        "validate_calibrated_stage2_action_context": (
            validate_calibrated_stage2_action_context
        ),
        "BLBStage2Env": BLBStage2Env,
        "BLBStage2EnvConfig": BLBStage2EnvConfig,
        "estimate_baseline_cost_stats": estimate_baseline_cost_stats,
        "FusionCountMap": FusionCountMap,
        "BLBStage2LayerwiseEnv": BLBStage2LayerwiseEnv,
        "BaselineCostStats": BaselineCostStats,
        "RewardWeights": RewardWeights,
        "calibrate_weights_from_baseline": calibrate_weights_from_baseline,
        "BLBStage2RLRunner": BLBStage2RLRunner,
        "BLBStage2TrainConfig": BLBStage2TrainConfig,
        "LayerImportanceEvaluator": LayerImportanceEvaluator,
        "_base_model": _base_model,
        "load_glue_dataset_equivalent": load_glue_dataset_equivalent,
        "seed_everything": seed_everything,
    }


def _recording_env_class(base_env_class):
    class RecordingBLBStage2Env(base_env_class):
        fixed_eval_trial_metrics: Optional[dict] = None

        def _aggregate_probe_trials(
                self,
                per_trial_loss,
                per_trial_metric1,
                per_trial_metric2,
                trial_seeds=None,
                ):
            self.fixed_eval_trial_metrics = {
                "loss": [float(value) for value in per_trial_loss],
                "metric1": [float(value) for value in per_trial_metric1],
                "metric2": [float(value) for value in per_trial_metric2],
                "trial_seeds": (
                    [int(value) for value in trial_seeds]
                    if trial_seeds is not None else []
                ),
            }
            return super()._aggregate_probe_trials(
                per_trial_loss,
                per_trial_metric1,
                per_trial_metric2,
                trial_seeds=trial_seeds,
            )

    return RecordingBLBStage2Env


def _tokenize_mrpc(data, tokenizer, *, seed: int):
    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding=False,
            max_length=128,
            return_tensors=None,
        )

    train_data = data["train"].shuffle(seed=int(seed)).map(tokenize_fn)
    validation_data = data["validation"].shuffle(seed=int(seed)).map(tokenize_fn)
    train_data = train_data.rename_column("label", "labels")
    validation_data = validation_data.rename_column("label", "labels")
    columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    train_data.set_format(type="torch", columns=columns)
    validation_data.set_format(type="torch", columns=columns)
    return train_data, validation_data


def _build_evaluator(args, deps: Mapping[str, Any], output_dir: Path):
    torch = deps["torch"]
    deps["seed_everything"](int(args.seed))
    base_model = str(args.base_model or deps["_base_model"](MODEL_TYPE, DATASET))
    tokenizer = deps["AutoTokenizer"].from_pretrained(
        base_model, trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    model = deps["AutoModelForSequenceClassification"].from_pretrained(
        base_model,
        num_labels=2,
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = torch.device(str(args.device))
    model.eval()
    model.to(device)

    data = deps["load_glue_dataset_equivalent"](
        DATASET,
        route_log_dir=str(output_dir / "logs"),
    )
    train_data, validation_data = _tokenize_mrpc(
        data, tokenizer, seed=int(args.seed),
    )
    if len(validation_data) != VALIDATION_SIZE:
        raise RuntimeError(
            f"MRPC validation_full has {len(validation_data)} rows, "
            f"expected {VALIDATION_SIZE}"
        )
    collator = deps["DataCollatorWithPadding"](
        tokenizer=tokenizer,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )
    resolver_run_dir = _prepare_stage1_record_layout(
        output_dir, str(args.stage1_record_id),
    )
    evaluator = deps["LayerImportanceEvaluator"](
        model=model,
        train_data=train_data,
        test_data=validation_data,
        data_collator=collator,
        batch_size=int(args.batch_size),
        stage1_rl_episodes=0,
        stage2_rl_episodes=0,
        stage1_rl_episodes_specified=True,
        stage2_rl_episodes_specified=True,
        run_output_dir=str(resolver_run_dir),
        stage2_fixed_config_source="stage1_result",
        decoupled_layout=True,
        stage1_run_id=str(args.stage1_record_id),
        skip_stage1_rl=True,
        skip_noise_rl=True,
        skip_final_eval=True,
        data_path=DATASET,
        stage2_limit_tolerance=0.005,
        stage2_stability_tolerance=1.2,
        stage2_k_trials=int(args.repeat),
        stage2_probe_size=VALIDATION_SIZE,
        stage2_rl_variant="blb_v3",
        blb_v3_inproc_rescale_optimizer_root=str(args.rescale_optimizer_root),
        blb_v3_fusion_count_action=True,
        rl_algo="ppo",
    )
    if int(evaluator.total_layers) != NUM_LAYERS:
        raise RuntimeError(
            f"loaded model has {evaluator.total_layers} layers, expected {NUM_LAYERS}"
        )
    return evaluator, base_model


def _resolve_stage1_best(evaluator) -> Tuple[np.ndarray, np.ndarray, str, str]:
    gelu, softmax, label, source = (
        evaluator._resolve_stage2_fixed_stage1_config()
    )
    gelu = np.asarray(gelu, dtype=int).reshape(-1)
    softmax = np.asarray(softmax, dtype=int).reshape(-1)
    if tuple(gelu.tolist()) != STAGE1_BEST_GELU:
        raise RuntimeError(
            "production Stage-2 resolver returned an unexpected Stage-1 GELU "
            f"vector: {gelu.tolist()}"
        )
    if tuple(softmax.tolist()) != FIXED_SOFTMAX:
        raise RuntimeError(
            "production Stage-2 resolver did not enforce Softmax degree 6: "
            f"{softmax.tolist()}"
        )
    if "stage1_record:bert large mrpc" not in str(source):
        raise RuntimeError(f"unexpected Stage-1 resolver source: {source!r}")
    return gelu, softmax, str(label), str(source)


def _clean_eval(
        evaluator,
        *,
        gelu: Sequence[int],
        softmax: Sequence[int],
        repeat: int,
        label: str,
        ) -> dict:
    evaluator.apply_configuration(
        np.asarray(gelu, dtype=int),
        np.asarray(softmax, dtype=int),
    )
    split_name = evaluator._resolve_eval_split(
        use_train=False, split="validation_full",
    )
    trials = []
    for _ in range(int(repeat)):
        loss, metric1, metric2, time_ms = evaluator._run_evaluation(
            evaluator.dataloaders[split_name],
            use_train=False,
            split_name=split_name,
        )
        trials.append({
            "loss": float(loss),
            "p": float(metric1),
            "s": float(metric2),
            "time_ms": float(time_ms),
        })
    packed = pack_repeat_evaluation(
        trials,
        evaluation_mode="clean_validation_full",
    )
    return {
        "label": str(label),
        "stage2_enabled": False,
        "gelu": [int(value) for value in gelu],
        "softmax": [int(value) for value in softmax],
        "repeat_evaluation": packed,
    }


def _build_stage2_runtime(
        args,
        deps: Mapping[str, Any],
        evaluator,
        *,
        fixed_gelu: Sequence[int],
        fixed_softmax: Sequence[int],
        clean_reference: Mapping[str, Any],
        ) -> Stage2Runtime:
    gelu = np.asarray(fixed_gelu, dtype=int).reshape(-1)
    softmax = np.asarray(fixed_softmax, dtype=int).reshape(-1)
    evaluator.apply_configuration(gelu, softmax)
    evaluator.reversible_handler.restore_layer_input_noise(
        layer_indices=list(range(int(evaluator.total_layers))),
    )

    profile = deps["resolve_stage2_profile"](
        DATASET,
        model_type=MODEL_TYPE,
        num_layers=int(evaluator.total_layers),
    )
    if str(profile) != "mrpc_large":
        raise RuntimeError(
            f"BERT-large MRPC resolved unexpected Stage-2 profile {profile!r}"
        )
    runner = deps["BLBStage2RLRunner"](evaluator)
    train_cfg = deps["BLBStage2TrainConfig"](
        total_episodes=1,
        rollout_size=1,
        seed=int(args.seed),
        profile=profile,
        num_trials_per_step=int(args.repeat),
        probe_batch_count=1,
        calibrate_baseline_samples=0,
        inproc_rescale_optimizer_root=str(args.rescale_optimizer_root),
        inproc_profile=profile,
        fusion_count_action=True,
    )
    validation_batches = runner._build_validation_full_batches(evaluator)
    expected_batches = int(
        np.ceil(VALIDATION_SIZE / max(1, int(args.batch_size)))
    )
    if len(validation_batches) != expected_batches:
        raise RuntimeError(
            "validation_full batch count mismatch: "
            f"{len(validation_batches)} != {expected_batches}"
        )
    bridge = runner._build_rescale_bridge(
        train_cfg, log=lambda message: print(message, flush=True),
    )
    calibrated = deps["load_calibrated_stage2_action_context"](
        rescale_optimizer_root=str(args.rescale_optimizer_root),
        dataset=profile,
        num_layers=int(evaluator.total_layers),
        gelu_per_layer=gelu.tolist(),
        softmax_per_layer=softmax.tolist(),
        snap_sf_to_noise_table=False,
    )
    deps["validate_calibrated_stage2_action_context"](
        calibrated,
        dataset=profile,
        num_layers=int(evaluator.total_layers),
        gelu_per_layer=gelu.tolist(),
        softmax_per_layer=softmax.tolist(),
        snap_sf_to_noise_table=False,
    )

    RecordingEnv = _recording_env_class(deps["BLBStage2Env"])
    base_env = RecordingEnv(
        handler=evaluator.reversible_handler,
        model=evaluator.model,
        probe_batches=validation_batches,
        rescale_bridge=bridge,
        baseline=deps["BaselineCostStats"](),
        reward_weights=deps["RewardWeights"](),
        acc_threshold=0.0,
        stab_threshold=float("inf"),
        max_sfs=calibrated.max_sfs,
        num_layers=int(evaluator.total_layers),
        gelu_degree=gelu,
        attn_degree=softmax,
        layers_attribute="model." + evaluator.layers_attribute,
        is_regression=bool(getattr(evaluator, "is_regression", False)),
        env_cfg=deps["BLBStage2EnvConfig"](
            profile=profile,
            num_trials_per_step=int(args.repeat),
            probe_batch_count=len(validation_batches),
            truncation_backend=str(args.truncation_backend),
            truncation_ring_bits=int(args.truncation_ring_bits),
            truncation_source_fractional_bits=int(
                args.truncation_source_fractional_bits
            ),
            borderline_retest_enabled=False,
            borderline_retest_trials_multiplier=1,
        ),
    )
    precomputed = {
        "total_bits_sum": int(calibrated.cost_stats.total_bits_sum),
        "total_fusion_count": int(calibrated.cost_stats.total_fusion_count),
        "avg_k": float(calibrated.cost_stats.avg_k),
    }
    baseline = deps["estimate_baseline_cost_stats"](
        base_env,
        sample_count=0,
        precomputed_baseline_signals=precomputed,
    )
    clean_stats = clean_reference["repeat_evaluation"]["stats"]
    baseline.loss_mean = float(clean_stats["loss_mean"])
    baseline.loss_std = float(clean_stats["loss_std"])
    baseline.metric1_mean = float(clean_stats["p_mean"])
    baseline.metric1_std = float(clean_stats["p_std"])
    baseline.metric2_mean = float(clean_stats["s_mean"])
    baseline.metric2_std = float(clean_stats["s_std"])
    base_env.baseline = baseline
    base_env.reward_weights = deps["calibrate_weights_from_baseline"](baseline)
    base_env.sync_degree_vectors_from_model()

    fusion_map = deps["FusionCountMap"].load(profile)
    expected_graphs = {
        "block2_mrpc_large",
        "block4",
        "block5_n1",
        "block5_n2",
        "block5_n4",
    }
    if set(fusion_map.graphs) != expected_graphs:
        raise RuntimeError(
            f"{profile}: unexpected fusion-map graphs "
            f"{sorted(fusion_map.graphs)}"
        )
    for graph_key, graph in fusion_map.graphs.items():
        fusion_counts = [
            int(option.fusion_count) for option in graph.options
        ]
        if fusion_counts != [0, 1]:
            raise RuntimeError(
                f"{profile}/{graph_key}: expected exactly fusion counts "
                f"[0, 1], got {fusion_counts}"
            )
    layerwise_env = deps["BLBStage2LayerwiseEnv"](
        base_env=base_env,
        fusion_map=fusion_map,
        baseline_action_vec=calibrated.baseline_action_vec,
        profile=profile,
    )
    return Stage2Runtime(
        base_env=base_env,
        layerwise_env=layerwise_env,
        fusion_map=fusion_map,
        calibrated_context=calibrated,
        profile=profile,
    )


def _selected_option_payload(
        fusion_map,
        *,
        graph_key: str,
        option_id: int,
        ) -> dict:
    graph = fusion_map.graphs[str(graph_key)]
    option = next(
        item for item in graph.options if int(item.option_id) == int(option_id)
    )
    return {
        "graph_key": str(graph_key),
        "option_id": int(option.option_id),
        "fusion_count": int(option.fusion_count),
        "boosted": bool(option.boosted),
        "slots": dict(option.slots),
        "explicit_field_values": (
            dict(option.explicit_field_values)
            if option.explicit_field_values else {}
        ),
    }


def _run_stage2_group(
        runtime: Stage2Runtime,
        group: GroupSpec,
        *,
        seed: int,
        ) -> dict:
    if not group.stage2_enabled or group.action_matrix is None:
        raise ValueError(f"group {group.name} has no Stage-2 action")
    env = runtime.layerwise_env
    base_env = runtime.base_env
    base_env.fixed_eval_trial_metrics = None
    env.reset(seed=int(seed))
    base_env.probe_noise_seed = int(seed)
    done = False
    terminal_info: Dict[str, Any] = {}
    started = time.perf_counter()
    for layer_idx, row in enumerate(group.action_matrix):
        _obs, _reward, done, terminal_info = env.step(row)
        if done != (layer_idx == len(group.action_matrix) - 1):
            raise RuntimeError(
                f"{group.name}: terminal state at unexpected layer {layer_idx}"
            )
    wall_seconds = float(time.perf_counter() - started)
    runtime_info = env.runtime_terminal_info
    if not isinstance(runtime_info, Mapping):
        raise RuntimeError(f"{group.name}: missing raw terminal runtime info")
    if not bool(runtime_info.get("forward_ran", False)):
        raise RuntimeError(
            f"{group.name}: model forward did not run: "
            f"{runtime_info.get('forward_skipped_reason') or runtime_info.get('error')}"
        )
    replan = runtime_info.get("replan_application") or {}
    if not bool(replan.get("model_uses_replan_config", False)):
        raise RuntimeError(
            f"{group.name}: model did not use the post-replan config"
        )
    if bool(runtime_info.get("invalid", False)):
        raise RuntimeError(f"{group.name}: terminal action was invalid")
    layer_summaries = env.layer_summaries
    if len(layer_summaries) != NUM_LAYERS:
        raise RuntimeError(
            f"{group.name}: got {len(layer_summaries)} layer summaries"
        )
    for row in layer_summaries:
        if not bool(row["all_valid"]):
            raise RuntimeError(
                f"{group.name}: layer {row['layer_idx']} contains invalid replan"
            )
        for block in row["blocks"]:
            block_replan = block.get("replan_application") or {}
            if not bool(block_replan.get("model_uses_replan_config", False)):
                raise RuntimeError(
                    f"{group.name}: layer {row['layer_idx']} block "
                    f"{block['block_idx']} did not use replan config"
                )

    trial_metrics = base_env.fixed_eval_trial_metrics
    if not isinstance(trial_metrics, Mapping):
        raise RuntimeError(f"{group.name}: missing per-trial metrics")
    trial_count = len(trial_metrics.get("loss") or [])
    if trial_count != int(base_env.env_cfg.num_trials_per_step):
        raise RuntimeError(
            f"{group.name}: got {trial_count} trials, expected "
            f"{base_env.env_cfg.num_trials_per_step}"
        )
    trials = [
        {
            "loss": float(trial_metrics["loss"][idx]),
            "p": float(trial_metrics["metric1"][idx]),
            "s": float(trial_metrics["metric2"][idx]),
            "time_ms": wall_seconds * 1000.0 / max(1, trial_count),
        }
        for idx in range(trial_count)
    ]
    packed = pack_repeat_evaluation(
        trials,
        evaluation_mode="stage2_rl_terminal_validation_full",
    )

    fusion_by_block = {2: 0, 3: 0, 4: 0, 5: 0}
    graph_rows: List[dict] = []
    schedule = env.schedule
    fusion_option_ids = terminal_info["fusion_option_ids"]
    for layer_idx, (summary, spec, option_ids) in enumerate(
            zip(layer_summaries, schedule, fusion_option_ids)):
        blocks = {int(row["block_idx"]): row for row in summary["blocks"]}
        for block_idx in fusion_by_block:
            if block_idx in blocks:
                fusion_by_block[block_idx] += int(
                    blocks[block_idx]["fusion_count"]
                )
        graph_keys = dict(spec.graph_keys_by_block)
        selected_options = {}
        for block_idx in (2, 4, 5):
            graph_key = str(graph_keys[block_idx])
            option_id = int(option_ids[block_idx])
            selected_options[str(block_idx)] = _selected_option_payload(
                runtime.fusion_map,
                graph_key=graph_key,
                option_id=option_id,
            )
        graph_rows.append({
            "layer": int(layer_idx),
            "graph_keys": {str(k): str(v) for k, v in graph_keys.items()},
            "selected_options": selected_options,
            "block_summaries": to_jsonable(
                summary["blocks"],
                stringify_unknown=True,
                preserve_native=True,
            ),
        })

    expected_fusion = {
        2: NUM_LAYERS,
        3: 0,
        4: NUM_LAYERS * int(group.block4_fusion or 0),
        5: NUM_LAYERS,
    }
    if fusion_by_block != expected_fusion:
        raise RuntimeError(
            f"{group.name}: actual fusion counts {fusion_by_block} "
            f"!= expected {expected_fusion}"
        )
    k_choices = list(terminal_info.get("k_choices") or [])
    if len(k_choices) != 5 * NUM_LAYERS:
        raise RuntimeError(
            f"{group.name}: got {len(k_choices)} active K choices, expected "
            f"{5 * NUM_LAYERS}"
        )
    if any(int(choice["k_value"]) != 13 for choice in k_choices):
        raise RuntimeError(f"{group.name}: at least one K choice is not 13")

    return {
        "label": group.label,
        "stage2_enabled": True,
        "stage1_config": group.stage1_config,
        "block4_fusion": int(group.block4_fusion),
        "fixed_block2_fusion": 1,
        "fixed_block5_fusion": 1,
        "truncation_k": 13,
        "action_matrix": [list(row) for row in group.action_matrix],
        "repeat_evaluation": packed,
        "trial_seeds": list(trial_metrics.get("trial_seeds") or []),
        "fusion_by_block": {
            str(block_idx): int(value)
            for block_idx, value in fusion_by_block.items()
        },
        "k_choices": to_jsonable(
            k_choices, stringify_unknown=True, preserve_native=True,
        ),
        "model_uses_replan_config": True,
        "final_config_fingerprint": str(
            runtime_info.get("final_config_fingerprint") or ""
        ),
        "materialization_failure_reason": runtime_info.get(
            "materialization_failure_reason"
        ),
        "calibrated_action_context": to_jsonable(
            runtime.calibrated_context.provenance,
            stringify_unknown=True,
            preserve_native=True,
        ),
        "graph_rows": graph_rows,
        "boosted_overrides": to_jsonable(
            terminal_info.get("boosted_overrides") or [],
            stringify_unknown=True,
            preserve_native=True,
        ),
        "installed_config": to_jsonable(
            runtime_info.get("decoded"),
            stringify_unknown=True,
            preserve_native=True,
        ),
        "replan_application": to_jsonable(
            replan,
            stringify_unknown=True,
            preserve_native=True,
        ),
    }


def _stats(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return result["repeat_evaluation"]["stats"]


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{int(digits)}f}"
    except (TypeError, ValueError):
        return ""


def _render_html(payload: Mapping[str, Any]) -> str:
    results = payload["group_results"]
    original = _stats(results["original_plaintext"])

    summary_rows = []
    for name in [group.name for group in build_group_specs()]:
        result = results[name]
        stats = _stats(result)
        summary_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(result['label'])}</strong><br><code>{name}</code></td>"
            f"<td>{stats['n']}</td>"
            f"<td>{_fmt(stats['loss_mean'])} +/- {_fmt(stats['loss_std'])}</td>"
            f"<td>{_fmt(stats['p_mean'])} +/- {_fmt(stats['p_std'])}</td>"
            f"<td>{_fmt(stats['s_mean'])} +/- {_fmt(stats['s_std'])}</td>"
            f"<td>{_fmt(float(stats['loss_mean']) - float(original['loss_mean']))}</td>"
            f"<td>{_fmt(float(stats['p_mean']) - float(original['p_mean']))}</td>"
            f"<td>{_fmt(float(stats['s_mean']) - float(original['s_mean']))}</td>"
            f"<td>{html.escape(str(result.get('fusion_by_block', '-')))}</td>"
            f"<td>{html.escape(str(result.get('truncation_k', '-')))}</td>"
            "</tr>"
        )

    trial_rows = []
    for name, result in results.items():
        for trial in result["repeat_evaluation"]["trials"]:
            trial_rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{trial['trial']}</td>"
                f"<td>{_fmt(trial['loss'])}</td>"
                f"<td>{_fmt(trial['p'])}</td>"
                f"<td>{_fmt(trial['s'])}</td>"
                "</tr>"
            )

    gate_rows = []
    layer_rows = []
    detail_sections = []
    for name, result in results.items():
        if not result.get("stage2_enabled"):
            continue
        context = result["calibrated_action_context"]
        gate_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(str(context['dataset']))}</td>"
            f"<td>{html.escape(str(context['archive_sha256']))}</td>"
            f"<td>{html.escape(str(result['model_uses_replan_config']))}</td>"
            f"<td><code>{html.escape(result['final_config_fingerprint'])}</code></td>"
            f"<td>{html.escape(str(result['materialization_failure_reason']))}</td>"
            "</tr>"
        )
        k_by_layer: Dict[int, Dict[int, int]] = {}
        for choice in result["k_choices"]:
            k_by_layer.setdefault(int(choice["layer_idx"]), {})[
                int(choice["block_idx"])
            ] = int(choice["k_value"])
        for row in result["graph_rows"]:
            selected = row["selected_options"]
            k = k_by_layer[int(row["layer"])]
            layer_rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{row['layer']}</td>"
                f"<td>{html.escape(selected['2']['graph_key'])} / "
                f"{selected['2']['fusion_count']} / {selected['2']['option_id']}</td>"
                f"<td>{html.escape(selected['4']['graph_key'])} / "
                f"{selected['4']['fusion_count']} / {selected['4']['option_id']}</td>"
                f"<td>{html.escape(selected['5']['graph_key'])} / "
                f"{selected['5']['fusion_count']} / {selected['5']['option_id']}</td>"
                f"<td>{html.escape(str(k))}</td>"
                "</tr>"
            )
        detail_payload = {
            "boosted_overrides": result["boosted_overrides"],
            "installed_config": result["installed_config"],
            "replan_application": result["replan_application"],
        }
        detail_sections.append(
            f"<details><summary>{html.escape(name)} actual installed configuration</summary>"
            f"<pre>{html.escape(json.dumps(detail_payload, indent=2, ensure_ascii=False))}</pre>"
            "</details>"
        )

    stage1 = payload["stage1_resolution"]
    provenance = payload["stage1_provenance"]
    selection = provenance["selection"]
    independent = provenance["independent_validation_evidence"]
    maps = payload["fusion_map_manifest"]
    map_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['graph_key'])}</td>"
        f"<td>{html.escape(str(row['fusion_counts']))}</td>"
        f"<td>{row['option_count']}</td>"
        f"<td><code>{html.escape(row['sha256'])}</code></td>"
        f"<td><code>{html.escape(row['path'])}</code></td>"
        "</tr>"
        for row in maps
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BERT-large MRPC Stage-1 to Stage-2 chain evaluation</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#17202a;background:#f7f9fb}}
main{{max-width:1500px;margin:auto}}h1,h2{{color:#102a43}}code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
table{{border-collapse:collapse;width:100%;background:white;margin:12px 0 24px}}th,td{{border:1px solid #cbd5e1;padding:7px 8px;font-size:12px;text-align:left;vertical-align:top}}
th{{background:#e8eef5}}.note{{border-left:4px solid #2563eb;background:#edf5ff;padding:10px 12px}}.pass{{color:#146c2e;font-weight:700}}
details{{background:white;border:1px solid #cbd5e1;padding:10px;margin:10px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}}
</style>
</head>
<body><main>
<h1>BERT-large MRPC: Stage-1 best to Stage-2 RL chain evaluation</h1>
<p>Generated: {html.escape(str(payload['generated_at_utc']))}</p>
<div class="note">The four noisy groups execute the production
<code>BLBStage2LayerwiseEnv</code> path. B2/B5 fusion is fixed to 1, B4 fusion
is the policy coordinate, Block1/Block3 SF stays on the calibrated RO baseline,
K is fixed to 13, and the post-replan/precision-boost configuration is installed
before each full-validation forward.</div>

<h2>Source and Stage-1 resolution</h2>
<table><tbody>
<tr><th>Git commit</th><td><code>{html.escape(payload['git_commit'])}</code></td></tr>
<tr><th>Git tree</th><td><code>{html.escape(payload['git_tree'])}</code></td></tr>
<tr><th>Model</th><td>{html.escape(payload['base_model'])}</td></tr>
<tr><th>Stage-2 profile</th><td>{html.escape(payload['profile'])}</td></tr>
<tr><th>Stage-1 source</th><td>{html.escape(stage1['source'])}</td></tr>
<tr><th>Stage-1 label</th><td>{html.escape(stage1['label'])}</td></tr>
<tr><th>Stage-1 GELU</th><td><code>{html.escape(str(stage1['gelu']))}</code></td></tr>
<tr><th>Stage-1 Softmax</th><td><code>{html.escape(str(stage1['softmax']))}</code></td></tr>
<tr><th>Historical config commit</th><td><code>{html.escape(selection['source_config_commit'])}</code></td></tr>
<tr><th>Historical validation report SHA-256</th><td><code>{html.escape(selection['source_validation_report_sha256'])}</code></td></tr>
<tr><th>Independent Stage-1-best validation</th><td>
loss={_fmt(independent['loss'])},
accuracy={_fmt(independent['metric1_accuracy'])},
weighted F1={_fmt(independent['metric2_weighted_f1'])},
N={int(independent['validation_full_size'])}, no Stage-2 noise
</td></tr>
<tr><th>Validation protocol</th><td>MRPC validation_full, 408 examples, repeated 5 times</td></tr>
</tbody></table>

<h2>Group comparison</h2>
<table><thead><tr><th>Group</th><th>N</th><th>Loss mean +/- std</th>
<th>Accuracy mean +/- std</th><th>Weighted F1 mean +/- std</th>
<th>Delta loss vs original</th><th>Delta accuracy</th><th>Delta F1</th>
<th>Actual fusion totals by block</th><th>K</th></tr></thead>
<tbody>{''.join(summary_rows)}</tbody></table>

<h2>Per-trial results</h2>
<table><thead><tr><th>Group</th><th>Trial</th><th>Loss</th><th>Accuracy</th><th>Weighted F1</th></tr></thead>
<tbody>{''.join(trial_rows)}</tbody></table>

<h2>Production-chain gates</h2>
<table><thead><tr><th>Group</th><th>Profile</th><th>RO archive SHA-256</th>
<th>Model uses replan config</th><th>Final config fingerprint</th><th>Failure</th></tr></thead>
<tbody>{''.join(gate_rows)}</tbody></table>

<h2>Fusion map manifest</h2>
<table><thead><tr><th>Graph</th><th>Fusion counts</th><th>Options</th><th>SHA-256</th><th>Path</th></tr></thead>
<tbody>{map_rows}</tbody></table>

<h2>Per-layer policy decision to map option</h2>
<p>Cells are <code>graph / actual fusion_count / option_id</code>. K dictionaries
show the values installed for active Blocks 1-5; layer 0 intentionally has no
Block1 action.</p>
<table><thead><tr><th>Group</th><th>Layer</th><th>Block2</th><th>Block4</th><th>Block5</th><th>K by block</th></tr></thead>
<tbody>{''.join(layer_rows)}</tbody></table>

<h2>Actual installed SF and replan evidence</h2>
{''.join(detail_sections)}
</main></body></html>"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-record-id", default=STAGE1_RECORD_ID)
    parser.add_argument(
        "--rescale-optimizer-root",
        default=str(REPO_ROOT / "Rescale_optimizer"),
    )
    parser.add_argument("--truncation-backend", default="binary")
    parser.add_argument("--truncation-ring-bits", type=int, default=43)
    parser.add_argument(
        "--truncation-source-fractional-bits", type=int, default=24,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if int(args.repeat) != 5:
        raise ValueError("this comparison contract requires exactly repeat=5")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.rescale_optimizer_root = str(
        Path(args.rescale_optimizer_root).resolve()
    )

    deps = _load_runtime_deps()
    evaluator, base_model = _build_evaluator(args, deps, output_dir)
    stage1_gelu, stage1_softmax, stage1_label, stage1_source = (
        _resolve_stage1_best(evaluator)
    )
    groups = build_group_specs(num_layers=NUM_LAYERS, k_value=13)

    results: Dict[str, dict] = {}
    results["original_plaintext"] = _clean_eval(
        evaluator,
        gelu=ORIGINAL_DEGREES,
        softmax=ORIGINAL_DEGREES,
        repeat=int(args.repeat),
        label=groups[0].label,
    )
    results["stage1_best_plaintext"] = _clean_eval(
        evaluator,
        gelu=stage1_gelu,
        softmax=stage1_softmax,
        repeat=int(args.repeat),
        label=groups[1].label,
    )

    all4_clean_reference = _clean_eval(
        evaluator,
        gelu=ALL4_GELU,
        softmax=FIXED_SOFTMAX,
        repeat=1,
        label="GELU4 clean calibration reference",
    )
    all4_runtime = _build_stage2_runtime(
        args,
        deps,
        evaluator,
        fixed_gelu=ALL4_GELU,
        fixed_softmax=FIXED_SOFTMAX,
        clean_reference=all4_clean_reference,
    )
    for group in groups[2:4]:
        results[group.name] = _run_stage2_group(
            all4_runtime, group, seed=int(args.seed),
        )
    all4_runtime.base_env.clear_installed_blb()

    stage1_runtime = _build_stage2_runtime(
        args,
        deps,
        evaluator,
        fixed_gelu=stage1_gelu,
        fixed_softmax=stage1_softmax,
        clean_reference=results["stage1_best_plaintext"],
    )
    for group in groups[4:6]:
        results[group.name] = _run_stage2_group(
            stage1_runtime, group, seed=int(args.seed),
        )
    stage1_runtime.base_env.clear_installed_blb()

    profile = str(stage1_runtime.profile)
    payload = {
        "schema_version": "stage1best_large_stage2_chain_eval_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "base_model": str(base_model),
        "model_type": MODEL_TYPE,
        "dataset": DATASET,
        "profile": profile,
        "validation_full_size": VALIDATION_SIZE,
        "repeat": int(args.repeat),
        "shared_noise_seed": int(args.seed),
        "stage1_resolution": {
            "record_id": str(args.stage1_record_id),
            "gelu": stage1_gelu.tolist(),
            "softmax": stage1_softmax.tolist(),
            "label": stage1_label,
            "source": stage1_source,
        },
        "stage1_provenance": _stage1_record_provenance(
            str(args.stage1_record_id)
        ),
        "stage2_contract": {
            "decision_path": "BLBStage2LayerwiseEnv",
            "block2_fusion": 1,
            "block5_fusion": 1,
            "block4_fusion": "per-layer policy coordinate",
            "block1_sf": "calibrated RO baseline",
            "block3_sf": "calibrated RO baseline",
            "truncation_k": 13,
            "truncation_backend": str(args.truncation_backend),
        },
        "fusion_map_manifest": _map_manifest(profile),
        "group_results": to_jsonable(
            results, stringify_unknown=True, preserve_native=True,
        ),
    }
    json_path = output_dir / "stage1best_large_stage2_chain_eval.json"
    html_path = output_dir / "stage1best_large_stage2_chain_eval.html"
    write_json_file(json_path, payload)
    html_path.write_text(_render_html(payload), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "html": str(html_path),
        "groups": list(results),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
