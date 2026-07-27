#!/usr/bin/env python3
"""Run and aggregate the BERT-base MRPC Stage-2 3x6 configuration grid."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(
    os.environ.get("RFR_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blb_stage2_rl.eval_metrics import pack_repeat_evaluation  # noqa: E402
from blb_stage2_rl.truncation_levels import (  # noqa: E402
    K_LEVELS,
    validate_exact_k_domain,
)
from json_utils import to_jsonable, write_json_file  # noqa: E402
from scripts import run_stage1best_large_stage2_chain_eval as chain  # noqa: E402


MODEL_TYPE = "bert-base"
DATASET = "mrpc"
PROFILE = "mrpc"
NUM_LAYERS = 12
VALIDATION_SIZE = 408
ALL4_GELU = (4,) * NUM_LAYERS
FIXED_SOFTMAX = (6,) * NUM_LAYERS
DEFAULT_RECORD_ID = "bert large mrpc 1 20260725"
FusionProfile = tuple[str, str, tuple[int, int, int]]
TruncationProfile = tuple[str, str, tuple[int, int, int, int, int]]

FUSION_PROFILES: tuple[FusionProfile, ...] = (
    ("f000", "Fusion B2/B4/B5=0/0/0", (0, 0, 0)),
    ("f101", "Fusion B2/B4/B5=1/0/1", (1, 0, 1)),
    ("f111", "Fusion B2/B4/B5=1/1/1", (1, 1, 1)),
)
TRUNCATION_PROFILES: tuple[TruncationProfile, ...] = (
    ("k13", "all K=13", (13, 13, 13, 13, 13)),
    ("k8", "all K=8", (8, 8, 8, 8, 8)),
    ("k6", "all K=6", (6, 6, 6, 6, 6)),
    (
        "high",
        "high precision B1/B2/B3/B4/B5=11/10/10/12/11",
        (11, 10, 10, 12, 11),
    ),
    (
        "medium",
        "medium precision B1/B2/B3/B4/B5=9/8/8/10/9",
        (9, 8, 8, 10, 9),
    ),
    (
        "low",
        "low precision B1/B2/B3/B4/B5=7/6/6/8/7",
        (7, 6, 6, 8, 7),
    ),
)


@dataclass(frozen=True)
class GroupSpec:
    name: str
    label: str
    fusion_by_block: tuple[int, int, int]
    k_by_block: tuple[int, int, int, int, int]

    @property
    def policy_representable(self) -> bool:
        return self.fusion_by_block[0] == 1 and self.fusion_by_block[2] == 1


def build_group_specs(*, num_layers: int = NUM_LAYERS) -> tuple[GroupSpec, ...]:
    if int(num_layers) < 1:
        raise ValueError("num_layers must be positive")
    validate_exact_k_domain(K_LEVELS)
    return tuple(
        GroupSpec(
            name=f"{fusion_key}_{k_key}",
            label=f"{fusion_label}; {k_label}",
            fusion_by_block=tuple(int(value) for value in fusion_values),
            k_by_block=tuple(int(value) for value in k_values),
        )
        for fusion_key, fusion_label, fusion_values in FUSION_PROFILES
        for k_key, k_label, k_values in TRUNCATION_PROFILES
    )


GROUP_SPECS = build_group_specs()
GROUP_ORDER = tuple(group.name for group in GROUP_SPECS)
GROUP_LABELS = {group.name: group.label for group in GROUP_SPECS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_value(*args: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _patch_chain_constants() -> None:
    chain.MODEL_TYPE = MODEL_TYPE
    chain.DATASET = DATASET
    chain.NUM_LAYERS = NUM_LAYERS
    chain.VALIDATION_SIZE = VALIDATION_SIZE
    chain.ALL4_GELU = ALL4_GELU
    chain.FIXED_SOFTMAX = FIXED_SOFTMAX


def _k_index(k_value: int) -> int:
    levels = tuple(int(value) for value in K_LEVELS)
    validate_exact_k_domain(levels)
    try:
        return levels.index(int(k_value))
    except ValueError as exc:
        raise ValueError(f"K={k_value} is absent from K_LEVELS={levels}") from exc


def build_policy_action_matrix(
        group: GroupSpec,
        *,
        num_layers: int = NUM_LAYERS,
        ) -> tuple[tuple[int, ...], ...]:
    if not group.policy_representable:
        raise ValueError(
            f"{group.name}: fusion profile is not policy-representable"
        )
    k_indices = tuple(_k_index(k_value) for k_value in group.k_by_block)
    row = (int(group.fusion_by_block[1]), *k_indices)
    return tuple(row for _ in range(int(num_layers)))


def apply_k_profile_to_full_vector(
        baseline_action_vec: Sequence[int],
        *,
        k_by_block: Sequence[int],
        num_layers: int = NUM_LAYERS,
        per_layer_fields: Sequence[tuple[int, str, str]] | None = None,
        ) -> np.ndarray:
    profile = tuple(int(value) for value in k_by_block)
    if len(profile) != 5:
        raise ValueError(f"k_by_block must contain B1-B5, got {profile}")
    result = np.asarray(baseline_action_vec, dtype=int).reshape(-1).copy()
    k_indices = {
        block_idx: _k_index(profile[block_idx - 1])
        for block_idx in range(1, 6)
    }
    if per_layer_fields is None:
        from blb_stage2_rl.action_space import per_layer_field_offsets

        fields = tuple(per_layer_field_offsets())
    else:
        fields = tuple(per_layer_fields)
    layer_width = len(fields)
    expected = layer_width * int(num_layers) + 1
    if result.size != expected:
        raise RuntimeError(
            f"baseline action has {result.size} slots, expected {expected}"
        )
    seen_blocks = set()
    for layer_idx in range(int(num_layers)):
        for field_offset, (block_idx, _field_name, kind) in enumerate(fields):
            if kind != "K":
                continue
            block = int(block_idx)
            if block not in k_indices:
                raise ValueError(f"unexpected K block index {block}")
            result[layer_idx * layer_width + field_offset] = k_indices[block]
            seen_blocks.add(block)
    if seen_blocks != set(range(1, 6)):
        raise RuntimeError(
            f"K fields cover blocks {sorted(seen_blocks)}, expected 1..5"
        )
    return result


def _build_runtime(
        args: argparse.Namespace,
        deps: Mapping[str, Any],
        evaluator: Any,
        clean_reference: Mapping[str, Any],
        ) -> chain.Stage2Runtime:
    gelu = np.asarray(ALL4_GELU, dtype=int)
    softmax = np.asarray(FIXED_SOFTMAX, dtype=int)
    evaluator.apply_configuration(gelu, softmax)
    try:
        evaluator.reversible_handler.restore_layer_input_noise(
            layer_indices=list(range(NUM_LAYERS)),
        )
    except Exception:
        pass

    profile = deps["resolve_stage2_profile"](
        DATASET,
        model_type=MODEL_TYPE,
        num_layers=NUM_LAYERS,
    )
    if str(profile) != PROFILE:
        raise RuntimeError(f"resolved profile {profile!r}, expected {PROFILE!r}")

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
            f"validation_full batches {len(validation_batches)} != {expected_batches}"
        )
    bridge = runner._build_rescale_bridge(
        train_cfg,
        log=lambda message: print(message, flush=True),
    )
    calibrated = deps["load_calibrated_stage2_action_context"](
        rescale_optimizer_root=str(args.rescale_optimizer_root),
        dataset=profile,
        num_layers=NUM_LAYERS,
        gelu_per_layer=list(ALL4_GELU),
        softmax_per_layer=list(FIXED_SOFTMAX),
        snap_sf_to_noise_table=False,
    )
    deps["validate_calibrated_stage2_action_context"](
        calibrated,
        dataset=profile,
        num_layers=NUM_LAYERS,
        gelu_per_layer=list(ALL4_GELU),
        softmax_per_layer=list(FIXED_SOFTMAX),
        snap_sf_to_noise_table=False,
    )

    RecordingEnv = chain._recording_env_class(deps["BLBStage2Env"])
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
        num_layers=NUM_LAYERS,
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
    required_graphs = {"block2_mrpc", "block4", "block5_n4"}
    missing = required_graphs.difference(fusion_map.graphs)
    if missing:
        raise RuntimeError(f"{profile}: missing fusion-map graphs {sorted(missing)}")
    for graph_key in sorted(required_graphs):
        counts = [
            int(option.fusion_count)
            for option in fusion_map.graphs[graph_key].options
        ]
        if counts != [0, 1]:
            raise RuntimeError(
                f"{profile}/{graph_key}: expected fusion counts [0, 1], got {counts}"
            )
    layerwise_env = deps["BLBStage2LayerwiseEnv"](
        base_env=base_env,
        fusion_map=fusion_map,
        baseline_action_vec=calibrated.baseline_action_vec,
        profile=profile,
    )
    return chain.Stage2Runtime(
        base_env=base_env,
        layerwise_env=layerwise_env,
        fusion_map=fusion_map,
        calibrated_context=calibrated,
        profile=profile,
    )


def _trial_payload(base_env: Any, wall_seconds: float) -> tuple[dict, list[int]]:
    values = base_env.fixed_eval_trial_metrics
    if not isinstance(values, Mapping):
        raise RuntimeError("terminal evaluation did not expose raw trial metrics")
    count = len(values.get("loss") or [])
    expected = int(base_env.env_cfg.num_trials_per_step)
    if count != expected:
        raise RuntimeError(f"got {count} trial metrics, expected {expected}")
    trials = [
        {
            "loss": float(values["loss"][idx]),
            "p": float(values["metric1"][idx]),
            "s": float(values["metric2"][idx]),
            "time_ms": wall_seconds * 1000.0 / max(1, count),
        }
        for idx in range(count)
    ]
    packed = pack_repeat_evaluation(
        trials,
        evaluation_mode="stage2_rl_terminal_validation_full",
    )
    seeds = [int(value) for value in values.get("trial_seeds") or []]
    if len(seeds) != expected:
        raise RuntimeError(f"got {len(seeds)} trial seeds, expected {expected}")
    return packed, seeds


def _k_evidence(
        action_vec: Sequence[int],
        runtime: chain.Stage2Runtime,
        *,
        expected_k_by_block: Sequence[int],
        ) -> list[dict]:
    from blb_stage2_rl.action_space import describe_action_vector

    expected = {
        block_idx: int(value)
        for block_idx, value in enumerate(expected_k_by_block, start=1)
    }
    if set(expected) != set(range(1, 6)):
        raise ValueError(
            f"expected_k_by_block must cover B1-B5, got {expected}"
        )
    description = describe_action_vector(
        np.asarray(action_vec, dtype=int),
        max_sfs=runtime.calibrated_context.max_sfs,
        num_layers=NUM_LAYERS,
        gelu_degree=list(ALL4_GELU),
        attn_degree=list(FIXED_SOFTMAX),
        profile=PROFILE,
    )
    rows = [
        {
            "layer": int(record["layer"]),
            "block": int(record["block_index"]),
            "slot_label": str(record["slot_label"]),
            "action_index": int(record["action_index"]),
            "k_value": int(record["effective_value"]),
        }
        for record in description["records"]
        if record.get("kind") == "K" and bool(record.get("effective"))
    ]
    if len(rows) != 5 * NUM_LAYERS:
        raise RuntimeError(
            f"found {len(rows)} effective K slots, expected {5 * NUM_LAYERS}"
        )
    wrong = [
        row for row in rows
        if row["k_value"] != expected[int(row["block"])]
    ]
    if wrong:
        raise RuntimeError(
            "K evidence differs from the requested per-block profile: "
            f"{wrong[:3]}"
        )
    return rows


def installed_k_evidence(
        decoded: Any,
        *,
        expected_k_by_block: Sequence[int],
        num_layers: int = NUM_LAYERS,
        expected_backend: str = "binary",
        ) -> list[dict]:
    expected = {
        block_idx: int(value)
        for block_idx, value in enumerate(expected_k_by_block, start=1)
    }
    if set(expected) != set(range(1, 6)):
        raise ValueError(
            f"expected_k_by_block must cover B1-B5, got {expected}"
        )
    rows = []
    for layer_idx in range(int(num_layers)):
        for block_idx in range(1, 6):
            cfgs = getattr(decoded, f"block{block_idx}_cfgs", None)
            if not isinstance(cfgs, Mapping) or layer_idx not in cfgs:
                raise RuntimeError(
                    f"installed cfg missing layer {layer_idx} block {block_idx}"
                )
            cfg = cfgs[layer_idx]
            actual_k = getattr(cfg, "output_truncation_k", None)
            if actual_k is None or int(actual_k) != expected[block_idx]:
                raise RuntimeError(
                    "installed K mismatch at "
                    f"L{layer_idx}.B{block_idx}: {actual_k!r} "
                    f"!= {expected[block_idx]}"
                )
            backend = str(
                getattr(cfg, "output_truncation_mode", "") or ""
            )
            if backend != str(expected_backend):
                raise RuntimeError(
                    "installed truncation backend mismatch at "
                    f"L{layer_idx}.B{block_idx}: {backend!r} "
                    f"!= {str(expected_backend)!r}"
                )
            rows.append({
                "layer": int(layer_idx),
                "block": int(block_idx),
                "k_value": int(actual_k),
                "backend": backend,
            })
    return rows


def _runtime_gate(info: Mapping[str, Any], *, expected_fusion: int) -> dict:
    if bool(info.get("invalid", False)):
        raise RuntimeError(f"terminal action invalid: {info.get('materialization_failure_reason')}")
    if not bool(info.get("forward_ran", False)):
        raise RuntimeError(
            f"model forward did not run: {info.get('forward_skipped_reason') or info.get('error')}"
        )
    replan = info.get("replan_application") or {}
    if not bool(replan.get("model_uses_replan_config", False)):
        raise RuntimeError("post-replan configuration was not installed in the model")
    signals = info.get("opt_signals")
    actual_fusion = int(getattr(signals, "total_fusion_count"))
    if actual_fusion != int(expected_fusion):
        raise RuntimeError(
            f"actual total fusion {actual_fusion} != expected {expected_fusion}"
        )
    return {
        "forward_ran": True,
        "model_uses_replan_config": True,
        "invalid": False,
        "actual_total_fusion_count": actual_fusion,
        "total_bits_sum": int(getattr(signals, "total_bits_sum")),
        "final_config_fingerprint": str(
            info.get("final_config_fingerprint") or ""
        ),
        "materialization_failure_reason": info.get(
            "materialization_failure_reason"
        ),
        "probe_diagnostics": to_jsonable(
            info.get("probe_diagnostics") or {},
            stringify_unknown=True,
            preserve_native=True,
        ),
    }


def _run_control_group(
        runtime: chain.Stage2Runtime,
        *,
        group: GroupSpec,
        seed: int,
        ) -> dict:
    if group.policy_representable:
        raise ValueError(f"{group.name}: expected a fusion=0 control group")
    base_env = runtime.base_env
    action_vec = apply_k_profile_to_full_vector(
        runtime.calibrated_context.baseline_action_vec,
        k_by_block=group.k_by_block,
    )
    base_env.clear_installed_blb()
    base_env.fixed_eval_trial_metrics = None
    base_env.reset(seed=int(seed))
    base_env.probe_noise_seed = int(seed)
    started = time.perf_counter()
    _state, reward, done, info = base_env.step(action_vec)
    wall_seconds = float(time.perf_counter() - started)
    if not done:
        raise RuntimeError(f"{group.name}: base environment did not terminate")
    gate = _runtime_gate(info, expected_fusion=0)
    packed, trial_seeds = _trial_payload(base_env, wall_seconds)
    action_k_rows = _k_evidence(
        action_vec,
        runtime,
        expected_k_by_block=group.k_by_block,
    )
    installed_k_rows = installed_k_evidence(
        info.get("decoded"),
        expected_k_by_block=group.k_by_block,
        expected_backend=str(base_env.env_cfg.truncation_backend),
    )
    base_env.clear_installed_blb()
    return {
        "name": group.name,
        "label": group.label,
        "path": "calibrated_baseline_action_vec -> BLBStage2Env.step",
        "policy_representable": False,
        "expected_fusion_by_block": {"2": 0, "4": 0, "5": 0},
        "actual_fusion_by_block": {"2": 0, "4": 0, "5": 0},
        "k_by_block": {
            str(block_idx): int(value)
            for block_idx, value in enumerate(group.k_by_block, start=1)
        },
        "effective_k_count": len(installed_k_rows),
        "action_k_choices": action_k_rows,
        "k_choices": installed_k_rows,
        "trial_seeds": trial_seeds,
        "repeat_evaluation": packed,
        "terminal_reward_diagnostic_only": float(reward),
        "runtime_gate": gate,
        "fusion_option_ids": [],
        "boosted_overrides": [],
        "boosted_override_count": 0,
        "wall_seconds": wall_seconds,
    }


def _run_candidate_group(
        runtime: chain.Stage2Runtime,
        *,
        group: GroupSpec,
        seed: int,
        ) -> dict:
    if not group.policy_representable:
        raise ValueError(f"{group.name}: expected a policy-representable group")
    env = runtime.layerwise_env
    base_env = runtime.base_env
    matrix = build_policy_action_matrix(group)
    base_env.clear_installed_blb()
    base_env.fixed_eval_trial_metrics = None
    env.reset(seed=int(seed))
    base_env.probe_noise_seed = int(seed)
    terminal_info: Dict[str, Any] = {}
    started = time.perf_counter()
    for layer_idx, row in enumerate(matrix):
        _obs, reward, done, terminal_info = env.step(row)
        if done != (layer_idx == NUM_LAYERS - 1):
            raise RuntimeError(
                f"{group.name}: unexpected terminal flag at layer {layer_idx}"
            )
    wall_seconds = float(time.perf_counter() - started)
    info = env.runtime_terminal_info
    if not isinstance(info, Mapping):
        raise RuntimeError(f"{group.name}: missing raw runtime terminal info")
    expected_fusion_total = sum(group.fusion_by_block) * NUM_LAYERS
    gate = _runtime_gate(info, expected_fusion=expected_fusion_total)
    packed, trial_seeds = _trial_payload(base_env, wall_seconds)
    action_k_rows = _k_evidence(
        env.pending_full_vector,
        runtime,
        expected_k_by_block=group.k_by_block,
    )
    installed_k_rows = installed_k_evidence(
        info.get("decoded"),
        expected_k_by_block=group.k_by_block,
        expected_backend=str(base_env.env_cfg.truncation_backend),
    )

    fusion_by_block = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for layer_summary in env.layer_summaries:
        if not bool(layer_summary["all_valid"]):
            raise RuntimeError(
                f"{group.name}: invalid layer replan at "
                f"{layer_summary['layer_idx']}"
            )
        for block in layer_summary["blocks"]:
            block_idx = int(block["block_idx"])
            fusion_by_block[block_idx] += int(block["fusion_count"])
            replan = block.get("replan_application") or {}
            if not bool(replan.get("model_uses_replan_config", False)):
                raise RuntimeError(
                    f"{group.name}: layer {layer_summary['layer_idx']} block "
                    f"{block_idx} did not install replan config"
                )
    expected = {
        1: 0,
        2: NUM_LAYERS * int(group.fusion_by_block[0]),
        3: 0,
        4: NUM_LAYERS * int(group.fusion_by_block[1]),
        5: NUM_LAYERS * int(group.fusion_by_block[2]),
    }
    if fusion_by_block != expected:
        raise RuntimeError(
            f"{group.name}: actual fusion {fusion_by_block} != {expected}"
        )
    boosted_overrides = list(terminal_info.get("boosted_overrides") or [])
    if len(boosted_overrides) != expected_fusion_total:
        raise RuntimeError(
            f"{group.name}: got {len(boosted_overrides)} boosted overrides, "
            f"expected {expected_fusion_total}"
        )
    base_env.clear_installed_blb()
    return {
        "name": group.name,
        "label": group.label,
        "path": "BLBStage2LayerwiseEnv -> BLBStage2Env.step(boosted_overrides)",
        "policy_representable": True,
        "action_matrix": [list(row) for row in matrix],
        "expected_fusion_by_block": {
            "2": expected[2],
            "4": expected[4],
            "5": expected[5],
        },
        "actual_fusion_by_block": {
            str(block): int(value) for block, value in fusion_by_block.items()
        },
        "k_by_block": {
            str(block_idx): int(value)
            for block_idx, value in enumerate(group.k_by_block, start=1)
        },
        "effective_k_count": len(installed_k_rows),
        "action_k_choices": action_k_rows,
        "k_choices": installed_k_rows,
        "trial_seeds": trial_seeds,
        "repeat_evaluation": packed,
        "terminal_reward_diagnostic_only": float(reward),
        "runtime_gate": gate,
        "fusion_option_ids": to_jsonable(
            terminal_info.get("fusion_option_ids") or [],
            stringify_unknown=True,
            preserve_native=True,
        ),
        "boosted_overrides": to_jsonable(
            boosted_overrides,
            stringify_unknown=True,
            preserve_native=True,
        ),
        "boosted_override_count": len(boosted_overrides),
        "wall_seconds": wall_seconds,
    }


def run_seed(args: argparse.Namespace) -> int:
    if int(args.repeat) != 5:
        raise ValueError("the stability comparison requires exactly five trials")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.rescale_optimizer_root = str(Path(args.rescale_optimizer_root).resolve())
    _patch_chain_constants()
    deps = chain._load_runtime_deps()
    evaluator_class = deps["LayerImportanceEvaluator"]

    def evaluator_factory(*factory_args, **factory_kwargs):
        factory_kwargs.setdefault("stage1_entropy_stop_threshold", 0.1)
        evaluator_module = sys.modules[evaluator_class.__module__]
        factory_kwargs["stage1_rl_episodes"] = int(
            evaluator_module.PPO_MAX_EPISODES
        )
        factory_kwargs["stage1_rl_episodes_specified"] = False
        factory_kwargs["stage2_rl_episodes_specified"] = False
        return evaluator_class(*factory_args, **factory_kwargs)

    deps["LayerImportanceEvaluator"] = evaluator_factory
    evaluator, base_model = chain._build_evaluator(args, deps, output_dir)
    clean_reference = chain._clean_eval(
        evaluator,
        gelu=ALL4_GELU,
        softmax=FIXED_SOFTMAX,
        repeat=1,
        label="GELU4 clean calibration reference",
    )
    runtime = _build_runtime(args, deps, evaluator, clean_reference)

    groups = {}
    for group in GROUP_SPECS:
        print(
            f"[grid] {group.name}: fusion={group.fusion_by_block} "
            f"K={group.k_by_block}",
            flush=True,
        )
        if group.policy_representable:
            result = _run_candidate_group(
                runtime,
                group=group,
                seed=int(args.seed),
            )
        else:
            result = _run_control_group(
                runtime,
                group=group,
                seed=int(args.seed),
            )
        groups[group.name] = result
        stats = result["repeat_evaluation"]["stats"]
        print(
            f"[grid] {group.name} complete: "
            f"loss={stats['loss_mean']:.6f}+/-{stats['loss_std']:.6f} "
            f"m1={stats['p_mean']:.6f}+/-{stats['p_std']:.6f} "
            f"m2={stats['s_mean']:.6f}+/-{stats['s_std']:.6f}",
            flush=True,
        )
    reference_seeds = groups[GROUP_ORDER[0]]["trial_seeds"]
    for name in GROUP_ORDER[1:]:
        if groups[name]["trial_seeds"] != reference_seeds:
            raise RuntimeError(
                f"paired trial seeds differ for {name}: "
                f"{groups[name]['trial_seeds']} vs {reference_seeds}"
            )

    payload = {
        "schema_version": "stage2_precision_stability_grid_seed_v1",
        "generated_at_utc": _utc_now(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "base_model": str(base_model),
        "model_type": MODEL_TYPE,
        "dataset": DATASET,
        "profile": PROFILE,
        "validation_full_size": VALIDATION_SIZE,
        "stage1_gelu": list(ALL4_GELU),
        "stage1_softmax": list(FIXED_SOFTMAX),
        "seed": int(args.seed),
        "repeat": int(args.repeat),
        "truncation_backend": str(args.truncation_backend),
        "groups": groups,
    }
    output_path = output_dir / "precision_stability_grid_seed_result.json"
    write_json_file(output_path, payload)
    print(json.dumps({
        "output": str(output_path),
        "seed": int(args.seed),
        "groups": list(groups),
    }, indent=2))
    return 0


def _sample_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _metric_trials(group: Mapping[str, Any], key: str) -> list[float]:
    return [
        float(row[key])
        for row in group["repeat_evaluation"]["trials"]
    ]


def _aggregate_group(seed_payloads: Sequence[Mapping[str, Any]], name: str) -> dict:
    metric_keys = ("loss", "p", "s")
    all_trials = []
    per_seed = []
    trial_seeds = []
    for payload in seed_payloads:
        group = payload["groups"][name]
        seed_row = {"seed": int(payload["seed"])}
        packed = group["repeat_evaluation"]
        stats = packed["stats"]
        for key in metric_keys:
            seed_row[f"{key}_mean"] = float(stats[f"{key}_mean"])
            seed_row[f"{key}_std"] = float(stats[f"{key}_std"])
        all_trials.extend(
            {
                "loss": float(row["loss"]),
                "p": float(row["p"]),
                "s": float(row["s"]),
                "time_ms": float(row.get("time_ms", 0.0)),
            }
            for row in packed["trials"]
        )
        seed_row["trial_seeds"] = [
            int(value) for value in group["trial_seeds"]
        ]
        trial_seeds.extend(seed_row["trial_seeds"])
        per_seed.append(seed_row)
    aggregate_repeat = pack_repeat_evaluation(
        all_trials,
        evaluation_mode="stage2_validation_full_5_seeds_x_5_trials",
    )
    summary = dict(aggregate_repeat["stats"])
    for key in metric_keys:
        seed_stds = [float(row[f"{key}_std"]) for row in per_seed]
        summary[f"{key}_within_seed_std_mean"] = _mean(seed_stds)
        summary[f"{key}_within_seed_std_max"] = max(seed_stds)
    first = seed_payloads[0]["groups"][name]
    fingerprints = sorted({
        str(payload["groups"][name]["runtime_gate"]["final_config_fingerprint"])
        for payload in seed_payloads
    })
    if len(fingerprints) != 1 or not fingerprints[0]:
        raise RuntimeError(
            f"{name}: expected one stable non-empty installed-config "
            f"fingerprint, got {fingerprints}"
        )
    k_choices = first["k_choices"]
    if any(
            payload["groups"][name]["k_choices"] != k_choices
            for payload in seed_payloads[1:]):
        raise RuntimeError(f"{name}: installed K evidence changed across seeds")
    fusion_by_block = first["actual_fusion_by_block"]
    if any(
            payload["groups"][name]["actual_fusion_by_block"] != fusion_by_block
            for payload in seed_payloads[1:]):
        raise RuntimeError(f"{name}: realized fusion changed across seeds")
    return {
        "name": name,
        "label": first["label"],
        "n": int(summary["n"]),
        "summary": summary,
        "repeat_evaluation": aggregate_repeat,
        "per_seed": per_seed,
        "trial_seeds": trial_seeds,
        "path": first["path"],
        "policy_representable": bool(first["policy_representable"]),
        "actual_fusion_by_block": fusion_by_block,
        "k_by_block": dict(first["k_by_block"]),
        "effective_k_count": int(first["effective_k_count"]),
        "k_choices": k_choices,
        "fingerprints": fingerprints,
        "all_runtime_gates_passed": all(
            payload["groups"][name]["runtime_gate"]["forward_ran"]
            and payload["groups"][name]["runtime_gate"]["model_uses_replan_config"]
            and not payload["groups"][name]["runtime_gate"]["invalid"]
            for payload in seed_payloads
        ),
    }


def _paired_comparison(
        seed_payloads: Sequence[Mapping[str, Any]],
        left: str,
        right: str,
        ) -> dict:
    metrics = {"loss": "loss", "accuracy": "p", "weighted_f1": "s"}
    result = {
        "left": left,
        "right": right,
        "label": f"{GROUP_LABELS[left]} minus {GROUP_LABELS[right]}",
        "metrics": {},
    }
    for label, key in metrics.items():
        diffs = []
        for payload in seed_payloads:
            left_group = payload["groups"][left]
            right_group = payload["groups"][right]
            if left_group["trial_seeds"] != right_group["trial_seeds"]:
                raise RuntimeError(
                    f"unpaired seeds for {left} and {right} in seed {payload['seed']}"
                )
            left_values = _metric_trials(left_group, key)
            right_values = _metric_trials(right_group, key)
            diffs.extend(
                float(a - b) for a, b in zip(left_values, right_values)
            )
        mean = _mean(diffs)
        std = _sample_std(diffs)
        se = std / math.sqrt(len(diffs)) if diffs else float("nan")
        t_975_df24 = 2.0639
        better = sum(
            value < 0.0 if key == "loss" else value > 0.0
            for value in diffs
        )
        result["metrics"][label] = {
            "mean_delta": mean,
            "std_delta": std,
            "ci95_low": mean - t_975_df24 * se,
            "ci95_high": mean + t_975_df24 * se,
            "paired_effect_snr": abs(mean) / std if std > 0.0 else float("inf"),
            "better_count": int(better),
            "n": len(diffs),
        }
    return result


def _constraint_status(groups: Mapping[str, Mapping[str, Any]]) -> dict:
    baseline = groups["f000_k13"]["summary"]
    loss_limit = float(baseline["loss_mean"]) * 1.001
    m1_limit = float(baseline["p_mean"]) * 0.999
    m2_limit = float(baseline["s_mean"]) * 0.999
    result = {
        "precision_tolerance": 0.001,
        "stability_multiplier": 2.0,
        "limits": {
            "loss_max": loss_limit,
            "accuracy_min": m1_limit,
            "weighted_f1_min": m2_limit,
        },
        "groups": {},
    }
    for name, group in groups.items():
        stats = group["summary"]
        precision = {
            "loss": float(stats["loss_mean"]) <= loss_limit,
            "accuracy": float(stats["p_mean"]) >= m1_limit,
            "weighted_f1": float(stats["s_mean"]) >= m2_limit,
        }
        stability = {}
        for key, label in (("loss", "loss"), ("p", "accuracy"), ("s", "weighted_f1")):
            baseline_std = float(baseline[f"{key}_within_seed_std_mean"])
            candidate_std = float(stats[f"{key}_within_seed_std_mean"])
            ratio = candidate_std / baseline_std if baseline_std > 0.0 else float("inf")
            stability[label] = {
                "ratio": ratio,
                "pass": ratio <= 2.0,
            }
        result["groups"][name] = {
            "precision": precision,
            "precision_all_pass": all(precision.values()),
            "stability": stability,
            "stability_all_pass": all(
                item["pass"] for item in stability.values()
            ),
        }
    return result


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _pass(flag: bool) -> str:
    return '<span class="pass">PASS</span>' if flag else '<span class="fail">FAIL</span>'


def _format_k_profile(k_by_block: Mapping[str, Any]) -> str:
    return "/".join(
        str(int(k_by_block[str(block_idx)]))
        for block_idx in range(1, 6)
    )


def _render_html(payload: Mapping[str, Any]) -> str:
    groups = payload["groups"]
    constraints = payload["constraint_reference"]
    summary_rows = []
    for name in GROUP_ORDER:
        group = groups[name]
        stats = group["summary"]
        gate = constraints["groups"][name]
        fusion = group["actual_fusion_by_block"]
        summary_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(group['label'])}</strong><br><code>{name}</code></td>"
            f"<td>{group['n']}</td>"
            f"<td>{_fmt(stats['loss_mean'])} +/- {_fmt(stats['loss_std'])}<br>"
            f"<small>within-seed std mean {_fmt(stats['loss_within_seed_std_mean'])}</small></td>"
            f"<td>{_fmt(stats['p_mean'])} +/- {_fmt(stats['p_std'])}<br>"
            f"<small>within-seed std mean {_fmt(stats['p_within_seed_std_mean'])}</small></td>"
            f"<td>{_fmt(stats['s_mean'])} +/- {_fmt(stats['s_std'])}<br>"
            f"<small>within-seed std mean {_fmt(stats['s_within_seed_std_mean'])}</small></td>"
            f"<td>B2={fusion.get('2', 0)}, B4={fusion.get('4', 0)}, B5={fusion.get('5', 0)}</td>"
            f"<td>B1/B2/B3/B4/B5={_format_k_profile(group['k_by_block'])}"
            f"<br><small>{group['effective_k_count']} installed slots</small></td>"
            f"<td>{_pass(gate['precision_all_pass'])}</td>"
            f"<td>{_pass(gate['stability_all_pass'])}</td>"
            "</tr>"
        )

    seed_rows = []
    for name in GROUP_ORDER:
        for row in groups[name]["per_seed"]:
            seed_rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td><td>{row['seed']}</td>"
                f"<td>{_fmt(row['loss_mean'])} +/- {_fmt(row['loss_std'])}</td>"
                f"<td>{_fmt(row['p_mean'])} +/- {_fmt(row['p_std'])}</td>"
                f"<td>{_fmt(row['s_mean'])} +/- {_fmt(row['s_std'])}</td>"
                "</tr>"
            )

    comparison_rows = []
    for comparison in payload["paired_comparisons"]:
        for metric, row in comparison["metrics"].items():
            comparison_rows.append(
                "<tr>"
                f"<td><code>{html.escape(comparison['left'])}</code> - "
                f"<code>{html.escape(comparison['right'])}</code></td>"
                f"<td>{html.escape(metric)}</td>"
                f"<td>{_fmt(row['mean_delta'])}</td>"
                f"<td>[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]</td>"
                f"<td>{_fmt(row['std_delta'])}</td>"
                f"<td>{_fmt(row['paired_effect_snr'], 3)}</td>"
                f"<td>{row['better_count']}/{row['n']}</td>"
                "</tr>"
            )

    stability_rows = []
    for name in GROUP_ORDER:
        gate = constraints["groups"][name]
        for metric, value in gate["stability"].items():
            stability_rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td><td>{html.escape(metric)}</td>"
                f"<td>{_fmt(value['ratio'], 3)}x</td><td>{_pass(value['pass'])}</td>"
                "</tr>"
            )

    audit_rows = []
    for name in GROUP_ORDER:
        group = groups[name]
        fusion = group["actual_fusion_by_block"]
        audit_rows.append(
            "<tr>"
            f"<td><code>{html.escape(name)}</code></td>"
            f"<td>{html.escape(group['path'])}</td>"
            f"<td>B2={fusion.get('2', 0)}, B4={fusion.get('4', 0)}, B5={fusion.get('5', 0)}</td>"
            f"<td>B1/B2/B3/B4/B5={_format_k_profile(group['k_by_block'])}"
            f" x 12 layers ({group['effective_k_count']} slots)</td>"
            f"<td>{_pass(group['all_runtime_gates_passed'])}</td>"
            f"<td>{len(group['fingerprints'])} unique across seeds</td>"
            "</tr>"
        )

    layer_rows = []
    for name in GROUP_ORDER:
        for layer_idx in range(NUM_LAYERS):
            k_by_block = {
                int(row["block"]): int(row["k_value"])
                for row in groups[name]["k_choices"]
                if int(row["layer"]) == layer_idx
            }
            fusion = groups[name]["actual_fusion_by_block"]
            layer_rows.append(
                "<tr>"
                f"<td><code>{html.escape(name)}</code></td>"
                f"<td>L{layer_idx}</td>"
                f"<td>"
                f"{int(fusion.get('2', 0) > 0)}/"
                f"{int(fusion.get('4', 0) > 0)}/"
                f"{int(fusion.get('5', 0) > 0)}</td>"
                f"<td>{int(k_by_block[1])}</td>"
                f"<td>{int(k_by_block[2])}</td>"
                f"<td>{int(k_by_block[3])}</td>"
                f"<td>{int(k_by_block[4])}</td>"
                f"<td>{int(k_by_block[5])}</td>"
                "</tr>"
            )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage-2 18-group precision and stability evaluation</title>
<style>
:root{{--ink:#18212b;--muted:#536273;--line:#cbd5df;--head:#e8eef4;--band:#f4f7fa;--blue:#135fa7;--green:#176a37;--red:#a12b2b}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#fff}}
header{{padding:28px 32px 20px;border-bottom:4px solid var(--blue);background:var(--band)}}main{{padding:0 32px 40px;max-width:1680px;margin:auto}}
h1{{font-size:28px;margin:0 0 8px;letter-spacing:0}}h2{{font-size:19px;margin:30px 0 8px;letter-spacing:0}}
p,li{{font-size:14px;line-height:1.55}}.meta{{color:var(--muted)}}.note{{border-left:4px solid var(--blue);padding:10px 14px;background:#eef5fb;margin:14px 0}}
table{{width:100%;border-collapse:collapse;margin:10px 0 22px}}th,td{{border:1px solid var(--line);padding:8px;vertical-align:top;font-size:12px}}th{{background:var(--head);text-align:left}}tbody tr:nth-child(even){{background:#fafbfd}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}}small{{color:var(--muted)}}.pass{{color:var(--green);font-weight:700}}.fail{{color:var(--red);font-weight:700}}
.key{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));border:1px solid var(--line);margin:16px 0}}.key div{{padding:12px;border-right:1px solid var(--line)}}.key div:last-child{{border-right:0}}.key strong{{display:block;font-size:18px;margin-top:4px}}
@media(max-width:900px){{header,main{{padding-left:14px;padding-right:14px}}.key{{grid-template-columns:1fr}}.key div{{border-right:0;border-bottom:1px solid var(--line)}}table{{display:block;overflow:auto}}}}
</style>
</head>
<body>
<header><h1>Stage-2 18 组 Fusion/K：精度与稳定性</h1>
<div class="meta">BERT-base MRPC | GELU=[4]x12 | Softmax=[6]x12 |
validation_full={payload['validation_full_size']} | 5 seeds x 5 trials/group |
generated {html.escape(payload['generated_at_utc'])}</div></header>
<main>
<div class="note"><strong>实验口径。</strong>18 组使用同一组配对 trial seeds。
Fusion 1/0/1 与 1/1/1 走生产 <code>BLBStage2LayerwiseEnv</code>；
Fusion 0/0/0 从 calibrated baseline action vector 只改 60 个有效 K 槽。
二者最终进入同一 <code>BLBStage2Env.step</code>、共用 optimizer write-back
与 installed-model inference。所有结果均要求真实 forward、真实
post-replan 安装、精确 fusion/K 审计且无 invalid。</div>

<div class="key">
<div>模型/数据<strong>BERT-base MRPC</strong></div>
<div>每组样本<strong>25 trials</strong></div>
<div>精度参考<strong>0.1%</strong></div>
<div>稳定性参考<strong>2.0x</strong></div>
</div>

<h2>25 次汇总</h2>
<table><thead><tr><th>组别</th><th>N</th><th>Loss mean +/- std</th>
<th>Accuracy mean +/- std</th><th>Weighted F1 mean +/- std</th>
<th>实际 fusion 总数</th><th>实际 K</th><th>精度门禁</th><th>稳定性门禁</th>
</tr></thead><tbody>{''.join(summary_rows)}</tbody></table>

<p class="meta">精度门禁仅作为当前 0.1% 约束的描述性复核：loss <= baseline x 1.001，
Accuracy/F1 >= baseline x 0.999。稳定性比值采用五个 seed 内 std 的均值，
并以 baseline 的对应值为 1.0；这是直观审计，不替代训练器的 bootstrap 统计门禁。</p>

<h2>配对差值与细微差异</h2>
<table><thead><tr><th>左组 - 右组</th><th>指标</th><th>平均差</th>
<th>95% paired CI</th><th>差值 std</th><th>|mean|/std</th><th>左组更优 trial</th>
</tr></thead><tbody>{''.join(comparison_rows)}</tbody></table>
<p class="meta">Loss 差值为负表示左组更好；Accuracy/F1 差值为正表示左组更好。
<code>|mean|/std</code> 越低，单次样本越难可靠区分配置。</p>

<h2>稳定性相对 baseline</h2>
<table><thead><tr><th>组别</th><th>指标</th><th>within-seed std 比值</th><th>200% 门禁</th>
</tr></thead><tbody>{''.join(stability_rows)}</tbody></table>

<h2>五个实验 seed</h2>
<table><thead><tr><th>组别</th><th>Seed</th><th>Loss</th><th>Accuracy</th><th>Weighted F1</th>
</tr></thead><tbody>{''.join(seed_rows)}</tbody></table>

<h2>真实动作与安装门禁</h2>
<table><thead><tr><th>组别</th><th>执行路径</th><th>实际 fusion</th><th>实际 K</th>
<th>forward/replan/valid</th><th>安装指纹</th></tr></thead>
<tbody>{''.join(audit_rows)}</tbody></table>

<h2>逐层动作审计（实际送入模型）</h2>
<details><summary>展开 18 组 x 12 层，共 216 行</summary>
<table><thead><tr><th>组别</th><th>层</th><th>Fusion B2/B4/B5</th>
<th>K B1</th><th>K B2</th><th>K B3</th><th>K B4</th><th>K B5</th></tr></thead>
<tbody>{''.join(layer_rows)}</tbody></table>
</details>

<h2>可复现性</h2>
<table><tbody>
<tr><th>Git commit</th><td><code>{html.escape(payload['git_commit'])}</code></td></tr>
<tr><th>Git tree</th><td><code>{html.escape(payload['git_tree'])}</code></td></tr>
<tr><th>Base model</th><td><code>{html.escape(payload['base_model'])}</code></td></tr>
<tr><th>Truncation backend</th><td><code>{html.escape(payload['truncation_backend'])}</code></td></tr>
<tr><th>Trial seeds</th><td><code>{html.escape(str(payload['seed_trial_seeds']))}</code></td></tr>
</tbody></table>
</main></body></html>"""


def aggregate(args: argparse.Namespace) -> int:
    paths = [Path(value).resolve() for value in args.inputs]
    seed_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    if len(seed_payloads) != 5:
        raise ValueError(f"expected five seed payloads, got {len(seed_payloads)}")
    seed_payloads.sort(key=lambda payload: int(payload["seed"]))
    commits = {str(payload["git_commit"]) for payload in seed_payloads}
    trees = {str(payload["git_tree"]) for payload in seed_payloads}
    if len(commits) != 1 or len(trees) != 1:
        raise RuntimeError(f"source mismatch: commits={commits}, trees={trees}")
    groups = {
        name: _aggregate_group(seed_payloads, name)
        for name in GROUP_ORDER
    }
    comparisons = [
        _paired_comparison(seed_payloads, name, "f000_k13")
        for name in GROUP_ORDER
        if name != "f000_k13"
    ]
    payload = {
        "schema_version": "stage2_precision_stability_grid_aggregate_v1",
        "generated_at_utc": _utc_now(),
        "git_commit": next(iter(commits)),
        "git_tree": next(iter(trees)),
        "base_model": seed_payloads[0]["base_model"],
        "model_type": MODEL_TYPE,
        "dataset": DATASET,
        "profile": PROFILE,
        "validation_full_size": VALIDATION_SIZE,
        "stage1_gelu": list(ALL4_GELU),
        "stage1_softmax": list(FIXED_SOFTMAX),
        "truncation_backend": seed_payloads[0]["truncation_backend"],
        "seed_trial_seeds": {
            str(payload["seed"]): payload["groups"][GROUP_ORDER[0]]["trial_seeds"]
            for payload in seed_payloads
        },
        "groups": groups,
        "paired_comparisons": comparisons,
        "constraint_reference": _constraint_status(groups),
        "source_files": [str(path) for path in paths],
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage2_18_group_precision_stability.json"
    html_path = output_dir / "stage2_18_group_precision_stability.html"
    write_json_file(json_path, payload)
    html_path.write_text(_render_html(payload), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "html": str(html_path),
        "groups": list(groups),
    }, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--base-model", default="")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--repeat", type=int, default=5)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--stage1-record-id", default=DEFAULT_RECORD_ID)
    run.add_argument(
        "--rescale-optimizer-root",
        default=str(REPO_ROOT / "Rescale_optimizer"),
    )
    run.add_argument("--truncation-backend", default="binary")
    run.add_argument("--truncation-ring-bits", type=int, default=43)
    run.add_argument(
        "--truncation-source-fractional-bits",
        type=int,
        default=24,
    )
    agg = subparsers.add_parser("aggregate")
    agg.add_argument("--output-dir", required=True)
    agg.add_argument("inputs", nargs="+")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        return run_seed(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
