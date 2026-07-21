#!/usr/bin/env python3
"""Replay two Block3 K choices through the production layerwise model path."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import torch

from blb_stage2_rl.action_space import (
    K_LEVELS,
    action_vector_to_cfgs,
    layer_dims,
    per_layer_field_offsets,
)
from blb_stage2_rl.baseline_bootstrap import (
    load_static_skeletons_baseline,
    static_skeletons_baseline_to_action,
)
from blb_stage2_rl.fusion_count_map import FusionCountMap
from blb_stage2_rl.layerwise_env import BLBStage2LayerwiseEnv
from json_utils import read_json_file, to_jsonable, write_json_file
from scripts.run_fusion_count_action_eval_rlpath import _build_evaluator, _build_seq_env


SEED = 20260721
GELU = [4] * 12
SOFTMAX = [6] * 12


def _cfg_payload(cfg):
    return to_jsonable(
        dataclasses.asdict(cfg),
        stringify_unknown=True,
        preserve_native=True,
    )


def _map_hashes(source_root: Path) -> dict:
    rows = {}
    map_dir = source_root / "blb_stage2_rl" / "fusion_maps" / "mrpc"
    for path in sorted(map_dir.glob("block*.json")):
        payload = read_json_file(path, default=None)
        if not isinstance(payload, dict) or "graph_key" not in payload or "options" not in payload:
            continue
        rows[str(payload["graph_key"])] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runroot", required=True)
    parser.add_argument("--local-glue-root", required=True)
    args = parser.parse_args()

    runroot = Path(args.runroot).resolve()
    source_root = runroot / "src"
    artifact_root = runroot / "artifacts"
    eval_args = argparse.Namespace(
        seed=SEED,
        base_model="textattack/bert-base-uncased-MRPC",
        dataset="mrpc",
        output_json=str(artifact_root / "block3_model_replay.json"),
        output_html=str(artifact_root / "unused.html"),
        prediction_jsonl="",
        batch_size=64,
        run_output_dir=str(artifact_root / "model_replay_runtime"),
        stage1_config_json=str(source_root / "glue_final_configs_best_ppo.json"),
        stage2_limit_tolerance=0.001,
        stage2_stability_tolerance=3.5,
        repeat=1,
        probe_size=408,
        rescale_optimizer_root=str(source_root / "Rescale_optimizer"),
    )

    print("[setup] loading evaluator/model/dataset", flush=True)
    evaluator = _build_evaluator(
        eval_args,
        stage1_gelu=GELU,
        stage1_softmax=SOFTMAX,
    )
    print("[setup] building canonical Stage-2 base environment", flush=True)
    legacy_seq_env, _baseline_metrics = _build_seq_env(
        eval_args,
        evaluator,
        stage1_gelu=GELU,
        stage1_softmax=SOFTMAX,
    )
    base_env = legacy_seq_env.base
    fusion_map = FusionCountMap.load("mrpc")
    static_baseline = load_static_skeletons_baseline(
        str(source_root / "Rescale_optimizer"),
        "mrpc",
        12,
        GELU,
        SOFTMAX,
    )
    baseline_vec, calibrated_max_sfs, baseline_cost, baseline_diag = (
        static_skeletons_baseline_to_action(
            static_baseline,
            snap_sf_to_noise_table=False,
        )
    )
    if base_env.max_sfs.by_layer_block_node != calibrated_max_sfs.by_layer_block_node:
        raise AssertionError("base environment does not use the calibrated RO baseline table")

    fields = per_layer_field_offsets()
    layer_width = len(layer_dims())
    block3_k_offset = next(
        idx
        for idx, (block, field_name, _kind) in enumerate(fields)
        if int(block) == 3 and str(field_name) == "output_truncation_k"
    )
    block3_non_k_offsets = [
        idx
        for idx, (block, field_name, _kind) in enumerate(fields)
        if int(block) == 3 and str(field_name) != "output_truncation_k"
    ]
    idx13 = int(K_LEVELS.index(13))
    idx8 = int(K_LEVELS.index(8))

    baseline_decoded = action_vector_to_cfgs(
        baseline_vec,
        calibrated_max_sfs,
        12,
        gelu_degree=GELU,
        attn_degree=SOFTMAX,
    )
    baseline_block3 = {
        str(layer): {
            "graph_key": str(static_baseline.per_block_layer[(3, layer)].graph_key),
            "fusion_count": int(static_baseline.per_block_layer[(3, layer)].fusion_count),
            "field_baseline_sfs": {
                str(key): int(value)
                for key, value in static_baseline.per_block_layer[
                    (3, layer)
                ].field_baseline_sfs.items()
            },
            "decoded_cfg": _cfg_payload(baseline_decoded.block3_cfgs[layer]),
        }
        for layer in range(12)
    }

    candidate_actions = {
        "block3_k13": [
            [0, idx13, idx13, idx13, idx13, idx13]
            for _ in range(12)
        ],
        "block3_k8": [
            [0, idx13, idx13, idx8, idx13, idx13]
            for _ in range(12)
        ],
    }

    results = {}
    for name, action_matrix in candidate_actions.items():
        print(f"[run] {name}", flush=True)
        env = BLBStage2LayerwiseEnv(
            base_env=base_env,
            fusion_map=fusion_map,
            baseline_action_vec=baseline_vec,
            profile="mrpc",
        )
        env.reset(seed=SEED)
        env.base.probe_noise_seed = SEED

        forward_logits = []
        installed_snapshots = []

        def capture_forward(_module, _inputs, output):
            logits = getattr(output, "logits", None)
            if logits is None:
                return
            forward_logits.append(logits.detach().float().cpu())
            installed = getattr(
                evaluator.reversible_handler,
                "block3_cfg_per_layer",
                {},
            )
            installed_snapshots.append({
                str(layer): _cfg_payload(cfg)
                for layer, cfg in sorted(installed.items())
            })

        hook = evaluator.model.register_forward_hook(capture_forward)
        info = {}
        reward = 0.0
        done = False
        try:
            for row in action_matrix:
                _obs, reward, done, info = env.step(row)
            if not done:
                raise AssertionError(f"{name} did not terminate after 12 layers")
        finally:
            hook.remove()

        if not forward_logits:
            raise AssertionError(f"{name} did not execute a real model forward")
        logits = torch.cat(forward_logits, dim=0)
        installed_first = installed_snapshots[0]
        if len(installed_first) != 12:
            raise AssertionError(
                f"{name} installed Block3 on {len(installed_first)} layers, expected 12"
            )
        expected_k = 13 if name.endswith("k13") else 8
        installed_k_values = sorted({
            int(cfg["output_truncation_k"])
            for cfg in installed_first.values()
        })
        if installed_k_values != [expected_k]:
            raise AssertionError(
                f"{name} installed K values {installed_k_values}, expected {expected_k}"
            )
        if any(snapshot != installed_first for snapshot in installed_snapshots):
            raise AssertionError(
                f"{name} installed Block3 cfg changed across model batches"
            )
        sf_chain_mismatch_layers = []
        for layer in range(12):
            expected_cfg = dict(baseline_block3[str(layer)]["decoded_cfg"])
            installed_cfg = dict(installed_first[str(layer)])
            expected_cfg.pop("output_truncation_k", None)
            installed_cfg.pop("output_truncation_k", None)
            if expected_cfg != installed_cfg:
                sf_chain_mismatch_layers.append(layer)
        if sf_chain_mismatch_layers:
            raise AssertionError(
                f"{name}: installed Block3 SF chain differs from the RO "
                f"baseline at layers {sf_chain_mismatch_layers}"
            )

        final_vec = env.pending_full_vector
        for layer in range(12):
            layer_start = layer * layer_width
            for relative_offset in block3_non_k_offsets:
                absolute = layer_start + relative_offset
                if int(final_vec[absolute]) != int(baseline_vec[absolute]):
                    raise AssertionError(
                        f"{name}: Block3 non-K action changed at "
                        f"layer={layer} offset={relative_offset}"
                    )
            k_absolute = layer_start + block3_k_offset
            expected_idx = idx13 if expected_k == 13 else idx8
            if int(final_vec[k_absolute]) != expected_idx:
                raise AssertionError(
                    f"{name}: Block3 K index mismatch at layer={layer}"
                )

        block3_replans = []
        for layer_summary in env.layer_summaries:
            matches = [
                row
                for row in layer_summary["blocks"]
                if int(row["block_idx"]) == 3
            ]
            if len(matches) != 1:
                raise AssertionError(
                    f"{name}: layer {layer_summary['layer_idx']} has "
                    f"{len(matches)} Block3 records"
                )
            row = matches[0]
            replan = dict(row.get("replan_application") or {})
            if not bool(row.get("valid")):
                raise AssertionError(
                    f"{name}: Block3 invalid at layer {layer_summary['layer_idx']}"
                )
            if int(row.get("fusion_count", -1)) != 0:
                raise AssertionError(
                    f"{name}: Block3 fusion_count={row.get('fusion_count')} not baseline 0"
                )
            if not bool(replan.get("model_uses_replan_config", False)):
                raise AssertionError(
                    f"{name}: Block3 replan not fully applied at "
                    f"layer {layer_summary['layer_idx']}"
                )
            block3_replans.append({
                "layer_idx": int(layer_summary["layer_idx"]),
                "graph_key": str(row["graph_key"]),
                "valid": bool(row["valid"]),
                "fusion_count": int(row["fusion_count"]),
                "total_bits": int(row["total_bits"]),
                "replan_application": to_jsonable(
                    replan,
                    stringify_unknown=True,
                    preserve_native=True,
                ),
                "optimizer_cfg_overrides": to_jsonable(
                    row.get("optimizer_cfg_overrides") or {},
                    stringify_unknown=True,
                    preserve_native=True,
                ),
            })

        runtime_terminal = env.runtime_terminal_info or {}
        metrics = runtime_terminal.get("metrics")
        result = {
            "name": name,
            "seed": SEED,
            "action_matrix": action_matrix,
            "expected_block3_k": expected_k,
            "block4_fusion_count_per_layer": 0,
            "fixed_block2_fusion_count_per_layer": 1,
            "fixed_block5_fusion_count_per_layer": 1,
            "pending_action_vector": [int(value) for value in final_vec.tolist()],
            "block3_non_k_actions_equal_ro_baseline": True,
            "block3_replans": block3_replans,
            "model_forward_batch_count": len(forward_logits),
            "model_forward_example_count": int(logits.shape[0]),
            "model_forward_logits": logits.tolist(),
            "model_forward_logits_sha256": hashlib.sha256(
                logits.numpy().tobytes()
            ).hexdigest(),
            "installed_block3_cfg_at_forward": installed_first,
            "installed_block3_layer_count": len(installed_first),
            "installed_block3_k_values": installed_k_values,
            "installed_sf_chain_matches_ro_baseline": True,
            "installed_sf_chain_mismatch_layers": sf_chain_mismatch_layers,
            "installed_cfg_stable_across_batches": True,
            "terminal_reward": float(reward),
            "terminal_forward_ran": bool(runtime_terminal.get("forward_ran", False)),
            "terminal_invalid": bool(runtime_terminal.get("invalid", False)),
            "terminal_apply_failed": bool(runtime_terminal.get("apply_failed", False)),
            "terminal_eval_failed": bool(runtime_terminal.get("eval_failed", False)),
            "terminal_replan_application": to_jsonable(
                runtime_terminal.get("replan_application") or {},
                stringify_unknown=True,
                preserve_native=True,
            ),
            "terminal_metrics": to_jsonable(
                metrics,
                stringify_unknown=True,
                preserve_native=True,
            ),
            "terminal_probe_diagnostics": to_jsonable(
                runtime_terminal.get("probe_diagnostics") or {},
                stringify_unknown=True,
                preserve_native=True,
            ),
        }
        if not result["terminal_forward_ran"]:
            raise AssertionError(f"{name}: terminal model forward did not run")
        if (
            result["terminal_invalid"]
            or result["terminal_apply_failed"]
            or result["terminal_eval_failed"]
        ):
            raise AssertionError(f"{name}: terminal failure flags present")
        if not bool(
            result["terminal_replan_application"].get(
                "model_uses_replan_config",
                False,
            )
        ):
            raise AssertionError(
                f"{name}: terminal replan config was not fully applied"
            )
        results[name] = result

    logits13 = torch.tensor(results["block3_k13"]["model_forward_logits"])
    logits8 = torch.tensor(results["block3_k8"]["model_forward_logits"])
    if logits13.shape != logits8.shape:
        raise AssertionError(
            f"candidate logit shapes differ: {tuple(logits13.shape)} "
            f"vs {tuple(logits8.shape)}"
        )
    delta = (logits13 - logits8).abs()
    comparison = {
        "same_shape": True,
        "shape": list(logits13.shape),
        "bitwise_equal": bool(torch.equal(logits13, logits8)),
        "allclose_atol_1e_8": bool(
            torch.allclose(logits13, logits8, rtol=0.0, atol=1e-8)
        ),
        "allclose_atol_1e_6": bool(
            torch.allclose(logits13, logits8, rtol=0.0, atol=1e-6)
        ),
        "max_abs_logit_delta": float(delta.max().item()),
        "mean_abs_logit_delta": float(delta.mean().item()),
        "changed_logit_count": int(torch.count_nonzero(delta).item()),
        "total_logit_count": int(delta.numel()),
    }
    if comparison["bitwise_equal"]:
        raise AssertionError(
            "Block3 K=13 and K=8 produced bitwise-identical model logits"
        )

    fusion_counts = sorted({
        int(static_baseline.per_block_layer[(3, layer)].fusion_count)
        for layer in range(12)
    })
    source_sync_raw = (runroot / "SOURCE_SYNC_COMMIT").read_text().strip()
    source_sync_commit = source_sync_raw.split("=", 1)[-1].strip()
    checked_out_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_sync_commit != checked_out_commit:
        raise AssertionError(
            f"source marker {source_sync_commit} != checkout {checked_out_commit}"
        )
    summary = {
        "schema_version": "block3_runtime_model_replay_v1",
        "source_sync_commit": source_sync_commit,
        "checked_out_commit": checked_out_commit,
        "dataset": "mrpc",
        "model": "textattack/bert-base-uncased-MRPC",
        "stage1_gelu": GELU,
        "stage1_softmax": SOFTMAX,
        "seed": SEED,
        "probe_size": int(eval_args.probe_size),
        "repeat": int(eval_args.repeat),
        "k_levels": [int(value) for value in K_LEVELS],
        "local_glue_root": str(Path(args.local_glue_root).resolve()),
        "baseline_archive": str(static_baseline.archive_path),
        "baseline_block3_fusion_count_unique": fusion_counts,
        "baseline_block3": baseline_block3,
        "baseline_cost": to_jsonable(
            dataclasses.asdict(baseline_cost),
            stringify_unknown=True,
            preserve_native=True,
        ),
        "baseline_diagnostics": to_jsonable(
            baseline_diag,
            stringify_unknown=True,
            preserve_native=True,
        ),
        "fusion_map_sources": _map_hashes(source_root),
        "results": results,
        "comparison": comparison,
        "gates": {
            "exact_source_snapshot": source_sync_commit == checked_out_commit,
            "ro_block3_baseline_loaded_once": True,
            "base_env_uses_calibrated_ro_max_sfs": True,
            "block3_baseline_fusion_count_zero": fusion_counts == [0],
            "block3_non_k_action_slots_frozen_to_baseline": True,
            "all_24_block3_replans_valid": all(
                all(row["valid"] for row in result["block3_replans"])
                for result in results.values()
            ),
            "all_24_block3_replans_applied": all(
                all(
                    row["replan_application"].get(
                        "model_uses_replan_config",
                        False,
                    )
                    for row in result["block3_replans"]
                )
                for result in results.values()
            ),
            "all_24_installed_cfgs_present_at_forward": all(
                result["installed_block3_layer_count"] == 12
                for result in results.values()
            ),
            "selected_k_reached_installed_cfg": all(
                result["installed_block3_k_values"]
                == [result["expected_block3_k"]]
                for result in results.values()
            ),
            "installed_sf_chain_matches_ro_baseline": all(
                result["installed_sf_chain_matches_ro_baseline"]
                for result in results.values()
            ),
            "real_model_forward_executed": all(
                result["terminal_forward_ran"]
                for result in results.values()
            ),
            "k_changes_real_model_logits": not comparison["bitwise_equal"],
        },
    }
    if not all(summary["gates"].values()):
        raise AssertionError(f"Block3 runtime gate failed: {summary['gates']}")

    output = artifact_root / "block3_model_replay.json"
    write_json_file(output, summary, ensure_ascii=False, indent=2, sort_keys=True)
    print("[pass] Block3 runtime/model replay gates all passed", flush=True)
    print(f"[result] output={output}", flush=True)
    print(f"[result] comparison={comparison}", flush=True)
    for name, result in results.items():
        print(f"[result] {name} metrics={result['terminal_metrics']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
