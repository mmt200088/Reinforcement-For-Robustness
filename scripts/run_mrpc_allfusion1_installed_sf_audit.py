#!/usr/bin/env python3
"""Run one canonical MRPC all-fusion1 install audit and render its report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from json_utils import read_json_file, to_jsonable  # noqa: E402
from scripts.fusion_count_action_eval_common import (  # noqa: E402
    parse_json_int_list,
    resolve_repo_path,
)
from scripts.fusion_count_installed_sf_audit import (  # noqa: E402
    SCHEMA_VERSION,
    InstalledConfigCapture,
    aggregate_prediction_rows,
    annotate_replan_removals,
    build_validation_row_lookup,
    render_audit_html,
    validate_allfusion1_result,
)


DEFAULT_GELU = [4] * 12
DEFAULT_SOFTMAX = [6] * 12
EXPECTED_GROUPS = (
    "all_fusion0",
    "block2_block5_all_layers_fusionmax",
    "block2_block4_block5_all_layers_fusion1",
)


def _load_rlpath_module():
    import scripts.run_fusion_count_action_eval_rlpath as rlpath

    return rlpath


def _load_action_config(path: Path) -> dict[str, Any]:
    payload = read_json_file(path)
    group = payload.get("group") or {}
    name = str(group.get("name") or path.stem)
    return {
        "name": name,
        "path": path,
        "group": group,
        "baseline_k_index": int(payload.get("baseline_k_index", 3)),
    }


def execute_live_fixed_action_audit(
    args,
    action_config: Mapping[str, Any],
    *,
    stage1_gelu: Sequence[int],
    stage1_softmax: Sequence[int],
    rlpath_module=None,
) -> dict[str, Any]:
    """Execute the existing canonical evaluator with one instance-level wrapper."""
    rlpath = rlpath_module or _load_rlpath_module()
    evaluator = rlpath._build_evaluator(
        args,
        stage1_gelu=stage1_gelu,
        stage1_softmax=stage1_softmax,
    )
    seq_env, baseline = rlpath._build_seq_env(
        args,
        evaluator,
        stage1_gelu=stage1_gelu,
        stage1_softmax=stage1_softmax,
    )
    bridge = seq_env.base.bridge
    original_apply = bridge.apply
    capture = InstalledConfigCapture(
        original_apply=original_apply,
        handler=evaluator.reversible_handler,
        expected_layers=range(12),
    )
    try:
        bridge.apply = capture.apply
        result = rlpath._run_group_canonical(
            seq_env,
            action_config,
            seed=int(args.seed),
        )
    finally:
        bridge.apply = original_apply

    capture_payload = capture.assert_complete()
    capture_payload["installed_config_rows"] = annotate_replan_removals(
        capture_payload["installed_config_rows"],
        result,
    )
    gate = validate_allfusion1_result(result)
    return {
        "result": result,
        "capture": capture_payload,
        "gate": gate,
        "baseline": baseline,
    }


def _prediction_paths(historical_root: Path) -> list[Path]:
    paths = sorted(historical_root.glob("runs/seed_*/predictions.jsonl"))
    if len(paths) != 5:
        raise ValueError(
            f"expected five historical prediction files under {historical_root}, "
            f"found {len(paths)}"
        )
    return paths


def _iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_jsonable(payload, stringify_unknown=True, preserve_native=True),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-config", required=True)
    parser.add_argument("--historical-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument(
        "--run-output-dir",
        default="experiments/server_command_runs/mrpc_allfusion1_sf_audit_tmp",
    )
    parser.add_argument(
        "--stage1-config-json",
        default="experiments/server_command_runs/mrpc_stage2_fixed_stage1_rlbest_20260627.json",
    )
    parser.add_argument("--dataset", default="mrpc")
    parser.add_argument("--model-type", default="bert-base")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--stage1-gelu", default=json.dumps(DEFAULT_GELU))
    parser.add_argument("--stage1-softmax", default=json.dumps(DEFAULT_SOFTMAX))
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--probe-size", type=int, default=408)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage2-limit-tolerance", type=float, default=0.001)
    parser.add_argument("--stage2-stability-tolerance", type=float, default=3.5)
    parser.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.dataset != "mrpc" or args.model_type != "bert-base":
        raise ValueError("this audit is fixed to BERT-base MRPC")
    if int(args.repeat) != 1 or int(args.probe_size) != 408:
        raise ValueError("live install audit requires repeat=1 and probe_size=408")

    action_path = resolve_repo_path(args.action_config)
    historical_root = resolve_repo_path(args.historical_root)
    output_json = resolve_repo_path(args.output_json)
    output_html = resolve_repo_path(args.output_html)
    args.base_model = args.base_model or "textattack/bert-base-uncased-MRPC"
    args.prediction_jsonl = ""
    stage1_gelu = parse_json_int_list(
        args.stage1_gelu,
        default=DEFAULT_GELU,
        name="--stage1-gelu",
    )
    stage1_softmax = parse_json_int_list(
        args.stage1_softmax,
        default=DEFAULT_SOFTMAX,
        name="--stage1-softmax",
    )
    if stage1_gelu != DEFAULT_GELU or stage1_softmax != DEFAULT_SOFTMAX:
        raise ValueError("this audit requires GELU=[4]*12 and Softmax=[6]*12")

    action_config = _load_action_config(action_path)
    if action_config["name"] != "block2_block4_block5_all_layers_fusion1":
        raise ValueError(f"unexpected action config {action_config['name']!r}")

    rlpath = _load_rlpath_module()
    live = execute_live_fixed_action_audit(
        args,
        action_config,
        stage1_gelu=stage1_gelu,
        stage1_softmax=stage1_softmax,
        rlpath_module=rlpath,
    )

    deps = rlpath._load_runtime_deps()
    source_data = deps["load_glue_dataset_equivalent"](
        "mrpc",
        route_log_dir=str(output_json.parent / "dataset_route_logs"),
    )
    row_lookup, labels = build_validation_row_lookup(source_data["validation"])
    if len(row_lookup) != 408 or sorted(row_lookup.values()) != list(range(408)):
        raise ValueError("MRPC validation source is not the expected 408 rows")

    prediction_paths = _prediction_paths(historical_root)
    validation_rows = aggregate_prediction_rows(
        _iter_jsonl(prediction_paths),
        row_lookup=row_lookup,
        labels=labels,
        expected_groups=EXPECTED_GROUPS,
        expected_trials=25,
    )
    if [row["validation_row_id"] for row in validation_rows] != list(range(408)):
        raise ValueError("aggregated validation rows are not exactly 0..407")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sync_commit": _git_head(),
        "protocol": {
            "model": "textattack/bert-base-uncased-MRPC",
            "dataset": "MRPC validation_full",
            "validation_row_range": [0, 407],
            "gelu": stage1_gelu,
            "softmax": stage1_softmax,
            "k": 13,
            "fusion": {"block2": 1, "block4": 1, "block5": 1},
            "live_audit_repeat": 1,
            "historical_trials_per_group": 25,
        },
        "gate": live["gate"],
        "capture": live["capture"],
        "live_result": live["result"],
        "baseline": live["baseline"],
        "historical_prediction_files": [str(path) for path in prediction_paths],
        "validation_rows": validation_rows,
    }
    _write_json(output_json, payload)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(render_audit_html(payload), encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_sync_commit": payload["source_sync_commit"],
        "output_json": str(output_json),
        "output_json_sha256": _sha256(output_json),
        "output_html": str(output_html),
        "output_html_sha256": _sha256(output_html),
        "validation_row_count": len(validation_rows),
        "installed_config_row_count": len(live["capture"]["installed_config_rows"]),
        "gate_passed": bool(live["gate"]["passed"]),
    }
    manifest_path = output_json.parent / "MANIFEST.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
