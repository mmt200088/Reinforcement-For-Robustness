#!/usr/bin/env python3
"""Run fixed fusion-count BLB action evaluations and build a comparison HTML.

This is an experiment driver. It does not launch RL. Each unique action vector
is evaluated through the standalone Paean BLB action final-eval path, then all
requested groups, including duplicate/no-op aliases, are folded into one report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "experiments" / "server_command_runs" / "fusion_count_map_action_eval_20260610"
DEFAULT_ACTION_DIR = DEFAULT_RUN_DIR / "action_configs"
DEFAULT_MAP_REPORT = DEFAULT_RUN_DIR / "fusion_count_map_report.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_DIR / "paean_outputs"
DEFAULT_HTML = REPO_ROOT / "reports" / "html_reports" / "20260610_mrpc_fusion_count_action_eval.html"

DEFAULT_STAGE1_GELU = [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
DEFAULT_STAGE1_SOFTMAX = [6] * 12
DEFAULT_MANUAL_NOISE = {
    "x": [30] * 12,
    "wq": [22] * 12,
    "wk": [22] * 12,
    "wv": [22] * 12,
    "wo": [22] * 12,
    "wffn1": [22] * 12,
    "wffn2": [22] * 12,
}


def _resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _json_int_list(raw: str | None, *, default: Sequence[int], name: str) -> List[int]:
    text = str(raw or "").strip()
    if not text:
        return [int(v) for v in default]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be a JSON list: {exc}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"{name} must be a JSON list")
    return [int(v) for v in payload]


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_action_configs(action_dir: Path) -> List[dict]:
    configs = []
    for path in sorted(action_dir.glob("*.json")):
        if path.name.startswith("._") or path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        action = payload.get("action_vec")
        slots = payload.get("slots")
        base = payload.get("base")
        legacy = payload.get("legacy_action_vec")
        has_executable_slots = isinstance(slots, (list, dict))
        has_executable_vec = isinstance(action, list)
        if not (has_executable_slots or has_executable_vec):
            continue
        group = payload.get("group") or {}
        name = str(group.get("name") or path.stem)
        hash_payload = (
            {
                "slots": slots,
                "base": base,
                "legacy_action_vec": legacy,
            }
            if has_executable_slots
            else [int(v) for v in action]
        )
        configs.append({
            "name": name,
            "path": path,
            "payload": payload,
            "action_hash": _json_hash(hash_payload),
            "group": group,
        })
    if not configs:
        raise RuntimeError(f"no action config JSON files found under {action_dir}")
    return configs


def _unique_configs(configs: Sequence[Mapping[str, Any]]) -> List[dict]:
    first_by_hash: Dict[str, dict] = {}
    for cfg in configs:
        first_by_hash.setdefault(str(cfg["action_hash"]), dict(cfg))
    return list(first_by_hash.values())


def _output_dir(output_root: Path, run_name: str) -> Path:
    return output_root / "mrpc" / "rl" / run_name


def _result_json_path(output_root: Path, run_name: str) -> Path:
    return _output_dir(output_root, run_name) / "final_eval" / "blb_action_final_eval_results_mrpc.json"


def _run_one(
    *,
    cfg: Mapping[str, Any],
    output_root: Path,
    repeat: int,
    batch_size: int,
    stage1_gelu: Sequence[int],
    stage1_softmax: Sequence[int],
    rescale_optimizer_root: str,
    force: bool,
    env: Mapping[str, str],
) -> Path:
    run_name = str(cfg["name"])
    result_path = _result_json_path(output_root, run_name)
    if result_path.is_file() and not force:
        print(f"[skip] {run_name}: existing result {result_path}")
        return result_path

    output_dir = _output_dir(output_root, run_name)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fusion_count_action_eval.log"
    cmd = [
        sys.executable,
        "-m",
        "Paean.run_final_eval",
        "--dataset",
        "mrpc",
        "--algorithm",
        "rl",
        "--model-type",
        "bert-base",
        "--batch-size",
        str(int(batch_size)),
        "--source",
        "manual",
        "--manual-stage1-gelu",
        json.dumps([int(v) for v in stage1_gelu]),
        "--manual-stage1-softmax",
        json.dumps([int(v) for v in stage1_softmax]),
        "--manual-stage2-noise",
        json.dumps(DEFAULT_MANUAL_NOISE, separators=(",", ":")),
        "--action-config",
        str(cfg["path"]),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        "--repeat",
        str(int(repeat)),
        "--rescale-invoker-kind",
        "in_process",
        "--rescale-optimizer-root",
        str(rescale_optimizer_root),
        "--require-rescale-optimizer",
        "--no-glue-submission",
        "--stage1-accuracy-tolerance",
        "0.005",
        "--stage2-limit-tolerance",
        "0.005",
        "--stage2-stability-tolerance",
        "0.005",
        "--stage2-k-trials",
        "5",
        "--stage2-probe-size",
        "408",
        "--foreground",
    ]
    print(f"[run] {run_name}: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command:\n" + " ".join(cmd) + "\n\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=dict(env)).returncode
    if rc != 0:
        raise RuntimeError(f"{run_name} failed with rc={rc}; see {log_path}")
    if not result_path.is_file():
        raise RuntimeError(f"{run_name} finished but result JSON is missing: {result_path}")
    return result_path


def _load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidate_results") or []
    if not candidates:
        raise RuntimeError(f"missing candidate_results in {path}")
    return payload


def _metric(result: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(result.get(key, default))
    except Exception:
        return float(default)


def _html_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    parts = ["<table><thead><tr>"]
    for h in headers:
        parts.append(f"<th>{html.escape(str(h))}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            if isinstance(cell, str) and cell.startswith("<"):
                parts.append(f"<td>{cell}</td>")
            else:
                parts.append(f"<td>{html.escape(str(cell))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _build_combined(
    *,
    configs: Sequence[Mapping[str, Any]],
    output_root: Path,
    map_report: Mapping[str, Any],
    stage1_gelu: Sequence[int],
    stage1_softmax: Sequence[int],
) -> dict:
    result_by_hash: Dict[str, dict] = {}
    canonical_by_hash: Dict[str, str] = {}
    for cfg in _unique_configs(configs):
        h = str(cfg["action_hash"])
        result = _load_result(_result_json_path(output_root, str(cfg["name"])))
        result_by_hash[h] = result
        canonical_by_hash[h] = str(cfg["name"])

    first_result = next(iter(result_by_hash.values()))
    baseline = first_result["baseline"]
    group_results = []
    for cfg in configs:
        h = str(cfg["action_hash"])
        source = result_by_hash[h]
        candidate = copy.deepcopy(source["candidate_results"][0])
        candidate["name"] = str(cfg["name"])
        candidate["action_hash"] = h
        candidate["canonical_run"] = canonical_by_hash[h]
        candidate["reused_from_canonical"] = canonical_by_hash[h] != str(cfg["name"])
        candidate["fusion_group"] = cfg["group"]
        candidate["action_config_path"] = str(cfg["path"])
        candidate["loss_delta_vs_baseline"] = _metric(candidate, "loss") - _metric(baseline, "loss")
        candidate["p_delta_vs_baseline"] = _metric(candidate, "p") - _metric(baseline, "p")
        candidate["s_delta_vs_baseline"] = _metric(candidate, "s") - _metric(baseline, "s")
        group_results.append(candidate)

    return {
        "schema_version": "fusion_count_action_eval_combined_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "map_report_context": {
            "profile": map_report.get("profile"),
            "stage1_gelu": map_report.get("stage1_gelu"),
            "stage1_softmax": map_report.get("stage1_softmax"),
            "baseline_k_value": map_report.get("baseline_k_value"),
            "schedule_occurrences": map_report.get("schedule_occurrences"),
        },
        "evaluation_protocol": {
            "split": "validation_full",
            "baseline": "original plaintext, original GELU/Softmax, no BLB noise",
            "stage2_groups": "manual Stage-1 GELU/Softmax, fixed baseline K, only fusion-count options varied",
            "manual_stage1_gelu": [int(v) for v in stage1_gelu],
            "manual_stage1_softmax": [int(v) for v in stage1_softmax],
            "unique_action_runs": len(result_by_hash),
            "requested_group_count": len(configs),
        },
        "baseline": baseline,
        "group_results": group_results,
    }


def _render_html(combined: Mapping[str, Any]) -> str:
    baseline = combined["baseline"]
    rows = [[
        "baseline_plaintext",
        "baseline",
        "",
        f"{_metric(baseline, 'loss'):.6f}",
        "0.000000",
        f"{_metric(baseline, 'p'):.6f}",
        "0.000000",
        f"{_metric(baseline, 's'):.6f}",
        "0.000000",
        "",
        "",
        "",
        "",
        "",
    ]]
    for result in combined["group_results"]:
        group = result.get("fusion_group", {}) or {}
        reused = "yes" if result.get("reused_from_canonical") else ""
        no_op = "yes" if group.get("no_op") else ""
        valid = result.get("rescale_optimizer", {}).get("invalid_count", "")
        verify = result.get("install_verification", {}).get("model_will_use_selected_cfg", "")
        rows.append([
            result["name"],
            group.get("family", ""),
            no_op,
            f"{_metric(result, 'loss'):.6f}",
            f"{_metric(result, 'loss_std'):.6f}",
            f"{_metric(result, 'p'):.6f}",
            f"{_metric(result, 'p_std'):.6f}",
            f"{_metric(result, 's'):.6f}",
            f"{_metric(result, 's_std'):.6f}",
            f"{_metric(result, 'loss_delta_vs_baseline'):+.6f}",
            f"{_metric(result, 'p_delta_vs_baseline'):+.6f}",
            f"{_metric(result, 's_delta_vs_baseline'):+.6f}",
            int(result.get("total_bits_sum", 0)),
            int(result.get("total_fusion_count", 0)),
            f"invalid={valid}; verified={verify}; reused={reused}; canonical={html.escape(str(result.get('canonical_run', '')))}",
        ])

    detail_rows = []
    for result in combined["group_results"]:
        group = result.get("fusion_group", {}) or {}
        detail_rows.append([
            result["name"],
            result.get("action_hash", "")[:12],
            json.dumps(group.get("fusion_count_by_graph", {}), ensure_ascii=False),
            json.dumps(group.get("option_by_graph", {}), ensure_ascii=False),
            result.get("action_config_path", ""),
        ])

    ctx = combined["map_report_context"]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>MRPC Fusion Count Action Evaluation</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:32px;color:#1f2933;background:#fbfcfd}",
        "h1,h2{color:#111827}.meta{color:#52606d}table{border-collapse:collapse;width:100%;margin:14px 0;background:white}",
        "th,td{border:1px solid #d9e2ec;padding:7px 9px;text-align:left;vertical-align:top;font-size:13px}",
        "th{background:#eef2f7}.note{background:#eef6ff;border-left:4px solid #2b6cb0;padding:10px 12px;margin:12px 0}",
        "code{background:#eef2f7;padding:1px 4px;border-radius:4px}",
        "</style></head><body>",
        "<h1>MRPC Fusion Count Fixed Action Evaluation</h1>",
        f"<p class='meta'>Generated: {html.escape(str(combined['generated_at_utc']))}</p>",
        "<div class='note'>baseline 是原明文模型：原始 GELU/Softmax、无 BLB noise、无函数替换。"
        "其它组固定最新 MRPC Stage-1 配置，只改变 fusion-count map option；K 全部固定在 baseline K=13。</div>",
        "<h2>Context</h2>",
        _html_table(
            ["profile", "Stage-1 GELU", "Stage-1 Softmax", "baseline K", "unique action runs", "requested groups"],
            [[
                ctx.get("profile"),
                json.dumps(ctx.get("stage1_gelu")),
                json.dumps(ctx.get("stage1_softmax")),
                ctx.get("baseline_k_value"),
                combined["evaluation_protocol"]["unique_action_runs"],
                combined["evaluation_protocol"]["requested_group_count"],
            ]],
        ),
        "<h2>Metrics and Stability</h2>",
        _html_table(
            [
                "group", "family", "no-op", "loss mean", "loss std", "Accuracy mean",
                "Accuracy std", "F1 mean", "F1 std", "loss Δ", "Accuracy Δ", "F1 Δ",
                "bits", "fusion", "diagnostics",
            ],
            rows,
        ),
        "<h2>Action Mapping</h2>",
        _html_table(["group", "action hash", "fusion count by graph", "option by graph", "action config"], detail_rows),
        "</body></html>",
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-dir", default=str(DEFAULT_ACTION_DIR))
    parser.add_argument("--map-report", default=str(DEFAULT_MAP_REPORT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-json", default=str(DEFAULT_RUN_DIR / "fusion_count_action_eval_results.json"))
    parser.add_argument("--output-html", default=str(DEFAULT_HTML))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage1-gelu", default=json.dumps(DEFAULT_STAGE1_GELU))
    parser.add_argument("--stage1-softmax", default=json.dumps(DEFAULT_STAGE1_SOFTMAX))
    parser.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Only collect existing Paean result JSON files.")
    args = parser.parse_args()

    action_dir = _resolve(args.action_dir)
    map_report_path = _resolve(args.map_report)
    output_root = _resolve(args.output_root)
    output_json = _resolve(args.output_json)
    output_html = _resolve(args.output_html)
    configs = _load_action_configs(action_dir)
    unique = _unique_configs(configs)
    print(f"[info] requested groups={len(configs)} unique action vectors={len(unique)}")
    stage1_gelu = _json_int_list(args.stage1_gelu, default=DEFAULT_STAGE1_GELU, name="--stage1-gelu")
    stage1_softmax = _json_int_list(args.stage1_softmax, default=DEFAULT_STAGE1_SOFTMAX, name="--stage1-softmax")
    if len(stage1_gelu) != len(stage1_softmax):
        raise SystemExit("--stage1-gelu and --stage1-softmax must have equal length")

    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not args.skip_run:
        for cfg in unique:
            _run_one(
                cfg=cfg,
                output_root=output_root,
                repeat=int(args.repeat),
                batch_size=int(args.batch_size),
                stage1_gelu=stage1_gelu,
                stage1_softmax=stage1_softmax,
                rescale_optimizer_root=str(args.rescale_optimizer_root),
                force=bool(args.force),
                env=env,
            )

    map_report = json.loads(map_report_path.read_text(encoding="utf-8"))
    combined = _build_combined(
        configs=configs,
        output_root=output_root,
        map_report=map_report,
        stage1_gelu=stage1_gelu,
        stage1_softmax=stage1_softmax,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_html.write_text(_render_html(combined), encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_json),
        "output_html": str(output_html),
        "unique_action_runs": combined["evaluation_protocol"]["unique_action_runs"],
        "requested_group_count": combined["evaluation_protocol"]["requested_group_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
