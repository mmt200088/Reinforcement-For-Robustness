#!/usr/bin/env python3
"""Audit final installed SFs for the six supported Stage-2 map profiles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from json_utils import read_json_file  # noqa: E402
from scripts.blb_verify_boosted_install import (  # noqa: E402
    _cfg_sf_projection,
    _evaluate,
    _install_and_inspect,
    _load_runtime_deps,
)


PROFILE_SPECS = (
    ("mrpc", "bert-base", "mrpc", 12),
    ("rte", "bert-base", "rte", 12),
    ("sst2", "bert-base", "sst2", 12),
    ("mrpc_large", "bert-large", "mrpc", 24),
    ("rte_large", "bert-large", "rte", 24),
    ("sst2_large", "bert-large", "sst2", 24),
)
METHOD_CHAIN = (
    "fusion map option/action_indices",
    "precision-boost explicit_field_values (option1)",
    "evaluate_action_for_cost",
    "Rescale Optimizer replan",
    "apply_optimizer_output_to_cfg plus block sync bindings",
    "final NoiseConfig consumed by bridge.apply",
)
PROVENANCE = "post_replan_optimizer_writeback_cfg_consumed_by_bridge_apply"


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _plain_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    candidate = getattr(value, "value", None)
    if candidate is not None and isinstance(candidate, (str, bool, int, float)):
        return candidate
    return str(value)


def _cfg_rows(cfg: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attr, value in vars(cfg).items():
        if attr == "output_truncation_mode" or attr.startswith("rotation_after_"):
            continue
        if attr == "output_truncation_k":
            rows.append({
                "point": attr,
                "scaling_factor": None,
                "truncation_k": None if value is None else int(value),
                "active": value is not None,
                "type": "truncation",
                "distribution": None,
                "N": None,
            })
            continue
        values = value if isinstance(value, tuple) else (value,)
        for index, item in enumerate(values):
            if item is not None and not hasattr(item, "scaling_factor"):
                continue
            point = f"{attr}[{index}]" if isinstance(value, tuple) else attr
            sf = getattr(item, "scaling_factor", None) if item is not None else None
            rows.append({
                "point": point,
                "scaling_factor": None if sf is None else int(sf),
                "truncation_k": None,
                "active": sf is not None,
                "type": type(item).__name__ if item is not None else "None",
                "distribution": _plain_scalar(getattr(item, "distribution", None)),
                "N": None if getattr(item, "N", None) is None else int(item.N),
            })
    return rows


def _override_value(value: Any) -> Any:
    if value is None:
        return None
    sf = getattr(value, "scaling_factor", None)
    if sf is not None:
        return int(sf)
    return _plain_scalar(value)


def _serialize_overrides(overrides: Iterable[Any]) -> list[dict[str, Any]]:
    return [{
        "cfg_attr": str(getattr(entry, "cfg_attr", "")),
        "graph_node": _plain_scalar(getattr(entry, "graph_node", None)),
        "source": str(getattr(entry, "source", "")),
        "old_value": _override_value(getattr(entry, "old_value", None)),
        "new_value": _override_value(getattr(entry, "new_value", None)),
    } for entry in overrides]


def _map_paths(profile: str) -> tuple[Path, ...]:
    root = REPO_ROOT / "blb_stage2_rl" / "fusion_maps" / profile
    return (
        root / f"block2_{profile}.json",
        root / "block4.json",
        root / "block5_n1.json",
        root / "block5_n2.json",
        root / "block5_n4.json",
    )


def _map_gate_row(profile: str, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    options = list(payload.get("options") or [])
    fusion_counts = [int(option.get("fusion_count", -1)) for option in options]
    build_meta = dict(payload.get("build_meta") or {})
    k_info = dict(build_meta.get("k_independence") or {})
    row = {
        "profile": profile,
        "graph_key": str(payload.get("graph_key")),
        "fusion_counts": fusion_counts,
        "num_options": len(options),
        "option0_id": int(options[0].get("option_id", -1)) if options else None,
        "option0_fusion": int(options[0].get("fusion_count", -1)) if options else None,
        "max_fusion": max(fusion_counts) if fusion_counts else None,
        "boosted_option_ids": [
            int(option.get("option_id", -1))
            for option in options
            if bool(option.get("boosted"))
        ],
        "precision_boost_applied": bool(build_meta.get("precision_boost_applied")),
        "precision_boost_phase2_applied": bool(
            build_meta.get("precision_boost_phase2_applied")
        ),
        "k_independent": bool(k_info.get("k_independent")),
        "map_path": str(path.relative_to(REPO_ROOT)),
    }
    expected = {
        "fusion_counts": [0, 1],
        "num_options": 2,
        "option0_id": 0,
        "option0_fusion": 0,
        "max_fusion": 1,
        "boosted_option_ids": [1],
        "precision_boost_applied": True,
        "precision_boost_phase2_applied": True,
        "k_independent": True,
    }
    for key, value in expected.items():
        if row[key] != value:
            raise ValueError(f"{profile}/{path.name}: {key}={row[key]!r}, expected {value!r}")
    return row


def _audit_option(ctx: Any, option: Mapping[str, Any]) -> dict[str, Any]:
    action_indices = [int(value) for value in option["action_indices"]]
    explicit = {
        str(key): int(value)
        for key, value in dict(option.get("explicit_field_values") or {}).items()
    }
    boosted = bool(option.get("boosted"))
    boosted_overrides = (
        {(int(ctx.block_idx), int(ctx.ref_layer)): explicit}
        if boosted and explicit
        else None
    )
    evaluation = _evaluate(ctx, action_indices, boosted_overrides=boosted_overrides)
    cfg, overrides, fused_still_installed = _install_and_inspect(evaluation, ctx)
    if cfg is None:
        raise RuntimeError(
            f"{ctx.graph_key} layer {ctx.ref_layer}: target config missing after replan"
        )
    projection = _cfg_sf_projection(cfg)
    plain_sum = None
    if boosted:
        plain_evaluation = _evaluate(ctx, action_indices, boosted_overrides=None)
        plain_cfg, _, _ = _install_and_inspect(plain_evaluation, ctx)
        if plain_cfg is None:
            raise RuntimeError(
                f"{ctx.graph_key} layer {ctx.ref_layer}: plain comparison config missing"
            )
        plain_sum = sum(_cfg_sf_projection(plain_cfg).values())
        if sum(projection.values()) <= plain_sum:
            raise ValueError(
                f"{ctx.graph_key} layer {ctx.ref_layer}: boosted installed SF sum "
                f"{sum(projection.values())} <= plain sum {plain_sum}"
            )
    if fused_still_installed:
        raise ValueError(
            f"{ctx.graph_key} layer {ctx.ref_layer}: fused rescale still installed: "
            f"{fused_still_installed}"
        )
    return {
        "layer": int(ctx.ref_layer),
        "actual_config_rows": _cfg_rows(cfg),
        "actual_active_sf": projection,
        "actual_active_sf_sum": sum(projection.values()),
        "plain_same_action_active_sf_sum": plain_sum,
        "optimizer_overrides": _serialize_overrides(overrides),
        "fused_still_installed": list(fused_still_installed),
        "provenance": PROVENANCE,
    }


def _audit_map(
    *,
    profile: str,
    num_layers: int,
    path: Path,
    payload: Mapping[str, Any],
    rescale_optimizer_root: str,
) -> dict[str, Any]:
    deps = _load_runtime_deps()
    fusion_enum = deps["fusion_enum"]
    graph_key = str(payload["graph_key"])
    block_idx = int(payload["block_idx"])
    gelu_degree = int(payload.get("gelu_degree", 4))
    attn_degree = int(payload.get("attn_degree", 2))
    output_options: list[dict[str, Any]] = []
    for option in payload["options"]:
        layers = []
        for layer in range(num_layers):
            ctx = fusion_enum.prepare_block_type_context(
                graph_key=graph_key,
                block_idx=block_idx,
                gelu_degree=gelu_degree,
                attn_degree=attn_degree,
                profile=profile,
                rescale_optimizer_root=rescale_optimizer_root,
                num_layers=num_layers,
                ref_layer=layer,
            )
            layers.append(_audit_option(ctx, option))
        representative = layers[0]["actual_config_rows"]
        identical = all(layer["actual_config_rows"] == representative for layer in layers[1:])
        if not identical:
            raise ValueError(f"{profile}/{graph_key}: final installed config differs by layer")
        output_options.append({
            "option_id": int(option["option_id"]),
            "fusion_count": int(option["fusion_count"]),
            "boosted": bool(option.get("boosted")),
            "map_slots": dict(option.get("slots") or {}),
            "explicit_field_values": dict(option.get("explicit_field_values") or {}),
            "layers": layers,
            "all_layers_identical": True,
            "representative_actual_config_rows": representative,
        })
    return {
        "graph_key": graph_key,
        "map_path": str(path.relative_to(REPO_ROOT)),
        "block_idx": block_idx,
        "gelu_degree": gelu_degree,
        "attn_degree": attn_degree,
        "build_meta": dict(payload.get("build_meta") or {}),
        "options": output_options,
    }


def _audit_profile(
    *,
    profile: str,
    model_type: str,
    dataset: str,
    num_layers: int,
    rescale_optimizer_root: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    maps = []
    gate_rows = []
    for path in _map_paths(profile):
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json_file(path)
        gate_rows.append(_map_gate_row(profile, path, payload))
        maps.append(_audit_map(
            profile=profile,
            num_layers=num_layers,
            path=path,
            payload=payload,
            rescale_optimizer_root=rescale_optimizer_root,
        ))
    return ({
        "profile": profile,
        "model_type": model_type,
        "dataset": dataset,
        "num_layers": num_layers,
        "stage1_gelu": [4] * num_layers,
        "stage1_softmax": [6] * num_layers,
        "maps": maps,
    }, gate_rows)


def _format_value(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "missing"
    if row.get("truncation_k") is not None:
        return f"K={int(row['truncation_k'])}"
    if row.get("scaling_factor") is not None:
        return f"SF={int(row['scaling_factor'])}"
    return "not installed / fused-away"


def _render_html(payload: Mapping[str, Any]) -> str:
    esc = html.escape
    profiles = list(payload["profiles"])
    parts = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<title>Six-profile fusion-count actual SF audit</title><style>",
        "body{font:14px/1.5 system-ui,-apple-system,sans-serif;margin:24px;color:#172033;background:#f7f8fa}",
        "main{max-width:1500px;margin:auto;background:white;padding:28px}",
        "h1,h2,h3{color:#101828}table{border-collapse:collapse;width:100%;margin:10px 0 24px}",
        "th,td{border:1px solid #d0d5dd;padding:6px 8px;vertical-align:top}",
        "th{background:#eef2f6;position:sticky;top:0}.ok{color:#067647;font-weight:700}",
        ".bad{color:#b42318;font-weight:700}.note{background:#eef6ff;border-left:4px solid #2563eb;padding:10px 12px}",
        ".muted{color:#667085}.tag{display:inline-block;padding:2px 6px;border-radius:4px;background:#ecfdf3;color:#067647;font-weight:600}",
        "code{background:#f2f4f7;padding:2px 4px;word-break:break-all}details{margin:12px 0 22px}",
        "summary{cursor:pointer;font-weight:700;font-size:16px}.active{background:#ecfdf3}",
        ".off{color:#667085;background:#f9fafb}td.num{text-align:right;font-variant-numeric:tabular-nums}",
        "</style></head><body><main>",
        "<h1>Six-profile fusion-count maps and actual post-replan SF</h1>",
        f"<p>Status: <span class=ok>{esc(str(payload['status']).upper())}</span> &nbsp; Source: <code>{esc(str(payload['source_commit']))}</code></p>",
        "<div class=\"note\"><b>Authoritative value:</b> every SF below comes from the final NoiseConfig after precision boost, replan, optimizer write-back, and block-specific sync. Map <code>slots</code> and pre-replan proposals are not used as the reported actual SF. A value marked not installed/fused-away does not inject noise into the model.</div>",
        "<p>Stage-2 base: GELU=4 in every layer, Softmax=6 in every layer, truncation K=13. Therefore the active Block5 graph in this mode is <code>block5_n4</code>; n1/n2 maps are retained and audited for future selectable Stage-1 configurations.</p>",
        "<h2>Coverage and gates</h2><table><thead><tr><th>Profile</th><th>Model</th><th>Dataset</th><th>Layers</th><th>Maps</th><th>Fusion domain</th><th>Current-source replan/install</th></tr></thead><tbody>",
    ]
    for profile in profiles:
        parts.append(
            "<tr>"
            f"<td>{esc(profile['profile'])}</td><td>{esc(profile['model_type'])}</td>"
            f"<td>{esc(profile['dataset'])}</td><td>{profile['num_layers']}</td>"
            f"<td>{len(profile['maps'])}</td><td>[0, 1] for all maps</td>"
            "<td class=ok>PASS</td></tr>"
        )
    parts.append("</tbody></table>")
    for profile in profiles:
        last_layer = int(profile["num_layers"]) - 1
        parts.append(
            f"<h2>{esc(profile['model_type'])} / {esc(profile['dataset'])} "
            f"<span class=tag>{esc(profile['profile'])}</span></h2>"
            "<table><thead><tr><th>Graph</th><th>Block</th><th>Degree</th><th>Options</th><th>Max fusion</th><th>Boosted option</th><th>All layers checked</th><th>Layer configs identical</th></tr></thead><tbody>"
        )
        for map_result in profile["maps"]:
            degree = (
                f"GELU n{map_result['gelu_degree']}"
                if int(map_result["block_idx"]) == 5
                else f"attn n{map_result['attn_degree']}"
            )
            parts.append(
                "<tr>"
                f"<td><code>{esc(map_result['graph_key'])}</code></td>"
                f"<td>{map_result['block_idx']}</td><td>{esc(degree)}</td>"
                "<td>0, 1</td><td>1</td><td>option 1</td>"
                f"<td>0..{last_layer}</td><td class=ok>YES</td></tr>"
            )
        parts.append("</tbody></table>")
        for map_result in profile["maps"]:
            options = {int(option["option_id"]): option for option in map_result["options"]}
            option0 = options[0]
            option1 = options[1]
            row0 = {row["point"]: row for row in option0["representative_actual_config_rows"]}
            row1 = {row["point"]: row for row in option1["representative_actual_config_rows"]}
            points = sorted(set(row0) | set(row1))
            parts.append(
                f"<details open><summary>{esc(map_result['graph_key'])}: actual option0 vs option1 final config</summary>"
                f"<p class=muted>Map: <code>{esc(map_result['map_path'])}</code>. Audited on every layer 0..{last_layer}. option1 fusion_count=1, precision boost=yes.</p>"
                "<table><thead><tr><th>Final cfg slot</th><th>Option 0 (fusion 0)</th><th>Option 1 (fusion 1)</th><th>SF delta</th><th>Distribution / N</th></tr></thead><tbody>"
            )
            for point in points:
                left = row0.get(point)
                right = row1.get(point)
                left_sf = left.get("scaling_factor") if left else None
                right_sf = right.get("scaling_factor") if right else None
                delta = int(right_sf) - int(left_sf) if left_sf is not None and right_sf is not None else None
                ref = right or left or {}
                dist = ref.get("distribution") or "-"
                n_value = ref.get("N")
                row_class = "active" if right and right.get("active") else "off"
                parts.append(
                    f"<tr class={row_class}><td><code>{esc(point)}</code></td>"
                    f"<td>{esc(_format_value(left))}</td><td>{esc(_format_value(right))}</td>"
                    f"<td class=num>{'-' if delta is None else delta}</td>"
                    f"<td>{esc(str(dist))} / {'-' if n_value is None else int(n_value)}</td></tr>"
                )
            parts.append("</tbody></table></details>")
    map_count = sum(len(profile["maps"]) for profile in profiles)
    parts.extend([
        "<h2>Interpretation</h2><ul>",
        f"<li>All {map_count} maps have exactly two choices: fusion_count 0 and 1. No profile contains fusion_count greater than 1.</li>",
        f"<li>All {map_count} boosted option1 configurations increased installed precision versus the same in-grid action and left no fused-away rescale installed.</li>",
        "<li>The tables show final runtime NoiseConfig values, including inactive points. K=13 is separate from map enumeration and remains the runtime truncation choice.</li>",
        "</ul></main></body></html>",
    ])
    return "".join(parts)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    source_commit = _git_head()
    profiles = []
    gate_rows = []
    for profile, model_type, dataset, num_layers in PROFILE_SPECS:
        profile_result, profile_gate_rows = _audit_profile(
            profile=profile,
            model_type=model_type,
            dataset=dataset,
            num_layers=num_layers,
            rescale_optimizer_root=args.rescale_optimizer_root,
        )
        profiles.append(profile_result)
        gate_rows.extend(profile_gate_rows)

    payload = {
        "schema": "six-profile-fusion-count-actual-sf-v1",
        "source_commit": source_commit,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_chain": list(METHOD_CHAIN),
        "stage2_base": "GELU=4 all layers; Softmax=6 all layers; K index 3 / K=13",
        "status": "pass",
        "elapsed_seconds": time.perf_counter() - started,
        "errors": [],
        "profiles": profiles,
    }
    map_gate = {
        "source_commit": source_commit,
        "status": "pass",
        "profiles": {
            profile: [row["graph_key"] for row in gate_rows if row["profile"] == profile]
            for profile, _, _, _ in PROFILE_SPECS
        },
        "rows": gate_rows,
        "errors": [],
    }
    counts = {
        "profiles": len(profiles),
        "maps": sum(len(profile["maps"]) for profile in profiles),
        "options": sum(
            len(map_result["options"])
            for profile in profiles
            for map_result in profile["maps"]
        ),
        "layer_option_replays": sum(
            len(option["layers"])
            for profile in profiles
            for map_result in profile["maps"]
            for option in map_result["options"]
        ),
        "actual_config_rows": sum(
            len(layer["actual_config_rows"])
            for profile in profiles
            for map_result in profile["maps"]
            for option in map_result["options"]
            for layer in option["layers"]
        ),
    }
    checks = {
        "audit_status_pass": payload["status"] == "pass",
        "map_gate_status_pass": map_gate["status"] == "pass",
        "profile_count_6": counts["profiles"] == 6,
        "map_count_30": counts["maps"] == 30,
        "all_fusion_domains_0_1": all(row["fusion_counts"] == [0, 1] for row in gate_rows),
        "all_layers_identical": all(
            option["all_layers_identical"]
            for profile in profiles
            for map_result in profile["maps"]
            for option in map_result["options"]
        ),
        "all_layer_counts_complete": all(
            len(option["layers"]) == int(profile["num_layers"])
            for profile in profiles
            for map_result in profile["maps"]
            for option in map_result["options"]
        ),
        "no_fused_rescale_installed": all(
            not layer["fused_still_installed"]
            for profile in profiles
            for map_result in profile["maps"]
            for option in map_result["options"]
            for layer in option["layers"]
        ),
        "all_provenance_final_cfg": all(
            layer["provenance"] == PROVENANCE
            for profile in profiles
            for map_result in profile["maps"]
            for option in map_result["options"]
            for layer in option["layers"]
        ),
    }
    if counts != {
        "profiles": 6,
        "maps": 30,
        "options": 60,
        "layer_option_replays": 1080,
        "actual_config_rows": 16848,
    }:
        raise ValueError(f"unexpected audit counts: {counts}")
    if not all(checks.values()):
        raise ValueError(f"audit checks failed: {checks}")

    _write_json(output_dir / "actual_sf_audit.json", payload)
    _write_json(output_dir / "map_gate.json", map_gate)
    _write_json(output_dir / "final_verification.json", {
        "status": "pass",
        "checks": checks,
        "counts": counts,
        "source_commit": source_commit,
    })
    (output_dir / "six_profile_fusion_count_actual_sf.html").write_text(
        _render_html(payload), encoding="utf-8"
    )
    (output_dir / "SOURCE_SYNC_COMMIT").write_text(source_commit + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "counts": counts, "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
