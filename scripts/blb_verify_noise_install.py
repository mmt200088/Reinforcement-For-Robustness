"""BLB noise-install verifier.

For a given RL action vector + Stage-1 config:
  1. Run a real ``rescale_optimizer.ReplanSession.replan`` for every (block, layer)
     config_name implied by the action.
  2. Apply the optimizer return back onto the action-decoded cfg via
     ``apply_optimizer_output_to_cfg`` (the same path env.step uses).
  3. Emit an HTML report listing, per (layer, block, graph_node), every noise
     injection that would be installed in the model: distribution, SF, N,
     variance σ², and whether the SF came from the action, the optimizer's
     replan, a fused-away point, or an effective rotation.

Two modes:
  ``--mode full`` (default): exercises the full bridge + function_handler chain
      and reports per-noise-point install plan with σ² from
      ``NOISE_VARIANCE_TABLE_BY_N``. **Requires torch + transformers** because
      function_handler imports torch.
  ``--mode smoke``: torch-free probe. Builds a ReplanSession against the
      profile and runs the all-baseline replan for each graph_key. Reports
      the optimizer-side return shape (valid / fusion_count / total_bits /
      cut_point_sf / effective_rotations) but not the model-side install
      plan. Useful when torch isn't available locally — proves the wiring is
      alive before running full mode on the server.

Outputs to ``reports/blb_opt/noise_install_verify/run_<timestamp>.html``
(or a custom ``--out`` path).
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from json_utils import read_json_file  # noqa: E402
from json_utils import to_jsonable  # noqa: E402
from report_format_utils import html_table  # noqa: E402

_NOISE_VARIANCE_TABLE_CACHE: Dict[int, Dict[int, Dict[str, float]]] | None = None


# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------
def _utcnow_slug() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _html_escape(s: Any) -> str:
    return html.escape(str(s), quote=True)


class _HtmlPartsWriter:
    def __init__(self, path: Path):
        self._handle = path.open("w", encoding="utf-8")

    def append(self, text: str) -> None:
        self._handle.write(str(text))
        self._handle.write("\n")

    def close(self) -> None:
        self._handle.close()


# ---------------------------------------------------------------------------
# Noise variance table — extracted via AST from function_handler.py so we
# don't need torch at script load time.
# ---------------------------------------------------------------------------
def load_noise_variance_table() -> Dict[int, Dict[int, Dict[str, float]]]:
    global _NOISE_VARIANCE_TABLE_CACHE
    if _NOISE_VARIANCE_TABLE_CACHE is not None:
        return _NOISE_VARIANCE_TABLE_CACHE
    src = (REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    raw: Optional[Dict[int, Dict[int, Tuple[float, float, float]]]] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_NOISE_STD_RAW":
                    raw = ast.literal_eval(node.value)
                    break
            if raw is not None:
                break
    if raw is None:
        raise RuntimeError("could not extract _NOISE_STD_RAW from function_handler.py")
    table: Dict[int, Dict[int, Dict[str, float]]] = {}
    for N, by_sf in raw.items():
        table[int(N)] = {}
        for sf, stds in by_sf.items():
            table[int(N)][int(sf)] = {
                "encoding": float(stds[0]) ** 2,
                "fresh": float(stds[1]) ** 2,
                "rescale": float(stds[2]) ** 2,
                "rotation": float(stds[2]) ** 2,
            }
    _NOISE_VARIANCE_TABLE_CACHE = table
    return table


def lookup_variance(
    table: Mapping[int, Mapping[int, Mapping[str, float]]],
    *,
    N: int,
    sf: int,
    distribution: str,
) -> Optional[float]:
    by_sf = table.get(int(N))
    if not by_sf:
        return None
    by_dist = by_sf.get(int(sf))
    if not by_dist:
        return None
    return by_dist.get(distribution.lower())


# ---------------------------------------------------------------------------
# Stage-1 / action resolution
# ---------------------------------------------------------------------------
def parse_stage1_config(text: str) -> Dict[str, List[int]]:
    """Parse a JSON Stage-1 config or a "gelu=4,4,...;softmax=2,3,..." string."""
    text = text.strip()
    if not text:
        raise ValueError("stage1 config is empty")
    if text.startswith("{"):
        doc = json.loads(text)
        return {
            k: [int(x) for x in v]
            for k, v in doc.items()
        }
    out: Dict[str, List[int]] = {}
    for part in text.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = [int(x) for x in v.split(",")]
    return out


# ---------------------------------------------------------------------------
# Smoke mode — torch-free
# ---------------------------------------------------------------------------
def run_smoke(args: argparse.Namespace) -> Path:
    """Optimizer-only verification: walk every graph in the static skeletons
    archive and run a baseline replan, capturing the return shape."""
    rescale_root = Path(args.rescale_optimizer_root).resolve()
    if str(rescale_root) not in sys.path:
        sys.path.insert(0, str(rescale_root))

    from rescale_optimizer import (
        ReplanSession,
        iter_stage2_graph_targets,
        load_static_skeleton_baselines,
    )

    profile = args.profile
    archive_path = rescale_root / "configs" / profile / f"static_skeletons_{profile}.json"
    if not archive_path.is_file():
        raise FileNotFoundError(f"static skeletons archive missing: {archive_path}")

    baselines = load_static_skeleton_baselines(archive_path)
    session = ReplanSession.from_profile(profile=profile, root=rescale_root)

    stage1 = parse_stage1_config(args.stage1)
    num_layers = int(args.num_layers)

    targets = iter_stage2_graph_targets(
        dataset=profile,
        num_layers=num_layers,
        stage1_config=stage1,
    )

    results: List[Dict[str, Any]] = []
    for tgt in targets:
        graph_key = tgt["graph_key"]
        if graph_key not in session.baselines:
            results.append({
                **tgt,
                "status": "missing_baseline",
                "valid": False,
            })
            continue
        out = session.replan(graph_key)
        compact = out.get("new_compact_config") or {}
        results.append({
            **tgt,
            "status": "ok",
            "valid": bool(out.get("valid")),
            "fusion_count": int(out.get("fusion_count", 0)),
            "total_bits": int((out.get("result") or {}).get("chain", {}).get("total_bits", 0)),
            "cut_point_sf": compact.get("cut_point_sf", []),
            "propagation_deltas": compact.get("propagation_deltas", []),
            "effective_rotations": compact.get("effective_rotations", []),
            "skeleton": (out.get("result") or {}).get("skeleton", []),
            "t_final": (out.get("result") or {}).get("t_final", []),
        })

    return write_smoke_html(args, baselines, results)


def write_smoke_html(
    args: argparse.Namespace,
    baselines: Mapping[str, Any],
    results: List[Dict[str, Any]],
) -> Path:
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "reports" / "blb_opt" / "noise_install_verify"
        / f"smoke_{args.profile}_{_utcnow_slug()}.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid_count = sum(1 for r in results if r.get("valid"))
    fusion_total = sum(int(r.get("fusion_count", 0)) for r in results)
    bits_total = sum(int(r.get("total_bits", 0)) for r in results)

    parts = _HtmlPartsWriter(out_path)
    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    parts.append(f"<title>BLB smoke verify — {_html_escape(args.profile)}</title>")
    parts.append(STYLE_BLOCK)
    parts.append("</head><body><main>")
    parts.append("<h1>BLB noise-install verifier — smoke (optimizer-only)</h1>")
    parts.append(
        "<p class='lead'>Torch-free probe. Confirms the in-process "
        "<code>ReplanSession</code> wires up and returns the expected shape "
        "for every (block, layer) config implied by the Stage-1 plan. "
        "For the full per-noise-point install plan, run <code>--mode full</code> "
        "on a machine with torch + transformers.</p>"
    )
    parts.append("<section><h2>Run inputs</h2><ul>")
    parts.append(f"<li>profile: <code>{_html_escape(args.profile)}</code></li>")
    parts.append(f"<li>num_layers: <code>{int(args.num_layers)}</code></li>")
    parts.append(f"<li>Rescale_optimizer root: <code>{_html_escape(args.rescale_optimizer_root)}</code></li>")
    parts.append(f"<li>Stage-1: <code>{_html_escape(args.stage1)}</code></li>")
    parts.append(
        f"<li>Available baselines in archive: {len(baselines)} graphs — "
        + ", ".join(f"<code>{_html_escape(k)}</code>" for k in sorted(baselines))
        + "</li>"
    )
    parts.append("</ul></section>")

    parts.append("<section><h2>Aggregate</h2><ul>")
    parts.append(f"<li>Targets: {len(results)}</li>")
    parts.append(f"<li>Valid replan: {valid_count} / {len(results)}</li>")
    parts.append(f"<li>Total fusion count: {fusion_total}</li>")
    parts.append(f"<li>Sum total_bits: {bits_total}</li>")
    parts.append("</ul></section>")

    parts.append("<section><h2>Per (block, layer)</h2>")
    summary_headers = [
        "config_name", "graph_key", "block", "layer",
        "status", "valid", "fusion", "total_bits",
        "cut points", "rotations",
    ]
    summary_rows = []
    for r in results:
        cls = "ok" if r.get("valid") else "bad"
        summary_rows.append([
            f"<code>{_html_escape(r.get('config_name'))}</code>",
            f"<code>{_html_escape(r.get('graph_key'))}</code>",
            int(r.get('block', 0)),
            int(r.get('layer', 0)),
            _html_escape(r.get('status')),
            bool(r.get('valid')),
            r.get('fusion_count', '—'),
            r.get('total_bits', '—'),
            len(r.get('cut_point_sf', [])),
            len(r.get('effective_rotations', [])),
        ])
    parts.append(html_table(summary_headers, summary_rows, row_classes=[
        "ok" if r.get("valid") else "bad" for r in results
    ], allow_html_cells=True))
    parts.append("</section>")

    parts.append("<section><h2>Per-config details</h2>")
    for r in results:
        cls = "card-ok" if r.get("valid") else "card-bad"
        parts.append(
            f"<details class='{cls}'><summary><code>{_html_escape(r.get('config_name'))}</code> "
            f"(graph <code>{_html_escape(r.get('graph_key'))}</code>) — "
            f"valid={bool(r.get('valid'))}, fusion={r.get('fusion_count', '?')}, "
            f"bits={r.get('total_bits', '?')}</summary>"
        )
        skeleton = r.get("skeleton") or []
        t_final = r.get("t_final") or []
        if skeleton:
            parts.append(
                f"<p>skeleton: <code>{_html_escape(skeleton)}</code> &nbsp;|&nbsp; "
                f"t_final: <code>{_html_escape(t_final)}</code></p>"
            )

        cut_pts = r.get("cut_point_sf") or []
        if cut_pts:
            parts.append("<h4>cut_point_sf</h4>")
            parts.append(html_table(
                ["i", "node", "type", "sf / sf_post", "sf_pre", "drop"],
                [
                    [
                        cp.get('i', ''),
                        f"<code>{_html_escape(cp.get('name', ''))}</code>",
                        _html_escape(cp.get('type', '')),
                        cp.get("sf_post", cp.get("sf")) if cp.get("sf_post", cp.get("sf")) is not None else '—',
                        cp.get('sf_pre', '—'),
                        cp.get('drop', '—'),
                    ]
                    for cp in cut_pts
                ],
                allow_html_cells=True,
            ))
        rotations = r.get("effective_rotations") or []
        if rotations:
            parts.append("<h4>effective_rotations</h4>")
            parts.append(html_table(
                ["node_id", "name", "after_cut_point", "sf", "count"],
                [
                    [
                        rot.get('node_id', ''),
                        f"<code>{_html_escape(rot.get('name', ''))}</code>",
                        rot.get('after_cut_point', ''),
                        rot.get('sf', ''),
                        rot.get('count', ''),
                    ]
                    for rot in rotations
                ],
                allow_html_cells=True,
            ))
        prop_deltas = r.get("propagation_deltas") or []
        if prop_deltas:
            parts.append("<h4>propagation_deltas</h4>")
            parts.append(html_table(
                ["node_id", "name", "type", "delta"],
                [
                    [
                        pd.get('node_id', ''),
                        f"<code>{_html_escape(pd.get('name', ''))}</code>",
                        _html_escape(pd.get('type', '')),
                        _html_escape(pd.get('delta', '')),
                    ]
                    for pd in prop_deltas
                ],
                allow_html_cells=True,
            ))
        parts.append("</details>")
    parts.append("</section>")
    parts.append("</main></body></html>")

    parts.close()
    return out_path


# ---------------------------------------------------------------------------
# Full mode — needs torch
# ---------------------------------------------------------------------------
def run_full(args: argparse.Namespace) -> Path:
    rescale_root = Path(args.rescale_optimizer_root).resolve()
    if str(rescale_root) not in sys.path:
        sys.path.insert(0, str(rescale_root))
    sys.path.insert(0, str(REPO_ROOT))

    # These imports need torch + transformers
    try:
        from blb_stage2_rl.action_space import (
            action_dims_for_config,
            action_vector_to_cfgs,
            build_optimizer_requests,
            load_max_sfs,
            make_all_max_action_vector,
            parse_config_name,
        )
        from blb_stage2_rl.optimizer_cost import materialize_decoded_action
        from rescale_optimizer_bridge import (
            InProcessInvoker,
            RescaleOptimizerBridge,
            _strip_layer_suffix,
            aggregate_optimizer_signals,
        )
    except ImportError as exc:
        raise RuntimeError(
            "full mode requires the project Python deps (torch, transformers, ...). "
            "Use --mode smoke for a torch-free probe, or run this script on a machine "
            "that can import blb_stage2_rl."
        ) from exc

    profile = args.profile
    stage1 = parse_stage1_config(args.stage1)
    num_layers = int(args.num_layers)
    gelu = stage1.get("gelu_degree_per_layer") or stage1.get("gelu")
    softmax = stage1.get("softmax_degree_per_layer") or stage1.get("softmax")
    if gelu is None or softmax is None:
        raise ValueError("stage1 config must specify both gelu / softmax degree per layer")

    max_sfs = load_max_sfs(profile=profile)
    if args.action_file:
        action_doc = read_json_file(Path(args.action_file))
        action_vec = list(action_doc.get("action_indices") or action_doc.get("action") or [])
        if not action_vec:
            raise ValueError(f"{args.action_file} did not contain 'action_indices' or 'action' list")
    else:
        action_vec = list(make_all_max_action_vector(num_layers=num_layers))

    decoded = action_vector_to_cfgs(
        action_vec,
        max_sfs,
        num_layers=num_layers,
        gelu_degree=gelu,
        attn_degree=softmax,
    )
    cfgs_dict = decoded.cfgs_dict()
    requests = build_optimizer_requests(profile, cfgs_dict)

    invoker = InProcessInvoker.from_profile(
        rescale_optimizer_root=str(rescale_root), profile=profile,
    )
    bridge = RescaleOptimizerBridge(invoker=invoker)
    outputs = bridge.evaluate_blocks(requests)
    signals = aggregate_optimizer_signals(outputs)

    invoker_baselines = invoker.baselines
    materialized = materialize_decoded_action(
        action_indices=action_vec,
        decoded=decoded,
        cfgs_dict=cfgs_dict,
        outputs=outputs,
        signals=signals,
        profile=profile,
        invoker_baselines=invoker_baselines,
        expected_config_names=list(requests),
    )
    if not materialized.model_ready:
        raise RuntimeError(
            "action cannot reach model: "
            f"{materialized.failure_reason}; "
            f"replan={materialized.replan_application}"
        )
    cfgs_dict = materialized.cfgs_dict
    replan_per_config = materialized.replan_application.get("per_config", {})
    noise_table = load_noise_variance_table()
    per_config_records: List[Dict[str, Any]] = []
    for cn, out in outputs.items():
        try:
            block_idx, _, layer_idx = parse_config_name(cn)
        except Exception:
            continue
        if layer_idx < 0:
            continue
        target_cfg = cfgs_dict[f"block{block_idx}"][int(layer_idx)]
        graph_key, _ = _strip_layer_suffix(cn)
        overrides = list(
            (replan_per_config.get(str(cn), {}) or {}).get("overrides", [])
        )

        noise_points = _enumerate_cfg_noise_points(target_cfg, noise_table=noise_table)
        per_config_records.append({
            "config_name": cn,
            "graph_key": graph_key,
            "block": int(block_idx),
            "layer": int(layer_idx),
            "valid": bool(out.valid),
            "fusion_count": int(out.fusion_count),
            "total_bits": int(out.total_bits),
            "compact": out.raw.get("new_compact_config") or {},
            "overrides": overrides,
            "noise_points": noise_points,
        })

    return write_full_html(
        args=args,
        action_vec=action_vec,
        stage1=stage1,
        signals=signals,
        records=per_config_records,
    )


def _enumerate_cfg_noise_points(cfg: Any, noise_table: Mapping) -> List[Dict[str, Any]]:
    """Walk a Block{N}NoiseConfig and report every active NoisePoint."""
    out: List[Dict[str, Any]] = []
    for name in vars(cfg).keys():
        if name.startswith("rotation_after_") or name == "output_truncation_mode":
            continue
        value = getattr(cfg, name)
        # Tuple-valued fields (square_rescales, gelu_power_rescales, gelu_coeff_mul_rescales)
        if isinstance(value, tuple):
            for idx, point in enumerate(value):
                if point is None:
                    continue
                if not hasattr(point, "scaling_factor"):
                    continue
                out.append({
                    "field": f"{name}[{idx}]",
                    "distribution": str(getattr(point, "distribution", "")),
                    "scaling_factor": int(point.scaling_factor),
                    "N": int(getattr(point, "N", 0)),
                    "variance": lookup_variance(
                        noise_table,
                        N=int(getattr(point, "N", 0)),
                        sf=int(point.scaling_factor),
                        distribution=str(getattr(point, "distribution", "")),
                    ),
                })
            continue
        if value is None:
            continue
        if not hasattr(value, "scaling_factor"):
            continue
        out.append({
            "field": name,
            "distribution": str(getattr(value, "distribution", "")),
            "scaling_factor": int(value.scaling_factor),
            "N": int(getattr(value, "N", 0)),
            "variance": lookup_variance(
                noise_table,
                N=int(getattr(value, "N", 0)),
                sf=int(value.scaling_factor),
                distribution=str(getattr(value, "distribution", "")),
            ),
        })
    # Rotation flags (their SF / N inherit from a bound rescale/fresh source)
    for name in vars(cfg).keys():
        if not name.startswith("rotation_after_"):
            continue
        if bool(getattr(cfg, name)):
            out.append({
                "field": name,
                "distribution": "rotation",
                "scaling_factor": None,
                "N": None,
                "variance": None,
            })
    return out


def write_full_html(
    *,
    args: argparse.Namespace,
    action_vec: Sequence[int],
    stage1: Mapping[str, Any],
    signals: Any,
    records: List[Dict[str, Any]],
) -> Path:
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "reports" / "blb_opt" / "noise_install_verify"
        / f"full_{args.profile}_{_utcnow_slug()}.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts = _HtmlPartsWriter(out_path)
    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    parts.append(f"<title>BLB noise-install verify — {_html_escape(args.profile)}</title>")
    parts.append(STYLE_BLOCK)
    parts.append("</head><body><main>")
    parts.append("<h1>BLB noise-install verifier — full mode</h1>")
    parts.append(
        "<p class='lead'>End-to-end: action → action_vector_to_cfgs → "
        "ReplanSession.replan → apply_optimizer_output_to_cfg → per-noise-point "
        "install plan. Every σ² shown comes from "
        "<code>NOISE_VARIANCE_TABLE_BY_N</code>; every optimizer return is a "
        "real <code>replan_with_user_actions</code> call (no heuristic fallback "
        "is reachable in the current build).</p>"
    )

    parts.append("<section><h2>Inputs</h2><ul>")
    parts.append(f"<li>profile: <code>{_html_escape(args.profile)}</code></li>")
    parts.append(f"<li>num_layers: {int(args.num_layers)}</li>")
    action_indices = [int(x) for x in action_vec]
    stage1_json = json.dumps(to_jsonable(stage1, preserve_native=True), indent=2)
    parts.append(f"<li>Stage-1: <pre>{_html_escape(stage1_json)}</pre></li>")
    parts.append(f"<li>action_vector ({len(action_indices)} indices): <pre>{_html_escape(json.dumps(action_indices))}</pre></li>")
    parts.append("</ul></section>")

    parts.append("<section><h2>Aggregate signals</h2><ul>")
    parts.append(f"<li>configs evaluated: {len(records)}</li>")
    parts.append(f"<li>any_invalid: <b>{bool(getattr(signals, 'any_invalid', False))}</b></li>")
    parts.append(f"<li>invalid block count: {int(getattr(signals, 'invalid_block_count', 0))}</li>")
    parts.append(f"<li>total fusion: {int(getattr(signals, 'total_fusion_count', 0))}</li>")
    parts.append(f"<li>sum total_bits: {int(getattr(signals, 'total_bits_sum', 0))}</li>")
    parts.append("</ul></section>")

    parts.append("<section><h2>Per (block, layer) install plan</h2>")
    for r in records:
        cls = "card-ok" if r["valid"] else "card-bad"
        parts.append(
            f"<details open class='{cls}'><summary><code>{_html_escape(r['config_name'])}</code> "
            f"— graph <code>{_html_escape(r['graph_key'])}</code>, "
            f"block {r['block']}, layer {r['layer']}, "
            f"valid={r['valid']}, fusion={r['fusion_count']}, "
            f"bits={r['total_bits']}, "
            f"noise points installed={sum(1 for p in r['noise_points'] if p.get('scaling_factor') is not None)}</summary>"
        )

        ovs = r.get("overrides") or []
        if ovs:
            parts.append("<h4>Optimizer overrides on cfg</h4>")
            parts.append(html_table(
                ["cfg_attr", "graph_node", "source", "old", "new"],
                [
                    [
                        f"<code>{_html_escape(ov.get('cfg_attr', ''))}</code>",
                        f"<code>{_html_escape(ov.get('graph_node') or '')}</code>",
                        _html_escape(ov.get('source', '')),
                        _html_escape(ov.get('old_value')),
                        _html_escape(ov.get('new_value')),
                    ]
                    for ov in ovs
                ],
                allow_html_cells=True,
            ))
        else:
            parts.append("<p><i>No optimizer overrides applied (action == optimizer outcome).</i></p>")

        nps = r.get("noise_points") or []
        if nps:
            parts.append("<h4>Noise points to install (post-override)</h4>")
            parts.append(html_table(
                ["cfg field", "distribution", "scaling_factor", "N", "σ² (variance)"],
                [
                    [
                        f"<code>{_html_escape(p['field'])}</code>",
                        _html_escape(p['distribution']),
                        p.get('scaling_factor', '—'),
                        p.get('N', '—'),
                        f"{p.get('variance'):.4e}" if isinstance(p.get("variance"), float) else "—",
                    ]
                    for p in nps
                ],
                allow_html_cells=True,
            ))
        else:
            parts.append("<p><i>No noise points to install for this config.</i></p>")

        compact = r.get("compact") or {}
        if compact:
            parts.append("<h4>Optimizer return — new_compact_config</h4>")
            compact_json = json.dumps(to_jsonable(compact, preserve_native=True), indent=2)
            parts.append(f"<pre>{_html_escape(compact_json[:5000])}</pre>")
        parts.append("</details>")
    parts.append("</section>")
    parts.append("</main></body></html>")

    parts.close()
    return out_path


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
STYLE_BLOCK = """
<style>
  body { background:#f5f7fa; color:#1a202c; font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","SF Pro Text",sans-serif; margin:0; }
  main { max-width: 1200px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0 0 16px; font-size: 26px; border-bottom: 2px solid #2b6cb0; padding-bottom: 10px; }
  h2 { margin: 28px 0 12px; font-size: 20px; border-left: 4px solid #2b6cb0; padding-left: 10px; }
  h4 { margin: 14px 0 6px; font-size: 14px; color: #2d3748; }
  p.lead { color: #4a5568; max-width: 980px; }
  ul { padding-left: 24px; }
  code { background: #edf2f7; border: 1px solid #cbd5e0; border-radius: 3px; padding: 0 4px; font: 12px Consolas, "SF Mono", Menlo, monospace; }
  pre { background: #0f172a; color: #e2e8f0; padding: 10px 14px; border-radius: 6px; overflow: auto; font: 12px/1.5 Consolas, "SF Mono", monospace; }
  table { border-collapse: collapse; margin: 8px 0 12px; min-width: 60%; }
  th, td { border: 1px solid #cbd5e0; padding: 4px 10px; text-align: left; font-size: 13px; }
  th { background: #e2e8f0; font-weight: 600; }
  tr.ok { background: #f0fff4; } tr.bad { background: #fff5f5; }
  details { background: #fff; border: 1px solid #cbd5e0; border-radius: 6px; padding: 10px 14px; margin: 10px 0; }
  details.card-ok { border-left: 4px solid #2f855a; }
  details.card-bad { border-left: 4px solid #c53030; }
  summary { cursor: pointer; font-weight: 600; }
</style>
"""


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="BLB noise-install verifier")
    ap.add_argument("--mode", choices=["full", "smoke"], default="full")
    ap.add_argument("--profile", default="mrpc")
    ap.add_argument("--num-layers", type=int, default=12)
    ap.add_argument(
        "--stage1",
        default='{"gelu_degree_per_layer":[4,4,4,4,4,4,4,4,4,4,4,4],"softmax_degree_per_layer":[4,4,4,4,4,4,4,4,4,4,4,4]}',
        help="JSON or 'gelu=4,4,...;softmax=2,3,...' (length = num_layers).",
    )
    ap.add_argument("--action-file", default=None,
                    help="Optional JSON with key 'action_indices' (list of ints). "
                         "If omitted, the all-max baseline action is used (full mode only).")
    ap.add_argument(
        "--rescale-optimizer-root",
        default=str(REPO_ROOT / "Rescale_optimizer"),
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.mode == "smoke":
        path = run_smoke(args)
    else:
        path = run_full(args)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
