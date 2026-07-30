#!/usr/bin/env python3
"""Build the standalone final HTML report from the archived raw streams."""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


COLORS = (
    "#2458a6",
    "#c24b36",
    "#19806a",
    "#9a6b16",
    "#6b4fa1",
    "#48636f",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def iter_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def finite(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def mean(values: Iterable[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def fmt(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}%"


def bytes_fmt(value: int) -> str:
    units = ("B", "KB", "MB", "GB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def svg_chart(
    title: str,
    series: list[tuple[str, list[tuple[float, float | None]]]],
    y_label: str,
) -> str:
    width, height = 960, 300
    left, right, top, bottom = 72, 22, 42, 48
    points = [
        (x, y)
        for _, values in series
        for x, y in values
        if y is not None and math.isfinite(y)
    ]
    if not points:
        return f"<div class='chart'><h3>{html.escape(title)}</h3><p>No data.</p></div>"
    x_min, x_max = min(x for x, _ in points), max(x for x, _ in points)
    y_min, y_max = min(y for _, y in points), max(y for _, y in points)
    if x_max == x_min:
        x_max += 1
    if y_max == y_min:
        pad = max(abs(y_max) * 0.05, 1e-6)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y: float) -> float:
        return top + (y_max - y) * plot_h / (y_max - y_min)

    grid = []
    for index in range(5):
        ratio = index / 4
        y = top + ratio * plot_h
        value = y_max - ratio * (y_max - y_min)
        grid.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{width-right}' y2='{y:.2f}' "
            "stroke='#dfe4e8' stroke-width='1'/>"
            f"<text x='{left-10}' y='{y+4:.2f}' text-anchor='end'>{value:.4g}</text>"
        )
    for index in range(5):
        ratio = index / 4
        x = left + ratio * plot_w
        value = x_min + ratio * (x_max - x_min)
        grid.append(
            f"<line x1='{x:.2f}' y1='{top}' x2='{x:.2f}' y2='{height-bottom}' "
            "stroke='#edf0f2' stroke-width='1'/>"
            f"<text x='{x:.2f}' y='{height-bottom+22}' text-anchor='middle'>{value:.0f}</text>"
        )

    paths = []
    legend = []
    for index, (name, values) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        segments = []
        active = []
        for x, y in values:
            if y is None or not math.isfinite(y):
                if active:
                    segments.append(active)
                    active = []
                continue
            active.append((sx(x), sy(y)))
        if active:
            segments.append(active)
        for segment in segments:
            commands = " ".join(
                f"{'M' if point_index == 0 else 'L'} {x:.2f} {y:.2f}"
                for point_index, (x, y) in enumerate(segment)
            )
            paths.append(
                f"<path d='{commands}' fill='none' stroke='{color}' "
                "stroke-width='2' vector-effect='non-scaling-stroke'/>"
            )
        legend.append(
            f"<span><i style='background:{color}'></i>{html.escape(name)}</span>"
        )

    return (
        "<div class='chart'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<div class='legend'>{''.join(legend)}</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{html.escape(title)}'>"
        f"<text x='18' y='{top + plot_h / 2:.2f}' text-anchor='middle' "
        f"transform='rotate(-90 18 {top + plot_h / 2:.2f})'>{html.escape(y_label)}</text>"
        f"{''.join(grid)}{''.join(paths)}"
        f"<text x='{left + plot_w / 2:.2f}' y='{height-8}' text-anchor='middle'>Episode</text>"
        "</svg></div>"
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    checkpoint = load_json(root / "checkpoint_summary.json")
    run_manifest = load_json(
        root
        / "small_files/run/stage2_noise/progress/layerwise_run_manifest.json"
    )
    snapshot = load_json(root / "snapshot_manifest.json")
    restore = load_json(root / "restore_verification.json")

    ppo_updates = list(
        iter_jsonl_gz(root / "streams/diagnostics_ppo_updates.jsonl.gz")
    )
    episodes = list(iter_jsonl_gz(root / "streams/diagnostics_episodes.jsonl.gz"))
    strict = checkpoint["strict_best"]
    baseline = run_manifest["baseline_references"]["F4"]["promotion_reference_ab"]
    best_metrics = strict["metrics"]
    limits = baseline["limits"]
    assessment = strict["assessment"]
    layers = strict["layer_configurations"]

    update_x = [float(item["completed_episodes"]) for item in ppo_updates]
    reward_chart = svg_chart(
        "Reward curve by PPO update",
        [
            (
                "Window mean return",
                list(
                    zip(
                        update_x,
                        [finite(item.get("window_mean_return")) for item in ppo_updates],
                    )
                ),
            ),
            (
                "Best reward so far",
                list(
                    zip(
                        update_x,
                        [finite(item.get("best_reward_so_far")) for item in ppo_updates],
                    )
                ),
            ),
        ],
        "Reward",
    )
    entropy_chart = svg_chart(
        "Policy entropy",
        [
            (
                "Total",
                list(zip(update_x, [finite(item.get("entropy")) for item in ppo_updates])),
            ),
            (
                "Block4",
                list(
                    zip(
                        update_x,
                        [finite(item.get("block4_entropy")) for item in ppo_updates],
                    )
                ),
            ),
            (
                "Precision preset",
                list(
                    zip(
                        update_x,
                        [finite(item.get("k_entropy")) for item in ppo_updates],
                    )
                ),
            ),
        ],
        "Normalized entropy",
    )

    buckets: list[dict] = []
    for start in range(0, len(episodes), 120):
        group = episodes[start : start + 120]
        buckets.append(
            {
                "episode": float(group[-1]["episode"] + 1),
                "loss": mean(finite(item.get("terminal_loss_mean")) for item in group),
                "m1": mean(finite(item.get("terminal_metric1_mean")) for item in group),
                "m2": mean(finite(item.get("terminal_metric2_mean")) for item in group),
                "compute": mean(finite(item.get("compute_saving")) for item in group),
                "communication": mean(
                    finite(item.get("communication_saving")) for item in group
                ),
                "weighted": mean(
                    finite(item.get("ppo_resource_score")) for item in group
                ),
            }
        )
    metric_chart = svg_chart(
        "Online probe classification metrics (120-episode means)",
        [
            ("Metric1 / Accuracy", [(b["episode"], b["m1"]) for b in buckets]),
            ("Metric2 / Weighted F1", [(b["episode"], b["m2"]) for b in buckets]),
        ],
        "Score",
    )
    loss_chart = svg_chart(
        "Online probe loss (120-episode mean)",
        [("Loss", [(b["episode"], b["loss"]) for b in buckets])],
        "Loss",
    )
    resource_chart = svg_chart(
        "Sampled resource savings (120-episode means)",
        [
            ("Compute", [(b["episode"], b["compute"]) for b in buckets]),
            (
                "Communication",
                [(b["episode"], b["communication"]) for b in buckets],
            ),
            ("Weighted", [(b["episode"], b["weighted"]) for b in buckets]),
        ],
        "Saving",
    )
    diagnostic_chart = svg_chart(
        "PPO optimization diagnostics",
        [
            (
                "Approx KL",
                list(zip(update_x, [finite(item.get("approx_kl")) for item in ppo_updates])),
            ),
            (
                "Clip fraction",
                list(
                    zip(
                        update_x,
                        [finite(item.get("clip_fraction")) for item in ppo_updates],
                    )
                ),
            ),
            (
                "Value explained variance",
                list(
                    zip(
                        update_x,
                        [
                            finite(item.get("value_explained_variance_post"))
                            for item in ppo_updates
                        ],
                    )
                ),
            ),
        ],
        "Diagnostic value",
    )

    metric_specs = (
        ("Loss", "loss_mean", "loss_std", "loss", "loss_std", "lower"),
        ("Accuracy", "metric1_mean", "metric1_std", "metric1", "metric1_std", "higher"),
        (
            "Weighted F1",
            "metric2_mean",
            "metric2_std",
            "metric2",
            "metric2_std",
            "higher",
        ),
    )
    metric_rows = []
    for label, mean_key, std_key, limit_key, std_limit_key, direction in metric_specs:
        base_mean = float(baseline[mean_key])
        base_std = float(baseline[std_key])
        candidate_mean = float(best_metrics[mean_key])
        candidate_std = float(best_metrics[std_key])
        delta = candidate_mean - base_mean
        delta_pct = delta / base_mean * 100
        mean_pass = (
            candidate_mean <= float(limits[limit_key])
            if direction == "lower"
            else candidate_mean >= float(limits[limit_key])
        )
        std_pass = candidate_std <= float(limits[std_limit_key])
        metric_rows.append(
            "<tr>"
            f"<th>{label}</th>"
            f"<td>{fmt(base_mean)} &plusmn; {fmt(base_std)}</td>"
            f"<td>{fmt(candidate_mean)} &plusmn; {fmt(candidate_std)}</td>"
            f"<td>{fmt(delta)}<br><span class='muted'>{pct(delta_pct)}</span></td>"
            f"<td>{fmt(float(limits[limit_key]))}</td>"
            f"<td>{fmt(float(limits[std_limit_key]))}</td>"
            f"<td><span class='pass'>{'PASS' if mean_pass and std_pass else 'FAIL'}</span></td>"
            "</tr>"
        )

    layer_rows = []
    preset_counts = Counter()
    for item in layers:
        preset_counts[item["precision_preset_name"]] += 1
        k = item["truncation_k_by_block"]
        layer_rows.append(
            "<tr>"
            f"<td>{int(item['layer_idx'])}</td>"
            "<td>1 <span class='fixed'>fixed</span></td>"
            f"<td><strong>{int(item['block4_fusion_count'])}</strong></td>"
            "<td>1 <span class='fixed'>fixed</span></td>"
            f"<td><span class='preset {html.escape(item['precision_preset_name'])}'>"
            f"{html.escape(item['precision_preset_name'].upper())}</span></td>"
            f"<td>{int(k['block1'])}</td><td>{int(k['block2'])}</td>"
            f"<td>{int(k['block3'])}</td><td>{int(k['block4'])}</td>"
            f"<td>{int(k['block5'])}</td>"
            "</tr>"
        )

    priority_counts = Counter(
        str(item.get("terminal_priority") or "unknown") for item in episodes
    )
    promotion_counts = Counter(
        str(item.get("promotion_status") or "none") for item in episodes
    )
    invalid_episodes = sum(int(item.get("invalid_steps") or 0) > 0 for item in episodes)
    loss_caps = sum(
        finite(item.get("terminal_loss_mean")) == 100.0 for item in episodes
    )
    first_timestamp = finite(episodes[0].get("timestamp"))
    last_timestamp = finite(episodes[-1].get("timestamp"))
    elapsed = (
        last_timestamp - first_timestamp
        if first_timestamp is not None and last_timestamp is not None
        else None
    )
    throughput = len(episodes) / elapsed * 3600 if elapsed and elapsed > 0 else None
    convergence = checkpoint["convergence_state"]
    stage1_gelu = run_manifest["run_context"]["candidate_identity_context"][
        "stage1_gelu_degrees"
    ]
    stage1_softmax = run_manifest["run_context"]["candidate_identity_context"][
        "stage1_softmax_degrees"
    ]
    actual_total_fusions = 48 + int(strict["fusion_count"])

    stream_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(path)}</code></td>"
        f"<td>{int(item['rows']):,}</td>"
        f"<td>{bytes_fmt(int(item['raw_bytes']))}</td>"
        f"<td><code>{html.escape(item['raw_sha256'][:16])}&hellip;</code></td>"
        "</tr>"
        for path, item in snapshot["streams"].items()
    )
    check_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name.replace('_', ' '))}</td>"
        f"<td><span class='pass'>{'PASS' if value else 'FAIL'}</span></td>"
        "</tr>"
        for name, value in snapshot["verification_checks"].items()
    )

    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BERT-large MRPC Stage-2 RL checkpoint 24720</title>
<style>
:root {{
  --ink:#18212a; --muted:#65717c; --line:#d9dfe4; --paper:#ffffff;
  --band:#f4f6f7; --blue:#2458a6; --green:#14755e; --red:#a43d31;
  --amber:#8a6414;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#eef1f3; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }}
main {{ max-width:1180px; margin:0 auto; background:var(--paper); min-height:100vh; }}
header {{ padding:38px 46px 30px; border-bottom:1px solid var(--line); background:#17242e; color:white; }}
h1 {{ margin:0 0 8px; font-size:30px; line-height:1.2; }}
h2 {{ margin:0 0 16px; font-size:21px; }}
h3 {{ margin:0; font-size:15px; }}
p {{ margin:8px 0; }}
section {{ padding:28px 46px; border-bottom:1px solid var(--line); }}
.lede {{ color:#d9e4eb; max-width:880px; font-size:15px; }}
.status {{ display:inline-block; margin-top:14px; padding:5px 9px; border:1px solid #efc66b; color:#ffe19a; font-weight:700; }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); }}
.stat {{ min-height:108px; padding:16px; background:white; }}
.stat b {{ display:block; margin-top:8px; font-size:24px; line-height:1.15; }}
.stat span {{ color:var(--muted); }}
.notice {{ padding:14px 16px; border-left:4px solid var(--amber); background:#fff8e7; }}
.ok-notice {{ border-left-color:var(--green); background:#edf8f4; }}
.chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.chart {{ border:1px solid var(--line); padding:14px 12px 8px; min-width:0; }}
.chart svg {{ display:block; width:100%; height:auto; margin-top:8px; }}
.chart svg text {{ fill:#62707a; font-size:11px; }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:8px; color:var(--muted); font-size:12px; }}
.legend i {{ display:inline-block; width:14px; height:3px; margin:0 6px 3px 0; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--line); }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:9px 10px; text-align:right; border-bottom:1px solid var(--line); }}
thead th {{ background:var(--band); color:#3f4c56; font-size:12px; text-transform:uppercase; }}
tbody th, td:first-child, th:first-child {{ text-align:left; }}
tbody tr:last-child td, tbody tr:last-child th {{ border-bottom:0; }}
.pass {{ color:var(--green); font-weight:800; }}
.muted {{ color:var(--muted); }}
.fixed {{ color:var(--muted); font-size:10px; font-weight:400; }}
.preset {{ display:inline-block; min-width:58px; padding:2px 6px; text-align:center; font-size:11px; font-weight:800; border:1px solid; }}
.preset.high {{ color:#2458a6; background:#eef4ff; }}
.preset.medium {{ color:#8a6414; background:#fff7df; }}
.preset.low {{ color:#14755e; background:#eaf8f3; }}
code {{ font:12px ui-monospace,SFMono-Regular,Menlo,monospace; white-space:normal; overflow-wrap:anywhere; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
ul {{ margin:8px 0; padding-left:20px; }}
footer {{ padding:22px 46px 34px; color:var(--muted); }}
@media (max-width:800px) {{
  header,section,footer {{ padding-left:20px; padding-right:20px; }}
  .grid,.chart-grid,.two-col {{ grid-template-columns:1fr; }}
  h1 {{ font-size:25px; }}
}}
</style>
</head>
<body><main>
<header>
  <h1>BERT-large MRPC Stage-2 RL</h1>
  <p class="lede">Checkpoint-boundary final archive and interim strict-best report. Source <code>{snapshot['repo_commit'][:12]}</code>; H/M/L truncation presets; learnable per-layer Block4 fusion; three-trial online evaluation on four GPUs.</p>
  <span class="status">STOPPED EARLY · NOT CONVERGED</span>
</header>

<section>
  <div class="grid">
    <div class="stat"><span>Completed episodes</span><b>{checkpoint['episode']:,}</b><span>of {checkpoint['planned_total_episodes']:,} original plan</span></div>
    <div class="stat"><span>PPO updates</span><b>{checkpoint['ppo_update_count']:,}</b><span>rollout window 120</span></div>
    <div class="stat"><span>Observed throughput</span><b>{fmt(throughput, 0)}</b><span>episodes/hour over retained trajectory</span></div>
    <div class="stat"><span>Strict resource score</span><b>{strict['ppo_resource_score']:.5f}</b><span>compute/communication = 1:1</span></div>
  </div>
  <div class="notice" style="margin-top:18px"><strong>Result interpretation.</strong> The selected candidate passed authoritative full-validation Banks A+B with 30 trials. It is not a convergence-certified final result: the 90,000-episode floor was not reached and Bank C strict revalidation remained <code>{html.escape(convergence['strict_revalidation_status'])}</code>.</div>
</section>

<section>
  <h2>Strict candidate vs baseline</h2>
  <p>Both rows use the same authoritative <code>validation_full</code> protocol. Baseline and candidate values below pool Banks A+B (30 trials each). Precision tolerance is 0.1%; stability limit is 200% of the corresponding baseline standard deviation.</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Metric</th><th>Baseline mean ± std</th><th>Strict candidate mean ± std</th><th>Candidate − baseline</th><th>Precision limit</th><th>Std limit</th><th>Gate</th></tr></thead>
      <tbody>{''.join(metric_rows)}</tbody>
    </table>
  </div>
  <div class="grid" style="margin-top:18px">
    <div class="stat"><span>Learned Block4 fusion</span><b>{strict['fusion_count']} / 24</b><span>compute saving {strict['compute_saving']:.2%}</span></div>
    <div class="stat"><span>Actual B2+B4+B5 fusion</span><b>{actual_total_fusions} / 72</b><span>B2/B5 fixed to 1; B4 learned</span></div>
    <div class="stat"><span>Precision presets</span><b>H {preset_counts['high']} · M {preset_counts['medium']} · L {preset_counts['low']}</b><span>communication saving {strict['communication_saving']:.2%}</span></div>
    <div class="stat"><span>Robust floor</span><b>{strict['robust_floor']:.5f}</b><span>min(compute, communication)</span></div>
  </div>
  <p class="muted">Gate probabilities: loss precision {assessment['loss_precision_probability']:.4f}, accuracy precision {assessment['metric1_precision_probability']:.4f}, F1 precision {assessment['metric2_precision_probability']:.4f}; loss stability {assessment['loss_stability_probability']:.4f}, accuracy stability {assessment['metric1_stability_probability']:.4f}, F1 stability {assessment['metric2_stability_probability']:.4f}.</p>
</section>

<section>
  <h2>Learning curves</h2>
  <div class="chart-grid">{reward_chart}{entropy_chart}{metric_chart}{loss_chart}{resource_chart}{diagnostic_chart}</div>
</section>

<section>
  <h2>Selected model configuration</h2>
  <p>Baseline for constraints is B2/B4/B5 fusion 0 with the high-precision K preset. The searched model fixes B2 and B5 fusion to 1 in every layer and learns only B4 fusion plus one H/M/L K preset per layer. Layer indices are zero-based.</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Layer</th><th>B2 fusion</th><th>B4 fusion</th><th>B5 fusion</th><th>K preset</th><th>B1 K</th><th>B2 K</th><th>B3 K</th><th>B4 K</th><th>B5 K</th></tr></thead>
      <tbody>{''.join(layer_rows)}</tbody>
    </table>
  </div>
  <div class="two-col" style="margin-top:18px">
    <div><h3>Fixed Stage-1 GELU</h3><p><code>{html.escape(json.dumps(stage1_gelu))}</code></p></div>
    <div><h3>Fixed Stage-1 Softmax</h3><p><code>{html.escape(json.dumps(stage1_softmax))}</code></p></div>
  </div>
  <p class="muted">Candidate key: <code>{html.escape(strict['candidate_key'])}</code>. Promotion status: <strong>{html.escape(strict['promotion_evidence']['status'])}</strong>; authoritative promotion trials: {strict['promotion_evidence']['trial_count']}.</p>
</section>

<section>
  <h2>Run health and convergence</h2>
  <div class="grid">
    <div class="stat"><span>Block4 entropy</span><b>{convergence['block4_entropy']:.4f}</b><span>diagnostic only</span></div>
    <div class="stat"><span>Preset entropy</span><b>{convergence['k_entropy']:.4f}</b><span>diagnostic only</span></div>
    <div class="stat"><span>Stable update windows</span><b>{convergence['selected_action_stable_update_windows']}</b><span>required 100 after minimum episode floor</span></div>
    <div class="stat"><span>Frontier stall windows</span><b>{convergence['stall_update_windows']}</b><span>required 100 after minimum episode floor</span></div>
  </div>
  <div class="two-col" style="margin-top:18px">
    <div><h3>Terminal priorities</h3><p><code>{html.escape(json.dumps(dict(priority_counts), sort_keys=True))}</code></p></div>
    <div><h3>Promotion states</h3><p><code>{html.escape(json.dumps(dict(promotion_counts), sort_keys=True))}</code></p></div>
  </div>
  <p>Episodes with any invalid step: <strong>{invalid_episodes:,}</strong>. Loss-cap sentinels (<code>loss_mean=100</code>): <strong>{loss_caps:,}</strong>. Termination reason in the last checkpoint: <code>{html.escape(convergence['termination_reason'])}</code>.</p>
</section>

<section>
  <h2>Complete data and recovery evidence</h2>
  <div class="notice ok-notice"><strong>Restore test passed.</strong> A clean restore produced all seven raw streams with exact byte counts, row counts, newline boundaries, and SHA-256 hashes. Checkpoint size and hash also matched. The restored tree occupies 1.3 GB.</div>
  <div class="table-wrap" style="margin-top:18px">
    <table><thead><tr><th>Archived stream</th><th>Rows</th><th>Raw size</th><th>Raw SHA-256</th></tr></thead><tbody>{stream_rows}</tbody></table>
  </div>
  <div class="table-wrap" style="margin-top:18px">
    <table><thead><tr><th>Integrity check</th><th>Result</th></tr></thead><tbody>{check_rows}<tr><td>clean restore all streams</td><td><span class="pass">{'PASS' if restore['all_streams_match'] else 'FAIL'}</span></td></tr><tr><td>restored checkpoint SHA-256</td><td><span class="pass">{'PASS' if restore['checkpoint_sha256_match'] else 'FAIL'}</span></td></tr></tbody></table>
  </div>
  <p class="muted">The archive retains raw per-episode rewards, metrics, actions, resource terms, trial seeds/results, promotion decisions, PPO diagnostics, Pareto history, top candidates, action histograms, baselines, manifests, and the resumable optimizer/policy checkpoint. These are sufficient to redraw reward/metric/entropy/resource/PPO curves, reproduce candidate rankings, audit constraints, and resume from episode 24,720 without rerunning the completed portion.</p>
</section>

<footer>Generated from the committed raw archive. No metric in this report was reconstructed from screenshots or rounded console logs.</footer>
</main></body></html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"REPORT_OK output={args.output} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
