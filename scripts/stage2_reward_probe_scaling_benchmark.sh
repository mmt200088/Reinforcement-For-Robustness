#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export GLUE_LOCAL_DATASET_DIR="${GLUE_LOCAL_DATASET_DIR:-/hy-tmp/glue_data}"

RUN_ID="stage2_reward_probe_scaling_$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="${ARTIFACT_DIR:-experiments/server_command_runs/${RUN_ID}}"
BENCH_EPISODES="${BENCH_EPISODES:-4}"
BENCH_BATCH_SIZES="${BENCH_BATCH_SIZES:-64 128 256}"
PROBE_SIZE="${PROBE_SIZE:-256}"
K_TRIALS="${K_TRIALS:-4}"
DEVICE_SPECS="${DEVICE_SPECS:-0;0,1;0,1,2;0,1,2,3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-2400}"

mkdir -p "$ARTIFACT_DIR"
exec > >(tee "${ARTIFACT_DIR}/benchmark_stdout.log") 2>&1

echo "[bench] Stage-2 reward probe GPU scaling benchmark"
echo "[bench] artifact_dir=${ARTIFACT_DIR}"
echo "[bench] episodes=${BENCH_EPISODES} k_trials=${K_TRIALS} probe_size=${PROBE_SIZE}"
echo "[bench] batch_sizes=${BENCH_BATCH_SIZES}"
echo "[bench] device_specs=${DEVICE_SPECS}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv \
  | tee "${ARTIFACT_DIR}/nvidia_pre_benchmark.csv"

IFS=';' read -r -a DEVICE_SPEC_ARRAY <<< "$DEVICE_SPECS"

for batch_size in $BENCH_BATCH_SIZES; do
  for device_spec in "${DEVICE_SPEC_ARRAY[@]}"; do
    device_spec="$(printf '%s' "$device_spec" | xargs)"
    [ -n "$device_spec" ] || continue
    gpu_count=$(python3 - <<PY
spec = "${device_spec}"
print(len([x for x in spec.split(",") if x.strip()]))
PY
)
    reward_arg=()
    if [ "$gpu_count" -ge 2 ]; then
      reward_arg=(--blb-v3-reward-devices "$device_spec")
    fi
    label="bs${batch_size}_g${gpu_count}"
    persistent_root="${ARTIFACT_DIR}/persistent_${label}"
    log_file="${ARTIFACT_DIR}/${label}.log"
    echo ""
    echo "================================================================================"
    echo "[bench] ${label}: CUDA_VISIBLE_DEVICES=${device_spec}"
    echo "================================================================================"
    rm -rf "$persistent_root"
    set +e
    CUDA_VISIBLE_DEVICES="$device_spec" timeout "$TIMEOUT_SECONDS" \
      bash llama_7B_LayerImportance.sh run rl \
        --preset mrpc-blb-stage2-rl \
        --persistent-root "$persistent_root" \
        --batch-size "$batch_size" \
        --stage2-search-episodes "$BENCH_EPISODES" \
        --stage2-rollout-size 60 \
        --stage2-k-trials "$K_TRIALS" \
        --stage2-probe-size "$PROBE_SIZE" \
        --stage2-save-interval 1000 \
        --stage2-eval-interval 1000 \
        --skip-final-eval \
        --fresh \
        "${reward_arg[@]}" 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    echo "[bench] ${label} rc=${rc}"
    progress_dir="${persistent_root}/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress"
    diag_dir="${progress_dir}/diagnostics"
    [ -f "${diag_dir}/episodes.jsonl" ] && cp "${diag_dir}/episodes.jsonl" "${ARTIFACT_DIR}/${label}_episodes.jsonl" || true
    [ -f "${progress_dir}/../pruning_search_log.txt" ] && cp "${progress_dir}/../pruning_search_log.txt" "${ARTIFACT_DIR}/${label}_pruning_search_log.txt" || true
    printf '{"label":"%s","batch_size":%s,"gpu_count":%s,"device_spec":"%s","rc":%s}\n' \
      "$label" "$batch_size" "$gpu_count" "$device_spec" "$rc" >> "${ARTIFACT_DIR}/runs.jsonl"
  done
done

python3 - "$ARTIFACT_DIR" <<'PY'
import html
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = []
for line in (root / "runs.jsonl").read_text().splitlines():
    if line.strip():
        runs.append(json.loads(line))

rows = []
for run in runs:
    label = run["label"]
    ep_path = root / f"{label}_episodes.jsonl"
    probe_walls = []
    speedups = []
    devices_seen = set()
    counts_seen = set()
    if ep_path.exists():
        for line in ep_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            wall = float(rec.get("terminal_probe_wall_seconds") or 0.0)
            if wall > 0:
                probe_walls.append(wall)
            sp = float(rec.get("terminal_probe_speedup") or 0.0)
            if sp > 0:
                speedups.append(sp)
            for dev in rec.get("terminal_probe_devices") or []:
                devices_seen.add(str(dev))
            counts = rec.get("terminal_probe_trial_counts") or []
            if counts:
                counts_seen.add(tuple(int(x) for x in counts))
    mean_wall = statistics.mean(probe_walls) if probe_walls else None
    median_wall = statistics.median(probe_walls) if probe_walls else None
    mean_speedup = statistics.mean(speedups) if speedups else None
    rows.append({
        **run,
        "probe_calls": len(probe_walls),
        "mean_wall": mean_wall,
        "median_wall": median_wall,
        "mean_speedup": mean_speedup,
        "devices_seen": sorted(devices_seen),
        "trial_splits": [list(x) for x in sorted(counts_seen)],
    })

completed = [r for r in rows if r["rc"] == 0 and r["mean_wall"] is not None]
best = min(completed, key=lambda r: r["mean_wall"]) if completed else None
(root / "benchmark_summary.json").write_text(
    json.dumps({"runs": rows, "best": best}, indent=2, ensure_ascii=False)
)
if best:
    (root / "best_batch_size.txt").write_text(str(best["batch_size"]) + "\n")

def fmt(x):
    return "" if x is None else f"{x:.4f}"

trs = []
for r in rows:
    trs.append(
        "<tr>"
        f"<td>{html.escape(r['label'])}</td>"
        f"<td>{r['batch_size']}</td>"
        f"<td>{r['gpu_count']}</td>"
        f"<td>{html.escape(r['device_spec'])}</td>"
        f"<td>{r['rc']}</td>"
        f"<td>{r['probe_calls']}</td>"
        f"<td>{fmt(r['mean_wall'])}</td>"
        f"<td>{fmt(r['median_wall'])}</td>"
        f"<td>{fmt(r['mean_speedup'])}</td>"
        f"<td>{html.escape(str(r['devices_seen']))}</td>"
        f"<td>{html.escape(str(r['trial_splits']))}</td>"
        "</tr>"
    )

best_html = (
    f"<p><strong>Best observed:</strong> {html.escape(best['label'])}, "
    f"mean probe wall {best['mean_wall']:.4f}s.</p>"
    if best else "<p><strong>Best observed:</strong> none; check failed runs.</p>"
)
page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage-2 Reward Probe GPU Scaling</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45;color:#1f2933}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d8dee4;padding:6px;vertical-align:top}}th{{background:#f6f8fa;text-align:left}}
code{{background:#f6f8fa;padding:2px 4px;border-radius:4px}}
</style></head><body>
<h1>Stage-2 Reward Probe GPU Scaling Benchmark</h1>
<p>This benchmark runs the real Stage-2 RL reward probe path with <code>K=4</code>
trials over the 256-example validation probe subset. For 4 GPUs, the expected
trial split is one independent trial per GPU.</p>
{best_html}
<table><thead><tr><th>run</th><th>batch</th><th>GPUs</th><th>visible devices</th><th>rc</th><th>probe calls</th><th>mean wall s</th><th>median wall s</th><th>mean speedup</th><th>devices seen</th><th>trial splits</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
</body></html>"""
(root / "stage2_reward_probe_scaling_report.html").write_text(page)
print(f"[bench] wrote {root / 'benchmark_summary.json'}")
print(f"[bench] wrote {root / 'stage2_reward_probe_scaling_report.html'}")
if best:
    print(f"[bench] best_batch_size={best['batch_size']} best_label={best['label']}")
PY
