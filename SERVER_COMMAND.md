# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."
export HF_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data
TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage1_ppo_queue_entropy0p1_${TS}"
STATE="/hy-tmp/stage1_ppo_queue_entropy0p1_${TS}"
mkdir -p "$OUT" "$STATE/logs"
SOURCE_COMMIT=$(git rev-parse HEAD)
{
  echo "HEAD=$SOURCE_COMMIT"
  echo "OUT=$OUT"
  echo "STATE=$STATE"
} | tee "$OUT/commit.txt"

cat > "$STATE/stage1_ppo_queue.sh" <<'QUEUE'
#!/usr/bin/env bash
set -u -o pipefail

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."
export HF_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data
export CUDA_VISIBLE_DEVICES=0,1,2,3

: "${STATE_DIR:?missing STATE_DIR}"
: "${OUT_DIR:?missing OUT_DIR}"
: "${SOURCE_COMMIT:?missing SOURCE_COMMIT}"

TASKS=(
  "base_rte|bert-base-rte-stage1-rl"
  "base_sst2|bert-base-sst2-stage1-rl"
  "large_mrpc|bert-large-mrpc-stage1-rl"
  "large_rte|bert-large-rte-stage1-rl"
  "large_sst2|bert-large-sst2-stage1-rl"
)

write_status() {
  local phase="$1"
  local task="${2:-}"
  local preset="${3:-}"
  local pid="${4:-}"
  local launch_log="${5:-}"
  local train_log="${6:-}"
  local run_dir="${7:-}"
  local message="${8:-}"
  python3 - "$STATE_DIR/status.json" \
    "$phase" "$task" "$preset" "$pid" "$launch_log" "$train_log" "$run_dir" "$message" <<'PY'
import datetime
import json
import sys

path, phase, task, preset, pid, launch_log, train_log, run_dir, message = sys.argv[1:10]
payload = {
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_commit": "__SOURCE_COMMIT__",
    "phase": phase,
    "task": task,
    "preset": preset,
    "training_pid": pid,
    "launch_log": launch_log,
    "training_log": train_log,
    "run_dir": run_dir,
    "message": message,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
  python3 - "$STATE_DIR/status.json" "$SOURCE_COMMIT" <<'PY'
import json
import sys

path, source_commit = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
payload["source_commit"] = source_commit
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
  cp -f "$STATE_DIR/status.json" "$OUT_DIR/status.json" 2>/dev/null || true
}

extract_after_colon() {
  local pattern="$1"
  local file="$2"
  sed -n "s/.*${pattern}：\\(.*\\)$/\\1/p" "$file" | tail -1
}

run_one() {
  local task="$1"
  local preset="$2"
  local launch_log="$STATE_DIR/logs/${task}_launch.log"
  local train_log=""
  local run_dir=""
  local pid=""

  write_status "launching" "$task" "$preset" "" "$launch_log" "" "" "starting launcher"
  {
    echo "=== TASK $task preset=$preset start $(date -Is) ==="
    echo "source_commit=$SOURCE_COMMIT"
    printf 'command='
    printf '%q ' bash llama_7B_LayerImportance.sh run rl \
      --preset "$preset" \
      --stage1-search-episodes 0 \
      --stage1-entropy-stop-threshold 0.1 \
      --stage1-rl-devices 0,1,2,3 \
      --rl-algo ppo \
      --fresh
    printf '\n'
    bash llama_7B_LayerImportance.sh run rl \
      --preset "$preset" \
      --stage1-search-episodes 0 \
      --stage1-entropy-stop-threshold 0.1 \
      --stage1-rl-devices 0,1,2,3 \
      --rl-algo ppo \
      --fresh
    launcher_rc=$?
    echo "launcher_rc=$launcher_rc"
    exit "$launcher_rc"
  } > "$launch_log" 2>&1
  local launcher_rc=$?
  if [ "$launcher_rc" -ne 0 ]; then
    write_status "failed" "$task" "$preset" "" "$launch_log" "" "" "launcher failed rc=$launcher_rc"
    return "$launcher_rc"
  fi

  pid=$(extract_after_colon "进程号（PID）" "$launch_log")
  train_log=$(sed -n "s/.*查看日志：tail -f \\(.*\\)$/\\1/p" "$launch_log" | tail -1)
  local latest_run_file
  latest_run_file=$(extract_after_colon "LATEST_RUN_DIR" "$launch_log")
  if [ -n "$latest_run_file" ] && [ -f "$latest_run_file" ]; then
    run_dir=$(cat "$latest_run_file")
  fi
  if [ -z "$pid" ]; then
    write_status "failed" "$task" "$preset" "" "$launch_log" "$train_log" "$run_dir" "could not parse training pid"
    return 90
  fi

  write_status "running" "$task" "$preset" "$pid" "$launch_log" "$train_log" "$run_dir" "training pid running"
  echo "$task|$preset|$pid|$launch_log|$train_log|$run_dir" >> "$STATE_DIR/task_pids.tsv"

  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    write_status "running" "$task" "$preset" "$pid" "$launch_log" "$train_log" "$run_dir" "training pid still running"
  done

  sleep 10
  cp -f "$launch_log" "$OUT_DIR/${task}_launch.log" 2>/dev/null || true
  [ -n "$train_log" ] && [ -f "$train_log" ] && tail -300 "$train_log" > "$OUT_DIR/${task}_train_tail.log" || true

  if [ -n "$train_log" ] && [ -f "$train_log" ] && grep -q "Stage-1 entropy convergence reached" "$train_log"; then
    write_status "completed" "$task" "$preset" "$pid" "$launch_log" "$train_log" "$run_dir" "entropy convergence reached"
    echo "TASK_COMPLETED $task $(date -Is)" >> "$STATE_DIR/events.log"
    return 0
  fi

  if [ -n "$run_dir" ] && [ -f "$run_dir/COMPLETED" ]; then
    write_status "completed" "$task" "$preset" "$pid" "$launch_log" "$train_log" "$run_dir" "COMPLETED marker present"
    echo "TASK_COMPLETED_MARKER $task $(date -Is)" >> "$STATE_DIR/events.log"
    return 0
  fi

  write_status "failed" "$task" "$preset" "$pid" "$launch_log" "$train_log" "$run_dir" "training exited without entropy convergence marker"
  return 91
}

main() {
  echo "queue_start=$(date -Is)" | tee "$STATE_DIR/events.log"
  echo "source_commit=$SOURCE_COMMIT" | tee -a "$STATE_DIR/events.log"
  write_status "starting" "" "" "" "" "" "" "queue starting"
  for spec in "${TASKS[@]}"; do
    IFS='|' read -r task preset <<< "$spec"
    run_one "$task" "$preset"
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "queue_failed task=$task rc=$rc time=$(date -Is)" | tee -a "$STATE_DIR/events.log"
      write_status "failed" "$task" "$preset" "" "$STATE_DIR/logs/${task}_launch.log" "" "" "queue stopped on task failure rc=$rc"
      cp -f "$STATE_DIR/status.json" "$OUT_DIR/status.json" 2>/dev/null || true
      cp -f "$STATE_DIR/events.log" "$OUT_DIR/events.log" 2>/dev/null || true
      exit "$rc"
    fi
  done
  echo "queue_completed=$(date -Is)" | tee -a "$STATE_DIR/events.log"
  write_status "completed" "" "" "" "" "" "" "all tasks completed"
  cp -f "$STATE_DIR/status.json" "$OUT_DIR/status.json" 2>/dev/null || true
  cp -f "$STATE_DIR/events.log" "$OUT_DIR/events.log" 2>/dev/null || true
}

main "$@"
QUEUE

chmod +x "$STATE/stage1_ppo_queue.sh"
STATE_DIR="$STATE" OUT_DIR="$OUT" SOURCE_COMMIT="$SOURCE_COMMIT" \
  nohup bash "$STATE/stage1_ppo_queue.sh" > "$STATE/logs/queue_wrapper.log" 2>&1 &
QUEUE_PID=$!
echo "$QUEUE_PID" > "$STATE/queue.pid"
{
  echo "HEAD=$SOURCE_COMMIT"
  echo "queue_pid=$QUEUE_PID"
  echo "state_dir=$STATE"
  echo "out_dir=$OUT"
  echo "tasks=base_rte,base_sst2,large_mrpc,large_rte,large_sst2"
  echo "algorithm=ppo"
  echo "stage1_search_episodes=0"
  echo "stage1_entropy_stop_threshold=0.1"
  echo "stage1_rl_devices=0,1,2,3"
  echo "fresh=true"
  echo "status_json=$STATE/status.json"
  echo "wrapper_log=$STATE/logs/queue_wrapper.log"
} | tee "$OUT/SUMMARY.txt"
echo "=== Stage-1 PPO convergence queue launched ==="
```

## metadata

- **任务**：启动 Stage-1 PPO 收敛队列，串行运行
  `bert-base-rte-stage1-rl`、`bert-base-sst2-stage1-rl`、
  `bert-large-mrpc-stage1-rl`、`bert-large-rte-stage1-rl`、
  `bert-large-sst2-stage1-rl`。
- **算法**：只用 PPO。GRPO 已永久禁用，命令显式传 `--rl-algo ppo`。
- **停止条件**：每个任务都用 `--stage1-search-episodes 0` 和
  `--stage1-entropy-stop-threshold 0.1`，即不设 episode 上限，PPO update 后
  policy entropy 低于 `0.1` 才算收敛完成。
- **运行方式**：队列 wrapper 写到 `/hy-tmp/stage1_ppo_queue_entropy0p1_<ts>/`，
  服务器 agent 启动后立即返回；wrapper 解析每个 launcher 输出的训练 PID，
  轮询该 PID，看到 `Stage-1 entropy convergence reached` 或 `COMPLETED`
  marker 后启动下一个任务。
- **状态文件**：`/hy-tmp/stage1_ppo_queue_entropy0p1_<ts>/status.json`、
  `events.log`、`logs/*_launch.log` 和 `queue_wrapper.log`。本地同步摘要在
  `experiments/server_command_runs/stage1_ppo_queue_entropy0p1_<ts>/SUMMARY.txt`。
- **协议**：服务器只 `git pull`、运行、产出/回传 artifacts；源码改动都在本地。把 `$OUT/` 回传本地。
