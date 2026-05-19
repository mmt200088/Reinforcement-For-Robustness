# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export GLUE_LOCAL_DATASET_DIR="${GLUE_LOCAL_DATASET_DIR:-/hy-tmp/glue_data}"

RUN_ID="stage2_rl_safe_curriculum_$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="experiments/server_command_runs/${RUN_ID}"
PERSIST_ROOT="Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005"
STAGE2_NOISE="${PERSIST_ROOT}/stage2_noise"
export ARTIFACT_DIR STAGE2_NOISE
mkdir -p "$ARTIFACT_DIR" logs
exec > >(tee "${ARTIFACT_DIR}/server_command_stdout.log") 2>&1

echo "[goal] Fix BLB Stage-2 sequential RL collapse as a research loop."
echo "[goal] Success requires no post-anchor loss cap, no sustained P1(acc), positive reward windows, and stable monitored metrics."

stop_rl_at_dir() {
  local PIDFILE="$1/rl.pid"
  [ -f "$PIDFILE" ] || { echo "[stop-rl] $1: no rl.pid"; return 0; }
  local RL_PID
  RL_PID="$(cat "$PIDFILE")"
  if [ -z "$RL_PID" ] || ! kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: pid=$RL_PID already dead"
    return 0
  fi
  echo "[stop-rl] $1: running pid=$RL_PID, SIGINT ..."
  kill -INT "$RL_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5 6; do sleep 10; kill -0 "$RL_PID" 2>/dev/null || break; done
  if kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: still alive after 60s, SIGTERM ..."
    kill -TERM "$RL_PID" 2>/dev/null || true
    for _ in 1 2 3; do sleep 10; kill -0 "$RL_PID" 2>/dev/null || break; done
  fi
  if kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: still alive after 90s, SIGKILL ..."
    kill -KILL "$RL_PID" 2>/dev/null || true
    sleep 3
  fi
  echo "[stop-rl] $1: stopped."
}

stop_rl_at_dir "$PERSIST_ROOT"
stop_rl_at_dir "${PERSIST_ROOT}_rdv2"

echo ""
echo "================================================================================"
echo "Step 1/5: git pull latest local source changes"
echo "================================================================================"
git pull --ff-only
echo "[git] HEAD = $(git rev-parse --short HEAD)"

echo ""
echo "================================================================================"
echo "Step 2/5: local/contract tests on server"
echo "================================================================================"
set +e
python -m unittest tests.test_sequential_smoke.WarmstartFixedRegressionTest tests.test_sequential_smoke.EntCoefScheduleRegressionTest -v 2>&1 | tee "${ARTIFACT_DIR}/test_sequential_smoke.log"
TEST1_RC=${PIPESTATUS[0]}
BLB_STRICT=0 python -m unittest discover -s tests -p "test_blb_*.py" -v 2>&1 | tee "${ARTIFACT_DIR}/test_blb_contracts.log"
TEST2_RC=${PIPESTATUS[0]}
set -e
if [ "$TEST1_RC" -ne 0 ] || [ "$TEST2_RC" -ne 0 ]; then
  echo "[abort] tests failed: sequential=$TEST1_RC contracts=$TEST2_RC"
  exit 10
fi

echo ""
echo "================================================================================"
echo "Step 3/5: GPU visibility"
echo "================================================================================"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${ARTIFACT_DIR}/nvidia_pre_rl.csv"
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "[nvidia-smi] visible GPUs = $N_GPUS"
if [ "$N_GPUS" -lt 2 ]; then
  echo "[abort] need >= 2 GPUs for this run; saw $N_GPUS"
  exit 11
fi

NVS_LOG="${ARTIFACT_DIR}/nvidia_smi_during_rl.csv"
(
  printf "timestamp,gpu_idx,util_pct,mem_used_mib\n" > "$NVS_LOG"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits >> "$NVS_LOG" 2>/dev/null || true
    sleep 15
  done
) &
NVS_PID=$!
trap "kill $NVS_PID 2>/dev/null || true" EXIT

echo ""
echo "================================================================================"
echo "Step 4/5: fresh 600-episode dual-GPU safe-curriculum RL smoke"
echo "================================================================================"
set +e
CUDA_VISIBLE_DEVICES=0,1 bash llama_7B_LayerImportance.sh run rl \
  --preset mrpc-blb-stage2-rl \
  --stage2-search-episodes 600 \
  --stage2-rollout-size 60 \
  --stage2-k-trials 5 \
  --blb-v3-warmstart-anchor-episodes 120 \
  --blb-v3-reward-devices 0,1 \
  --fresh 2>&1 | tee "${ARTIFACT_DIR}/rl_600_dual_gpu.log"
LAUNCH_RC=${PIPESTATUS[0]}
set -e
echo "[rl] launcher rc=$LAUNCH_RC"
if [ "$LAUNCH_RC" -ne 0 ]; then
  kill "$NVS_PID" 2>/dev/null || true
  trap - EXIT
  exit "$LAUNCH_RC"
fi

RL_PID_FILE="${PERSIST_ROOT}/rl.pid"
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  [ -s "$RL_PID_FILE" ] && break
  sleep 2
done
if [ ! -s "$RL_PID_FILE" ]; then
  echo "[fail] launcher returned success but did not write $RL_PID_FILE"
  kill "$NVS_PID" 2>/dev/null || true
  trap - EXIT
  exit 12
fi
RUN_PID="$(cat "$RL_PID_FILE")"
echo "[rl] background pid=$RUN_PID; waiting for completion before monitor"
while kill -0 "$RUN_PID" 2>/dev/null; do
  EPISODES_DONE=0
  if [ -f "${STAGE2_NOISE}/progress/diagnostics/episodes.jsonl" ]; then
    EPISODES_DONE=$(wc -l < "${STAGE2_NOISE}/progress/diagnostics/episodes.jsonl" | tr -d ' ')
  fi
  echo "[rl-monitor] pid=$RUN_PID alive; episodes_jsonl=$EPISODES_DONE; $(date -Is)"
  sleep 60
done
echo "[rl] background pid=$RUN_PID exited; running monitor"
kill "$NVS_PID" 2>/dev/null || true
trap - EXIT
RL_RC=0

echo ""
echo "================================================================================"
echo "Step 5/5: monitor reward/loss/priority/safe-neighbor/GPU evidence"
echo "================================================================================"
set +e
python - <<'PY'
import csv
import glob
import html
import json
import os
import re
from pathlib import Path

artifact = Path(os.environ["ARTIFACT_DIR"])
stage2_noise = Path(os.environ["STAGE2_NOISE"])
anchor = 120
summary = {
    "artifact_dir": str(artifact),
    "stage2_noise": str(stage2_noise),
    "anomalies": [],
    "warnings": [],
    "episodes_seen": 0,
    "post_anchor_episodes_seen": 0,
    "post_anchor_min_return": None,
    "post_anchor_min_loss_mean": None,
    "post_anchor_max_loss_mean": None,
    "post_anchor_p1_count": 0,
    "post_anchor_low_return_count": 0,
    "safe_neighbor_active_count": 0,
    "gpu_max_util": {},
    "ppo_updates": [],
}

detail_paths = sorted((stage2_noise / "details").glob("noise_ppo_step_info_*.txt"))
episode_re = re.compile(r"episode\s+(\d+).*?episode_return=([+-]?\d+(?:\.\d+)?)")
priority_re = re.compile(r"priority=(P\d)")
loss_re = re.compile(r"loss_mean=([+-]?\d+(?:\.\d+)?)")
safe_re = re.compile(r"safe_neighbor:\s+active=(True|False)\s+mutated_offsets=(\d+)\s+radius=(\d+)")
episodes = []
for path in detail_paths:
    text = path.read_text(encoding="utf-8", errors="replace")
    current = None
    for line in text.splitlines():
        m = episode_re.search(line)
        if m:
            current = {
                "episode": int(m.group(1)),
                "return": float(m.group(2)),
                "priority": None,
                "loss_mean": None,
                "safe_neighbor_active": False,
                "safe_neighbor_mutations": None,
                "safe_neighbor_radius": None,
            }
            pm = priority_re.search(line)
            if pm:
                current["priority"] = pm.group(1)
            episodes.append(current)
            continue
        if current is None:
            continue
        lm = loss_re.search(line)
        if lm:
            current["loss_mean"] = float(lm.group(1))
        sm = safe_re.search(line)
        if sm:
            current["safe_neighbor_active"] = sm.group(1) == "True"
            current["safe_neighbor_mutations"] = int(sm.group(2))
            current["safe_neighbor_radius"] = int(sm.group(3))

summary["episodes_seen"] = len(episodes)
post = [e for e in episodes if e["episode"] > anchor]
summary["post_anchor_episodes_seen"] = len(post)
if post:
    returns = [e["return"] for e in post]
    losses = [e["loss_mean"] for e in post if e["loss_mean"] is not None]
    summary["post_anchor_min_return"] = min(returns)
    summary["post_anchor_low_return_count"] = sum(1 for value in returns if value < 20.0)
    if losses:
        summary["post_anchor_min_loss_mean"] = min(losses)
        summary["post_anchor_max_loss_mean"] = max(losses)
    summary["post_anchor_p1_count"] = sum(1 for e in post if e["priority"] == "P1")
    summary["safe_neighbor_active_count"] = sum(1 for e in post if e["safe_neighbor_active"])
else:
    summary["anomalies"].append("No post-anchor episodes found in details logs.")

if any(e.get("loss_mean") is not None and e["loss_mean"] >= 99.0 for e in post):
    summary["anomalies"].append("Post-anchor loss_mean reached collapse cap >=99.")
if summary["post_anchor_p1_count"]:
    summary["anomalies"].append(f"Post-anchor P1(acc) episodes found: {summary['post_anchor_p1_count']}.")
if post and summary["safe_neighbor_active_count"] == 0:
    summary["anomalies"].append("No post-anchor safe_neighbor active episodes found.")

warning_path = stage2_noise / "warning.txt"
if warning_path.exists():
    warning_text = warning_path.read_text(encoding="utf-8", errors="replace")
    if "loss_mean=100" in warning_text or "P1" in warning_text:
        summary["anomalies"].append("warning.txt contains collapse/P1 evidence.")
    if warning_text.strip():
        summary["warnings"].append(warning_text[-4000:])

ppo_path = stage2_noise / "progress" / "diagnostics" / "ppo_updates.jsonl"
if ppo_path.exists():
    for line in ppo_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        summary["ppo_updates"].append(row)
    for row in summary["ppo_updates"]:
        ep = int(row.get("completed_episodes", row.get("episode", 0)) or 0)
        mean_ret = row.get("window_mean_return", row.get("mean_return"))
        if ep > anchor and mean_ret is not None and float(mean_ret) < 20.0:
            summary["anomalies"].append(f"PPO window after anchor has mean return < +20 at episode {ep}: {mean_ret}.")

nvs_path = artifact / "nvidia_smi_during_rl.csv"
if nvs_path.exists():
    with nvs_path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            idx = str(row.get("gpu_idx", "")).strip()
            try:
                util = float(str(row.get("util_pct", "0")).strip())
            except Exception:
                util = 0.0
            if idx:
                summary["gpu_max_util"][idx] = max(summary["gpu_max_util"].get(idx, 0.0), util)
if len(summary["gpu_max_util"]) < 2:
    summary["anomalies"].append("nvidia-smi sampler did not capture two GPUs.")
elif any(float(v) <= 0 for v in summary["gpu_max_util"].values()):
    summary["anomalies"].append(f"At least one GPU had zero max util: {summary['gpu_max_util']}.")

summary_path = artifact / "monitor_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

rows = [
    ("episodes_seen", summary["episodes_seen"]),
    ("post_anchor_episodes_seen", summary["post_anchor_episodes_seen"]),
    ("post_anchor_min_return", summary["post_anchor_min_return"]),
    ("post_anchor_low_return_count", summary["post_anchor_low_return_count"]),
    ("post_anchor_max_loss_mean", summary["post_anchor_max_loss_mean"]),
    ("post_anchor_p1_count", summary["post_anchor_p1_count"]),
    ("safe_neighbor_active_count", summary["safe_neighbor_active_count"]),
    ("gpu_max_util", summary["gpu_max_util"]),
    ("anomaly_count", len(summary["anomalies"])),
]
html_body = "\n".join(
    f"<tr><th>{html.escape(str(k))}</th><td><pre>{html.escape(json.dumps(v, ensure_ascii=False, indent=2))}</pre></td></tr>"
    for k, v in rows
)
anoms = "\n".join(f"<li>{html.escape(x)}</li>" for x in summary["anomalies"]) or "<li>None</li>"
(artifact / "server_monitor_report.html").write_text(
    "<!doctype html><meta charset='utf-8'><title>Stage2 RL safe curriculum monitor</title>"
    "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;line-height:1.45;margin:28px}"
    "table{border-collapse:collapse;width:100%;margin-top:16px}th,td{border:1px solid #ddd;padding:8px;vertical-align:top}"
    "th{text-align:left;background:#f6f6f6;width:260px}pre{white-space:pre-wrap;margin:0}</style>"
    "<h1>Stage2 RL safe curriculum monitor</h1>"
    f"<h2>Anomalies</h2><ul>{anoms}</ul><h2>Summary</h2><table>{html_body}</table>",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if summary["anomalies"]:
    raise SystemExit(20)
PY
MONITOR_RC=$?
set -e

cp "${STAGE2_NOISE}/pruning_search_log.txt" "${ARTIFACT_DIR}/pruning_search_log_tail_source.txt" 2>/dev/null || true
cp "${STAGE2_NOISE}/warning.txt" "${ARTIFACT_DIR}/warning.txt" 2>/dev/null || true
cp "${STAGE2_NOISE}/progress/diagnostics/ppo_updates.jsonl" "${ARTIFACT_DIR}/ppo_updates.jsonl" 2>/dev/null || true
cp "${STAGE2_NOISE}/progress/diagnostics/episodes.jsonl" "${ARTIFACT_DIR}/episodes.jsonl" 2>/dev/null || true

git add "$ARTIFACT_DIR"
git commit -m "Add server safe-curriculum RL monitor results" || true
git push || true

exit "$MONITOR_RC"
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：在 mrpc-blb-stage2-rl preset 下，依次跑（a）sequential smoke + BLB 契约测试，（b）nvidia-smi 双卡可见性确认，（c）fresh 600-episode sequential RL 烟测，启用两卡奖励探针并行（`--blb-v3-reward-devices 0,1`），并自动监控 post-anchor reward/loss/priority/safe_neighbor/GPU 使用。
- **更新时间**：2026-05-20
- **当前 goal**：这不是一次性短修。最终目标是 RL 能在 anchor 后持续正常训练，不出现 reward 崩塌、`loss_mean=100`、持续 P1(acc)、或 PPO/GPU/safe-neighbor 监控指标的异常跳变。若本次烟测失败，服务器只产出证据；真实代码修复继续在本地完成后通过 git 同步。
- **更新原因**：验证 2026-05-20 safe sequential curriculum 修复：
    1. sequential anchor 现在尊重 `warmstart_anchor_episodes`，除非显式设置 `force_baseline_episodes`。
    2. anchor/entropy/resume 使用 absolute episode，避免 resume 后重新进入 anchor。
    3. post-anchor 采样从 unrestricted 改为 baseline-neighborhood curriculum：每个 episode 只开放少量 effective full-vector offsets，其他 slots baseline-only。
    4. K 邻域按非单调 `K_LEVELS` 的值序处理，不按 action index 单调性假设处理。
    5. PPO buffer 保存并 replay 每个 transition 的 `action_level_mask`，PPO ratio 使用采样时同一 action support。
    6. details/status 增加 `safe_neighbor` 诊断，方便服务器端监控 mutated_offsets/radius。
- **历史背景**：上一次服务器崩塌发生在 episode 121：episodes 1-120 是 forced baseline，episode 121 是第一个 sampled action，`any_invalid=False` 但 `loss_mean=100` 且 priority=P1(acc)。这说明 optimizer invalid blacklist 不够，必须限制 terminal model-forward accuracy gate 前的探索空间。
- **历史相关修复**（本轮仍会被契约测试覆盖）：
    1. **2026-05-18 entropy schedule fix** (commit `0ca6de0`)：anchor 期 ent_coef=0，ramp 240ep，steady 0.02。
    2. **2026-05-19 policy init + Huber value loss** (commit `50ea91a`)：action_head orthogonal(0.01)、encoder √2、value_head 1.0；value loss MSE→Huber(δ=1)。
    3. **2026-05-19 two-GPU reward probe** (commit `54365b5`)：`blb_stage2_rl/probe_runner.py` + env/runner/sequential_runner/rl_tune/evaluator/launcher 串通；`--blb-v3-reward-devices 0,1` 即可两卡并行 K 个 trial。worker 0 复用 env 既有 model/handler/bridge；worker 1 deepcopy 主模型到 cuda:1，独立 handler/bridge/probe_batches；threading 并发，trial 顺序保持。**注：codex agent 仍在排查为何上一轮基准测试两卡 util 没起来**（详见 `experiments/server_command_runs/stage2_reward_probe_benchmark_20260519_202236/`）；本次 RL 训练会继续暴露同一问题供后续修。
    4. **2026-05-20 contract gate fix** (commit `46d9b01`)：修上一轮 contract gate 的 8 fail + 1 error，全部是 action→config 链路 + stale 测试。具体：
        - `max_sfs/mrpc.json` block 2 节点名从 `ctpt_kt_mask1/kt_mask2/qkt_merge_mask/wk/wq/q_mask{1,2}` 改成 `ctpt_rotKT_mask1/rotKT_mask2/mask/wq_wk/x_centered`，与 bridge 的 `default_block2_cfg_to_delta` 一致 —— 上一轮 `test_real_mrpc_all_max_*` 失败的根因（命名不一致 → 查表 miss → SF 退回默认 22 → optimizer 算出 q_bits=[51,66,56] invalid）。
        - `_BLOCK*_FIELDS` 把 2026-05-14 被删的 25 个 slot 以 *compat-extra* 方式恢复（block1+2、block2+11、block3+1、block4+5、block5+6 = 73/层），匹配 CLAUDE.md "rather than deleting" 指引；新增 `_COMPAT_EXTRA_FIELDS` set 让 `_is_action_field_effective` 把它们标 effective=False，cfg-build 路径继续把对应字段写 None。
        - `BASELINE_K_BY_BLOCK` 从 `{2:10, 4:10}` 还原成统一 13，让 all-max action 的 avg_k=13.0 与 baseline 一致（上一轮 `test_env_all_max` reward=41.5 ≠ 0 / `test_action_description` block2 K=13 都是这个引起的）。
        - `describe_action_vector` 给 L0B1 的所有 record 设 `value=None`（block 1 在 layer 0 不安装噪声，decoded SF 是默认值的伪信号）—— 这样 baseline-decode 测试的过滤器 `value is not None` 能正确排除它们。
        - `MaxSFsTable.get` 在 `_BLOCK_NODE_NAME_BY_FIELD` 没有映射时回退到 field_name 当节点名 —— 让 `static_skeletons_baseline_to_action` 注入到 `(layer, block, field_name)` 的 calibration 在 describe 里读得到（test_block4_wo_rescale 走的就是这条路）。
        - `env.py` any_invalid short-circuit 的 placeholder metrics 抬到 acc/stab 阈值之上，使 `priority=3` 与 docstring 承诺的 "cost-only reward" 一致（之前用 baseline 默认 0 会触发 acc_violation 把 priority 错降到 1）。
        - 两条 stale test 同步：`test_env_runs_forward_even_when_optimizer_invalid` 现在断言 `forward_ran=False / forward_skipped_reason="any_invalid_chain"` 匹配 2026-05-17 skip-forward 设计；`test_candidate_store_hash_fidelity` F2→F4 匹配 2026-05-16 F0/F1/F4 ladder；`test_env_all_max_action_uses_optimizer_baseline_scoring` 不再断言 reward=0（v2 reward 在 metric_ok+stab_ok 时强制 tier_bonus=40），改为断言 `cost_score/k_drop/bits_drop=0` + reward∈[35,45]。
- **本次改动汇总**（multi-GPU 部分）：
    1. `blb_stage2_rl/probe_runner.py`（新文件，~360 行）：`ProbeWorker` / `ProbeRunner` / `build_probe_runner` / `parse_device_ids` / `_split_round_robin` / `_trial_seed`。线程并发 + 日志诊断。
    2. `blb_stage2_rl/env.py`：`BLBStage2Env.__init__` 加可选 `probe_runner` 参数；step 的 install/clear、`_eval_on_probe` 在多卡模式下转给 runner，单卡模式 bitwise 不变。
    3. `blb_stage2_rl/runner.py`：`BLBStage2TrainConfig.reward_devices` 字段；`_apply_runtime_overrides_to_cfg` 从 `ev.blb_v3_reward_devices` 解析；`BLBStage2RLRunner.run` 在 env 构造后若 len(reward_devices)≥2 则 `build_probe_runner(primary_bridge=env.bridge, …)` 注入到 `env.probe_runner`。
    4. `blb_stage2_rl/sequential_runner.py`：在 `train_sequential` 的 base_env 构造后同样钩入 ProbeRunner（与 runner.py 对称，逻辑共享）。
    5. `rl_tune.py`：新增 `blb_v3_reward_devices=""` 形参，转发给 `LayerImportanceEvaluator`。
    6. `layer_importance_evaluator.py`：`__init__` 新增 `blb_v3_reward_devices=""` 形参 + 存到 `self.blb_v3_reward_devices`。
    7. `llama_7B_LayerImportance.sh`：新增 `--blb-v3-reward-devices STR` 启动器开关 + 透传到 `python rl_tune.py --blb_v3_reward_devices`。
    8. `tests/test_blb_chain_integrity.py`：新增 `ProbeRunnerHelpersTest`（8 个 pure-Python case）+ `ProbeRunnerTwoGPUTest`（双卡 smoke，自动 skipUnless ≥2 CUDA 设备）。本地全部 skip 通过；服务器有双卡时会真跑。
- **预期信号**（按强→弱排）：
    - **契约测试全部通过**（上一轮 99 个 case 里有 8 fail + 1 error，全部在 commit `46d9b01` 修复；本轮期望 fail=0 / error=0；29 个 chain-integrity case 继续全绿）。
    - **`nvidia-smi` 启动前快照显示 GPU 0/1 都在线，mem.used 极低**。
    - **训练启动后 `logs/nvidia_smi_<ts>.csv` 显示 GPU 0/1 两列 util_pct 都长期 > 0**（不是只有 GPU 0 在 100%、GPU 1 长期 0%）；典型节奏是模型 forward 期间两卡同步上 80%+，optimizer/PPO update 期间两卡都回落（因为这阶段单线程）。
    - **训练日志的 startup banner 含 `Multi-GPU reward probe enabled: devices=[0, 1]` 和 `worker 0/1` 行**（前者来自 runner.py 的 log，后者来自 build_probe_runner 的 log_fn）。
    - **每个 episode 的 wall-clock 应比同 preset 单卡基线快 ~1.5x–1.9x**（理论上限 2x，扣掉 deepcopy 后单次 worker 启动 + Python 线程 join 开销）。可以从 `details/noise_ppo_step_info_*.txt` 的时间戳 diff 看。
    - **本次同时验证 policy init + Huber 修复**：anchor 期 `entropy` 应平稳下降，sample 期（eps 120+）reward ≥ +25，clip_fraction << 0.7。
- **预期产物**：
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/`
        - `blb_stage2_status.json` / `diagnostics_summary.md` / `blb_stage2_rl_checkpoint_live.pt`
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/`
        - `details/noise_ppo_step_info_*.txt` / `warning.txt` / `pruning_search_log.txt`
    - `logs/nvidia_smi_<ts>.csv`（**这个是 multi-GPU 用量证据**；每 15s 采样，包含两卡 util_pct + mem_used_mib）
    - `/tmp/blb_test_output.log`（契约测试日志，agent 失败时贴这个）
- **预期耗时**：~5-6 小时（单卡 ~9 小时 → 双卡奖励探针约 1.7x 加速；首 60 ep 锚定阶段加速最少，因为 forced-baseline 时 K 个 trial 都跑 baseline 行动）。

## 切换到其他常用任务时（备查，agent 不读这一段）

需要换任务时，**直接覆盖上面的 active command 代码块** + 改这里的元信息。下面只是常用命令样板，不会被执行：

- 续训（同 preset 不带 `--fresh`，自动检测持久化目录、恢复 forbidden_mask）：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --blb-v3-reward-devices 0,1`
- 单独 final-eval（最新 best）：
  `bash Paean/run_final_eval.sh --preset mrpc-final-eval-only --action-config "$RUN_DIR/stage2_noise/progress/diagnostics/best_action_vec.json"`
- 单卡回滚（不传 `--blb-v3-reward-devices` 即可，逻辑等价于历史路径）：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh`
- 离线诊断某个 action 的 invalid_blocks：
  `python scripts/blb_diagnose_invalid_blocks.py --action-config <path> --output-dir reports/blb_opt/invalid_blocks/<name>`
- 多 seed 扫（5 seeds，隔离持久化目录）：
  `bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 1,2,3,4,5 trial1 --fresh`

## 服务器 agent 期望

- agent 只读这个文件的**第一个 ```bash 代码块**，其余 markdown 全部忽略。
- agent 应该在仓库根目录 `bash` 执行（不要 `cd`，所有路径已经按相对仓库根写好）。
- 如果该文件未变更（git hash 未动），agent 不应重复触发同一命令 —— 由 agent 侧做幂等。
- 本次脚本会主动停掉正在跑的 RL（基于 `<slug>/rl.pid`），所以 agent 不需要额外的 pre-kill 钩子。
- `set -e` + `trap kill $NVS_PID` 已经处理掉了背景 nvidia-smi 采样进程的清理 —— 即便 RL 中途崩溃，sampler 也会被信号杀掉。
