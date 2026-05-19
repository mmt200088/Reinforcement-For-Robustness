# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -e

# ----------------------------------------------------------------------
# 0) GPUShare server env (per CLAUDE.md "GPUShare server state" section).
#    Safe no-op on other environments — these are exports only.
# ----------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export GLUE_LOCAL_DATASET_DIR="${GLUE_LOCAL_DATASET_DIR:-/hy-tmp/glue_data}"

# ----------------------------------------------------------------------
# 1) 优雅停掉前一轮（如果还在跑）。同时扫主目录 + 历史 _rdv2 临时目录，
#    防止服务器上有任何残留进程。
# ----------------------------------------------------------------------
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
  for i in 1 2 3 4 5 6; do sleep 10; kill -0 "$RL_PID" 2>/dev/null || break; done
  if kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: still alive after 60s, SIGTERM ..."
    kill -TERM "$RL_PID" 2>/dev/null || true
    for i in 1 2 3; do sleep 10; kill -0 "$RL_PID" 2>/dev/null || break; done
  fi
  if kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: still alive after 90s, SIGKILL ..."
    kill -KILL "$RL_PID" 2>/dev/null || true
    sleep 3
  fi
  echo "[stop-rl] $1: stopped."
}
stop_rl_at_dir "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005"
stop_rl_at_dir "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005_rdv2"

# ----------------------------------------------------------------------
# 2) Pull latest code so the server runs the policy-init + Huber + multi-GPU
#    probe commits. CLAUDE.md "GPUShare server state" notes the local
#    HTTP/1.1 + protocol.version 0 git config; if a fresh checkout is needed
#    those settings must already be in repo .git/config.
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "Step 1/4: git pull (refresh local repo with latest local commits)"
echo "================================================================================"
git pull --ff-only || {
  echo "[warn] git pull --ff-only failed; continuing with whatever HEAD has."
}
echo "[git] HEAD = $(git rev-parse --short HEAD)"

# ----------------------------------------------------------------------
# 3) Contract tests. Gates the RL run — any red test aborts before the
#    9-hour training. Now includes (a) prior chain-integrity + sequential
#    policy init tests, (b) the new ProbeRunner helpers (split / seed /
#    parse_device_ids), and (c) the two-GPU smoke test that only runs when
#    nvidia exposes >= 2 cards.
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "Step 2/4: contract tests (chain + policy init + probe runner)"
echo "================================================================================"
BLB_STRICT=0 python -m unittest discover -s tests -p "test_blb_*.py" -v 2>&1 | tee /tmp/blb_test_output.log
TEST_RC=${PIPESTATUS[0]}
if [ "$TEST_RC" -ne 0 ]; then
  echo ""
  echo "[abort] contract tests FAILED (rc=$TEST_RC). Not starting RL." >&2
  echo "        Full log: /tmp/blb_test_output.log" >&2
  exit "$TEST_RC"
fi
echo "[ok] contract tests passed."

# ----------------------------------------------------------------------
# 4) Verify both GPUs are visible. Snapshot before training so the log
#    captures the starting state. The active-utilisation proof comes from
#    the background sampler in step 5 below.
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "Step 3/4: nvidia-smi snapshot (pre-RL — both GPUs must be visible & free)"
echo "================================================================================"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "[nvidia-smi] visible GPUs = $N_GPUS"
if [ "$N_GPUS" -lt 2 ]; then
  echo "[abort] need >= 2 GPUs for multi-GPU probe; saw $N_GPUS. Re-export CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

# ----------------------------------------------------------------------
# 5) Launch BLB Stage-2 sequential RL (fresh) with two-GPU reward probe.
#
#    Flags vs the prior single-GPU run:
#      + --blb-v3-reward-devices 0,1
#        → BLBStage2RLRunner builds a ProbeRunner that fans the K=5 trials
#          across cuda:0 and cuda:1 (split [0,2,4] / [1,3]); per-trial seeds
#          are deterministic from base_seed+trial_idx.
#      + CUDA_VISIBLE_DEVICES=0,1
#        → both cards visible to torch; without this only cuda:0 is exposed.
#
#    Carrying over from prior fixes (commits ebca10d / ed66325 / 50ea91a):
#      · entropy schedule (anchor=0, ramp 240ep, steady=0.02)
#      · action_head orthogonal(0.01) init (warmstart bias survives encoder drift)
#      · Huber value loss (caps shared-trunk perturbation from huge raw returns)
#      · contract tests on apply_optimizer_output_to_cfg / Q-K sync / live SF read
#      · per-slot mode warmstart bias gain=3.5
#      · forced baseline anchor (120 ep)
#      · v2-style clipped+tier reward
#      · ForbiddenActionMask + rejection-sample
#
#    Background sampler proves both GPUs are actually busy (every 15s logs
#    util% + mem.used). File lives next to logs/ so it's grep-friendly.
# ----------------------------------------------------------------------
mkdir -p logs
NVS_LOG="logs/nvidia_smi_$(date +%Y%m%d_%H%M%S).csv"
echo "[nvidia-smi-sampler] writing every 15s → $NVS_LOG"
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
echo "Step 4/4: BLB Stage-2 Sequential RL (fresh) — --blb-v3-reward-devices 0,1"
echo "================================================================================"
CUDA_VISIBLE_DEVICES=0,1 bash llama_7B_LayerImportance.sh run rl \
  --preset mrpc-blb-stage2-rl \
  --stage2-k-trials 5 \
  --blb-v3-reward-devices 0,1 \
  --fresh
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：在 mrpc-blb-stage2-rl preset 下，依次跑（a）契约测试套件、（b）nvidia-smi 双卡可见性确认、（c）fresh 一轮 6000-episode sequential RL，**首次启用两卡奖励探针并行**（`--blb-v3-reward-devices 0,1`）。
- **更新时间**：2026-05-20
- **更新原因**：合并四项尚未在服务器上跑过的修复 + 新功能：
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
