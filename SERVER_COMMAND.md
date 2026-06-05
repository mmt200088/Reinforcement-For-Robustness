# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command  (fusion safe-neighbor curriculum A/B — 两组全训练对比)

```bash
set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."
export HF_HOME=/hy-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data

# ============================================================================
# A/B：fusion-count Stage-2 RL，对比「加 / 不加」block 粒度 safe-neighbor 课程。
#   A = curriculum ON  (--blb-v3-fusion-neighbor-curriculum 1，新默认，本次修复)
#   B = curriculum OFF (--blb-v3-fusion-neighbor-curriculum 0，旧的无限制行为=对照组)
# 唯一变量就是这一个 flag；seed/preset/K/probe/baseline 全相同 → 干净对照。
# 顺序跑（每组独占 4 卡 K=4），跑完一组把产物拷出来，再跑下一组（--fresh 清掉上一组工作目录）。
# 课程在 0.5×EPISODES 后完全打开（mask==开放 mask，全空间可达），所以全训练才能看出
# 「打开后是否仍健康」+「最终配置质量」。
# ----------------------------------------------------------------------------
# EPISODES：A/B 规模旋钮。默认 6000 = 预设自带的「完整一轮」（每组约数小时，崩溃/健康信号
#   足够清楚地分出胜负）。要直接做里程碑级全规模对比就把它改成 60000（每组可能 1-2 天，两组顺序≈数天）。
EPISODES=6000
# ============================================================================

TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/fusion_ab_${TS}"
mkdir -p "$OUT"
CANON_STAGE2="Parting Chapter/stage2"          # 解耦 stage2 工作目录根；combo=bert base mrpc

run_variant () {
  local tag="$1" curr="$2"
  echo "==================== [A/B] variant=$tag curriculum=$curr episodes=$EPISODES ===================="
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash llama_7B_LayerImportance.sh run rl \
    --preset mrpc-blb-stage2-rl \
    --blb-v3-fusion-count-action 1 \
    --blb-v3-fusion-neighbor-curriculum "$curr" \
    --stage2-search-episodes "$EPISODES" \
    --stage2-k-trials 4 \
    --stage2-probe-size 256 \
    --batch-size 512 \
    --blb-v3-reward-devices 0,1,2,3 \
    --fresh 2>&1 | tee "$OUT/${tag}_launch.log"
  # 启动器是后台 nohup，会立刻返回；PID/run dir 写在 <stage2>/LATEST_{PID,RUN_DIR}
  sleep 12
  local pid rundir
  pid="$(cat "${CANON_STAGE2}/LATEST_PID" 2>/dev/null || true)"
  rundir="$(cat "${CANON_STAGE2}/LATEST_RUN_DIR" 2>/dev/null || true)"
  echo "[A/B] $tag launched: PID=$pid  run_dir=$rundir"
  if [ -z "$pid" ]; then echo "[A/B][FATAL] 没拿到 PID，启动可能失败，看 ${tag}_launch.log"; return 1; fi
  # 等后台训练结束（满训练，不设人为超时）
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done
  echo "[A/B] $tag training process $pid exited."
  # 把这一组产物拷出来（排除大 checkpoint），下一组 --fresh 会清掉工作目录
  mkdir -p "$OUT/$tag"
  if [ -n "$rundir" ] && [ -d "$rundir" ]; then
    rsync -a --exclude='*.pt' --exclude='__pycache__' "$rundir/" "$OUT/$tag/run/" 2>/dev/null \
      || cp -r "$rundir" "$OUT/$tag/run"
  else
    echo "[A/B][warn] run_dir 不存在，尝试 glob 解耦 progress"
    cp -r ${CANON_STAGE2}/*/progress "$OUT/$tag/run" 2>/dev/null || true
  fi
  echo "[A/B] $tag artifacts -> $OUT/$tag/run"
  # 拷出后清掉 combo 工作目录，保证下一组绝对从头训练（两组都是 fusion=_fusioncount_v1
  # 变体，不清的话下一组可能 resume 上一组的 checkpoint）。带模式守卫，避免误删。
  if [ -n "$rundir" ] && [[ "$rundir" == *"/stage2/"*mrpc* ]] && [ -d "$rundir" ]; then
    rm -rf "$rundir"; echo "[A/B] cleared working dir $rundir (record/ 归档不受影响)"
  fi
}

run_variant curr_on  1   || { echo "[A/B] curr_on 失败，停止"; exit 1; }
run_variant curr_off 0   || { echo "[A/B] curr_off 失败，但仍尝试对比已完成部分"; }

# A/B 对比报告（torch-free，读两组 episodes.jsonl → 并排 HTML/JSON + 结论）
python3 scripts/blb_fusion_ab_compare.py \
  --run-a "$OUT/curr_on/run"  --label-a "curriculum ON" \
  --run-b "$OUT/curr_off/run" --label-b "curriculum OFF" \
  --anchor 80 --window 200 \
  --out "$OUT/fusion_ab_report.html" 2>&1 | tee "$OUT/ab_compare.log" || true

echo "==================== A/B DONE -> $OUT （把这个目录 commit+push 回来） ===================="
ls -la "$OUT"
```

## metadata

- **本次目标（2026-06-05，用户已批准 + 加约束）**：验证 block 粒度 safe-neighbor 课程是否真的
  解决了 fusion 模式释放 anchor 后的坍塌。两组**全训练**对照：A=加课程（新默认），B=不加（旧无限制）。
- **用户硬约束（已在实现里满足并单测锁死）**：课程**不能永久屏蔽任何配置**，必须能搜到全空间，
  限制只能**逐渐打开**。实现：每 episode 只让少数 block 偏离 baseline，可变 block 数(1→47)和半径(1→6)
  随进度线性放开，到 `0.5×EPISODES` 后 `fully_open` → mask 与开放 mask 逐字节相同。证明见
  `tests/test_blb_fusion_curriculum.py::FullSpaceReachabilityTest`（16 个 torch-free 测试全过）。
- **唯一变量**：`--blb-v3-fusion-neighbor-curriculum 1/0`。其余（preset、seed=42、K=4 四卡 probe、
  probe-size 256、baseline、warmstart）两组完全一致 → 干净对照。
- **隔离方式**：顺序跑，每组独占 4 卡（K=4）。用解耦 canonical 工作目录
  `Parting Chapter/stage2/bert base mrpc/`（这样 Stage-1 前置能正常从 `stage1/record/` 读到）；
  跑完立刻把 `LATEST_RUN_DIR` 指向的产物拷到 `$OUT/<tag>/run`，下一组 `--fresh` 再清掉工作目录。
  完成的 run 会归档到只读 `stage2/record/`（--fresh 不动 record），数据有双重保险。
- **EPISODES 旋钮**：默认 6000（预设自带「完整一轮」，每组数小时，足够分出胜负）。要里程碑级全规模
  就改成 60000（每组 1-2 天，两组顺序≈数天）。坍塞信号 ep~120 就出现，所以即使 6000 也能清楚看到
  B 崩 / A 不崩。
- **判读**：看 `$OUT/fusion_ab_report.html` 的 verdict + 曲线。期望 A：anchor 释放后 P1 不爆、
  reward 维持高位、fusion_count 受控、`loss_mean=100` 罕见；B：复现坍塌（tail P1≈99%、reward≈-5）。
  comparator 的判定：A tail-P1<25% 且 P3>40%，同时 B tail-P1>50% 或 loss-cap>40% → 「课程有用」。
- **回传**：把整个 `experiments/server_command_runs/fusion_ab_<ts>/` commit+push（含两组 run 产物、
  HTML/JSON 对比、launch/compare 日志）。
- **协议**：服务器只 `git pull`、运行、产出/回传 artifacts；源码改动都在本地（本次课程实现已在
  commit `ed797b1`，服务器先 pull 到含该提交的 HEAD 再跑）。

### 上一条已完成（fusion full-build，commit 0122eb2）

7 张 fusion 图已建好并验证通过（block4 跑了 12.7h，`superset_pass=True`，OOM 修复生效，
block1/block4 `[0]→[0,1]`，block5_n2/n4 `[0,1,2]`）。**图无需重建。** 上次包装脚本的
`KeyError: 'options'` 是 heredoc glob 读到了 `_summary.json` sidecar，解析时跳过 `_*.json` 即可
（`FusionCountMap.load` 本来就跳过，所以 RL 运行不受影响）。本次 A/B 直接用这些图。
