# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."
export HF_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data
TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/fusion_smoke_${TS}"
mkdir -p "$OUT"
echo "HEAD=$(git rev-parse HEAD)" | tee "$OUT/commit.txt"

echo "=== [0/2] confirm canonical fusion maps are present (load smoke, torch-free) ==="
python tests/test_blb_fusion_count_map.py -v 2>&1 | tee "$OUT/fusion_unit.log"
G0=${PIPESTATUS[0]}
ls -la blb_stage2_rl/fusion_maps/mrpc/ 2>&1 | tee -a "$OUT/SUMMARY_pre.txt"

echo "=== [1/2] F1 SMOKE: fusion-count Stage-2 RL, ~300 episodes, 4 GPU ==="
# Validates the fusion runtime path end-to-end: env/policy(max_step_dim=2)/open-mask/
# forced-baseline anchor (option0,K=baseline)/(option,K)->block SF expansion->replan->
# terminal forward+reward. Stage-2-only reads Stage-1 degrees from a Stage-1 record
# (any record works — the map covers gelu {0,1,2,4}; block3 is frozen). If no Stage-1
# record exists, see metadata note (run a short Stage-1 first, or pass --stage2-fixed-config).
CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 3600 bash llama_7B_LayerImportance.sh run rl \
  --mode stage2-only \
  --preset mrpc-blb-stage2-rl \
  --blb-v3-fusion-count-action 1 \
  --stage2-search-episodes 300 \
  --stage2-k-trials 4 \
  --stage2-probe-size 256 \
  --batch-size 512 \
  --blb-v3-reward-devices 0,1,2,3 \
  --fresh 2>&1 | tee "$OUT/smoke.log"
G1=${PIPESTATUS[0]}

RUNDIR=$(grep -oE "Parting Chapter/stage2/[^\"]+/progress" "$OUT/smoke.log" | tail -1 || true)
[ -z "$RUNDIR" ] && RUNDIR=$(ls -dt "Parting Chapter/stage2/"*/progress 2>/dev/null | head -1 || true)
echo "RUNDIR=$RUNDIR" | tee -a "$OUT/commit.txt"
[ -n "$RUNDIR" ] && cp -f "$RUNDIR/diagnostics/episodes.jsonl" "$OUT/episodes.jsonl" 2>/dev/null || true
[ -n "$RUNDIR" ] && cp -f "$RUNDIR/diagnostics/diagnostics_summary.md" "$OUT/diagnostics_summary.md" 2>/dev/null || true

echo "=== [2/2] SUMMARY ==="
{
  echo "HEAD=$(git rev-parse HEAD)"
  echo "fusion_unit_exit=$G0     (0 = map load + torch-free units pass)"
  echo "smoke_exit=$G1           (0 = RL run finished; 124 = timeout)"
  echo "RUNDIR=$RUNDIR"
  echo "--- fusion mode actually engaged? (expect this line) ---"
  grep -i "Fusion-count action ENABLED" "$OUT/smoke.log" | head
  echo "--- 4-GPU reward probe engaged? ---"
  grep -iE "reward probe enabled|probe-runner.*worker|trial split" "$OUT/smoke.log" | head
  echo "--- crashes / fusion-path errors (should be EMPTY) ---"
  grep -iE "Traceback|AttributeError|fusion_num_options|slot_dims|full_vec_offsets|KeyError|ValueError" "$OUT/smoke.log" | head -25
  echo "--- reward / priority / collapse sentinels tail ---"
  grep -iE "reward|loss_mean=100|P1\\(acc\\)|best_reward|episode " "$OUT/smoke.log" | tail -40
  echo "--- episodes.jsonl tail (reward/priority/fusion per episode) ---"
  [ -f "$OUT/episodes.jsonl" ] && tail -5 "$OUT/episodes.jsonl" || echo "(no episodes.jsonl)"
} | tee "$OUT/SUMMARY.txt"
echo "=== DONE -> $OUT ==="
```

## metadata

- **任务**：**Task 8 — fusion-count 运行期 F1 smoke**。验证 Tasks 5–7（schedule + env + runner 接线）的
  fusion 运行期路径端到端能跑：policy `max_step_dim=2`（slot0=fusion_option、slot1=K）、open 掩码、
  forced-baseline anchor 强制 `(option0, baseline-K)`、`(option,K)→block SF vec` 展开 → replan → 终局 forward+reward。
  canonical map 已在仓内（`blb_stage2_rl/fusion_maps/mrpc/*.json`，commit 26441b9）。
- **代码**：本地 Tasks 5–7 已提交（schedule 2df8054 / runner c4a5391）。fusion 是 opt-in
  flag `--blb-v3-fusion-count-action 1`；不开时旧 per-slot 路径不变。
- **成功标准（F1，看 SUMMARY.txt）**：
  1. `fusion_unit_exit=0`。
  2. `smoke_exit=0`（跑完 300 episode）或 `124`（timeout 但中途健康也可接受，看下面健康信号）。
  3. **`Fusion-count action ENABLED` 这行必须出现**（证明真的走了 fusion 分支、map 已加载）。
  4. **crashes/fusion-path errors 段应为空**——特别是不能有 `AttributeError ... 'FusionStepSpec' has no attribute 'slot_dims'/'full_vec_offsets'`（这会说明某处逐槽机制没被 fusion 分支挡住）。
  5. 四卡 reward probe engaged（`reward probe enabled` / `trial split`）。
  6. reward 不是常数 `-150`、不出现持续 `loss_mean=100` 坍塌；anchor 期（前 ~60 episode）best 应贴近 baseline。
  7. `episodes.jsonl` 有逐 episode 记录、reward/priority 正常推进。
- **若失败**：把 `smoke.log` + `SUMMARY.txt`（+ `episodes.jsonl` 若有）带回本地。常见两类：
  - **fusion-path bug**（AttributeError/KeyError 等）→ 本地修对应分支，push，rerun。这是这次 smoke 的主要目的。
  - **前置条件**（无 Stage-1 record / `--mode` / 解耦路径）→ 与 fusion 代码无关；服务器可先跑一个短 Stage-1
    （`run rl --mode stage1-only --preset bert-base-mrpc-stage1-rl --fresh`，几百 episode）产出 record，再跑本 smoke；
    或把报错带回，我改用 `--stage2-fixed-config` 固定 stage-1 degrees。
- **协议**：服务器只 `git pull`、运行、产出/`push`/回传 artifacts；源码改动都在本地。把 `$OUT/` 回传本地。
```
