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

echo "=== [0/2] full BLB contract gate (catches any block3-removal regression + fusion units) ==="
# block 3 was removed from the baseline + optimizer requests (2026-06-03); this gate
# confirms nothing that asserted block3-in-baseline/cost broke, and runs the 20 fusion
# map/schedule tests under real torch.
BLB_STRICT=0 python -m unittest discover -s tests -p "test_blb_*.py" -v 2>&1 | tee "$OUT/contract_gate.log"
G0=${PIPESTATUS[0]}

echo "=== [1/2] F1 SMOKE: fusion-count Stage-2 RL, ~300 episodes, 4 GPU ==="
# Now that block 3 is removed from the baseline, the run no longer needs block3_exp_n6,
# so it can read any existing Stage-1 record directly (gelu in {0,1,2,4} is map-covered;
# softmax is irrelevant — block3 frozen+removed). Validates the fusion runtime path:
# policy(max_step_dim=2)/open-mask/anchor(option0,K=baseline)/(option,K)->block SF
# expansion->replan->terminal forward+reward.
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
[ -n "$RUNDIR" ] && cp -f "$RUNDIR/diagnostics/episodes.jsonl" "$OUT/episodes.jsonl" 2>/dev/null || true

echo "=== [2/2] SUMMARY ==="
{
  echo "HEAD=$(git rev-parse HEAD)"
  echo "contract_gate_exit=$G0   (0 = all test_blb_*.py pass under torch; block3 removal + fusion clean)"
  echo "smoke_exit=$G1           (0 = RL run finished; 124 = timeout)"
  echo "RUNDIR=$RUNDIR"
  echo "--- contract gate tail (failures/errors if any) ---"
  grep -iE "FAILED|ERROR|OK|Ran [0-9]+ tests" "$OUT/contract_gate.log" | tail -8
  echo "--- fusion mode engaged? (expect this line) ---"
  grep -i "Fusion-count action ENABLED" "$OUT/smoke.log" | head
  echo "--- 4-GPU reward probe engaged? ---"
  grep -iE "reward probe enabled|probe-runner.*worker|trial split" "$OUT/smoke.log" | head
  echo "--- crashes / fusion-path errors (should be EMPTY) ---"
  grep -iE "Traceback|AttributeError|FusionStepSpec|slot_dims|full_vec_offsets|block3_exp_n6|KeyError" "$OUT/smoke.log" | head -25
  echo "--- reward / collapse sentinels tail ---"
  grep -iE "reward|loss_mean=100|P1\\(acc\\)|best_reward|episode " "$OUT/smoke.log" | tail -40
  echo "--- episodes.jsonl tail ---"
  [ -f "$OUT/episodes.jsonl" ] && tail -5 "$OUT/episodes.jsonl" || echo "(no episodes.jsonl)"
} | tee "$OUT/SUMMARY.txt"
echo "=== DONE -> $OUT ==="
```

## metadata

- **任务**：**Task 8 — fusion-count 运行期 F1 smoke**（验证 Tasks 5–7），并验证刚做的 **block3 baseline 移除**。
- **本轮关键修复（block3 baseline 移除）**：前两轮 smoke 都挂在 fusion 之前——
  `_resolve_stage2_fixed_stage1_config` 把 softmax 写死成 `FIXED_SOFTMAX_DEGREE=6`，
  而重生成的 skeleton 里 `block3_exp_n6` 是 `success=false`（只有 n2..n5 有效），baseline handoff 取不到 n6 就崩。
  按用户指令「baseline 动作也不再包含 block3」，本地已把 block3 从 **baseline 抽取**
  (`baseline_bootstrap.load_static_skeletons_baseline` 的块循环 `(1,2,3,4,5)→(1,2,4,5)`) 与
  **optimizer requests**（`action_space.build_optimizer_requests` 跳过 block3）双双移除。block3 本来就：
  不进 RL schedule（C）、不装模型噪声（bridge 不调）、现在也不进 baseline/cost → **彻底移除、n6 依赖消失**。
  cost 仍一致（baseline 与 action 两边都没有 block3）。
- **成功标准（SUMMARY.txt）**：
  1. `contract_gate_exit=0`（block3 移除没碰坏任何 baseline/cost 契约测试；20 个 fusion 测试在 torch 下过）。
  2. `smoke_exit=0`（跑完 300 episode）或 `124`（timeout 但中途健康）。
  3. **`Fusion-count action ENABLED` 出现**；crashes 段**为空**（尤其没有 `block3_exp_n6` 报错、没有
     `AttributeError FusionStepSpec ... slot_dims/full_vec_offsets`）。
  4. 四卡 probe engaged；reward 非常数 -150、无持续 `loss_mean=100` 坍塌、anchor 期 best≈baseline。
- **若仍失败**：
  - contract gate 红 → 某 baseline/cost 测试断言了 block3，本地按「block3 已移除」更新该测试，push，rerun。
  - smoke 在 fusion 之后崩（fusion-path bug）→ 本地修对应分支，push，rerun。**这才是 smoke 的主目标。**
  - 仍缺 Stage-1 record → 先跑短 Stage-1（`run rl --mode stage1-only --preset bert-base-mrpc-stage1-rl --fresh`）。
- **协议**：服务器只 `git pull`、运行、产出/回传 artifacts；源码改动都在本地。把 `$OUT/` 回传本地。
```
