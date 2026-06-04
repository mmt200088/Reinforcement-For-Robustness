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
export CUDA_VISIBLE_DEVICES=0,1,2,3

TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_fusion_reward_${TS}"
mkdir -p "$OUT"
SOURCE_COMMIT=$(git rev-parse HEAD)
echo "HEAD=$SOURCE_COMMIT" | tee "$OUT/commit.txt"
nvidia-smi -L 2>/dev/null | tee "$OUT/gpus.txt" || true

# ---- Phase 0: what stage-1 records exist (stage2-only needs one for bert-base mrpc) ----
{ echo "=== stage1 records ==="; ls -la "Parting Chapter"/stage1/record/ 2>/dev/null || echo "(no stage1 record dir)"; } | tee "$OUT/stage1_records.txt"

# ---- Phase 1: full contract gate (torch) + fusion reward tests; record rc, do NOT abort ----
echo "=== contract gate $(date -Is) ===" | tee "$OUT/test_gate.log"
BLB_STRICT=0 python3 -m unittest discover -s tests -p "test_blb_*.py" -v >> "$OUT/test_gate.log" 2>&1
echo "contract_gate_rc=$?" | tee -a "$OUT/test_gate.log"
python3 -m unittest tests.test_blb_fusion_reward -v >> "$OUT/test_gate.log" 2>&1
echo "fusion_reward_test_rc=$?" | tee -a "$OUT/test_gate.log"
tail -n 40 "$OUT/test_gate.log" | tee "$OUT/test_gate_tail.log"

# ---- Phase 2: real fusion-count Stage-2 RL smoke with the NEW reward (K=4, 4-GPU) ----
echo "=== fusion stage2 smoke $(date -Is) ===" | tee "$OUT/train.log"
bash llama_7B_LayerImportance.sh run rl \
  --mode stage2-only \
  --preset mrpc-blb-stage2-rl \
  --blb-v3-fusion-count-action 1 \
  --stage2-k-trials 4 \
  --stage2-probe-size 256 \
  --batch-size 512 \
  --blb-v3-reward-devices 0,1,2,3 \
  --stage2-search-episodes 600 \
  --fresh >> "$OUT/train.log" 2>&1
echo "train_rc=$?" | tee -a "$OUT/train.log"

# ---- Phase 3: collect artifacts for local inspection ----
PROG=$(ls -dt "Parting Chapter"/stage2/*/progress 2>/dev/null | head -1)
{
  echo "HEAD=$SOURCE_COMMIT"
  echo "out_dir=$OUT"
  echo "progress_dir=$PROG"
} | tee "$OUT/SUMMARY.txt"
if [ -n "${PROG:-}" ] && [ -d "$PROG" ]; then
  cp -f "$PROG"/blb_stage2_status.json "$OUT/" 2>/dev/null || true
  cp -f "$PROG"/blb_stage2_report.md "$OUT/" 2>/dev/null || true
  cp -f "$PROG"/blb_stage2_best_action_full.json "$OUT/" 2>/dev/null || true
  cp -f "$PROG"/blb_stage2_baseline_action_full.json "$OUT/" 2>/dev/null || true
  cp -f "$PROG"/blb_stage2_error.txt "$OUT/" 2>/dev/null || true
  cp -f "$PROG"/warning.txt "$OUT/" 2>/dev/null || true
  cp -rf "$PROG"/diagnostics "$OUT/diagnostics" 2>/dev/null || true
  tail -n 3000 "$PROG"/diagnostics/episodes.jsonl > "$OUT/episodes_tail.jsonl" 2>/dev/null || true
fi
tail -n 500 "$OUT/train.log" > "$OUT/train_tail.log" 2>/dev/null || true
# quick markers for triage
{ echo "=== markers ==="; grep -nE "Fusion-count action ENABLED|fast-reward|num_trials_per_step|reward probe enabled|trial split|警告|warning|Traceback|Error|loss_mean=100|invalid_steps" "$OUT/train.log" | head -80; } | tee "$OUT/markers.txt" || true
echo "=== done $(date -Is) ===" | tee -a "$OUT/SUMMARY.txt"

# ---- Phase 4: push artifacts back so local can pull + inspect ----
git config --local http.version HTTP/1.1 2>/dev/null || true
git config --local protocol.version 0 2>/dev/null || true
git add "$OUT" 2>/dev/null || true
git -c user.email=server@run -c user.name=server commit -q -m "Stage-2 fusion reward smoke artifacts ${TS}" 2>/dev/null || true
git push origin jk_standard_rl 2>&1 | tail -3 || true
echo "=== Stage-2 fusion reward validation finished ==="
```

## metadata

- **任务（取代之前的 Stage-1 PPO 队列）**：真实验证 2026-06-04 的 Stage-2 fusion-count
  **动作 + reward 大改**。两段：
  1. **契约门**：`BLB_STRICT=0 python -m unittest discover -s tests -p "test_blb_*.py"`
     （含需要 torch 的 `test_blb_action_mask` / `test_blb_stage2_rl_regressions`，捕捉 reward
     改动在 torch 路径下的回归）+ 新增 `tests/test_blb_fusion_reward.py`（含对真实
     committed 融合图的集成测试）。记录 rc，不因失败中止。
  2. **真实 fusion Stage-2 smoke**：`run rl --mode stage2-only --preset mrpc-blb-stage2-rl
     --blb-v3-fusion-count-action 1 --stage2-k-trials 4 --blb-v3-reward-devices 0,1,2,3
     --stage2-search-episodes 600 --fresh`，新 reward（per-block 加权 P3 cost、total_bits
     删除、K=4 跨卡 std 门、warmstart 2.5）。
- **前置**：stage2-only 需要 bert-base mrpc 的 Stage-1 record（之前的 fusion smoke 用到过，
  应已存在；Phase 0 会把 `Parting Chapter/stage1/record/` 列出来，若缺会在 train.log 报错）。
- **要回看的信号**（local 会据此检查 bug）：契约门全绿；train.log 出现
  `Fusion-count action ENABLED`、四卡 probe、`fast-reward ... disabled`、K=4 trial split；
  `episodes_tail.jsonl` 里 reward 分三档（P1≈[-5,0]/P2≈[15,25]/P3≈[40,45]），cost 只在 P3、
  随 fusion/K 节省变化（看 `cost_score`/`fusion_cost_norm`），P2 由 std 触发（`stab_violation`>0），
  无 `loss_mean=100` 坍塌 / 无 `invalid_steps` 异常 / 无 Traceback。
- **产出**：`experiments/server_command_runs/stage2_fusion_reward_<ts>/`（commit.txt、test_gate.log、
  train.log、train_tail.log、markers.txt、status.json、report.md、episodes_tail.jsonl、diagnostics/）。
  命令末尾尝试 commit+push 回 `jk_standard_rl`（best-effort）。
- **协议**：服务器只 `git pull`、运行、产出/回传 artifacts；源码改动都在本地。
