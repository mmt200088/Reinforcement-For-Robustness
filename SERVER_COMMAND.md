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
OUT="experiments/server_command_runs/stage2_fusion_hybrid10_${TS}"
mkdir -p "$OUT"
SOURCE_COMMIT=$(git rev-parse HEAD)
echo "HEAD=$SOURCE_COMMIT" | tee "$OUT/commit.txt"
nvidia-smi -L 2>/dev/null | tee "$OUT/gpus.txt" || true
nproc 2>/dev/null | tee "$OUT/nproc.txt" || true

# ---- Phase 1: contract gate (validates the new step-1 deep-rescale decode + no regressions) ----
echo "=== contract gate $(date -Is) ===" | tee "$OUT/test_gate.log"
BLB_STRICT=0 python3 -m unittest discover -s tests -p "test_blb_*.py" -v >> "$OUT/test_gate.log" 2>&1
echo "contract_gate_rc=$?" | tee -a "$OUT/test_gate.log"
python3 -m unittest tests.test_blb_fusion_reward -v >> "$OUT/test_gate.log" 2>&1
echo "fusion_reward_test_rc=$?" | tee -a "$OUT/test_gate.log"
tail -n 40 "$OUT/test_gate.log" | tee "$OUT/test_gate_tail.log"

# ---- Phase 2: REBUILD the 7 fusion-count maps with the deepened rescale sweep (preprocessing) ----
WK=$(nproc 2>/dev/null || echo 16)
echo "=== rebuild fusion maps (workers=$WK) $(date -Is) ===" | tee "$OUT/build.log"
python3 scripts/blb_build_fusion_count_map.py --profile mrpc \
  --out-dir blb_stage2_rl/fusion_maps/mrpc \
  --report "$OUT/fusion_map_build.html" --workers "$WK" >> "$OUT/build.log" 2>&1
echo "build_rc=$?" | tee -a "$OUT/build.log"
tail -n 40 "$OUT/build.log" | tee "$OUT/build_tail.log"
# summarize the NEW maps — did fusion counts get richer than the old [0]/[0,1]?
python3 - <<'PY' | tee "$OUT/map_summary.txt"
import json, glob, os
for p in sorted(glob.glob("blb_stage2_rl/fusion_maps/mrpc/*.json")):
    d=json.load(open(p)); fc=[o["fusion_count"] for o in d["options"]]; bm=d.get("build_meta",{})
    print(f"{os.path.basename(p):16s} #opt={len(d['options']):3d} fusion_counts={sorted(set(fc))} "
          f"enum_total={bm.get('enum_total_combos')} valid={bm.get('valid_configs')} wall={bm.get('wall_seconds')}")
PY

# ---- Phase 3: real fusion-count Stage-2 RL smoke with the NEW maps + NEW reward (K=4, 4-GPU) ----
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

# ---- Phase 4: collect artifacts ----
PROG=$(ls -dt "Parting Chapter"/stage2/*/progress 2>/dev/null | head -1)
{ echo "HEAD=$SOURCE_COMMIT"; echo "out_dir=$OUT"; echo "progress_dir=$PROG"; } | tee "$OUT/SUMMARY.txt"
if [ -n "${PROG:-}" ] && [ -d "$PROG" ]; then
  cp -f "$PROG"/blb_stage2_status.json "$OUT/" 2>/dev/null || true
  cp -f "$PROG"/blb_stage2_report.md "$OUT/" 2>/dev/null || true
  cp -f "$PROG"/blb_stage2_best_action_full.json "$OUT/" 2>/dev/null || true
  cp -f "$PROG"/blb_stage2_error.txt "$OUT/" 2>/dev/null || true
  cp -rf "$PROG"/diagnostics "$OUT/diagnostics" 2>/dev/null || true
  tail -n 3000 "$PROG"/diagnostics/episodes.jsonl > "$OUT/episodes_tail.jsonl" 2>/dev/null || true
fi
tail -n 500 "$OUT/train.log" > "$OUT/train_tail.log" 2>/dev/null || true
{ echo "=== markers ==="; grep -nE "Fusion-count action ENABLED|fast-reward|num_trials_per_step|reward probe enabled|trial split|警告|warning|Traceback|Error|loss_mean=100|invalid_steps" "$OUT/train.log" | head -80; } | tee "$OUT/markers.txt" || true
echo "=== done $(date -Is) ===" | tee -a "$OUT/SUMMARY.txt"

# ---- Phase 5: push the rebuilt maps + artifacts back ----
git config --local http.version HTTP/1.1 2>/dev/null || true
git config --local protocol.version 0 2>/dev/null || true
git add blb_stage2_rl/fusion_maps/mrpc/*.json "$OUT" 2>/dev/null || true
git -c user.email=server@run -c user.name=server commit -q -m "Rebuild fusion maps (deep rescale) + stage2 smoke artifacts ${TS}" 2>/dev/null || true
git push origin jk_standard_rl 2>&1 | tail -3 || true
echo "=== Stage-2 deep-rescale rebuild + smoke finished ==="
```

## metadata

- **任务**：验证 2026-06-04 的 10 档 hybrid 档位 + N=16384 + 放宽 pinning，三段：
  1. **契约门**：`test_blb_*.py`（含 torch 的 action_space 测试；`HybridDecodeTest` 断言全 SF 槽 10 档、
     baseline 30→30,28,26,24,22,20,19,18,17,16、rescale idx0=None / max idx→baseline、低 baseline-SF snap 到 10、
     `_block_default_N` 恒 16384）+ fusion reward 测试。记录 rc，不中止。
  2. **重建 7 张融合图**（预处理）：`blb_build_fusion_count_map.py --profile mrpc`。全 SF 槽 10 档 hybrid（顶部 step-2、
     底部 step-1，到 baseline-14），**pinning 放宽成只看 fusion_count**（total_bits 已移出 reward）→ 只枚举真正影响
     fusion 的槽（rescale + 少数 source），其余钉在 baseline SF → build 应回到 ~分钟级。`map_summary.txt` 看
     fusion_counts 是否比旧的 `[0]`/`[0,1]` 更丰富。**builder 的 option0==baseline 断言守住 baseline 不变性**。
  3. **真实 fusion Stage-2 smoke**（用新图 + 新 reward，K=4，4 卡，600 episode）。
- **重要**：600 episode 只是**查 bug / 能否正确运行**的 smoke，RL 一般要几万轮才有起色，这里**不**评判训练曲线是否优秀。
- **要回看的信号**：契约门全绿；`map_summary.txt` 里至少部分 block 的 fusion_counts 变多（验证加深 rescale 确实带来更多 fusion）；
  train.log 出现 `Fusion-count action ENABLED`、四卡 K=4 probe、`fast-reward disabled`；`episodes_tail.jsonl` reward 分三档、
  cost 只在 P3、P2 由 std 触发；无 `loss_mean=100` 坍塌 / 无 Traceback / 无 invalid 爆发。
- **前置**：stage2-only 需要 bert-base mrpc 的 Stage-1 record（之前 smoke 用过，应已存在）。
- **产出**：`experiments/server_command_runs/stage2_fusion_deepR_<ts>/`（test_gate.log、build.log、map_summary.txt、
  fusion_map_build.html、train.log、markers.txt、status.json、report.md、episodes_tail.jsonl、diagnostics/）+ 重建后的
  `blb_stage2_rl/fusion_maps/mrpc/*.json`。命令末尾 commit+push 回 `jk_standard_rl`（含新图，best-effort）。
- **协议**：服务器只 `git pull`、运行、产出/回传 artifacts（图是生成物，非手改源码）；源码改动都在本地。
```
