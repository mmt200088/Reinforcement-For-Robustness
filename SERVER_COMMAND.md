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
OUT="experiments/server_command_runs/stage2_fusion_soundpin_${TS}"
mkdir -p "$OUT"
SOURCE_COMMIT=$(git rev-parse HEAD)
echo "HEAD=$SOURCE_COMMIT" | tee "$OUT/commit.txt"
nvidia-smi -L 2>/dev/null | tee "$OUT/gpus.txt" || true
nproc 2>/dev/null | tee "$OUT/nproc.txt" || true

# snapshot the committed maps as the soundness REFERENCE before rebuilding in place
mkdir -p "$OUT/maps_ref_committed"
cp -f blb_stage2_rl/fusion_maps/mrpc/*.json "$OUT/maps_ref_committed/" 2>/dev/null || true

# ---- Phase 1: contract gate (decode + fusion reward + pin-criterion regression) ----
echo "=== contract gate $(date -Is) ===" | tee "$OUT/test_gate.log"
BLB_STRICT=0 python3 -m unittest discover -s tests -p "test_blb_*.py" -v >> "$OUT/test_gate.log" 2>&1
echo "contract_gate_rc=$?" | tee -a "$OUT/test_gate.log"
python3 -m unittest tests.test_blb_fusion_reward -v >> "$OUT/test_gate.log" 2>&1
echo "fusion_reward_test_rc=$?" | tee -a "$OUT/test_gate.log"
tail -n 40 "$OUT/test_gate.log" | tee "$OUT/test_gate_tail.log"

# ---- Phase 2: REBUILD fusion maps with the SOUND (fusion_count, total_bits) pin ----
# Joint cartesian over fusion-relevant encodes recovers the joint-encode fusion
# (every block's fusion>0 option lowers 2-4 encodes together; a fusion-only probe
# would lose it). --max-enum-combos routes block4 (~7e8 combos, fusion-degenerate)
# to a degeneracy probe (all-min corner + samples) -> correct baseline-only map,
# while block2 / block5_n4 (~6.6e7 each) build FULLY. ALL maps are regenerated so
# their action_indices use the new uniform-10 index convention.
WK=$(nproc 2>/dev/null || echo 16)
echo "=== rebuild fusion maps (workers=$WK, budget=100M) $(date -Is) ===" | tee "$OUT/build.log"
python3 scripts/blb_build_fusion_count_map.py --profile mrpc \
  --out-dir blb_stage2_rl/fusion_maps/mrpc \
  --max-enum-combos 100000000 \
  --degeneracy-probe-samples 3000 \
  --report "$OUT/fusion_map_build.html" --workers "$WK" >> "$OUT/build.log" 2>&1
echo "build_rc=$?" | tee -a "$OUT/build.log"
tail -n 80 "$OUT/build.log" | tee "$OUT/build_tail.log"

# summarize the NEW maps — did fusion counts get richer than the old [0]/[0,1]?
python3 - <<'PY' | tee "$OUT/map_summary.txt"
import json, glob, os
# builder asserts option0==baseline internally, so a successful build already
# guarantees the baseline invariant; here we just surface fusion richness.
for p in sorted(glob.glob("blb_stage2_rl/fusion_maps/mrpc/*.json")):
    d=json.load(open(p)); fc=[o["fusion_count"] for o in d["options"]]; bm=d.get("build_meta",{})
    print(f"{os.path.basename(p):16s} #opt={len(d['options']):3d} fusion_counts={sorted(set(fc))} "
          f"enum_total={bm.get('enum_total_combos')} valid={bm.get('valid_configs')} "
          f"pinned={len(bm.get('pinned_positions',[]))} over_budget={bm.get('over_budget_degenerate',False)} "
          f"wall={bm.get('wall_seconds')}")
PY

# ---- Phase 2b: SOUNDNESS AUDIT — rebuilt fusion sets must be a SUPERSET of the
#      committed ground-truth (deeper sweep may ADD fusion, must never LOSE any). ----
python3 - "$OUT/maps_ref_committed" "$OUT/soundness_audit.txt" <<'PY'
import json, glob, os, sys
ref_dir, out_txt = sys.argv[1], sys.argv[2]
lines=[]; ok=True
for p in sorted(glob.glob("blb_stage2_rl/fusion_maps/mrpc/*.json")):
    gk=os.path.basename(p)
    new=set(int(o["fusion_count"]) for o in json.load(open(p))["options"])
    rp=os.path.join(ref_dir, gk)
    old=set(int(o["fusion_count"]) for o in json.load(open(rp))["options"]) if os.path.exists(rp) else set()
    superset = old <= new
    ok = ok and superset
    lines.append(f"{gk:16s} old={sorted(old)} new={sorted(new)} {'OK' if superset else 'REGRESSION(lost fusion)'}")
hdr=f"soundness_audit superset_pass={ok}"
open(out_txt,"w").write(hdr+"\n"+"\n".join(lines)+"\n")
print(hdr); [print(' ',l) for l in lines]
PY
echo "audit_done $(date -Is)"

# ---- Phase 3: real fusion-count Stage-2 RL smoke with the NEW maps + reward (K=4, 4-GPU) ----
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
git -c user.email=server@run -c user.name=server commit -q -m "Rebuild fusion maps (sound joint pin) + stage2 smoke artifacts ${TS}" 2>/dev/null || true
git push origin jk_standard_rl 2>&1 | tail -3 || true
echo "=== Stage-2 sound-pin rebuild + smoke finished ==="
```

## metadata

- **任务**：验证并落地「融合图 pinning 必须用 `(fusion_count, total_bits)` 联合判据」的修复，三段：
  1. **契约门**：`test_blb_*.py`（含 torch 的 `HybridDecodeTest` + 新增 `PinClassificationCriterionTest`，
     后者锁死联合-encode 融合必须被枚举的判据）+ `test_blb_fusion_reward`。记录 rc，不中止。
  2. **重建 7 张融合图（核心）**：`blb_build_fusion_count_map.py --profile mrpc --max-enum-combos 100000000`。
     - **为什么必须联合枚举**：committed ground-truth 显示每个 block 的 `fusion=1` 选项都是把 2–4 个**非 rescale
       encode SF 一起往下压**（rescale 全在 baseline）得到的，例 block2 fc=1 = `inv_std_fresh 28→20, gamma 20→16,
       wk 22→16`。任何单个 encode 单独动都不改 fusion，所以「只看 fusion 的 solo 探针」会把它们全部 pin 住、
       融合图坍塌成 `fusion={0}`（2026-06-04 那次放宽就是这个 bug，已回退）。`total_bits` 在这里只是
       **build 期的过度枚举代理**（不进 reward），保证所有 encode 都进联合笛卡尔积。
     - **预算/退化探针**：block4 在 10 档深扫下 ~7×10⁸ 组合（数十小时），但它本就 fusion-退化。`--max-enum-combos`
       把这种超预算 block 交给退化探针（全-最小-SF 角点 + 3000 随机样本）：若无任何组合触发融合 → 写出
       **新档位约定下正确的 baseline-only 图**；若探到融合 → **直接报错**拒绝写错图。**不能沿用旧 committed 图**，
       因为旧图的 `action_indices` 用的是旧档位约定（如 idx 4），在新 10 档解码下会解错。
     - block1≈8.1e5、block2≈6.6e7、block4→探针、block5_n0≈8.1e4、n1≈8.1e5、n2≈7.3e6、n4≈6.6e7：除 block4 外全部**完整重建**。
       `map_summary.txt` 里 `opt0_allmax` 必须为 True（option0=全档位上限=baseline，新约定）。
  2b. **soundness 审计**：重建前把 committed 图快照到 `maps_ref_committed/`，重建后断言每个 block 的
      `fusion_counts` 是旧集合的**超集**（深扫只能加 fusion，不能丢）。`superset_pass=True` 才算过；
      任何 `REGRESSION(lost fusion)` 都要查。
  3. **真实 fusion Stage-2 smoke**（新图 + 新 reward，K=4，4 卡，600 episode）。
- **重要**：600 episode 只是**查 bug / 能否正确运行**的 smoke；RL 一般要几万轮才有起色，这里**不**评判训练曲线优秀与否。
- **预计耗时**：重建受 block2 + block5_n4（各 ~6.6e7）主导，按核数从约 1.5 小时到数小时；block4 走探针只要秒级；
  之后接 600-ep smoke。整段是一次性预处理 + 冒烟，长一点没关系（服务器可并行）。
- **要回看的信号**：契约门全绿（尤其 `PinClassificationCriterionTest`、`HybridDecodeTest`）；
  `soundness_audit.txt` `superset_pass=True`；`map_summary.txt` 里 block2/block5_* 的 `fusion_counts` ⊇ `[0,1]`
  且 `opt0_allmax=True`，block4 `over_budget=True` 且 `fusion=[0]`；train.log 出现 `Fusion-count action ENABLED`、
  四卡 K=4 probe、`fast-reward disabled`；`episodes_tail.jsonl` reward 分三档、cost 只在 P3、P2 由 std 触发；
  无 `loss_mean=100` 坍塌 / 无 Traceback / 无 invalid 爆发。
- **前置**：stage2-only 需要 bert-base mrpc 的 Stage-1 record（之前 smoke 用过，应已存在）。
- **产出**：`experiments/server_command_runs/stage2_fusion_soundpin_<ts>/`（test_gate.log、build.log、map_summary.txt、
  soundness_audit.txt、fusion_map_build.html、train.log、markers.txt、status.json、report.md、episodes_tail.jsonl、
  diagnostics/、maps_ref_committed/）+ 重建后的 `blb_stage2_rl/fusion_maps/mrpc/*.json`。命令末尾 commit+push 回
  `jk_standard_rl`（含新图，best-effort）。
- **协议**：服务器只 `git pull`、运行、产出/回传 artifacts（图是生成物，非手改源码）；源码改动都在本地。
```
