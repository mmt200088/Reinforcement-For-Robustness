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
OUT="experiments/server_command_runs/stage2_fusion_fullbuild_${TS}"
mkdir -p "$OUT"
SOURCE_COMMIT=$(git rev-parse HEAD)
echo "HEAD=$SOURCE_COMMIT" | tee "$OUT/commit.txt"
nvidia-smi -L 2>/dev/null | tee "$OUT/gpus.txt" || true
WK=$(nproc 2>/dev/null || echo 16); echo "nproc=$WK" | tee "$OUT/nproc.txt"

git config --local http.version HTTP/1.1 2>/dev/null || true
git config --local protocol.version 0 2>/dev/null || true
push_now () { # $1 = commit message
  git add blb_stage2_rl/fusion_maps/mrpc/*.json "$OUT" 2>/dev/null || true
  git -c user.email=server@run -c user.name=server commit -q -m "$1" 2>/dev/null || true
  git push origin jk_standard_rl 2>&1 | tail -3 || true
}

# snapshot the committed (old-convention) maps as the soundness REFERENCE
mkdir -p "$OUT/maps_ref_committed"
cp -f blb_stage2_rl/fusion_maps/mrpc/*.json "$OUT/maps_ref_committed/" 2>/dev/null || true

# ---- Phase 1: contract gate (decode + fusion reward + pin-criterion regression) ----
echo "=== contract gate $(date -Is) ===" | tee "$OUT/test_gate.log"
BLB_STRICT=0 python3 -m unittest discover -s tests -p "test_blb_*.py" -v >> "$OUT/test_gate.log" 2>&1
echo "contract_gate_rc=$?" | tee -a "$OUT/test_gate.log"
tail -n 25 "$OUT/test_gate.log" | tee "$OUT/test_gate_tail.log"

# ---- Phase 2: build the 6 cheaper block-types FULLY (block2/block5_n4 ~6.6e7 each
#      ~80min, rest minutes; ~3h total). block4 is excluded here and built separately
#      below. Pushing these first means a block4 interruption never costs these six.
#      --max-enum-combos 0 = unlimited: none of these is degenerate (all fuse), so
#      build every one fully (no probe / no abort even if a slot count shifts). ----
echo "=== build 6 cheaper maps (workers=$WK) $(date -Is) ===" | tee "$OUT/build_feasible.log"
python3 scripts/blb_build_fusion_count_map.py --profile mrpc \
  --out-dir blb_stage2_rl/fusion_maps/mrpc \
  --only block1_mrpc,block2_mrpc,block5_n0,block5_n1,block5_n2,block5_n4 \
  --max-enum-combos 0 \
  --report "$OUT/fusion_map_build_feasible.html" --workers "$WK" >> "$OUT/build_feasible.log" 2>&1
echo "build_feasible_rc=$?" | tee -a "$OUT/build_feasible.log"
tail -n 40 "$OUT/build_feasible.log" | tee "$OUT/build_feasible_tail.log"
push_now "Rebuild 6 feasible fusion maps (sound joint pin) ${TS}"

# ---- Phase 3: build block4 FULLY (729M combos, NOT degenerate -> fusion=[0,1]) ----
# Confirmed not-degenerate by the prior degeneracy probe; --max-enum-combos 0 =
# unlimited -> full sound joint enumeration. ~13.5h on 96 cores (~15000 evals/s).
echo "=== build block4 FULL (workers=$WK) $(date -Is) ===" | tee "$OUT/build_block4.log"
python3 scripts/blb_build_fusion_count_map.py --profile mrpc \
  --out-dir blb_stage2_rl/fusion_maps/mrpc \
  --only block4 \
  --max-enum-combos 0 \
  --report "$OUT/fusion_map_build_block4.html" --workers "$WK" >> "$OUT/build_block4.log" 2>&1
echo "build_block4_rc=$?" | tee -a "$OUT/build_block4.log"
tail -n 40 "$OUT/build_block4.log" | tee "$OUT/build_block4_tail.log"
push_now "Rebuild block4 fusion map FULL (sound joint pin) ${TS}"

# summarize ALL 7 NEW maps
python3 - <<'PY' | tee "$OUT/map_summary.txt"
import json, glob, os
for p in sorted(glob.glob("blb_stage2_rl/fusion_maps/mrpc/*.json")):
    d=json.load(open(p)); fc=[o["fusion_count"] for o in d["options"]]; bm=d.get("build_meta",{})
    o0=d["options"][0]["action_indices"] if d["options"] else []
    conv="NEW" if (o0 and max(o0)>=9) else "OLD"
    print(f"{os.path.basename(p):16s} #opt={len(d['options']):3d} fusion_counts={sorted(set(fc))} conv={conv} "
          f"enum_total={bm.get('enum_total_combos')} valid={bm.get('valid_configs')} "
          f"pinned={len(bm.get('pinned_positions',[]))} wall={bm.get('wall_seconds')}")
PY

# ---- Phase 4: SOUNDNESS AUDIT — new fusion sets must be a SUPERSET of committed
#      ground-truth (deeper sweep may ADD fusion, must never LOSE any). ----
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
push_now "Fusion map summary + soundness audit ${TS}"

# ---- Phase 5: real fusion-count Stage-2 RL smoke with the NEW maps (K=4, 4-GPU) ----
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

# ---- Phase 6: collect artifacts ----
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
push_now "Stage-2 fusion full-build smoke artifacts ${TS}"
echo "=== Stage-2 full-build (block4 included) + smoke finished ==="
```

## metadata

- **背景**：上一轮（`stage2_fusion_soundpin_20260604_191907`，96 核）证明退化探针**工作正常**：block4
  `enum_total=729000000 > 100M budget`，探针在更深的 10 档扫中**找到了 fusion=[0,1]**（block4 在旧浅扫下
  是退化的 `[0]`，深扫解锁了融合），builder **正确拒绝**输出 shortcut 图。同时 block1 也从 `[0]→[0,1]`、
  block2 仍 `[0,1]`——**深扫确实带来更多 fusion**。结论：block4 不退化，**必须完整建**。服务器只推了失败日志，
  committed 图仍是旧约定（一致，不是半坏状态）。
- **本轮任务**：把 7 张图**全部按新 10 档约定 + sound 联合 pin 重建**，分两步保命：
  1. **6 张可行图**（block1/block2/block5_n0/n1/n2/n4，各 < 1e8）先完整建 + **先推**（block2 ~81min，
     block5_n4 ~81min，其余分钟级，合计 ~3h）。
  2. **block4 完整建**（729M 组合，`--max-enum-combos 0` 不设上限 → 完整 sound 联合枚举）。96 核 ~15000 evals/s
     → **约 13.5 小时**。建完**单独推**，所以即使 block4 被中断也不丢前 6 张。
- **为什么是 13.5h 而不是走捷径**：fusion 是 2–4 个 encode **联合**压出来的（rescale 不动），唯一**可证无遗漏**的
  办法就是对所有 `(fusion,total_bits)` 相关 encode 做**联合笛卡尔积**。block4 有 6 个 encode（10 档）× 3 rescale
  （9 档）= 729M，没有可证 sound 的捷径（除非引入"SF 越低融合越多"的单调性假设，那是另一套实现）。用户优先级是
  **毫无遗漏**，所以这里直接完整建。**注**：reward 改动**不**需要重建图（图只管 action→SF），只有档位/解码/skeleton
  变了才需重建——所以这 13.5h 是低频成本，不是每次调 reward 都付。
- **soundness 审计**：重建前快照旧图，全建完后断言每个 block 的 `fusion_counts` ⊇ 旧集合（深扫只能加、不能丢）。
  `superset_pass=True` 才算过。
- **要回看的信号**：契约门全绿；`map_summary.txt` 7 张图全 `conv=NEW`、fusion ⊇ 旧（block1/block4 现应为 `[0,1]`，
  block2/block5_* ≥ `[0,1]`，可能更多）；`soundness_audit.txt` `superset_pass=True`；train.log 出现
  `Fusion-count action ENABLED`、四卡 K=4 probe、`fast-reward disabled`；episodes reward 分三档、cost 只在 P3、
  P2 由 std 触发；无 `loss_mean=100` / Traceback / invalid 爆发。
- **600 episode smoke 只是查 bug/能否跑**，不评训练曲线（RL 要几万轮才有起色）。smoke 在所有图推送之后才跑，
  所以即使 smoke 出问题，图也已经安全落库。
- **前置**：stage2-only 需 bert-base mrpc 的 Stage-1 record（应已存在）。
- **协议**：服务器只 `git pull`、运行、产出/回传 artifacts（图是生成物）；源码改动都在本地。
