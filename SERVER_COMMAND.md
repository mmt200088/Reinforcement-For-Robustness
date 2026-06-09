# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command  (stage1 改+replan 改 → 重建 fusion 图 + 重推 baseline，然后 curriculum A/B 全训练)

```bash
set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export HF_HOME=/hy-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data

# ============================================================================
# 上一轮（stage2_rebuild_ab_20260607_195411）结果：fusion 图已全部按新 replan 重建并
# 提交（1ad078c：block1=(1,[0]) K-only、block5_n2/n4 从[0,1,2]→[0,1]，图门禁通过），
# 但 A/B 两组都在启动瞬间崩溃：
#   FileNotFoundError: 未找到 combo='bert base mrpc' 的 Stage-1 record（Parting Chapter/stage1/record）
# 根因：解耦 stage2 从 stage1/record/<combo>/final_config.json 读前置 Stage-1 degrees，
# 但服务器上 MRPC 的 stage1/record 不存在（最近一次 Stage-1 跑的是 RTE；MRPC 新 degrees
# 只写进了 approx_per_dataset.json，没归档成解耦 record）。错误提示里的 --stage2-fixed-config
# 是死代码（rl_tune 参数未接线）。本轮修复：用已提交的 approx_per_dataset.json 里的 MRPC
# degrees 合成出那条 Stage-1 record（gelu [1,2,1,1,1,1,1,1,2,1,1,1] / softmax [6]*12），再跑 A/B。
# 这是服务器侧「生成产物」，不动任何训练源码。
# 流程：同步/接口自检 → (图已建好,默认跳过) → 图门禁 → 合成缺失的 Stage-1 record → A/B → 对比。
# ----------------------------------------------------------------------------
EPISODES=6000        # A/B 规模：6000 足够跑完课程全生命周期(ramp 0.5×=3000 收, 3000-6000 全开)；里程碑级改 60000
REBUILD_MAPS=0       # 图已按新 replan 建好且提交(1ad078c) → 默认跳过。只有 skeleton 再变才改回 1。
WORKERS="$( n=$(nproc 2>/dev/null || echo 8); echo $(( n > 16 ? 16 : n )) )"   # 图构建是 CPU 活，不占 GPU
# 自动探测可用 GPU（上一轮 A/B 只看到 device 0）：用实际有的卡，K=卡数(封顶8)，1-卡也能跑。
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"; [ -z "$NGPU" ] && NGPU=1; [ "$NGPU" -lt 1 ] && NGPU=1
DEVS="$(seq -s, 0 $((NGPU-1)))"           # 0  或  0,1,2,3 ...
KTRIALS="$NGPU"; [ "$KTRIALS" -gt 8 ] && KTRIALS=8
echo "[gpu] 探测到 $NGPU 张卡 -> CUDA_VISIBLE_DEVICES=$DEVS, stage2-k-trials=$KTRIALS"
# ============================================================================

TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_rebuild_ab_${TS}"
mkdir -p "$OUT"
SKEL="Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json"
MAPS_DIR="blb_stage2_rl/fusion_maps/mrpc"
CANON_STAGE2="Parting Chapter/stage2"          # 解耦 stage2：LATEST_PID/RUN_DIR 落在这里；combo=bert base mrpc

echo "==================== [phase0] 同步 + 接口自检 ===================="
git rev-parse HEAD | tee "$OUT/HEAD.txt"
git log --oneline -5
# RO 接口仍向后兼容（旧导出都在 + 新增 fusion 导出可用）。打印新默认融合策略，留证。
python3 - <<'PY' 2>&1 | tee "$OUT/ro_interface.txt" || { echo "[FATAL] RO 接口导入失败"; exit 1; }
import rescale_optimizer as r
print("RO 导入 OK；DEFAULT_FUSION_POLICY =", r.DEFAULT_FUSION_POLICY)
for k in ["block1_mrpc","block2_mrpc","block4","block5_n1","block5_n2","block5_n4"]:
    print(f"  默认融合对 {k:12s} = {r.resolve_allowed_fusion_pairs(k, r.DEFAULT_FUSION_POLICY)}")
PY
# 回显前置 Stage-1 degrees（应为 {1,2}，无 0；degree-0 现在会在 bootstrap 直接报错）
python3 - <<'PY' 2>&1 | tee "$OUT/stage1_degrees.txt"
import glob, json, os
recs = sorted(glob.glob("Parting Chapter/stage1/record/*/final_config.json"))
if not recs:
    print("[warn] 没找到 stage1/record，A/B 启动时 runner 会按自身逻辑解析前置 Stage-1")
for rec in recs:
    c = json.load(open(rec)); g = c.get("gelu_degree_per_layer")
    print(os.path.basename(os.path.dirname(rec)), "gelu =", g, "| 含 degree0:", 0 in (g or []))
PY

echo "==================== [phase1] 按新 replan 默认重建 fusion 图（幂等：比 skeleton 新则跳过）===================="
cp -a "$MAPS_DIR" "$OUT/old_maps" 2>/dev/null || true       # 旧图备份，便于 diff/取证
build_map () {
  local gk="$1"; local mp="${MAPS_DIR}/${gk}.json"
  if [ -f "$mp" ] && [ "$mp" -nt "$SKEL" ]; then
    echo "[maps] $gk 已比 skeleton 新 -> 跳过（视为已按新策略重建）"; return 0; fi
  echo "[maps] building $gk ..."
  python scripts/blb_build_fusion_count_map.py --profile mrpc --only "$gk" \
    --out-dir "$MAPS_DIR" --rescale-optimizer-root Rescale_optimizer \
    --num-layers 12 --workers "$WORKERS" --max-enum-combos 0 \
    2>&1 | tee "$OUT/build_${gk}.log"
}
if [ "$REBUILD_MAPS" = 1 ]; then
  # 快的先建，block4(~12.7h 全枚举) 放最后；中途崩也保住其它图。block5_n0 已停用(不在 BLOCK_TYPES)。
  for gk in block1_mrpc block2_mrpc block5_n1 block5_n2 block5_n4 block4; do
    build_map "$gk" || { echo "[FATAL] fusion 图 $gk 构建失败"; exit 1; }
  done
fi

echo "==================== [phase2] 图门禁（FusionCountMap.load 强制 option0==baseline）+ 新旧对比 ===================="
python3 - <<'PY' 2>&1 | tee "$OUT/map_gate.txt" || { echo "[FATAL] 图门禁失败：option0 不是 baseline 或图损坏"; exit 1; }
import glob, json, os
from blb_stage2_rl.fusion_count_map import FusionCountMap
FusionCountMap.load("mrpc")        # 加载即强制每张图 option_id0==fusion_count0 baseline，坏图会抛
print("FusionCountMap.load('mrpc') OK — 所有图 option0==baseline。")
def summarize(d):
    o = d["options"]; return len(o), sorted({x["fusion_count"] for x in o})
new = "blb_stage2_rl/fusion_maps/mrpc"; old = os.environ.get("OUTOLD","")
print("\n图           新(n_opt, fusion_counts)        旧(n_opt, fusion_counts)")
for f in sorted(glob.glob(new+"/*.json")):
    b = os.path.basename(f)
    if b.startswith("_"): continue
    nn = summarize(json.load(open(f)))
    op = os.path.join(old, b) if old else ""
    oo = summarize(json.load(open(op))) if op and os.path.exists(op) else "—"
    print(f"  {b:16s} {str(nn):28s} {oo}")
PY
OUTOLD="$OUT/old_maps" python3 - <<'PY' 2>&1 | tee -a "$OUT/map_gate.txt" || true
import glob, json, os
old = os.environ.get("OUTOLD",""); new = "blb_stage2_rl/fusion_maps/mrpc"
def summarize(d):
    o = d["options"]; return len(o), sorted({x["fusion_count"] for x in o})
print("\n[新 replan 策略对 fusion 图的影响]")
for f in sorted(glob.glob(new+"/*.json")):
    b = os.path.basename(f)
    if b.startswith("_"): continue
    nn = summarize(json.load(open(f)))
    op = os.path.join(old, b)
    oo = summarize(json.load(open(op))) if os.path.exists(op) else "(无旧图)"
    flag = "  <-- 变了" if str(nn) != str(oo) else ""
    print(f"  {b:16s} 旧={oo}  新={nn}{flag}")
PY
cp -a "$MAPS_DIR" "$OUT/new_maps"

echo "==================== [phase2.5] 合成缺失的 MRPC Stage-1 record（修上一轮 A/B 启动崩溃根因）===================="
# 解耦 stage2 从 stage1/record/<combo>/final_config.json 读前置 Stage-1 degrees；服务器上 MRPC 的
# 这条 record 不存在（最近 Stage-1 跑的是 RTE）。从已提交的 approx_per_dataset.json 合成它（真实结果，非假跑）。
python3 - <<'PY' 2>&1 | tee "$OUT/stage1_record_synth.txt" || { echo "[FATAL] Stage-1 record 合成失败"; exit 1; }
import json, os, glob, datetime
rec_root = "Parting Chapter/stage1/record"; combo = "bert base mrpc"
existing = [d for d in glob.glob(os.path.join(rec_root, combo + " *")) if os.path.isdir(d)]
if existing:
    print("[skip] 已存在真实 Stage-1 record，不覆盖：", [os.path.basename(d) for d in existing]); raise SystemExit(0)
ap = json.load(open("Model_analysis/configs/approx_per_dataset.json"))
s1 = ap["mrpc"]["stage1"]; gelu = [int(x) for x in s1["gelu"]]; softmax = [int(x) for x in s1["softmax"]]
assert 0 not in gelu, f"degree-0 不应出现: {gelu}"
date = datetime.datetime.now().strftime("%Y%m%d")
rec_dir = os.path.join(rec_root, f"{combo} 1 {date}")   # run_layout 约定 run_id = "{combo} {N} {YYYYMMDD}"
os.makedirs(rec_dir, exist_ok=True)
cfg = {"gelu_degree_per_layer": gelu, "softmax_degree_per_layer": softmax,
       "_synthesized_from": "Model_analysis/configs/approx_per_dataset.json",
       "_note": "MRPC Stage-1 final config (real result); synthesized into decoupled record so stage2 can resolve it."}
json.dump(cfg, open(os.path.join(rec_dir, "final_config.json"), "w"), ensure_ascii=False, indent=2)
print("[ok] 合成 Stage-1 record:", rec_dir, "| gelu =", gelu, "| softmax =", softmax)
PY

echo "==================== [phase3] curriculum A/B（curr_on=加课程 / curr_off=不加；启动即自动推 baseline）===================="
# A/B 唯一变量：--blb-v3-fusion-neighbor-curriculum 1/0。其余(preset/seed/K/probe/新图/新 baseline)全一致。
# 顺序跑，每组独占 4 卡 K=4。每组启动时 runner 用 static_skeletons_baseline_to_action 自动推 baseline
# 动作（= 验证「自动化代码」），并做带噪 baseline preflight；若 baseline 自动化坏了，curr_on 会快速报错退出。
copy_run_artifacts () {     # 解耦 run dir 是 combo 目录，BLB 产物在 {combo}/progress/，稳妥地找 diagnostics/
  local rundir="$1" dest="$2"; mkdir -p "$dest"
  local diagdir; diagdir=$(find "$rundir" -type d -name diagnostics 2>/dev/null | head -1)
  if [ -n "$diagdir" ]; then rsync -a --exclude='*.pt' --exclude='__pycache__' "$(dirname "$diagdir")/" "$dest/"
  else rsync -a --exclude='*.pt' --exclude='__pycache__' "$rundir/" "$dest/"; fi
}
run_variant () {
  local tag="$1" curr="$2"
  echo "-------------------- [A/B] variant=$tag curriculum=$curr episodes=$EPISODES --------------------"
  CUDA_VISIBLE_DEVICES=$DEVS bash llama_7B_LayerImportance.sh run rl \
    --preset mrpc-blb-stage2-rl \
    --blb-v3-fusion-count-action 1 \
    --blb-v3-fusion-neighbor-curriculum "$curr" \
    --stage2-search-episodes "$EPISODES" \
    --stage2-k-trials "$KTRIALS" \
    --stage2-probe-size 256 \
    --batch-size 512 \
    --blb-v3-reward-devices "$DEVS" \
    --fresh 2>&1 | tee "$OUT/${tag}_launch.log"
  sleep 12   # 启动器后台 nohup 立刻返回；PID/run dir 写在 <stage2>/LATEST_{PID,RUN_DIR}
  local pid rundir
  pid="$(cat "${CANON_STAGE2}/LATEST_PID" 2>/dev/null || true)"
  rundir="$(cat "${CANON_STAGE2}/LATEST_RUN_DIR" 2>/dev/null || true)"
  echo "[A/B] $tag launched: PID=$pid  run_dir=$rundir"
  if [ -z "$pid" ]; then echo "[A/B][FATAL] 没拿到 PID（启动失败/baseline 自动化报错？看 ${tag}_launch.log）"; return 1; fi
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done      # 等满训练，不设人为超时
  echo "[A/B] $tag training process $pid exited."
  if [ -n "$rundir" ] && [ -d "$rundir" ]; then copy_run_artifacts "$rundir" "$OUT/$tag/run"
  else echo "[A/B][warn] run_dir 不存在，尝试 glob progress"; cp -r ${CANON_STAGE2}/*/progress "$OUT/$tag/run" 2>/dev/null || true; fi
  echo "[A/B] $tag artifacts -> $OUT/$tag/run"
  # 拷出后清掉 combo 工作目录，保证下一组绝对从头(两组都是 _fusioncount_v1 变体，不清可能 resume)。带守卫防误删。
  if [ -n "$rundir" ] && [[ "$rundir" == *"/stage2/"*mrpc* ]] && [ -d "$rundir" ]; then
    rm -rf "$rundir"; echo "[A/B] cleared working dir $rundir（record/ 归档不受影响）"
  fi
}
run_variant curr_on  1   || { echo "[A/B] curr_on 失败（很可能 baseline 自动化/图/preflight 门禁触发），停止"; exit 1; }
run_variant curr_off 0   || { echo "[A/B] curr_off 失败，但仍尝试对已完成部分出对比"; }

echo "==================== [phase4] A/B 对比报告 ===================="
python3 scripts/blb_fusion_ab_compare.py \
  --run-a "$OUT/curr_on/run"  --label-a "curriculum ON" \
  --run-b "$OUT/curr_off/run" --label-b "curriculum OFF" \
  --anchor 80 --window 200 \
  --out "$OUT/fusion_ab_report.html" 2>&1 | tee "$OUT/ab_compare.log" || true

echo "==================== DONE ===================="
echo "[push] 请回传：(1) 重建后的 canonical 图  git add \"$MAPS_DIR\"   (2) 运行产物  git add \"$OUT\""
ls -la "$OUT"
```

## metadata

### 本次目标（2026-06-07）
Stage-1 删了 degree-0 动作后重跑，**最终配置更新**（gelu `[1,2,1,1,1,1,1,1,2,1,1,1]`，无 0）；
同时 **replan 逻辑改了**（新增默认融合策略）。两者都让旧 fusion 图 / baseline 失效，必须先重建，
再跑之前那组 curriculum A/B。本脚本把「重建 + 重推 baseline + A/B」串成一条龙，并用重推 baseline
这次机会**实测自动化代码是否真的能自动、正确地从 skeleton 得到 baseline 动作**（无需手改 SF）。

### 本地已逐项核对（写命令前做的验证）
1. **接口未变（用户的判断成立）**：`ReplanSession.replan / from_profile / __init__` 签名在 `ca072b0`
   **没动**；返回 dict 的键全保留（仅新增一个 `allowed_fusion_pairs`，我方代码可忽略）。`__init__.py`
   导出是**纯新增**（`DEFAULT_FUSION_POLICY/FusionPair/resolve_allowed_fusion_pairs/...`），旧导出都在。
2. **build 与 runtime 一致**：我方 build(`fusion_enum`→`InProcessInvoker`→`RescaleOptimizerBridge`→
   `replan_variables`) 和 runtime 都**不显式传** `allowed_fusion_pairs` → 都吃 `ReplanSession.replan`
   的新默认 `DEFAULT_FUSION_POLICY` → 两边规则一致，重建出来的图与训练时 replan 行为对齐。**无需改任何调用代码。**
3. **新默认融合策略（`_DEFAULT_ALLOWED_FUSION_PAIRS`）变严了**，会改变 fusion 图：
   - `block1_mrpc = []`（**完全禁融合** → block1 图会塌成 K-only / 单 option）
   - `block2_mrpc / block4 / block5_n1/n2/n4 = [(1,2)]`（只许 (1,2) 这一对融合）
   - 旧图（不限融合）：block1/2/4/n1=`[0,1]`、n2/n4=`[0,1,2]`；新策略下多半会变少
     （尤其 n2/n4 的 `fusion=2`(需两次融合)在新策略下不再可达）。phase2 会把新旧对比打印出来。
   - ⚠️ 这是**用户自己改的 replan 默认**，等于「fusion 动作空间整体变小、block1 不再有融合动作」。
     如果这不是你想要的（比如想让 RL 端继续无限制融合），需要在 RO 端调 `_DEFAULT_ALLOWED_FUSION_PAIRS`
     或在我方 build/runtime 显式传 `allowed_fusion_pairs="all"`（那是改代码，我没动）。否则按现状重建即对齐。
4. **新 skeleton + 新 replan 的 baseline 本地实测有效**（torch-free，直接调 RO）：
   `block1[30,30,27] block2[28,28,28,28] block4[35,31,31,31] block5_n1[31,28,31] block5_n2[31,31,31,31]
   block5_n4[31,31,31,31,31]` → **全 valid，fusion_count=0**（option0==baseline 成立）。地基没问题。
5. **degree-0 已停用且与新 Stage-1 一致**：新 gelu 用 {1,2}，所以只需 block5_n1/n2（+ 不分度的 block1/2/4）；
   block5_n4 一并重建(degree4 仍 ALLOWED，防将来)；block5_n0 维持停用(不在 BLOCK_TYPES)。

### 自动化检验（用户的第二诉求）落在哪
- `static_skeletons_baseline_to_action`（baseline_bootstrap）是**唯一**的 baseline 自动推导入口，
  **fusion 图构建**(`fusion_enum.py:384`)和 **A/B 启动**(`runner.py:897`)都用它：
  - phase1 图能建出来 = SF 抽取/baseline 标定自动化正确（option0 就是它推的 baseline）。
  - phase3 curr_on 启动会自动推「全 47 (layer,block)」baseline 动作 + 带噪 preflight，写
    `blb_stage2_baseline_action_full.{json,md}`；坏了就会快速报错退出（= 门禁）。产物会被拷进 `$OUT` 供你复核。
- 也就是说：**全程没有手改 baseline SF 的步骤**，baseline 完全由代码从新 skeleton 自动得到。

### A/B 设计（同上一版，硬约束不变）
- 唯一变量 `--blb-v3-fusion-neighbor-curriculum 1/0`；其余(preset/seed=42/K=4 四卡 probe/probe-size 256/
  新图/新 baseline/warmstart)两组完全一致 → 干净对照。
- 课程**不永久屏蔽任何配置**、全空间可达、限制逐渐打开（`0.5×EPISODES` 后 `fully_open`→mask 与开放 mask
  逐字节相同），已被 `tests/test_blb_fusion_curriculum.py::FullSpaceReachabilityTest` 单测锁死。
- 隔离：顺序跑，每组独占 4 卡；用解耦 canonical 工作目录 `Parting Chapter/stage2/bert base mrpc/`
  （Stage-1 前置能从 `stage1/record/` 读到）；跑完拷产物到 `$OUT/<tag>/run`，下一组 `--fresh` 清工作目录。
- 路径已核对：`LATEST_PID/RUN_DIR` 落在 `Parting Chapter/stage2/`（=`RUN_GROUP_DIR`），run dir 是 combo 目录，
  BLB 产物在其 `progress/` 下，所以拷贝用 `find diagnostics/` 兜底，comparator 的 `<run>/diagnostics/episodes.jsonl` 才对得上。
- **EPISODES 旋钮**：默认 6000（课程 ramp 0.5× 在 3000 收，3000-6000 是全开期，足以看出「打开后是否仍健康」+
  崩溃信号 ep~120 就现）；要里程碑级全规模改 60000（每组 1-2 天）。
- 判读：看 `$OUT/fusion_ab_report.html` 的 verdict + 曲线。期望 A：anchor 释放后 P1 不爆、reward 维持、
  fusion_count 受控、`loss_mean=100` 罕见；B：复现坍塌(tail P1≈99%、reward≈-5)。注意：新融合策略让动作空间变小，
  B 的坍塌可能比上次温和——这本身也是 A/B 要量的。

### 幂等 / 安全
- 图构建按 block 逐个 `--only`，比 skeleton 新就跳过 → 重触发可续(block4 12.7h 不白跑两遍)。
- `--max-enum-combos 0` = 全枚举(block4 不走捷径；新策略下若 block4 仍可融合，捷径本就会拒绝)。
- 旧图先备份到 `$OUT/old_maps`，新图另存 `$OUT/new_maps`，canonical `blb_stage2_rl/fusion_maps/mrpc/` 被覆盖更新。
- 协议：服务器只 pull/跑/回传 artifacts；源码改动都在本地。本次涉及的代码(课程 `ed797b1`、degree-0 停用
  `469474d`)都已在本地提交，服务器 pull 到含这些 + RO 改动(`ca072b0`)的 HEAD 再跑。

### 回传
把**重建后的 `blb_stage2_rl/fusion_maps/mrpc/`** 和整个 `experiments/server_command_runs/stage2_rebuild_ab_<ts>/`
（两组 run 产物 + HTML/JSON 对比 + 各 build/launch/compare 日志 + 新旧图对比）commit+push 回来。

### 历史（已完成，供参考）
- 旧 fusion full-build（`0122eb2`，旧 replan 策略）：7 张图建好；**因 replan 改了，这批图本次作废、需重建**
  （本脚本 phase1 干这事）。block4 当时 12.7h、`superset_pass=True`。
- 上一版 A/B 命令（curriculum ON/OFF）**未在服务器跑过**——因 Stage-1/replan 变更，本版在其前面加了
  「重建图 + 重推 baseline + 接口自检」并修正了产物拷贝路径，A/B 主体逻辑沿用。
