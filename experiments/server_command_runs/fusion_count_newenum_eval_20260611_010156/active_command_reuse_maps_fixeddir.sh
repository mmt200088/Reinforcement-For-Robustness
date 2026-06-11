set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export HF_HOME=/hy-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data

# ============================================================================
# 本轮（2026-06-11）顺序执行：
#  ⓪ SF 档位改为「统一间隔1、最多15档、无下限」（用户 spec）→ 旧图 action_indices
#    语义作废，REBUILD_MAPS=1 必须全量重建（--max-enum-combos 0，不走捷径）。
#  ① 枚举提速到极致（结果零妥协）：
#     - 直连 replan 快路径（fusion_enum_fast）：跳过 cfg 对象/bridge 缓存 deepcopy/
#       输出回写等全部中介，每组合 ≈0.10ms（本地实测；金路径 ≈10×）。模板由金路径
#       自动派生（逐槽两档探测 + sync 哨兵发现 Q/K 镜像），构建前强制做
#       baseline+corner+随机 的金vs快逐字段等价门禁，任何不一致直接 FATAL。
#     - block1 / block5_n1 用 --enum-path both：两条路径各自全量枚举，最终选项列表
#       必须逐项相等（最强交叉验证，分钟级）。其余大图用 fast + 128 随机对拍。
#     - WORKERS 放开到 nproc-2（此前 16 的封顶是误留；replan 是纯 CPU 整数运算，
#       GPU 帮不上——要上 GPU 只能重写成本真值源，正确性红线不允许）。
#     - 预计 @64核: block1 ~6s, b5_n1 ~1min(both), b5_n2 ~2min, block2 ~6min,
#       b5_n4 ~25min, block4 ~2h；构建日志每 30s 打进度+ETA。
#  ② Stage-2 episode 并行 1卡 vs N卡 确定性门禁（同 seed 300ep；逐窗 rollout_sig
#    逐字相同 + episodes.jsonl 数值逐项相同 + 加速比实测）。
#  ③ 门禁 PASS → 自动启动 60000-episode curriculum-ON fusion 长跑（里程碑）。
# 失败处理：任一 FATAL 即停（不烧 60k 预算）；门禁 FAIL 时回传对拍证据。
# ----------------------------------------------------------------------------
REBUILD_MAPS=0       # step-1×15 新档位 → 旧图作废，必须全量重建
GATE_EPISODES=300    # 门禁短跑规模：anchor80 + 220 post，5 个 PPO 窗口
LONG_EPISODES=60000  # 门禁通过后的里程碑长跑
KTRIALS=5            # K 固定为 5（与卡数解耦——这是确定性要求的一部分，勿改回 K=NGPU）
WORKERS="$( n=$(nproc 2>/dev/null || echo 8); m=$(( n - 2 )); [ "$m" -lt 1 ] && m=1; [ "$m" -gt 128 ] && m=128; echo "$m" )"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"; [ -z "$NGPU" ] && NGPU=1; [ "$NGPU" -lt 1 ] && NGPU=1
DEVS="$(seq -s, 0 $((NGPU-1)))"
echo "[gpu] 探测到 $NGPU 张卡 -> DEVS=$DEVS, K=$KTRIALS(固定); 枚举 WORKERS=$WORKERS (nproc=$(nproc 2>/dev/null || echo '?'))"
# ============================================================================

TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_grid_gate_60k_${TS}"
mkdir -p "$OUT"
SKEL="Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json"
MAPS_DIR="blb_stage2_rl/fusion_maps/mrpc"
CANON_STAGE2="Parting Chapter/stage2"

echo "==================== [phase0] 同步自检 ===================="
git rev-parse HEAD > "$OUT/HEAD.txt" 2>&1; cat "$OUT/HEAD.txt"; git log --oneline -5
python3 - <<'PY' 2>&1 | tee "$OUT/selfcheck.txt" || { echo "[FATAL] 自检失败"; exit 1; }
import rescale_optimizer as r
print("RO 导入 OK；DEFAULT_FUSION_POLICY =", r.DEFAULT_FUSION_POLICY)
import sys; sys.path.insert(0, "blb_stage2_rl")
import action_space as asp
assert not hasattr(asp, "MIN_SF_FLOOR"), "无下限（user spec）"
assert asp.LEVELS_F == asp.LEVELS_W == asp.LEVELS_MS == asp.LEVELS_R == 15
assert [asp.sf_from(i, 30, 15) for i in range(14, -1, -1)] == list(range(30, 15, -1))
assert asp.distinct_sf_level_indices(kind="F", levels=15, max_sf=20, N=16384) == [0] + list(range(5, 15))
assert asp.distinct_sf_level_indices(kind="R", levels=15, max_sf=30, N=16384) == list(range(1, 15))
print("step-1×15 档位规则 OK（间隔1/最多15档/无下限/同值档预去重）")
import fusion_enum_fast as fef
import itertools
lens = [3, 4, 5]
full = list(itertools.product(*[range(n) for n in lens]))
assert [tuple(c) for c in fef.iter_combo_range(lens, 7, 31)] == full[7:31]
print("fusion_enum_fast unranking OK")
from seed_utils import derive_probe_seed, derive_policy_step_seed, PREFLIGHT_EPISODE
print("stage2 seed_utils OK; preflight episode =", PREFLIGHT_EPISODE)
PY
# 前置 Stage-1 record（缺失则从已提交 degrees 合成；幂等）
python3 - <<'PY' 2>&1 | tee "$OUT/stage1_record_synth.txt" || { echo "[FATAL] Stage-1 record 处理失败"; exit 1; }
import json, os, glob, datetime
rec_root = "Parting Chapter/stage1/record"; combo = "bert base mrpc"
existing = [d for d in glob.glob(os.path.join(rec_root, combo + " *")) if os.path.isdir(d)]
if existing:
    print("[skip] 已存在 Stage-1 record：", [os.path.basename(d) for d in existing]); raise SystemExit(0)
ap = json.load(open("Model_analysis/configs/approx_per_dataset.json"))
s1 = ap["mrpc"]["stage1"]; gelu = [int(x) for x in s1["gelu"]]; softmax = [int(x) for x in s1["softmax"]]
assert 0 not in gelu, f"degree-0 不应出现: {gelu}"
date = datetime.datetime.now().strftime("%Y%m%d")
rec_dir = os.path.join(rec_root, f"{combo} 1 {date}")
os.makedirs(rec_dir, exist_ok=True)
json.dump({"gelu_degree_per_layer": gelu, "softmax_degree_per_layer": softmax,
           "_synthesized_from": "Model_analysis/configs/approx_per_dataset.json"},
          open(os.path.join(rec_dir, "final_config.json"), "w"), ensure_ascii=False, indent=2)
print("[ok] 合成 Stage-1 record:", rec_dir, "| gelu =", gelu)
PY

echo "==================== [phase1] step-1×15 全量重建 fusion 图（直连 replan 快路径 + 等价门禁）===================="
cp -a "$MAPS_DIR" "$OUT/old_maps" 2>/dev/null || true
if [ "$REBUILD_MAPS" = 1 ]; then
  # 仅在确实重建时才清空图目录（rm -rf 必须留在守卫内，防误删有效图）。
  rm -rf "$MAPS_DIR"
  mkdir -p "$MAPS_DIR"
  # 全量枚举（--max-enum-combos 0，不走捷径）。小图 both = 金/快两路各自全量、
  # 最终选项必须逐项相等（最强交叉验证）；大图 fast + 128 随机金vs快对拍门禁
  #（不一致即 FATAL）。小图在前：交叉验证先趟雷，大图不白跑。
  for gk in block1_mrpc block5_n1 block5_n2 block2_mrpc block5_n4 block4; do
    EPATH="fast"
    case "$gk" in block1_mrpc|block5_n1) EPATH="both" ;; esac
    echo "[maps] building $gk (enum-path=$EPATH, workers=$WORKERS) ..."
    python scripts/blb_build_fusion_count_map.py --profile mrpc --only "$gk" \
      --out-dir "$MAPS_DIR" --rescale-optimizer-root Rescale_optimizer \
      --num-layers 12 --workers "$WORKERS" --max-enum-combos 0 \
      --enum-path "$EPATH" --fast-verify-random 128 \
      > "$OUT/build_${gk}.log" 2>&1 || { echo "[FATAL] fusion 图 $gk 构建失败，见 build_${gk}.log"; tail -30 "$OUT/build_${gk}.log"; exit 1; }
    grep -E "\[fast\] template OK|\[both\] fast == golden|options=|wall=|rate=" "$OUT/build_${gk}.log" | tail -4
  done
fi

echo "==================== [phase2] 图门禁（option0==baseline）+ 新旧对比 ===================="
OUTDIR="$OUT" python3 - <<'PY' 2>&1 | tee "$OUT/map_gate.txt" || { echo "[FATAL] 图门禁失败"; exit 1; }
import glob, json, os
from blb_stage2_rl.fusion_count_map import FusionCountMap
FusionCountMap.load("mrpc")
print("FusionCountMap.load('mrpc') OK — 所有图 option0==baseline。")
def summarize(p):
    d = json.load(open(p)); o = d["options"]
    return len(o), sorted({x["fusion_count"] for x in o}), d.get("build_meta", {}).get("enum_total_combos")
old = "%s/old_maps" % os.environ.get("OUTDIR", "")
print("\n图              新(n_opt, fusion, enum_total)          旧(n_opt, fusion, enum_total)")
for f in sorted(glob.glob("blb_stage2_rl/fusion_maps/mrpc/*.json")):
    b = os.path.basename(f)
    if b.startswith("_"): continue
    nn = summarize(f)
    op = os.path.join(old, b)
    oo = summarize(op) if os.path.exists(op) else "—"
    print(f"  {b:16s} {str(nn):36s} {oo}")
PY
cp -a "$MAPS_DIR" "$OUT/new_maps"

echo "==================== [phaseG] Stage-2 episode 并行确定性门禁：1卡 vs ${NGPU}卡 ===================="
GOUT="$OUT/stage2_ngpu_gate"; mkdir -p "$GOUT"
run_gate () {   # tag, visible devs, --stage2-rl-devices 值
  local tag="$1" vis="$2" devspec="$3" pid rundir t0 t1
  echo "-------- [gate] $tag CUDA_VISIBLE_DEVICES=$vis stage2-rl-devices=$devspec episodes=$GATE_EPISODES --------"
  CUDA_VISIBLE_DEVICES="$vis" bash llama_7B_LayerImportance.sh run rl \
    --preset mrpc-blb-stage2-rl \
    --blb-v3-fusion-count-action 1 \
    --blb-v3-fusion-neighbor-curriculum 1 \
    --stage2-search-episodes "$GATE_EPISODES" \
    --stage2-k-trials "$KTRIALS" \
    --stage2-probe-size 256 \
    --batch-size 512 \
    --stage2-rl-devices "$devspec" \
    --fresh 2>&1 | tee "$GOUT/${tag}_launch.log"
  sleep 12
  pid="$(cat "${CANON_STAGE2}/LATEST_PID" 2>/dev/null || true)"
  rundir="$(cat "${CANON_STAGE2}/LATEST_RUN_DIR" 2>/dev/null || true)"
  [ -z "$pid" ] && { echo "[gate][FATAL] $tag 没拿到 PID"; return 1; }
  t0=$(date +%s); while kill -0 "$pid" 2>/dev/null; do sleep 20; done; t1=$(date +%s)
  echo "$((t1 - t0))" > "$GOUT/${tag}_walltime_s.txt"
  # 逐窗签名：workers= 在行尾，截取与卡数无关的前缀做 byte-diff
  grep -rhoE "window_start=[0-9]+ episodes=[0-9]+ rollout_sig=[0-9a-f]+" "$rundir" 2>/dev/null \
    | sort -u > "$GOUT/${tag}_sigs.txt" || true
  grep -rh "\[ANOMALY\]" "$rundir" 2>/dev/null > "$GOUT/${tag}_anomaly.txt" || true
  local diag; diag=$(find "$rundir" -type f -name episodes.jsonl 2>/dev/null | head -1)
  [ -n "$diag" ] && cp "$diag" "$GOUT/${tag}_episodes.jsonl"
  if [ -n "$rundir" ] && [[ "$rundir" == *"/stage2/"*mrpc* ]] && [ -d "$rundir" ]; then rm -rf "$rundir"; fi
}
run_gate g1 0       0          || { echo "[FATAL] 门禁 g1 失败"; exit 1; }
run_gate gN "$DEVS" "$DEVS"    || { echo "[FATAL] 门禁 gN 失败"; exit 1; }

echo "==== [gate] 判读 ====" | tee "$GOUT/verdict.txt"
GATE_PASS=1
if [ -s "$GOUT/g1_sigs.txt" ] && diff "$GOUT/g1_sigs.txt" "$GOUT/gN_sigs.txt" > "$GOUT/sig_diff.txt" 2>&1; then
  echo "[gate][PASS] rollout_sig 逐窗逐字相同（1卡 == ${NGPU}卡）" | tee -a "$GOUT/verdict.txt"
else
  GATE_PASS=0; echo "[gate][FAIL] rollout_sig 不一致或为空 → 看 sig_diff.txt" | tee -a "$GOUT/verdict.txt"
fi
if [ -s "$GOUT/g1_anomaly.txt" ] || [ -s "$GOUT/gN_anomaly.txt" ]; then
  GATE_PASS=0; echo "[gate][FAIL] 出现 [ANOMALY]（fusion 图存在 invalid 动作?）" | tee -a "$GOUT/verdict.txt"
fi
python3 - <<PY 2>&1 | tee -a "$GOUT/verdict.txt"
import json
fields = ["episode","total_reward","terminal_reward","terminal_priority","terminal_loss_mean",
          "terminal_loss_std","terminal_metric1_mean","terminal_metric2_mean","fusion_count","total_bits"]
def rows(p):
    return [tuple(json.loads(l).get(f) for f in fields) for l in open(p)]
try:
    a = rows("$GOUT/g1_episodes.jsonl"); b = rows("$GOUT/gN_episodes.jsonl")
    same = (len(a) == len(b)) and all(x == y for x, y in zip(a, b))
    print(f"[gate] episodes.jsonl 数值逐项对比: {'PASS（完全一致）' if same else 'FAIL（存在差异）'}  n={len(a)}/{len(b)}")
    if not same:
        for i,(x,y) in enumerate(zip(a,b)):
            if x != y: print("  first diff @", i, x, "vs", y); break
        raise SystemExit(2)
except FileNotFoundError as e:
    print("[gate][FAIL] episodes.jsonl 缺失:", e); raise SystemExit(2)
PY
[ $? -ne 0 ] && GATE_PASS=0
g1s=$(cat "$GOUT/g1_walltime_s.txt" 2>/dev/null || echo 0); gNs=$(cat "$GOUT/gN_walltime_s.txt" 2>/dev/null || echo 0)
python3 -c "
g1=$g1s; gN=$gNs; ep=$GATE_EPISODES; nd=$NGPU
print(f'1-GPU : {ep/(g1/3600):.0f} ep/h ({g1}s)') if g1>0 else print('1-GPU : n/a')
print(f'{nd}-GPU : {ep/(gN/3600):.0f} ep/h ({gN}s)  speedup={g1/gN:.2f}x (ideal {nd}x)') if (g1>0 and gN>0) else 0
" | tee -a "$GOUT/verdict.txt"

if [ "$GATE_PASS" != 1 ]; then
  echo "[STOP] 门禁未通过——不启动 60k。请回传 $OUT 全部产物供本地诊断。"; exit 1
fi

echo "==================== [phase60k] 门禁 PASS → 启动 ${LONG_EPISODES}-episode curriculum-ON fusion 长跑 ===================="
CUDA_VISIBLE_DEVICES=$DEVS bash llama_7B_LayerImportance.sh run rl \
  --preset mrpc-blb-stage2-rl \
  --blb-v3-fusion-count-action 1 \
  --blb-v3-fusion-neighbor-curriculum 1 \
  --stage2-search-episodes "$LONG_EPISODES" \
  --stage2-k-trials "$KTRIALS" \
  --stage2-probe-size 256 \
  --batch-size 512 \
  --stage2-rl-devices "$DEVS" \
  --fresh 2>&1 | tee "$OUT/long60k_launch.log"
sleep 12
PID60="$(cat "${CANON_STAGE2}/LATEST_PID" 2>/dev/null || true)"
RUN60="$(cat "${CANON_STAGE2}/LATEST_RUN_DIR" 2>/dev/null || true)"
[ -z "$PID60" ] && { echo "[FATAL] 60k 启动失败，看 long60k_launch.log"; exit 1; }
echo "PID=$PID60  run_dir=$RUN60  started=$(date -Is)" | tee "$OUT/long60k_RUNNING.txt"
# 监控循环：每 30 分钟记录健康快照（rolling reward / P1 P2 P3 / fusion / 进度）
DIAG60=""; for _i in 1 2 3 4 5; do DIAG60=$(find "$RUN60" -type f -name episodes.jsonl 2>/dev/null | head -1); [ -n "$DIAG60" ] && break; sleep 60; done
while kill -0 "$PID60" 2>/dev/null; do
  sleep 1800
  python3 - <<PY >> "$OUT/long60k_health.log" 2>&1 || true
import json, datetime, collections
try:
    eps=[json.loads(l) for l in open("$DIAG60")][-600:]
    pr=collections.Counter(int(e.get("terminal_priority",0) or 0) for e in eps)
    rw=sum(float(e.get("total_reward",0) or 0) for e in eps)/max(1,len(eps))
    fu=sum(float(e.get("fusion_count",0) or 0) for e in eps)/max(1,len(eps))
    last=eps[-1].get("episode") if eps else -1
    print(f"{datetime.datetime.now().isoformat()} ep={last} rolling600: reward={rw:.3f} P1={pr.get(1,0)} P2={pr.get(2,0)} P3={pr.get(3,0)} fusion={fu:.2f}")
except Exception as e:
    print(datetime.datetime.now().isoformat(), "health probe error:", e)
PY
done
echo "[60k] training process exited at $(date -Is)" | tee -a "$OUT/long60k_RUNNING.txt"
# 回收产物（不含 .pt 大件）
copy_run_artifacts () {
  local rundir="$1" dest="$2"; mkdir -p "$dest"
  local diagdir; diagdir=$(find "$rundir" -type d -name diagnostics 2>/dev/null | head -1)
  if [ -n "$diagdir" ]; then rsync -a --exclude='*.pt' --exclude='__pycache__' "$(dirname "$diagdir")/" "$dest/"
  else rsync -a --exclude='*.pt' --exclude='__pycache__' "$rundir/" "$dest/"; fi
}
[ -n "$RUN60" ] && [ -d "$RUN60" ] && copy_run_artifacts "$RUN60" "$OUT/long60k/run"
tail -5 "$OUT/long60k_health.log" 2>/dev/null || true

echo "==================== DONE ===================="
echo "[push] 请回传：(1) 新 canonical 图  git add \"$MAPS_DIR\"   (2) 全部运行产物  git add \"$OUT\""
ls -la "$OUT"
