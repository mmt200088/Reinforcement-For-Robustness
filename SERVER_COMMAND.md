# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command  (ADR-012 边界与探索修复：近界渐变档+边缘复测+policy-K探针+ε下限 → 确定性门禁 → PASS 自动接 60k)

```bash
set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export HF_HOME=/hy-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data

# ============================================================================
# 本轮（2026-06-12，ADR-012）：第 2 次 60k 仍收敛到 fusion=0，但取证完全改写了结论：
#   - 拆分预算有效：on-policy fusion episode 对同期无融合邻居的优势全程为正
#     (+0.05~+0.19，每阶段、单翻转/多翻转都正)；
#   - 真凶 = P1 悬崖税：1327 个 P1 里 1226 个全是 fusion episode 的"边缘型"
#     (m1∈[0.833,0.858]，0 个灾难型；无融合 episode 整个 60k 只有 1 个 P1)。
#     fusion+深K 把真实 m1 推到容忍度边界，256 样本探针的抽样噪声(σ≈0.0018)
#     把它概率性砸进 -46 悬崖：期望优势 = +0.117 − 8.4%×46 ≈ −3.8 → 策略理性弃疗；
#   - 三类探针全部失效：b2 被强制基线 K 抵消(净+0.07)、b5 净 −0.86(负面教材!)、
#     b4 恒 −46×100 次(反 fusion 泛化)；
#   - upd≈700 起 entropy=0.000 / clip=0.000：策略完全冻结，最后 18k episode 纯浪费。
# 本轮四项修复（ADR-012，代码已在本提交内；图无需重建）：
#   ① 近界渐变档(reward.near_miss_*)：非 invalid 的指标失败若最差通道亏损
#     ≤1×|baseline−阈值| 宽度，tier 从悬崖改为 35→15 线性渐变(典型上轮 P1 从
#     −7 → ~26.6)；priority 仍是 1(选择/排名语义不变；渐变上限 35 < P3 下限 40，
#     P3 永远压住近界)。带外/灾难型保留旧悬崖。
#   ② 边缘复测(env.borderline_retest)：边缘指标失败用盐化确定性种子做一次
#     2×trials 全新复测，复测结论替代首测 → 概率性误杀率平方级下降，真违规
#     照样挂。只在确定性探针路径生效(probe_noise_seed)，1==N 不受影响。
#   ③ 探针 v2：只强制目标块类型的 option=1；K 与其它块全部跟随当前策略+课程
#     (目标 option 档注入 mask)→ 探针变成"在当前策略上追加融合"的干净反事实，
#     优势 ≈ +1.4 而非被基线 K 抵消的 +0.07。轮换改为 b2→b5(剔除必死的 b4)。
#   ④ ε 探索下限(policy 混合分布，采样与 PPO 回放同分布)：option 槽 0.05、
#     K 槽 0.02 → 策略永远不能在 fusion 选择上变成确定性(根治 entropy=0 冻结)。
#   ⑤ 提速（2026-06-12，画像驱动）：上轮 60k 实测每 episode 墙钟 = 探针 2.69s
#     (78%) + rollout 0.74s (21%) + replan 0.009s；窗口负载不均仅 2.8%；PPO
#     更新 ≈1.4h/13.85h。最大安全收益 = 每卡 2 个 worker（episode 结果只依赖
#     全局序号→worker 指派无关，正是 1==N 的根基）：一个 worker 的 CPU 侧
#     rollout/簿记与同卡兄弟的 GPU 探针重叠。两个 RNG 原子单元加同卡锁：
#     (manual_seed→sample) 与 (reseed_noise→单个 trial forward)，交错顺序不
#     影响各 trial 噪声流。默认 1（与旧行为逐位一致）；gN 门禁与 60k 用 2，
#     g1 保持 1 worker 作参照——同一条 byte-diff 同时验证卡数与 worker 数两个
#     不变量。预期 ~1.3-1.4×（13.5h → ~10h）。
# 容忍度沿用用户 spec：stability 500% + 指标 0.5%。
# 顺序：phase0 自检(ADR-012 断言+三件测试) → phase2 图门禁(REBUILD_MAPS=0) →
# phaseG 1卡vsN卡确定性门禁(探针出现性判读改为动态检测——上轮发现 gate 与长跑
# anchor 不同) → PASS 自动接 60k。
# ----------------------------------------------------------------------------
REBUILD_MAPS=0       # 图在 2026-06-11 已按 step-1×15 重建并 push（本两轮只改 RL 代码，动作→SF 解码未变）
GATE_EPISODES=300    # 门禁短跑规模：anchor80 + 220 post，5 个 PPO 窗口
LONG_EPISODES=60000  # 门禁通过后的里程碑长跑
KTRIALS=5            # K 固定为 5（与卡数解耦——这是确定性要求的一部分，勿改回 K=NGPU）
ANCHOR_EPISODES=80   # 与 mrpc-blb-stage2-rl preset 保持一致；探针从 anchor 后第一个 episode 开始
FUSION_PROBE_INTERVAL=200
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
# ---- ADR-011/012 断言：预算拆分 + 探针轮换(2,5) + 近界渐变 + 复测 + ε ----
import fusion_curriculum as fcur
assert fcur.FUSION_PROBE_BLOCK_ROTATION == (2, 5), "ADR-012: b4 必须已剔除"
assert fcur.fusion_probe_target_block(80, anchor_episodes=80, interval=200) == 2
assert fcur.fusion_probe_target_block(280, anchor_episodes=80, interval=200) == 5
assert fcur.fusion_probe_target_block(480, anchor_episodes=80, interval=200) == 2
assert fcur.fusion_probe_target_block(81, anchor_episodes=80, interval=200) is None
import fusion_cost, reward as rwd
assert abs(rwd.FUSION_COST_BUDGET_FRACTION - 2.0/3.0) < 1e-12
ch = [fusion_cost.BlockChoice(2, "block2_mrpc", 1, 1, 13)]
r = fusion_cost.compute_fusion_cost_saving(ch, fusion_w=rwd.FUSION_COST_W, trunc_w=rwd.TRUNC_COST_W)
assert abs(r.fusion_norm - 1.0) < 1e-12 and r.trunc_norm == 0.0
w = rwd.RewardWeights()
assert (w.near_miss_tier_cap, w.near_miss_tier_floor, w.near_miss_band) == (35.0, 15.0, 1.0)
base = rwd.BaselineCostStats(total_bits_sum=1000, total_fusion_count=0, avg_k=13.0,
                             loss_mean=0.34, loss_std=0.002, metric1_mean=0.8672,
                             metric2_mean=0.8672, metric1_std=0.001, metric2_std=0.001)
w2 = rwd.RewardWeights(baseline_metric1=0.8672, baseline_metric2=0.8672)
class _O:
    any_invalid = False; total_bits_sum = 1000; total_fusion_count = 0
def _r(m1):
    m = rwd.EpisodeMetrics(loss_mean=0.34, loss_std=0.002, metric1_mean=m1,
                           metric2_mean=m1, metric1_std=0.001, metric2_std=0.001)
    return rwd.compute_reward(m, _O(), action_avg_k=13.0, baseline=base, weights=w2,
                              acc_threshold=0.858, acc_threshold_m2=0.858,
                              stab_threshold=0.05, external_cost_score=0.0,
                              external_cost_rank=0.0)
bnm = _r(0.8540)   # 上轮典型边缘 P1
assert bnm.priority == 1 and bnm.near_miss and 20.0 < bnm.reward < 35.0
assert _r(0.8672).reward >= 40.0 > bnm.reward                  # P3 永远压住近界
assert _r(0.3200).reward < -4.0 and not _r(0.3200).near_miss   # 灾难型保留悬崖
print("ADR-012 近界渐变档断言 OK（上轮典型边缘P1 -7 -> %.1f）" % bnm.reward)
import sys as _sys
_sys.path.insert(0, ".")
from blb_stage2_rl.env import BLBStage2EnvConfig
ec = BLBStage2EnvConfig()
assert ec.borderline_retest_enabled and ec.borderline_retest_trials_multiplier == 2
from blb_stage2_rl.sequential_runner import SequentialTrainConfig as _STC
stc = _STC()
assert abs(stc.fusion_exploration_epsilon - 0.05) < 1e-12
assert abs(stc.fusion_exploration_epsilon_k - 0.02) < 1e-12
print("ADR-012 复测/ε 默认值断言 OK")
# ---- workers-per-device（2026-06-12 提速）断言 ----
from blb_stage2_rl.parallel_runner import expand_device_ids_for_workers
assert expand_device_ids_for_workers([0, 1, 2, 3, 4], 2) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
assert expand_device_ids_for_workers([0, 1], 1) == [0, 1]   # wpd=1 == 旧行为
from blb_stage2_rl.runner import BLBStage2TrainConfig as _BTC
assert _BTC().stage2_workers_per_device == 1
print("workers-per-device 断言 OK（默认 1 = 旧行为；gN/60k 用 2）")
PY
echo "==================== [phase0b] ADR-012 单元测试（torch 在位：ε混合/复测/近界档/轮换） ===================="
for f in test_blb_fusion_curriculum test_blb_fusion_reward test_blb_fusion_exploration; do
  python3 "tests/${f}.py" > "$OUT/unittest_${f}.log" 2>&1 || { echo "[FATAL] ${f} 失败"; tail -20 "$OUT/unittest_${f}.log"; exit 1; }
  tail -1 "$OUT/unittest_${f}.log"
done
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
run_gate () {   # tag, visible devs, --stage2-rl-devices 值, workers-per-device
  local tag="$1" vis="$2" devspec="$3" wpd="${4:-1}" pid rundir t0 t1
  echo "-------- [gate] $tag CUDA_VISIBLE_DEVICES=$vis stage2-rl-devices=$devspec wpd=$wpd episodes=$GATE_EPISODES --------"
  CUDA_VISIBLE_DEVICES="$vis" bash llama_7B_LayerImportance.sh run rl \
    --preset mrpc-blb-stage2-rl \
    --blb-v3-fusion-count-action 1 \
    --blb-v3-fusion-neighbor-curriculum 1 \
    --stage2-workers-per-device "$wpd" \
    --stage2-search-episodes "$GATE_EPISODES" \
    --stage2-k-trials "$KTRIALS" \
    --stage2-probe-size 256 \
    --batch-size 512 \
    --stage2-rl-devices "$devspec" \
    --blb-v3-warmstart-anchor-episodes "$ANCHOR_EPISODES" \
    --stage2-stability-tolerance 5.0 \
    --stage2-limit-tolerance 0.005 \
    --blb-v3-fusion-probe-interval "$FUSION_PROBE_INTERVAL" \
    --blb-v3-fusion-exploration-epsilon 0.05 \
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
# g1 = 最简参照（1 worker 总量）；gN = 生产配置（N 卡 × 2 worker/卡）。
# 同一条 byte-diff 同时验证「卡数无关」与「worker 数无关」两个不变量。
run_gate g1 0       0       1  || { echo "[FATAL] 门禁 g1 失败"; exit 1; }
run_gate gN "$DEVS" "$DEVS" 2  || { echo "[FATAL] 门禁 gN 失败"; exit 1; }

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
# ADR-012 探针出现性（动态检测，anchor 无关——上轮发现 gate 与长跑 anchor 不同）：
# 300ep 门禁内必须出现 >=2 个 forced_fusion_probe episode；前两个目标依次是
# b2、b5（轮换 2->5，b4 已剔除），且各自 fusion_count>=12（目标块 12 层全部
# 强制 option1；probe-v2 下其余块跟随策略+课程，只增不减）。
python3 - <<PY 2>&1 | tee -a "$GOUT/verdict.txt"
import json
probes = []
for l in open("$GOUT/gN_episodes.jsonl"):
    d = json.loads(l)
    mode = str(d.get("exploration_mode", ""))
    if mode.startswith("forced_fusion_probe_"):
        probes.append((int(d.get("episode", -1)), mode, int(d.get("fusion_count", -1))))
probes.sort()
print(f"[gate] forced_fusion_probe episodes in gate run: {probes}")
ok = len(probes) >= 2
if ok:
    (e1, m1, f1), (e2, m2, f2) = probes[0], probes[1]
    ok = (
        m1.endswith("_b2") and f1 >= 12
        and m2.endswith("_b5") and f2 >= 12
        and (e2 - e1) == int("$FUSION_PROBE_INTERVAL")
    )
print(f"[gate] probe presence/rotation/fc check: {'OK' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 2)
PY
[ $? -ne 0 ] && { GATE_PASS=0; echo "[gate][FAIL] fusion 探针未按计划出现" | tee -a "$GOUT/verdict.txt"; }
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
  --blb-v3-warmstart-anchor-episodes "$ANCHOR_EPISODES" \
  --stage2-stability-tolerance 5.0 \
  --stage2-limit-tolerance 0.005 \
  --blb-v3-fusion-probe-interval "$FUSION_PROBE_INTERVAL" \
  --blb-v3-fusion-exploration-epsilon 0.05 \
  --stage2-workers-per-device 2 \
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
```

## metadata

### 本次目标（2026-06-12，ADR-012 边界与探索修复重跑）

1. **背景**：第 2 次 60k（artifacts `stage2_grid_gate_60k_20260612_004130`）门禁 PASS、跑满 60000，但仍收敛到 fusion=0。取证改写结论：ADR-011 的预算拆分**有效**（on-policy fusion 优势全程为正 +0.05~+0.19），真凶是 **P1 悬崖税**——1327 个 P1 中 1226 个是 fusion episode 的边缘型（m1∈[0.833,0.858]，0 个灾难型；无融合 episode 仅 1 个 P1），fusion+深K 把真实 m1 推到容忍度边界，256 样本探针噪声(σ≈0.0018)概率性把它砸进 −46 悬崖，期望优势 −3.8。叠加：b2 探针被强制基线 K 抵消（净 +0.07）、b5 探针净 −0.86（负面教材）、b4 探针 100/100 P1（反 fusion 泛化）、upd≈700 起 entropy=0.000/clip=0.000（策略冻结 18k episode）。
2. **本提交四项修复**（ADR-012；详见 `docs/adr/ADR-012-*.md`）：①近界渐变档（near_miss tier 35→15，典型上轮 P1 −7→约 26.6；priority 仍 1，P3 下限 40 永远压住）；②边缘复测（盐化确定性种子 2×trials 全新复测替代首测，误杀率平方级下降，1==N 保持）；③探针 v2（只强制目标块 option=1，K+其余块跟随当前策略+课程；轮换 b2→b5，b4 剔除）；④ε 探索下限（option 槽 0.05 / K 槽 0.02 混合分布，采样与 PPO 回放同分布，根治 entropy=0 冻结）。
3. **流程**：REBUILD_MAPS=0 → 自检（ADR-012 断言+3 件单测，torch 在位）→ 图门禁 → 1卡vsN卡确定性门禁（探针出现性判读改为动态检测：前两个探针 = b2、b5，fc≥12，间隔=interval）→ PASS 自动接 60k（容忍度沿用 5.0/0.005）。
4. **60k 观察重点**：①探针 episode 的 reward 应明显高于邻近 episode（b2 探针 ≈ +1.4，不再被基线 K 抵消）；②on-policy fusion 尝试不应再在中途归零（ε 下限 + 渐变档保证）；③entropy 永不为 0、clip_fraction 不长期为 0；④P3 episode 的 fusion_count 应随训练增长（理论最优 ≈ b2+b5 全开 24 个 fusion + 适度 K，reward ≈ 41+）；⑤`borderline_retest` 字段出现率与翻转率（复测改判比例）；⑥near_miss(P1) 占比可接受（渐变档下它是探索成本，不是事故）。

### 关键事实（给人看的）

- **新期望优势算术**（修复后）：从收敛态翻转 1 个 b2 fusion = +0.117，其 P1 误杀税 ≈ 0.2%×6 ≈ 0.01（复测+渐变档双保险）→ 净 +0.1 量级，PPO 可学。b2+b5 全融合 +1.78 ≫ P3 margin 上限 0.5；K 单独 ≤1.5，fusion 是登顶 P3 的必经之路。
- **ε 混合是真策略分布**：sample 与 evaluate（含 PPO 回放）用同一混合，log-prob 比率良定；eps=0 时逐位还原旧分布（单测锁定）。
- **探针正常计分**：probe-v2 走采样分支 + 目标槽覆写 + evaluate_action 重算 log_prob/value；动作来自全有效图必 valid。
- **近界渐变不破硬优先级**：priority 仍为 1，best 选择/候选排序的 tuple rank 不变；渐变只作用于 PPO 标量，且上限 35 < P3 下限 40（cost 永远不能补偿精度违规——mental-model item 7 保持）。
- `--blb-v3-reward-devices`（K-split）不再使用；`--stage2-rl-devices`（互斥）；`KTRIALS` 固定 5；噪声/策略/更新/复测按 (seed, 全局episode, …) 键控。
- **workers-per-device（提速，画像驱动）**：上轮实测探针占 78%（GPU-bound）、rollout 21%（CPU 重）→ `--stage2-workers-per-device 2` 让同卡两 worker 互相掩盖空隙；RNG 原子单元（seed→sample、reseed→trial forward）持同卡锁，trial 级交错不改变各自噪声流 → 任意 worker 数结果逐字节一致（gN 用 2 vs g1 用 1 的 byte-diff 直接验证）。显存 ≈2×4GB/卡 ≪ 32GB。预期 60k 13.5h→~10h。straggler 实测仅 2.8%（不做动态分配）；PPO 更新 1.4h 维持现状（改数值精度有训练语义风险，不动）。

### 预期产物

- `$OUT/selfcheck.txt`（ADR-012 断言）+ `unittest_*.log` ×3
- `$OUT/map_gate.txt`、`$OUT/stage2_ngpu_gate/`（g1/gN sigs、sig_diff、episodes.jsonl ×2、verdict.txt 含动态探针判读）
- `$OUT/long60k/run/`（diagnostics 全套，无 .pt）+ `long60k_health.log`

### 幂等 / 安全

- Stage-1 record 合成幂等。门禁两跑各自 `--fresh` 且拷出产物后清工作目录。
- 60k 用 `--fresh`；若 agent 会话中断，训练进程(nohup)继续，下次触发只做产物回收。
- 门禁任何 FAIL 都不启动 60k。

### 历史（已完成，供参考）

- 2026-06-12：ADR-011 修复后第 2 次 60k 跑满（artifacts `stage2_grid_gate_60k_20260612_004130`，1==5 PASS、3.62×、4333 ep/h）仍 fusion=0 → 取证定位 P1 悬崖税/探针 K 抵消/entropy 冻结 → 本轮 ADR-012。
- 2026-06-11：step-1×15 全量重建 6 图 + 确定性门禁 PASS（1==5 逐字、3.62×）+ 第 1 次 60k fusion=0 坍缩（artifacts `stage2_grid_gate_60k_20260611_031751`，诊断→ADR-011）。
- 2026-06-10：5 卡 A/B 完成（artifacts `stage2_rebuild_ab_20260610_013046`，提交 0457fc0）；stageA Stage-1 1vs5 确定性 PASS、3.75×。
- 2026-06-07~09：fusion 图按新 replan 策略重建（1ad078c）；A/B 启动崩溃修复（c16d0f7：合成 MRPC Stage-1 record）；Stage-1 多卡确定性+提速（a1cf152/15c16ad）。
- 2026-06-05~06：fusion 课程上线（ed797b1）、degree-0 停用（469474d）、RO 默认融合策略（957ff7a 轮）。
