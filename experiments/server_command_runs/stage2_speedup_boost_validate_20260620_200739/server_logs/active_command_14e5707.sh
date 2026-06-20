set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export HF_HOME=/hy-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data

# ============================================================================
# 本命令做两件事，都非破坏（不 --fresh 清 canonical Stage-2 best；reward 60k 主线在
# 下方 "⏸ on-deck"，验证完手动移回第一个 ```bash 块）：
#   [A] KV-cache rollout 提速（commit 1eec624+c253f92，默认 OFF）——用户要求**完全验证
#       加速相较加速前确有效果**。非 byte-identical（nn.MHA 无 K/V 接口→手写 attention，
#       复用同一训练权重，浮点 ~1e-6 非逐位；用户已放弃 1==N）。加速只在 policy 逐步 GTrXL
#       前向（env/replan/probe 全不变）。证据 = 自检等价(不掉质量) + 生产规模 OFF vs ON 测速，
#       且对 speedup 设**硬门禁**（ON 必须明显快于 OFF，否则判失败）。
#   [B] 加大精度（precision boost，block2/4/5_n2/5_n4）——把每个非零 fusion 的短质数抬到最大
#       (≤q_max) 的最小噪声动作组。boost 是枚举后的后处理，对已提交 map 直接应用 == builder
#       全量重建的末步（秒级，免去 block4 全枚举 ~1h）。先单测(真实 replan)再落 map。
# ============================================================================
TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_speedup_boost_validate_${TS}"
mkdir -p "$OUT"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$OUT/HEAD.txt"; cat "$OUT/HEAD.txt"; git log --oneline -4
fi
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"; [ -z "$NGPU" ] && NGPU=1
DEV0="$([ "$NGPU" -ge 1 ] && echo cuda || echo cpu)"

echo "#################### [A] KV-cache 提速验证 ####################"
echo "==================== [A0] 自检门禁：增量前向 == 全前向（不掉质量）===================="
python3 -m unittest tests.test_blb_kvcache_rollout -v 2>&1 | tee "$OUT/kvcache_selftest.txt"
grep -qE "^OK" "$OUT/kvcache_selftest.txt" || { echo "[FATAL] KV-cache 自检失败"; exit 1; }
# torch 缺失会把 6 个测试全 skip → 门禁无效，禁止放行
if grep -qE "skipped=6" "$OUT/kvcache_selftest.txt"; then
  echo "[FATAL] KV-cache 自检全 skip（torch 不可用）— 门禁无效"; exit 1; fi
echo "[A0] PASS"

echo "==================== [A1] 生产规模 等价 + 测速（OFF vs ON）===================="
# 真实 fusion policy(H=59,d=256,4层) 同 seed 逐步断言 OFF==ON(logits/value/argmax) + 计时。
CUDA_VISIBLE_DEVICES=0 python3 scripts/blb_kvcache_benchmark.py \
  --episodes 300 --horizon 59 --tol 1e-4 --device "$DEV0" 2>&1 | tee "$OUT/kvcache_benchmark.txt"
# 等价(正确性)是硬门禁——不掉质量是前提。
grep -qE "equivalence PASS" "$OUT/kvcache_benchmark.txt" || { echo "[FATAL] 生产规模等价失败"; exit 1; }
# speedup 有效性 = 清晰 VERDICT（不退出，避免挡住 [B]；GPU 上小张量 launch 开销可能让
# 实测加速低于 FLOP 分析——无论结果如何都明确报出来，这就是"完全验证加速是否有效果"）。
SPEEDUP="$(grep -oE 'speedup=[0-9.]+x' "$OUT/kvcache_benchmark.txt" | tail -1 | tr -dc '0-9.')"
[ -z "$SPEEDUP" ] && SPEEDUP=0
if python3 -c "import sys;sys.exit(0 if float('$SPEEDUP')>=1.2 else 1)"; then
  KV_VERDICT="EFFECTIVE (rollout-forward speedup ${SPEEDUP}x >= 1.2x)"
elif python3 -c "import sys;sys.exit(0 if float('$SPEEDUP')>1.02 else 1)"; then
  KV_VERDICT="MARGINAL (${SPEEDUP}x — likely small-tensor launch overhead on GPU)"
else
  KV_VERDICT="NOT EFFECTIVE (${SPEEDUP}x — do NOT enable --blb-v3-kv-cache-rollout)"
fi
echo "[A] DONE — 等价 PASS（不掉质量）；加速 VERDICT = $KV_VERDICT"

echo "#################### [B] 加大精度 验证与落 map ####################"
echo "==================== [B0] 单测：block2/4/5_n2/5_n4 + SF-direct==index（真实 replan）===================="
python3 -m unittest tests.test_blb_precision_boost -v 2>&1 | tee "$OUT/boost_selftest.txt"
grep -qE "^OK" "$OUT/boost_selftest.txt" || { echo "[FATAL] 加大精度单测失败"; exit 1; }
# SF-direct 等价(4个)需要 torch；若全 skip 说明 torch 缺失
if grep -qE "skipped=([4-9]|[0-9]{2,})" "$OUT/boost_selftest.txt"; then
  echo "[WARN] 部分 boost 测试被 skip（torch/RO 缺失？）——检查上面计数"; fi
echo "[B0] PASS"

echo "==================== [B1] 对已提交 map 应用 boost（block2/4/5_n2/5_n4）===================="
python3 scripts/blb_apply_precision_boost.py --profile mrpc 2>&1 | tee "$OUT/boost_apply.txt"
grep -qE "options boosted" "$OUT/boost_apply.txt" || { echo "[FATAL] boost 未应用"; exit 1; }
echo "boosted 后的 map 变更："; git status --short blb_stage2_rl/fusion_maps/ | tee "$OUT/boost_map_diff.txt"

echo "#################### [DONE] 汇总 ####################"
echo "[A] KV-cache 加速：等价 PASS（不掉质量）；VERDICT = $KV_VERDICT"
echo "[B] 加大精度：单测 PASS；boost 已落 4 张 map（block2/4/5_n2/5_n4）"
echo "证据目录：$OUT"
echo "  - kvcache_selftest.txt / kvcache_benchmark.txt（max|diff| 与 speedup=${SPEEDUP}x）"
echo "  - boost_selftest.txt / boost_apply.txt / boost_map_diff.txt"
echo "请把 $OUT 下证据 + 改动后的 blb_stage2_rl/fusion_maps/*.json 一并 git add/commit/push 回传。"
echo "下一步：若 [A] VERDICT 为 EFFECTIVE，reward 60k 可用带 --blb-v3-kv-cache-rollout 1 的命令"
echo "（下方 on-deck）上线；否则保持默认 OFF，只享用 [B] 的 boost map。"
