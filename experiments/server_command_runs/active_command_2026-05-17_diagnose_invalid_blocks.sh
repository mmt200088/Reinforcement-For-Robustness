set -e

# ----------------------------------------------------------------------
# 诊断当前 RL-best ep203 的 8 个 invalid_chain 究竟落在哪些 (layer, block)
# 上，以及优化器给出的具体 reason。脚本是纯诊断（torch + Rescale_optimizer
# 都需要，所以必须在跑训练的 conda 环境里执行）。
#
# 输出：
#   reports/blb_opt/invalid_blocks/rl_best_ep203_buggy_reward/{report.md,report.json}
# 每行一条记录：(L, B) · graph_key · 失败原因 · 该 block 内每个 slot 的 SF/K 决策。
# 如果你想再对 baseline action 跑同样的诊断（应该是 0 invalid，做 sanity），
# 把下面的 ACTION_JSON 改成 baseline_action_vec.json 路径再触发一次即可。
# ----------------------------------------------------------------------

ACTION_JSON="Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/diagnostics/best_action_vec.json"
OUT_DIR="reports/blb_opt/invalid_blocks/rl_best_ep203_buggy_reward"

if [ ! -f "$ACTION_JSON" ]; then
  echo "[FATAL] action JSON not found: $ACTION_JSON" >&2
  exit 1
fi

echo "================================================================================"
echo "BLB invalid-chain diagnosis · action=$ACTION_JSON"
echo "================================================================================"

python scripts/blb_diagnose_invalid_blocks.py \
    --action-config "$ACTION_JSON" \
    --profile mrpc \
    --num-layers 12 \
    --rescale-optimizer-root Rescale_optimizer \
    --output-dir "$OUT_DIR"

echo ""
echo "================================================================================"
echo "DONE · 报告已写入 $OUT_DIR/{report.md, report.json}"
echo "本地 pull 后看 reports/blb_opt/invalid_blocks/rl_best_ep203_buggy_reward/report.md"
echo "================================================================================"
