# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."
TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage1_approx_reuse_${TS}"
mkdir -p "$OUT"
echo "HEAD=$(git rev-parse HEAD)" | tee "$OUT/commit.txt"

echo "=== [1/3] bit-identity correctness test (CPU torch.equal) ==="
python tests/test_stage1_approx_reuse.py 2>&1 | tee "$OUT/correctness_test.log"
CT=${PIPESTATUS[0]}

echo "=== [2/3] BLB Stage-2 regression (shared-file safety) ==="
BLB_STRICT=0 python tests/test_blb_action_mask.py 2>&1 | tee "$OUT/blb_action_mask.log"; B1=${PIPESTATUS[0]}
BLB_STRICT=0 python tests/test_blb_stage2_rl_regressions.py 2>&1 | tee "$OUT/blb_regression.log"; B2=${PIPESTATUS[0]}

echo "=== [3/3] bert-large speedup benchmark (identical logits + speedup) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/stage1_approx_reuse_benchmark.py \
  --model-type bert-large --num-episodes 40 --batch-size 32 --seq-len 128 \
  --output-dir "$OUT" 2>&1 | tee "$OUT/benchmark.log"
BM=${PIPESTATUS[0]}

# Also run bert-base for a second data point (cheap).
echo "=== [3b] bert-base benchmark ==="
CUDA_VISIBLE_DEVICES=0 python scripts/stage1_approx_reuse_benchmark.py \
  --model-type bert-base --num-episodes 40 --batch-size 32 --seq-len 128 \
  --output-dir "${OUT}_base" 2>&1 | tee "$OUT/benchmark_base.log"; BMB=${PIPESTATUS[0]}

{
  echo "correctness_test_exit=$CT   (0 = identical logits proven)"
  echo "blb_action_mask_exit=$B1"
  echo "blb_regression_exit=$B2"
  echo "benchmark_bert_large_exit=$BM  (0 = identical logits + benchmark done)"
  echo "benchmark_bert_base_exit=$BMB"
  echo "output_dir=$OUT"
} | tee "$OUT/SUMMARY.txt"
echo "=== DONE -> $OUT ==="
```

## metadata

- **任务**：验证 bert-large（及 bert-base）Stage-1 推理加速改动 —— approx-module 复用缓存。改动只在本地 `function_handler.py`，目标是「加速版与不加速版结果完全一致，但更快」。
- **⚠️ 本次临时替换了之前的 Stage-2 60000 轮 active command**。Stage-2 那条命令仍在 git 历史里（上一版 SERVER_COMMAND.md），验证跑完后可以从历史恢复再继续 Stage-2。
- **改了什么**（全部 bit-identical，不改任何算术）：
  - `replace_layer_softmax`：Stage-1 每个 episode 不再为每层重建 `BertSelfAttentionWithAproximation`（CPU kaiming-init + state_dict copy + GPU transfer），改为按层缓存、只就地更新 `degree/lower_bound`。bert-large 每 episode 省下 24 次注意力模块重建。
  - `replace_layer_gelu` + `PolynomialGELU`：按 `(layer, degree)` 缓存 GELU 模块并缓存 coeff 张量，去掉每次 forward 的 `torch.tensor(...)` host→device 拷贝。
  - `_approx_attn_is_fresh_equivalent` 守卫：缓存模块一旦带上 BLB per-instance hook 就回退到原始重建路径 → **BLB Stage-2 路径 bit-for-bit 不变**。
- **为什么完全等价**：搜索期 BERT 权重冻结（只训练 GTrXL policy）；`GELU_MAP/SOFTMAX_MAP` 永不暴露 original 哨兵 `-1`，所以 `restore_*` 从不触发。只复制过一次冻结权重的模块，与每 episode 重建出来的模块逐位相同。
- **本次验证三步**：
  1. `tests/test_stage1_approx_reuse.py`：在 CPU（确定性）上对 reuse=ON / reuse=OFF 两个模型跑一串变化的 GELU/Softmax degree 配置，断言 logits `torch.equal`（逐位相等）；并断言 reuse 真的减少了模块重建次数；并验证 fresh-equivalence 守卫在带 hook 时回退。
  2. `tests/test_blb_action_mask.py` + `tests/test_blb_stage2_rl_regressions.py`：因为改的是共享文件 `function_handler.py`，跑 BLB Stage-2 回归确认没被影响。
  3. `scripts/stage1_approx_reuse_benchmark.py`：用真实 bert-large（24 层 / hidden 1024）+ 合成 batch，跑 install+forward 热路径，reuse ON vs OFF 各计时，断言两路 logits 一致（含 NaN 对齐），输出 per-episode install/forward/total 墙钟、speedup、模块重建计数。合成输入只影响时间不影响等价性，且 install+forward 成本由张量形状而非数值决定。
- **成功标准**：
  - `correctness_test_exit=0`（逐位相等已证明）。
  - `benchmark_bert_large_exit=0`（基准里 logits 一致；脚本检测到任何不一致会以非零退出）。
  - BLB 回归两个 exit 都为 0（共享文件没引入 Stage-2 回归）。
  - benchmark JSON 里 `identical_logits=true`、`episode_speedup>1`（install_speedup 通常远大于 1）。
- **主要输出**：`experiments/server_command_runs/stage1_approx_reuse_<timestamp>/`
  - `SUMMARY.txt`（各步 exit code）
  - `correctness_test.log`
  - `blb_action_mask.log` / `blb_regression.log`
  - `benchmark.log` + `stage1_approx_reuse_benchmark.json`（bert-large）
  - `benchmark_base.log` + `..._base/stage1_approx_reuse_benchmark.json`（bert-base）
- **协议**：服务器只负责 `git pull`、运行实验、产出/`push` artifacts；真实源码修改都在本地完成并通过 git 同步，不在服务器改源码。
- **若验证 PASS**：本地把 `function_handler.py` 改动确认为 Stage-1 默认行为（已是 `reuse_approx_modules=True`），无需进一步代码改动；随后可恢复 Stage-2 60000 轮 active command 继续 Stage-2，或按需切到 bert-large Stage-1 RL 正式跑。
- **若验证 FAIL**：把失败日志带回本地分析，定位是等价性破坏还是基准环境问题；修复在本地完成后再 push、server pull、rerun。
```
