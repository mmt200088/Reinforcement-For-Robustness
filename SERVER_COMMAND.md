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
TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_degree0_verify_${TS}"
mkdir -p "$OUT"
echo "HEAD=$(git rev-parse HEAD)" | tee "$OUT/commit.txt"

echo "=== [1/3] full BLB contract gate (regression + degree-0 tests, WITH torch) ==="
# On the server torch IS available, so the 25 torch-import 'errors' seen locally
# become real runs. test_blb_degree0_stage2.py (new) runs both classes here:
#   - Degree0RescaleOptimizerContractTest (block5_n0 valid; +gelu_coeff rejected)
#   - Degree0BaselineExtractionTest (real-archive degree-0 baseline extraction)
BLB_STRICT=0 python -m unittest discover -s tests -p "test_blb_*.py" -v 2>&1 | tee "$OUT/contract_gate.log"
G1=${PIPESTATUS[0]}

echo "=== [1b] degree-0 test file alone (explicit, easy to read) ==="
BLB_STRICT=0 python -m unittest tests.test_blb_degree0_stage2 -v 2>&1 | tee "$OUT/degree0_tests.log"
G1B=${PIPESTATUS[0]}

echo "=== [2/3] noise-install full verify: MIXED degree-0 stage-1 (layers 0/4/8 = ReLU) ==="
# Drives action_vector_to_cfgs(gelu has 0) -> build_block5_cfg_from_action ->
# make_block5_default_config(gelu_degree=0) -> bridge.evaluate_blocks (block5_n0
# cost) -> apply_optimizer_output_to_cfg, all under REAL torch. softmax kept at 2
# (block3_exp_n2 has a successful baseline; n6 does not).
python scripts/blb_verify_noise_install.py --mode full --profile mrpc --num-layers 12 \
  --stage1 '{"gelu_degree_per_layer":[0,1,2,4,0,1,2,4,0,1,2,4],"softmax_degree_per_layer":[2,2,2,2,2,2,2,2,2,2,2,2]}' \
  --out "$OUT/noise_install_mixed.html" 2>&1 | tee "$OUT/noise_install_mixed.log"
G2=${PIPESTATUS[0]}

echo "=== [3/3] noise-install full verify: ALL-ReLU stage-1 (every layer degree 0) ==="
python scripts/blb_verify_noise_install.py --mode full --profile mrpc --num-layers 12 \
  --stage1 '{"gelu_degree_per_layer":[0,0,0,0,0,0,0,0,0,0,0,0],"softmax_degree_per_layer":[2,2,2,2,2,2,2,2,2,2,2,2]}' \
  --out "$OUT/noise_install_allrelu.html" 2>&1 | tee "$OUT/noise_install_allrelu.log"
G3=${PIPESTATUS[0]}

{
  echo "HEAD=$(git rev-parse HEAD)"
  echo "contract_gate_exit=$G1        (0 = no failures/errors across the whole BLB suite)"
  echo "degree0_tests_exit=$G1B       (0 = degree-0 RO contract + baseline extraction pass)"
  echo "noise_install_mixed_exit=$G2  (0 = degree-0 cfg-build + block5_n0 cost ran without crash)"
  echo "noise_install_allrelu_exit=$G3"
  echo "output_dir=$OUT"
  echo "--- block5_n0 per-config valid/cost lines (mixed run) ---"
  grep -iE "block5_n0|block5_n|\"valid\"|valid=|invalid|degree" "$OUT/noise_install_mixed.log" | head -50
} | tee "$OUT/SUMMARY.txt"
echo "=== DONE -> $OUT ==="
```

## metadata

- **任务**：验证本地新加的 **Stage-2 degree-0 (ReLU / block5_n0) 支持**（commit `2f9862e`）。
  Stage-1 某层 GELU degree=0 表示「用 ReLU 替换 GELU」，对应 Stage-2 图是
  `block5_n0`（只有 LN tail + Wffn1，无多项式 GELU 节点）。此前 Stage-2 没有
  degree-0 路径，任何含 degree-0 的 Stage-1 都会让 Stage-2 失败。
- **改了什么**（4 源文件 + 1 新测试，全部本地 torch-free 验证过）：
  - `blb_stage2_rl/baseline_bootstrap.py`：`ALLOWED_GELU_DEGREES=(0,1,2,4)`；block5
    SOURCE 映射补 `inv_std`（n0 的 SOURCE 节点名是 `inv_std` 不是 `x_mean`）→
    `x_centered_fresh_sf`。
  - `rescale_optimizer_bridge.py`：`ctpt_gelu_coeff` delta 加 `degree>=1` 门控
    （degree 0 不能发，否则 RO 判 invalid）；`DEFAULT_CFG_TO_T_NEW_MAP` 新增
    `block5_n0`（x_centered_fresh, normalize_result_rescale, wffn1_result_rescale）。
  - `blb_stage2_rl/action_space.py`：`_build_block5_action` 不再把 degree 0 钳成 1；
    `_block_default_N` block5 `<=1→N=8192`；`gelu_coeff_sf` 在 degree 0 标无效。
  - `function_handler.py`：`make_block5_default_config` 放开 degree 0；
    `replace_layer_block5_noise` 识别 `nn.ReLU` → 只装 LN tail+Wffn1、跳过 GELU 包裹。
  - `tests/test_blb_degree0_stage2.py`：RO 契约 + 真实 archive 抽取回归测试。
- **本次三步验证**：
  1. **全量 BLB 契约门** `test_blb_*.py`：服务器有 torch，本地那 25 个 torch-import
     error 在此变成真实运行；新测试 `test_blb_degree0_stage2` 的两个 class 都会跑
     （RO 契约 + 真实 archive degree-0 baseline 抽取）。成功标准：0 failures / 0 errors。
  2. **noise-install full（混合 degree-0 stage-1）**：layers 0/4/8 = ReLU。真实 torch
     下跑通 `action_vector_to_cfgs(含 degree 0)` → `make_block5_default_config(0)` →
     `bridge.evaluate_blocks`（block5_n0 成本）→ `apply_optimizer_output_to_cfg`，
     并枚举 degree-0 cfg 的噪声点。产出 HTML。softmax 用 2（`block3_exp_n2` 有成功
     baseline；n6 没有）。
  3. **noise-install full（全 ReLU stage-1）**：12 层全 degree 0，压一遍全 n0 路径。
- **成功标准**：
  - `contract_gate_exit=0` 且 `degree0_tests_exit=0`（degree-0 RO 契约 + baseline
    抽取通过；全套无回归）。
  - `noise_install_mixed_exit=0` / `noise_install_allrelu_exit=0`（degree-0 的
    cfg 构造 + block5_n0 成本在真实 torch 下不崩）。
  - HTML / 日志里 block5_n0 的 per-config `valid=true`（degree-0 层成本链合法）。
- **主要输出**：`experiments/server_command_runs/stage2_degree0_verify_<timestamp>/`
  - `SUMMARY.txt`（各步 exit code + block5_n0 valid/cost 摘要）
  - `contract_gate.log` / `degree0_tests.log`
  - `noise_install_mixed.{log,html}` / `noise_install_allrelu.{log,html}`
- **本轮范围说明**：本命令在真实 torch 下覆盖 degree-0 的 **cfg 构造 + 成本 +
  baseline + 噪声点枚举 + 全量回归**。`replace_layer_block5_noise` 的 **模型前向
  ReLU 安装分支**（一个简单的 `isinstance(nn.ReLU)` 分支）本轮未由真实模型 forward
  覆盖——它最自然的验证方式是一次带 degree-0 stage-1 的真实 Stage-2 RL 跑；等本轮
  通过后再单独安排（或在下一次 Stage-1 RL 选出 degree 0 后由 Stage-2 自然触发）。
- **协议**：服务器只 `git pull`、运行、产出/`push` artifacts；源码改动都在本地完成
  并经 git 同步，不在服务器改源码。把 `SUMMARY.txt` + 两个 HTML + 日志 push 回来。
- **若 FAIL**：把失败日志带回本地分析定位，本地修复后再 push、server pull、rerun。
```
