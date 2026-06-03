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
OUT="experiments/server_command_runs/fusion_map_build_${TS}"
mkdir -p "$OUT"
echo "HEAD=$(git rev-parse HEAD)" | tee "$OUT/commit.txt"

echo "=== [1/3] fusion-count map unit tests (torch-free core; runs under torch here too) ==="
python tests/test_blb_fusion_count_map.py -v 2>&1 | tee "$OUT/fusion_unit.log"
G1=${PIPESTATUS[0]}

echo "=== [2/3] BUILD all 7 block-type fusion-count maps (real replan, multi-core) ==="
# Enumerates effective chain slots per block-type, runs real Rescale_optimizer
# replan, groups by realized fusion_count, keeps the minimum-installed-noise set
# (option 0 == baseline by construction). Needs torch (cfg dataclasses live in
# function_handler) but NO model forward / GLUE. Writes maps into the repo:
#   blb_stage2_rl/fusion_maps/mrpc/{block1_mrpc,block2_mrpc,block4,block5_n0,n1,n2,n4}.json
python scripts/blb_build_fusion_count_map.py --profile mrpc \
  --out-dir blb_stage2_rl/fusion_maps/mrpc \
  --report "$OUT/fusion_map_build.html" \
  --workers "$(nproc)" 2>&1 | tee "$OUT/build.log"
G2=${PIPESTATUS[0]}
cp -f blb_stage2_rl/fusion_maps/mrpc/_summary.json "$OUT/_summary.json" 2>/dev/null || true

echo "=== [3/3] SUMMARY ==="
{
  echo "HEAD=$(git rev-parse HEAD)"
  echo "fusion_unit_exit=$G1   (0 = torch-free map/NoiseOrder/grouping unit tests pass)"
  echo "build_exit=$G2         (0 = all 7 block-type maps built without crash)"
  echo "--- per-type options / fusion_counts / K-independence (F0 gate) ---"
  grep -E "^  block|^max_num_options|WARN" "$OUT/build.log" | tail -40
  echo "--- map files written ---"
  ls -la blb_stage2_rl/fusion_maps/mrpc/ 2>/dev/null
} | tee "$OUT/SUMMARY.txt"
echo "=== DONE -> $OUT ; maps in blb_stage2_rl/fusion_maps/mrpc/ ==="
```

## metadata

- **任务**：构建 Stage-2 **fusion-count 映射表**（设计 `docs/superpowers/specs/2026-06-03-stage2-fusion-count-action-design.md`，
  计划 `docs/superpowers/plans/2026-06-03-stage2-fusion-count-action.md` 的 Task 1–4）。
- **背景**：把 Stage-2 RL 每个 block 的动作从「24 个 per-slot SF 头」改成「`(fusion_option, K)`」。
  `fusion_option` 由这张离线映射表展开成现有 full SF vec。本命令只负责**离线构建映射表**；
  运行期接入（policy/env/runner）是后续 Task 5–7，等本表建好回传后再做。
- **构建逻辑**（每种 block-type，共 7 种，block3 冻结不建）：
  1. 复用 runner 的 `load_static_skeletons_baseline` + `static_skeletons_baseline_to_action` 取**校准过的
     max_sfs** + baseline（保证 decode 与运行期一致，避开通用 `load_max_sfs` 的 degree-1 假象）。
  2. 枚举 effective chain 槽（**rescale 只枚举 SF 值 index 1..levels-1，绝不枚举 index 0=None=丢操作**，依据心智模型第 2 条；其余槽做整轴探针，不改 `(fusion_count,total_bits)` 的钉在 max=最小噪声）。
  3. 每组合走**真实 replan** → 跳过 invalid → `apply_optimizer_output_to_cfg` → 算 **post-override 实际安装方差**。
  4. 按 realized `fusion_count` 分组，每组取最小安装方差集（按安装方案去重），按 `(fusion,var,bits,lex)` 排序，all-max baseline 自然落 option 0（守卫断言）。
- **本次是第 3 轮 RE-RUN（修两处：一处补全产物、一处真 bug）**：
  - 前两轮已修：① active-rescale 缓存空（`__file__` 路径在服务器 temp-dir 布局下静默吞错）→ 改用显式 ro_root 预热 + 报错停；
    ② 枚举了 rescale index 0(None) 让 RL「丢 rescale」，被优化器接受成支配 baseline 的配置 → 已排除 index 0。
  - 本轮新修：③ `decode_block_slots` 之前用动作字段名读 cfg 属性（名字对不上）→ 每个 option 的 `slots` 都是 `{}`，本轮改用 `_field_level_values` 按动作 index 直接解 SF，**slots 现在应有值**；④ `FusionCountMap.load` 现在跳过 `_*.json` sidecar（防 `_summary.json` 撑爆加载）。
  - **block1_mrpc / block4 单 option（fusion=[0]）是已确认的真实退化**（穷举 1485/2025、40320/91125 个 valid 配置全 fusion=0），用户已接受；不是 bug。
- **成功标准（F0 门槛，spec §8）**：
  - `fusion_unit_exit=0` 且 `build_exit=0`。
  - 每种 block-type 都建出 `>=1` 个 option，且 option 0 是 baseline（builder 内部断言 baseline 还原 all-max，否则报错停）。
  - **每类型日志的 `rescales=[...]` 非空**（block1=2、block2=4、block4=3、block5_n0=2…）；为空会报错停。
  - **预期结果（已确认，不是 stop 条件）**：block1_mrpc / block4 = `#options=1 fusion=[0]`（真实退化）；其余 5 类 = `#options=2 fusion=[0,1]`。`K-indep=True`。
  - 本轮额外检查：随便打开一个 `block*.json`，每个 option 的 `slots` 字段**应非空**（如 `{"gelu_out_sf":30,...}`）——这是本轮 ③ 的修复点。
  - 仅当出现**新异常**（某类型 option 病态大、option 0 != baseline 触发守卫报错、build 崩）才停下复审。
- **产物**：
  - 映射 JSON：`blb_stage2_rl/fusion_maps/mrpc/*.json`（运行期直接读）。
  - 报告/日志：`experiments/server_command_runs/fusion_map_build_<ts>/`（`SUMMARY.txt` / `build.log` / `fusion_map_build.html`）。
- **回传（本轮按「Claude 本地提升」走，因服务器 GitHub 推送不稳）**：把 `blb_stage2_rl/fusion_maps/mrpc/*.json`
  连同 `$OUT/` 报告**回传到本地**（如放进 `$OUT/fusion_maps_snapshot/`），由本地 Claude 把 7 个 `block*.json` 提升到
  canonical `blb_stage2_rl/fusion_maps/mrpc/` 并 commit/push（不含 `_summary.json`）。服务器不必自己 commit map。
- **协议**：服务器只 `git pull`、运行、产出/`push` artifacts；源码改动都在本地完成并经 git 同步，不在服务器改源码。
- **若 FAIL / 门槛异常**：把 `build.log` + `SUMMARY.txt` 带回本地分析定位，本地修复后再 push、server pull、rerun。
```
