# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command

```bash
set -e

# ----------------------------------------------------------------------
# 1) 优雅停掉前一轮（如果还在跑）。同时扫主目录 + 历史 _rdv2 临时目录，
#    防止服务器上有任何残留进程。本次起回滚 _rdv2 后缀，回到单目录形式
#    （用户反馈：多目录维护成本更高，--fresh 强制重启已足够防混用）。
# ----------------------------------------------------------------------
stop_rl_at_dir() {
  local PIDFILE="$1/rl.pid"
  [ -f "$PIDFILE" ] || { echo "[stop-rl] $1: no rl.pid"; return 0; }
  local RL_PID
  RL_PID="$(cat "$PIDFILE")"
  if [ -z "$RL_PID" ] || ! kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: pid=$RL_PID already dead"
    return 0
  fi
  echo "[stop-rl] $1: running pid=$RL_PID, SIGINT ..."
  kill -INT "$RL_PID" 2>/dev/null || true
  for i in 1 2 3 4 5 6; do sleep 10; kill -0 "$RL_PID" 2>/dev/null || break; done
  if kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: still alive after 60s, SIGTERM ..."
    kill -TERM "$RL_PID" 2>/dev/null || true
    for i in 1 2 3; do sleep 10; kill -0 "$RL_PID" 2>/dev/null || break; done
  fi
  if kill -0 "$RL_PID" 2>/dev/null; then
    echo "[stop-rl] $1: still alive after 90s, SIGKILL ..."
    kill -KILL "$RL_PID" 2>/dev/null || true
    sleep 3
  fi
  echo "[stop-rl] $1: stopped."
}

# 主目录（本次起 canonical）
stop_rl_at_dir "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005"
# 历史 _rdv2 临时目录（已废弃；如果服务器上还有跑就停掉）
stop_rl_at_dir "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005_rdv2"

# ----------------------------------------------------------------------
# 2) 先跑 torch-required 测试套件，硬卡链路 + 新增 policy init 不变式。
#    任何一个测试红了就 abort，不进 RL（防止用旧二进制白跑 9 小时）。
#    - test_blb_chain_integrity.py（19 cases）：apply_optimizer_output_to_cfg
#      四类回写 + Block2 Q/K binding + _sample_gaussian_for_point live read
#      + 本轮新增 SequentialPolicyInitTest（action_head ‖W‖<0.5 / 不动 bias /
#        warmstart bias margin>2.5）
#    - 其余 torch-free test_blb_*.py 顺带过一遍
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "Step 1/2: run contract tests (chain integrity + sequential policy init)"
echo "================================================================================"
BLB_STRICT=0 python -m unittest discover -s tests -p "test_blb_*.py" -v 2>&1 | tee /tmp/blb_test_output.log
TEST_RC=${PIPESTATUS[0]}
if [ "$TEST_RC" -ne 0 ]; then
  echo ""
  echo "[abort] contract tests FAILED (rc=$TEST_RC). Not starting RL." >&2
  echo "        Full log: /tmp/blb_test_output.log" >&2
  exit "$TEST_RC"
fi
echo ""
echo "[ok] contract tests passed."

# ----------------------------------------------------------------------
# 3) --fresh 重跑 BLB Stage-2 sequential RL（本次：policy init 修复 + Huber value loss）
#
#    上一轮（commit 0ca6de0, entropy schedule fix）anchor 阶段（eps 0-119）
#    reward = +29.86 → +31.20 ✓ baseline 正常；ent_coef schedule 生效（update
#    1-2 都是 0.00000）；但 sample 一开始（eps 120+）reward 立刻塌到 -7.89
#    （terminal=-5）。bug 报告：
#    `reports/stage2_rl/bug_reports/2026-05-19_entropy_schedule_sampling_collapse/`
#
#    根因（值损主导共享 trunk 把 warmstart bias 冲垮）：
#      · PPO update 1: policy_loss=-0.054, value_loss=60.86, clip_fraction=0.72
#      · PPO update 2: policy_loss=-0.009, value_loss=33.99, clip_fraction=0.66
#      · value_loss / policy_loss ≈ 1126x：共享 encoder 的梯度几乎全部来自
#        value head（returns~+37 而 V_init~0 → MSE 起始 ~1369，loss×0.5×backprop
#        通过 shared trunk 把 encoder 拉得很猛）
#      · action_head.weight 用 PyTorch 默认 Kaiming（‖W‖~0.55）→ encoder 演化后
#        |W@h|~4-9，远超 warmstart bias +3.5 → policy 漂离 baseline
#      · clip_fraction=0.72 是直接证据：2/3 的 trajectory 的 ratio 在 [0.8,1.2]
#        外，说明 policy 每个 update 都在剧烈变化
#      · ep 120 第一个 sample：约 14/59 个 slot 偏离 baseline → 14 fusion →
#        acc 跌穿
#
#    本次三项修复（commit 待定）：
#
#      (a) **action_head.weight 用 orthogonal(gain=0.01)** 初始化（legacy
#          noise_rl_module_v2.py line 1066 同款 trick）。让 ‖W_action‖ 在初始
#          阶段几乎为 0，bias 项独占 logit → warmstart bias 不被 encoder 演化
#          冲垮，能稳住很多个 PPO update。
#      (b) **encoder + value_head 用 orthogonal init**（gain=√2 / 1.0），bias 全 0
#          —— 标准 actor-critic 初始化，配合 (a) 让初始策略可预测。
#      (c) **value loss MSE → Huber(delta=1.0)**（v2 line 1886 同款）：
#          未归一化 returns~+37 时，MSE 的梯度幅度=37，Huber 截到 delta=1。
#          shared trunk 收到的 value 梯度幅度立刻降 ~30×，policy_grad 的相对
#          地位不再被压住。
#
#    保留不动（前几次已经验证 OK 的）：
#      · entropy schedule（anchor=0 / ramp 240ep / steady=0.02）—— commit 0ca6de0
#      · per-slot mode warmstart bias gain=3.5 —— commit 4097bea
#      · forced baseline anchor（120 ep）—— commit 4097bea
#      · v2-style clipped+tier reward —— commit b97ca83
#      · ForbiddenActionMask + rejection-sample —— commit 42cfbe4
# ----------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "Step 2/2: BLB Stage-2 Sequential RL (fresh) — policy init fix + Huber value loss"
echo "================================================================================"
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
```

---

## 元信息（meta，给人看的，agent 忽略）

- **任务**：先跑契约测试（含本轮新增 policy init 不变式），通过后 fresh 跑一轮 6000-episode sequential RL，验证 policy init + Huber value loss 修复
- **更新时间**：2026-05-19
- **更新原因**：上一轮（commit `0ca6de0`, entropy schedule fix）entropy schedule 部分**生效**（anchor 期 ent_coef=0 已落到 PPO update 中），但 sample 一开始 reward 仍立刻塌到 -7.89。bug 报告：`reports/stage2_rl/bug_reports/2026-05-19_entropy_schedule_sampling_collapse/`。
    新根因：value loss 主导共享 trunk 把 warmstart bias 冲垮。PPO update 1 `value_loss=60.86 / policy_loss=-0.054 → 1126x 比例`。`clip_fraction=0.72` 是直接证据 —— 2/3 trajectory 的 ratio 落在 [0.8,1.2] 外，说明 policy 每个 update 都在剧烈变化。配合 `action_head.weight` 用 PyTorch 默认 Kaiming（‖W‖~0.55），encoder 演化后 `|W@h|~4-9` 远超 warmstart bias +3.5 → policy 漂离 baseline → ep 120 一来就出 14 fusion + acc 跌穿。
- **本次改动汇总**：
    1. `blb_stage2_rl/sequential_policy.py`：`BLBStage2SequentialPolicy.__init__` 末尾调用新 `_init_weights()`。encoder 各 Linear 用 orthogonal(√2)；value_head orthogonal(1.0)；**action_head orthogonal(0.01)**（legacy v2 line 1066 同款 trick）。
    2. `blb_stage2_rl/sequential_policy.py`：`sequential_ppo_update` 内的 `value_loss = F.mse_loss(...)` 改为 `F.huber_loss(..., delta=1.0)`（legacy v2 line 1886 同款）。Huber 截断未归一化 returns 的梯度，shared trunk 收到的 value 梯度幅度立刻降 ~30×。
    3. `tests/test_blb_chain_integrity.py`：新增 `SequentialPolicyInitTest`（4 个 case）—— action_head ‖W‖<0.5 / action_head.bias 初始为 0 / value_head 正确初始化 / 带 warmstart bias=3.5 时随机 state 上 preferred logit margin > 2.5。
- **预期效果**（这次要看的信号 —— 越前面越强）：
    - **契约测试 19/19 + 新 policy init test 4/4 通过**（如果挂了立即 abort，不进训练）。
    - **PPO update 1 的 `value_loss` 应该 << 60.86**（Huber 把梯度幅度从 37 截到 1，shared trunk 不再被 value loss 主导）。
    - **PPO update 1 的 `clip_fraction` 应该 << 0.72**（policy 每个 update 不再剧烈变化 → 大部分 ratio 落在 [0.8,1.2] 内）。
    - **anchor 期（eps 0-119）entropy 平稳下降**（gain=0.01 weight init + 0 entropy bonus + forced baseline → policy 高度集中 baseline）。
    - **sample 期（eps 120+）reward ≥ +25**（policy 仍坐落 baseline 附近，sampled action 大多和 baseline 几乎一样 → fusion 数应 ≤ 2，acc 不动）。
    - **训练中期（eps 1000-5000）**：policy 慢慢从 baseline 邻域开始探索，total_bits 渐降，reward 可能突破 +45。
    - **训练后期（eps 5000+）**：reward 应稳定在 +40+，best 可能 +50+。
    - 用户提示：这个 RL 至少要 5 万轮才会显著收敛，前 6000 episode 主要看 **policy 是否稳坐 baseline 附近、不崩**，不期望立刻找到比 baseline 强很多的 action。
- **预期产物**（**回滚单目录**）：
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/`
        - `blb_stage2_status.json` —— 实时状态板
        - `diagnostics/diagnostics_summary.md` —— 中文诊断摘要
        - `blb_stage2_rl_checkpoint_live.pt` —— policy + optimizer + forbidden_mask_records
    - `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/`
        - `details/noise_ppo_step_info_<a>-<b>.txt`（带 `terminal_metrics: loss_mean=X loss_std=Y m1=Z`）
        - `warning.txt` —— 奖励暴跌警告
        - `pruning_search_log.txt` —— 主日志（启动头部能看到 `preferred per slot (mode over 59 steps) = [...]` 和 `强制 baseline 锚点: 前 N 个 episode...`）
- **预期耗时**：~8-9 小时。前 60 个 episode 跑 baseline action 也要 5 trials × 4 probe forward 加 59 个 optimizer call，但 PPO update 仍按 60-episode rollout 算。

### Stage-1 → Stage-2 degree 适配（用户问题 #1）

经过审查代码链：
- `_resolve_stage2_fixed_stage1_config` → `resolve_stage1_only` 从 `glue_final_configs_best_ppo.json` 的 `bert-base.mrpc.stage1.gelu/softmax` 读出 per-layer 向量。
- 传给 `BLBStage2RLRunner.run(fixed_gelu, fixed_softmax)` → 传给 `BLBStage2Env(gelu_degree=fixed_gelu, attn_degree=fixed_softmax)`。
- `BLBStage2Env._normalize_degree_vector` 把 length-L 向量直接保留（不会塌成 scalar=4）。
- `evaluate_action_for_cost(..., gelu_degree=self.gelu_degree, attn_degree=self.attn_degree)` 把向量传到 `action_vector_to_cfgs(..., gelu_degree=..., attn_degree=...)`。
- `action_vector_to_cfgs` 每层用 `_degree_for_layer(gelu_degree, li, ...)` 拿出该层的 degree，构造 Block3/5 cfg。
- `make_config_name` 用 `cfg.degree` (Block3) / `cfg.gelu_degree` (Block5) 拼 graph_key，所以每个 (layer, block) 送进 optimizer 时用的是 per-layer graph。
- **Paean/blb_action_eval.py** 走的是同一条 `action_vector_to_cfgs` 路径，per-layer 向量在 line 273-274 显式传入。

→ **训练 / final-eval 都已经在用 per-layer stage-1 degree**。之前 `report.md` 里出现的 `block5_n4` / `block3_exp_n4` 是 **诊断脚本的 bug**（读 JSON 路径错了，回落到 `[4]*12`），不是训练代码的 bug。本次也修了诊断脚本（`scripts/blb_diagnose_invalid_blocks.py:_stage1_degrees_from_meta`）。

如果对训练日志里实际用的 graph_key 还不放心，跑起来之后看 `details/noise_ppo_step_info_*.txt`，每条 `invalid_blocks` 行里都有 `graph=block5_n1_L0` / `graph=block3_exp_n2_L0` 之类字段，能直接验证 per-layer 是否正确。

## 切换到其他常用任务时（备查，agent 不读这一段）

需要换任务时，**直接覆盖上面的 active command 代码块** + 改这里的元信息。下面只是常用命令样板，不会被执行：

- 续训（同 preset 不带 `--fresh`，自动检测持久化目录、恢复 forbidden_mask）：
  `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl`
- 单独 final-eval（最新 best）：
  `bash Paean/run_final_eval.sh --preset mrpc-final-eval-only --action-config "$RUN_DIR/stage2_noise/progress/diagnostics/best_action_vec.json"`
- 单独 final-eval（baseline）：
  `bash Paean/run_final_eval.sh --preset mrpc-blb-baseline-fixed --run-name baseline_blb_s1t0.005`
- 离线诊断某个 action 的 invalid_blocks：
  `python scripts/blb_diagnose_invalid_blocks.py --action-config <path> --output-dir reports/blb_opt/invalid_blocks/<name>`
- 多 seed 扫（5 seeds，隔离持久化目录）：
  `bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 1,2,3,4,5 trial1 --fresh`

## 服务器 agent 期望

- agent 只读这个文件的**第一个 ```bash 代码块**，其余 markdown 全部忽略。
- agent 应该在仓库根目录 `bash` 执行（不要 `cd`，所有路径已经按相对仓库根写好）。
- 如果该文件未变更（git hash 未动），agent 不应重复触发同一命令 —— 由 agent 侧做幂等。
- 本次脚本会主动停掉正在跑的 RL（基于 `<slug>/rl.pid`），所以 agent 不需要额外的 pre-kill 钩子。
