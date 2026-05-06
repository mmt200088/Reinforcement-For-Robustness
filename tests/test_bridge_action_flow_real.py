"""动作流真调用核查：``RescaleOptimizerBridge.evaluate(...)`` 现在应该：

  1. 接受 ``"block1_mrpc_L0"`` 这种带 ``_L<i>`` 后缀的 RL 端 config_name，
     自动剥成 graph_key ``"block1_mrpc"`` 喂 invoker（Bug A）。
  2. 当 caller 没传 ``t_new`` 时，自动从 cfg 派生 t_new 喂 invoker（Bug B）。
  3. 派生出的 t_new 真的影响 optimizer 的 chain：用同一 graph_key 但 cfg 里
     rescale SF 选择不同 ⇒ total_bits 应当不同。

跑法：

    python tests/test_bridge_action_flow_real.py
"""
from __future__ import annotations

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Rescale_optimizer"))

# 不引入 torch / transformers：给 function_handler 装 stub
if "function_handler" not in sys.modules:
    stub = types.ModuleType("function_handler")
    for name in ("Block1NoiseConfig", "Block2NoiseConfig", "Block3NoiseConfig",
                 "Block4NoiseConfig", "Block5NoiseConfig"):
        setattr(stub, name, type(name, (), {}))
    sys.modules["function_handler"] = stub


def main() -> int:
    import rescale_optimizer_bridge as M

    # 简单 cfg：只用 ducktyping，不依赖真 BLB cfg 类（本测试只关心 SF 字段抽取）
    class FakeNP:
        def __init__(self, sf):
            self.scaling_factor = int(sf)

    class FakeBlock1Cfg:
        # 必选字段（被 default_block1_cfg_to_delta 用到）
        def __init__(self, *, gelu_out_sf=30,
                     wffn2_sf=22, mean_inv_d_sf=22, var_inv_d_sf=22,
                     mean_rescale_sf=34, var_rescale_sf=34):
            self.gelu_out_fresh = FakeNP(gelu_out_sf)
            self.wffn2_encode = FakeNP(wffn2_sf)
            self.mean_inv_d_encode = FakeNP(mean_inv_d_sf)
            self.var_inv_d_encode = FakeNP(var_inv_d_sf)
            # rescale 字段（cfg_to_t_new_from_table 会用 mean / var）
            self.mean_result_rescale = FakeNP(mean_rescale_sf)
            self.var_result_rescale = FakeNP(var_rescale_sf)
            # 其它 rescale（不在 mrpc skeleton 中，不进 t_new）
            self.wffn2_result_rescale = None
            self.square_result_rescale = None

    inv = M.InProcessInvoker.from_profile(
        rescale_optimizer_root=os.path.join(ROOT, "Rescale_optimizer"),
        profile="mrpc",
    )
    bridge = M.RescaleOptimizerBridge(invoker=inv)

    # ---------- 测试 1：layered config_name 不再 KeyError（Bug A 修复） ----------
    cfg_baseline = FakeBlock1Cfg(mean_rescale_sf=34, var_rescale_sf=34)
    out1 = bridge.evaluate(
        config_name="block1_mrpc_L0",   # 带 _L<i> 后缀
        block_name="block1",
        cfg=cfg_baseline,
    )
    assert out1.config_name == "block1_mrpc_L0", (
        f"config_name 应保留原始 layered 形式，实际 {out1.config_name!r}"
    )
    assert out1.valid, f"baseline cfg 应该 valid，实际 invalid_chain={out1.invalid_chain}"
    assert out1.raw.get("_graph_key") == "block1_mrpc", (
        f"raw 应回写 _graph_key=block1_mrpc，实际 {out1.raw.get('_graph_key')!r}"
    )
    assert out1.raw.get("_t_new_used") == [30, 34, 34], (
        f"baseline cfg 自动派生的 t_new 应等于 baseline t=[30,34,34]，"
        f"实际 {out1.raw.get('_t_new_used')!r}"
    )
    assert out1.raw.get("_t_new_source") == "cfg_derived"
    print(f"[OK] Bug A 修复：layered config_name 跑通；total_bits={out1.total_bits}")

    # ---------- 测试 2：cfg 改 mean_rescale_sf ⇒ optimizer 看到不同 t_new（Bug B 修复） ----------
    cfg_lower_mean = FakeBlock1Cfg(mean_rescale_sf=30, var_rescale_sf=34)  # t_new[1]: 34→30
    out2 = bridge.evaluate(
        config_name="block1_mrpc_L1",
        block_name="block1",
        cfg=cfg_lower_mean,
    )
    assert out2.raw.get("_t_new_used") == [30, 30, 34], (
        f"cfg 改 mean_rescale=30 后 t_new 应为 [30,30,34]，实际 {out2.raw.get('_t_new_used')!r}"
    )
    # baseline t=[30,34,34]，新 t=[30,30,34]：stage 1 q=70-30=40 (vs 36)，stage 2 q=2*30+20-34=46 (vs 54)
    # 总 bits: head 60 + 40 + 46 + tail 60 = 206 ≠ baseline 210
    print(f"[OK] Bug B 修复：t_new 改了 → total_bits 从 {out1.total_bits} → {out2.total_bits}")
    assert out2.total_bits != out1.total_bits, (
        f"cfg 改 rescale_sf 后 total_bits 应不同（baseline {out1.total_bits} vs new {out2.total_bits}）"
    )

    # ---------- 测试 3：用户显式 t_new 仍生效（不被 cfg 派生覆盖） ----------
    out3 = bridge.evaluate(
        config_name="block1_mrpc_L2",
        block_name="block1",
        cfg=cfg_lower_mean,           # cfg 派生会得 [30,30,34]
        t_new=[30, 34, 34],            # 用户强制指定 baseline t
    )
    assert out3.raw.get("_t_new_used") == [30, 34, 34]
    assert out3.raw.get("_t_new_source") == "user_provided"
    assert out3.total_bits == out1.total_bits, (
        f"显式 t_new=baseline 应得 baseline total_bits={out1.total_bits}，实际 {out3.total_bits}"
    )
    print(f"[OK] 用户显式 t_new 优先：total_bits={out3.total_bits}（与 baseline 一致）")

    # ---------- 测试 4：evaluate_blocks 多层并行也能跑 ----------
    requests = {
        "block1_mrpc_L0": ("block1", cfg_baseline),
        "block1_mrpc_L1": ("block1", cfg_lower_mean),
        "block1_mrpc_L2": ("block1", FakeBlock1Cfg(mean_rescale_sf=36, var_rescale_sf=36)),
    }
    outs = bridge.evaluate_blocks(requests)
    assert set(outs.keys()) == set(requests.keys())
    # 每条 cfg 对应不同 t_new ⇒ 至少有 2 个不同 total_bits
    bits_set = set(o.total_bits for o in outs.values())
    print(f"[OK] evaluate_blocks 三层独立：total_bits = {sorted(o.total_bits for o in outs.values())}")
    assert len(bits_set) >= 2, "三层不同 cfg 应有至少 2 个不同 total_bits"

    # ---------- 测试 5：disable auto_t_new_from_cfg 时退回 baseline 行为 ----------
    bridge_no_auto = M.RescaleOptimizerBridge(invoker=inv, auto_t_new_from_cfg=False)
    out5 = bridge_no_auto.evaluate(
        config_name="block1_mrpc_L0",
        block_name="block1",
        cfg=cfg_lower_mean,            # cfg 改了 mean_rescale_sf=30
    )
    # auto 关闭 ⇒ invoker 走 baseline t ⇒ total_bits 应等于 out1
    assert out5.total_bits == out1.total_bits, (
        f"auto 关闭时应走 baseline t，total_bits 应等于 {out1.total_bits}，实际 {out5.total_bits}"
    )
    print(f"[OK] auto_t_new_from_cfg=False 时退回 baseline 行为")

    print()
    print("=" * 70)
    print(" 动作流核查：Bug A + Bug B 全部修复，行为正确")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
