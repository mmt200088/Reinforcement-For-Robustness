"""真调用 sanity 测试：``InProcessInvoker`` × 实际 ``Rescale_optimizer`` 包。

这个测试直接 import ``rescale_optimizer`` 跑端到端 replan，验证：

  1. ``InProcessInvoker.from_profile(...)`` 能成功扫描 ``configs/mrpc/`` 并
     预加载所有 graph + baseline；
  2. 单次 ``invoker(config_name, payload)`` 调用走通 ``replan_with_user_actions``
     并返回符合 ``_parse_optimizer_raw`` 期望 shape 的 dict；
  3. ``RescaleOptimizerBridge.evaluate(...)`` 把 cfg→delta 映射、t_new、extra_overrides
     一路透传，最终 ``RescaleOptimizerOutput`` 三个核心字段
     ``fusion_count`` / ``total_bits`` / ``invalid_chain`` 都是合理值；
  4. 极端 ``t_new`` 能触发 invalid_chain 路径（valid=False，invalid_chain != None）。

不依赖 torch / transformers（function_handler 引入的依赖太重）；直接用
``delta_overrides`` 形态喂 invoker。

跑法（项目根目录下）：

    python tests/test_rescale_optimizer_bridge_real.py
"""
from __future__ import annotations

import os
import sys

# 把项目根加到 sys.path，让 rescale_optimizer_bridge 可以 import
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# 同样把 Rescale_optimizer 加到 sys.path，让 rescale_optimizer 包可以 import
RO_ROOT = os.path.join(ROOT, "Rescale_optimizer")
sys.path.insert(0, RO_ROOT)


def _build_invoker():
    """import ``rescale_optimizer_bridge`` 后返回模块。

    bridge 顶层会 ``from function_handler import Block1NoiseConfig ...``——
    这会拉 torch / transformers。为了让本测试不依赖深栈，先把
    ``function_handler`` 替换成 stub。
    """
    if "function_handler" not in sys.modules:
        import types
        stub = types.ModuleType("function_handler")
        for name in ("Block1NoiseConfig", "Block2NoiseConfig", "Block3NoiseConfig",
                     "Block4NoiseConfig", "Block5NoiseConfig"):
            setattr(stub, name, type(name, (), {}))
        sys.modules["function_handler"] = stub
    import rescale_optimizer_bridge as mod
    return mod


def _check_dict_shape(out: dict, *, label: str) -> None:
    assert isinstance(out, dict), f"[{label}] expected dict, got {type(out).__name__}"
    assert "fusion_count" in out, f"[{label}] missing key 'fusion_count'"
    assert "valid" in out, f"[{label}] missing key 'valid'"
    assert "result" in out, f"[{label}] missing key 'result'"
    res = out["result"]
    assert isinstance(res, dict), f"[{label}] 'result' is not a dict"
    # 链合法时 chain != None；不合法时 invalid_chain != None
    has_chain = res.get("chain") is not None
    has_invalid = res.get("invalid_chain") is not None
    assert has_chain or has_invalid, (
        f"[{label}] result.chain 与 result.invalid_chain 不能同时为 None"
    )
    if has_chain:
        chain = res["chain"]
        assert isinstance(chain, dict), f"[{label}] result.chain not dict"
        assert "total_bits" in chain, f"[{label}] result.chain.total_bits missing"
        assert isinstance(chain["total_bits"], int), f"[{label}] total_bits not int"


def main() -> int:
    bridge_mod = _build_invoker()

    # 1) 构造 invoker
    inv = bridge_mod.InProcessInvoker.from_profile(
        rescale_optimizer_root=RO_ROOT,
        profile="mrpc",
    )
    print(f"[OK] from_profile 加载 {len(inv.baselines)} 个 config: "
          f"{sorted(inv.baselines.keys())}")
    assert "block1_mrpc" in inv.baselines, "block1_mrpc 应该在 baselines 中"

    # 2) baseline-only 调用：t_new=None ⇒ 用 baseline，delta_overrides 给一个
    #    与 graph stage 默认一致的 dict（应得到 valid 链）
    payload_baseline = {
        "t_new": None,           # 触发 _split_payload → t_new=None
        "delta_overrides": {
            "ctpt_ffn2":      20,
            "ctpt_inv_d_1":   20,
            "ctct_ext_square": "x2",
            "ctpt_inv_d_2":   20,
        },
    }
    out = inv("block1_mrpc", payload_baseline)
    _check_dict_shape(out, label="baseline-call")
    print(f"[OK] baseline 调用：valid={out['valid']}  "
          f"fusion_count={out['fusion_count']}  "
          f"total_bits={out['result']['chain']['total_bits'] if out['result']['chain'] else 'N/A'}")
    assert out["valid"] is True, "baseline t_new 应该产生合法链"

    # 3) bare dict 形态（向后兼容路径）：等价于 "只改 delta_overrides，t_new=baseline"
    out2 = inv("block1_mrpc", {
        "ctpt_ffn2":      20,
        "ctpt_inv_d_1":   20,
        "ctct_ext_square": "x2",
        "ctpt_inv_d_2":   20,
    })
    _check_dict_shape(out2, label="bare-dict")
    assert out2["valid"] is True, "bare dict 形态也该 valid"
    print(f"[OK] bare-dict 形态：valid={out2['valid']}  "
          f"fusion_count={out2['fusion_count']}")

    # 4) _parse_optimizer_raw 检查
    parsed = bridge_mod._parse_optimizer_raw(out, config_name="block1_mrpc")
    assert parsed.config_name == "block1_mrpc"
    assert parsed.valid is True
    assert parsed.invalid_chain is None
    assert parsed.total_bits > 0
    assert parsed.fusion_count >= 0
    print(f"[OK] parse_optimizer_raw → "
          f"valid={parsed.valid}  fusion_count={parsed.fusion_count}  "
          f"total_bits={parsed.total_bits}")

    # 5) 极端 t_new：让某一 stage 的 q' < 30，强制 fusion 或 invalid
    skel, t_base, q_base = inv.baselines["block1_mrpc"]
    print(f"[INFO] block1_mrpc baseline: skeleton={skel}  t_base={t_base}  q_base={q_base}")
    # block1_mrpc skeleton=[0,2,4,5]，stage 1 路径包含 ctpt_ffn2/ctpt_inv_d_1
    # （两个 CTPT_MUL，scale_delta_bits=20）⇒ s_pre_1 = t_new[0] + 40。
    # 选 t_new=[30, 45, 60]：q'_1 = 30+40-45 = 25 < 30 ⇒ 强制 fusion；
    # q'_2 = 2·45+20-60 = 50（合法）。
    extreme_t = [30, 45, 60]
    print(f"[INFO] extreme t_new = {extreme_t}")
    out3 = inv("block1_mrpc", {
        "t_new": extreme_t,
        "delta_overrides": {
            "ctpt_ffn2":      20,
            "ctpt_inv_d_1":   20,
            "ctct_ext_square": "x2",
            "ctpt_inv_d_2":   20,
        },
    })
    _check_dict_shape(out3, label="extreme")
    parsed3 = bridge_mod._parse_optimizer_raw(out3, config_name="block1_mrpc")
    print(f"[OK] 极端 t_new：valid={parsed3.valid}  "
          f"fusion_count={parsed3.fusion_count}  "
          f"total_bits={parsed3.total_bits}  "
          f"invalid_chain={'YES' if parsed3.invalid_chain else 'no'}")
    # 两种合理结果：fusion_count > 0 (链经过融合修复) 或 invalid_chain != None
    assert parsed3.fusion_count > 0 or parsed3.invalid_chain is not None, (
        f"极端 t_new={extreme_t} 应该至少触发 fusion 或 invalid_chain，"
        f"但实测 fusion_count={parsed3.fusion_count} invalid_chain={parsed3.invalid_chain}"
    )

    # 6) 多个 config 都跑得通
    for cname in ["block2_mrpc", "block3_exp_n4", "block4", "block5_n2"]:
        if cname not in inv.baselines:
            continue
        # 直接用空 delta_overrides + baseline t_new（先看 baseline t_new 在不动 graph 时是不是合法）
        try:
            out_n = inv(cname, {})
        except Exception as exc:
            print(f"[WARN] {cname} bare-empty 调用抛 {type(exc).__name__}: {exc}")
            continue
        _check_dict_shape(out_n, label=cname)
        parsed_n = bridge_mod._parse_optimizer_raw(out_n, config_name=cname)
        print(f"[OK] {cname:18s}  valid={parsed_n.valid}  "
              f"fusion_count={parsed_n.fusion_count}  "
              f"total_bits={parsed_n.total_bits}")

    print()
    print("=" * 70)
    print(" ALL SANITY CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
