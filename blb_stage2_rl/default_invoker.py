"""HeuristicStubInvoker —— 当 ``Rescale_optimizer`` 子项目不可用时的兜底实现。

仓库当前的 ``Rescale_optimizer/`` 目录可能是空的（用户单独维护）。为了让新版
BLB stage 2 RL 在 *没有* 外部子项目的情况下仍能跑通端到端训练循环，
我们提供一个轻量级的"启发式 stub"：

  * ``fusion_count``        ≈ 选择的 rescale slot 数（粗略代理）
  * ``total_bits``          ≈ Σ (cfg 里所有 SF 之和)
  * ``invalid_chain``       ≈ None；除非 SF 有显著低于安全阈值的情况
  * ``effective_rotations`` ≈ 空（不反写 rotation flag）

这套兜底估计有两个核心优点：
  1. **可微分**：``total_bits`` 随每个 SF idx 单调增加，cost reward 仍能正确指导
     RL 找到"更小 SF → 更小 total_bits"的最优解；
  2. **不需要 forward**：完全基于 cfg 字段计算，开销可忽略。

切到真实 ``Rescale_optimizer``：用户在 ``rl_tune.py`` / 训练入口上把
``stage2_rl_rescale_invoker_kind="subprocess"`` 即可；当前默认 ``"heuristic"``。
"""
from __future__ import annotations

from typing import Any, Mapping


def _gather_sf_attributes(cfg: Any) -> Mapping[str, int]:
    """收集 cfg 上所有 ``NoisePoint`` 字段的 ``scaling_factor``。"""
    out = {}
    for name, val in vars(cfg).items():
        if val is None:
            continue
        # NoisePoint
        sf = getattr(val, "scaling_factor", None)
        if sf is not None:
            try:
                out[str(name)] = int(sf)
            except (TypeError, ValueError):
                continue
            continue
        # tuple[NoisePoint] (block3.square_rescales / block5.gelu_*_rescales)
        if isinstance(val, tuple):
            for k, p in enumerate(val):
                if p is None:
                    continue
                sf = getattr(p, "scaling_factor", None)
                if sf is None:
                    continue
                try:
                    out[f"{name}[{k}]"] = int(sf)
                except (TypeError, ValueError):
                    continue
    return out


def _count_rescale_slots(cfg: Any) -> int:
    """粗略计数 cfg 上启用了多少个 ``*_rescale`` 字段（用作 fusion_count 代理）。"""
    n = 0
    for name, val in vars(cfg).items():
        if val is None:
            continue
        if name.endswith("_rescale") or "_rescale" in name:
            if hasattr(val, "scaling_factor"):
                n += 1
            elif isinstance(val, tuple):
                n += sum(1 for p in val if p is not None and hasattr(p, "scaling_factor"))
    return n


class HeuristicStubInvoker:
    """与 ``RescaleOptimizerInvoker`` Protocol 兼容的启发式实现。

    用法：
        bridge = RescaleOptimizerBridge(invoker=HeuristicStubInvoker())
    """

    def __init__(
            self,
            *,
            invalid_total_bits_threshold: int = 0,
            cfg_lookup: dict = None,
            ):
        self._invalid_total_bits_threshold = int(invalid_total_bits_threshold)
        # 上层在每次 evaluate 之前会通过 ``register_cfg`` 注册当前 (config_name, cfg)；
        # 这样我们就能从 cfg 抽 SF。如果没注册，退化到 delta_overrides 里读 int 值。
        self._cfg_lookup = dict(cfg_lookup) if cfg_lookup else {}

    def register_cfg(self, config_name: str, cfg: Any) -> None:
        self._cfg_lookup[str(config_name)] = cfg

    def clear_cfg_registry(self) -> None:
        self._cfg_lookup.clear()

    def __call__(self, config_name: str, payload: Any) -> dict:
        """兼容两种 invoker payload 形态（与 ``rescale_optimizer_bridge`` 一致）：
        bare ``delta_overrides`` dict 或 ``{"t_new": [...], "delta_overrides": {...}}``。
        """
        # 拆 payload
        if isinstance(payload, Mapping) and (
            "t_new" in payload or "delta_overrides" in payload
        ):
            delta_overrides = payload.get("delta_overrides") or {}
        else:
            delta_overrides = payload or {}
        config_name = str(config_name)
        cfg = self._cfg_lookup.get(config_name)
        sf_attrs: Mapping[str, int]
        if cfg is not None:
            sf_attrs = _gather_sf_attributes(cfg)
            fusion_count = _count_rescale_slots(cfg)
        else:
            # 从 delta_overrides 抽 int 值；"x2" 标记按 0 计入
            sf_attrs = {}
            for k, v in (delta_overrides or {}).items():
                if isinstance(v, (int, float)):
                    sf_attrs[str(k)] = int(v)
            fusion_count = sum(1 for v in (delta_overrides or {}).values() if v == "x2")

        total_bits = sum(int(s) for s in sf_attrs.values())

        invalid_chain = None
        if total_bits <= self._invalid_total_bits_threshold:
            invalid_chain = {
                "reason": "total_bits_below_threshold",
                "total_bits": total_bits,
                "threshold": int(self._invalid_total_bits_threshold),
            }

        return {
            "fusion_count": int(fusion_count),
            "valid": bool(invalid_chain is None),
            "result": {
                "valid": bool(invalid_chain is None),
                "chain": {"total_bits": int(total_bits)},
                "invalid_chain": invalid_chain,
            },
            "new_compact_config": {
                "effective_rotations": [],   # heuristic 不反写 rotation flag
                "config_name": config_name,
            },
            "delta_overrides": dict(delta_overrides or {}),
            "sf_attributes": dict(sf_attrs),
        }
