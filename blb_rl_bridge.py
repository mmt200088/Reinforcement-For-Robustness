"""BLB 噪声与 RL 动作的桥接层。

把 RL agent 输出的"动作（scaling factor 选择）"转成 BLB Block 1-5 + first-input
的 ``Block*NoiseConfig``，并通过 ``ReversibleLayerHandler.replace_layer_block*_noise``
完整安装到模型上；forward 完成后通过 ``clear()`` 一键还原。

设计要点：
  * 与现有 ``function_handler.py`` 的 BLB API 完全独立 —— 桥接层只是"调度器"，
    不引入新的噪声机制。
  * BLB 与 legacy 在 ``ReversibleLayerHandler`` 里已强制互斥（``apply()`` 触发
    ``_check_blb_legacy_conflict``，残留 legacy 时抛 RuntimeError）。
  * 所有 σ² 都通过 ``NOISE_VARIANCE_TABLE_BY_N`` 查表得到（不写死）。
  * Block 3 / Block 5 是 degree-aware 的，桥接层的 ``apply()`` 要求传入
    每层的 degree（softmax / GELU），保证 cfg.degree 与 attention.degree /
    PolynomialGELU.degree 严格匹配。

典型用法（RL stage 2 一个回合）：

    bridge = BLBNoiseRLBridge(handler, layers_attribute="model.bert.encoder.layer")

    # 1) RL agent 输出一个"动作"（每层每个 noise 点选一个 scaling factor）
    action = agent.sample_action(state)

    # 2) 把动作翻译成 cfg 字典（业务侧可自定义具体的翻译规则）
    block1_cfgs = {
        i: build_block1_cfg_from_action(action.block1[i], N=8192)
        for i in range(num_layers)
    }
    # block2_cfgs / block3_cfgs / block4_cfgs / block5_cfgs 同理 ...

    # 3) 一次性把所有噪声装上模型
    bridge.apply(
        first_input_sf=action.first_input_sf, first_input_N=16384,
        block1_cfgs=block1_cfgs, block2_cfgs=block2_cfgs,
        block3_cfgs=block3_cfgs, block4_cfgs=block4_cfgs, block5_cfgs=block5_cfgs,
    )

    # 4) forward → reward
    logits = model(input_ids, attention_mask=mask).logits
    reward = compute_reward(logits, labels)

    # 5) 还原（必须！否则下一个回合会复合上去）
    bridge.clear()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict

from function_handler import (
    NoisePoint,
    Block1NoiseConfig,
    Block2NoiseConfig,
    Block3NoiseConfig,
    Block4NoiseConfig,
    Block5NoiseConfig,
    make_block1_default_config,
    make_block2_default_config,
    make_block3_default_config,
    make_block4_default_config,
    make_block5_default_config,
    NOISE_VARIANCE_TABLE_BY_N,
)


# ---------------------------------------------------------------------------
# RL 动作允许的 scaling factor 集合（每个 distribution 各一份默认）。
# 业务侧可以自由覆盖；这里只给一个起点，与 N=8192 / N=16384 表的列保持一致。
# ---------------------------------------------------------------------------
def _default_allowed_sfs(distribution: str) -> Tuple[int, ...]:
    """Return all scaling factors present in ``NOISE_VARIANCE_TABLE_BY_N``
    that have a positive variance for the given distribution.

    用作 RL 离散动作空间的默认候选集。RL agent 输出 [0, len(allowed_sfs)) 的
    离散索引，桥接层将其映射到具体的 scaling_factor。
    """
    # 取 N=16384 表（覆盖更广），按 scale_bits 升序
    table = NOISE_VARIANCE_TABLE_BY_N[16384]
    sfs = [
        sf for sf in sorted(table.keys())
        if table[sf].get(distribution, 0.0) > 0.0
    ]
    return tuple(sfs)


BLB_DEFAULT_ALLOWED_SFS_FRESH = _default_allowed_sfs("fresh")
BLB_DEFAULT_ALLOWED_SFS_ENCODE = _default_allowed_sfs("encoding")
BLB_DEFAULT_ALLOWED_SFS_RESCALE = _default_allowed_sfs("rescale")


def discrete_action_to_sf(action_idx: int, allowed_sfs: Sequence[int]) -> int:
    """RL 离散动作索引 → scaling factor。"""
    idx = int(action_idx)
    if idx < 0 or idx >= len(allowed_sfs):
        raise ValueError(
            f"action_idx={idx} 超出 allowed_sfs (len={len(allowed_sfs)}) 范围"
        )
    return int(allowed_sfs[idx])


def discrete_action_to_optional_sf(
        action_idx: int,
        allowed_sfs: Sequence[int],
        off_token: int = -1,
        ) -> Optional[int]:
    """与 ``discrete_action_to_sf`` 同，但允许"关闭"动作。

    ``action_idx == off_token`` 时返回 ``None``（用于 rescale 关闭的语义）。
    """
    if int(action_idx) == int(off_token):
        return None
    return discrete_action_to_sf(action_idx, allowed_sfs)


# ===========================================================================
# 桥接控制器
# ===========================================================================

class BLBNoiseRLBridge:
    """RL 阶段把 BLB 噪声装上 / 卸下的统一入口。

    不持有模型本身，只持有 ``ReversibleLayerHandler``；apply / clear 都是幂等的
    操作（重复 apply 会覆盖；重复 clear 不会出错）。
    """

    def __init__(
            self,
            reversible_handler,
            layers_attribute: str = "model.bert.encoder.layer",
            ):
        self.handler = reversible_handler
        self.layers_attribute = str(layers_attribute)
        # 跟踪当前装了哪些 (layer_idx, block_name)，clear() 时按需还原。
        # block_name ∈ {"block1","block2","block3","block4","block5","first_input"}
        self._installed: Dict[int, set] = {}

    # ------------------------------------------------------------------
    # 安装：把所有 BLB 噪声一次性挂到模型
    # ------------------------------------------------------------------
    def apply(
            self,
            *,
            first_input_sf: Optional[int] = None,
            first_input_N: int = 16384,
            first_input_layers: Sequence[int] = (0,),
            block1_cfgs: Optional[Dict[int, Block1NoiseConfig]] = None,
            block2_cfgs: Optional[Dict[int, Block2NoiseConfig]] = None,
            block3_cfgs: Optional[Dict[int, Block3NoiseConfig]] = None,
            block4_cfgs: Optional[Dict[int, Block4NoiseConfig]] = None,
            block5_cfgs: Optional[Dict[int, Block5NoiseConfig]] = None,
            ) -> None:
        """安装一次 RL 动作对应的所有 BLB 噪声。

        Args:
            first_input_sf:    layer 0 入口 fresh 噪声 scaling_factor；None = 不加
            first_input_N:     first-input 用哪一张 N 表
            first_input_layers: 默认只装 (0,)；用户可改成更多层
            block1_cfgs:       {layer_idx: Block1NoiseConfig}；None / {} = 不装 Block 1
            block2_cfgs..block5_cfgs 同上

        每个 cfg 直接调用 ``handler.replace_layer_block*_noise`` 完成实际安装；
        Block 3 / Block 5 走 ``cfg_per_layer`` 路径以支持每层不同 degree。
        BLB / legacy 互斥校验由 handler 内部完成（残留 legacy 噪声会抛 RuntimeError）。
        """
        # ---------- 1) first-input fresh ----------
        if first_input_sf is not None:
            self.handler.replace_blb_first_input_noise(
                scaling_factor=int(first_input_sf),
                N=int(first_input_N),
                layer_indices=list(first_input_layers),
                layer_name=self.layers_attribute,
            )
            for li in first_input_layers:
                self._installed.setdefault(int(li), set()).add("first_input")

        # ---------- 2) Block 1 / 2 / 4：按 cfg 分组批量安装 ----------
        for block_name, cfgs, install_method in (
                ("block1", block1_cfgs, self.handler.replace_layer_block1_noise),
                ("block2", block2_cfgs, self.handler.replace_layer_block2_noise),
                ("block4", block4_cfgs, self.handler.replace_layer_block4_noise),
                ):
            if not cfgs:
                continue
            # 不同层的 cfg 不一定相同，按 id() 分组以减少 install 调用次数。
            buckets: Dict[int, Tuple[object, list]] = {}
            for layer_idx, cfg in cfgs.items():
                key = id(cfg)
                if key not in buckets:
                    buckets[key] = (cfg, [])
                buckets[key][1].append(int(layer_idx))
            for cfg_obj, layer_indices in buckets.values():
                install_method(
                    layer_indices=layer_indices,
                    layer_name=self.layers_attribute,
                    cfg=cfg_obj,
                )
                for li in layer_indices:
                    self._installed.setdefault(int(li), set()).add(block_name)

        # ---------- 3) Block 3 / 5：cfg_per_layer 路径（degree-aware） ----------
        if block3_cfgs:
            self.handler.replace_layer_block3_noise(
                layer_indices=list(block3_cfgs.keys()),
                layer_name=self.layers_attribute,
                cfg_per_layer=dict(block3_cfgs),
            )
            for li in block3_cfgs:
                self._installed.setdefault(int(li), set()).add("block3")

        if block5_cfgs:
            self.handler.replace_layer_block5_noise(
                layer_indices=list(block5_cfgs.keys()),
                layer_name=self.layers_attribute,
                cfg_per_layer=dict(block5_cfgs),
            )
            for li in block5_cfgs:
                self._installed.setdefault(int(li), set()).add("block5")

    # ------------------------------------------------------------------
    # 还原：一次性把 apply() 装的 BLB 噪声全部脱掉
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """还原本次 RL 步骤装的所有 BLB 噪声，恢复到 apply 之前的状态。

        还原顺序按 BLB 数据流的"反向"做：
          block5 → block4 → block3 → block2 → block1 → first_input
        这样 LN 替身（NoisyBlock1/4LayerNorm）能被正确剥到原 LN。
        """
        if not self._installed:
            return

        # 按 block 反向收集 layer 列表
        per_block: Dict[str, list] = {
            "block5": [], "block4": [], "block3": [],
            "block2": [], "block1": [], "first_input": [],
        }
        for li, blocks in self._installed.items():
            for b in blocks:
                if b in per_block:
                    per_block[b].append(int(li))

        if per_block["block5"]:
            self.handler.restore_layer_block5_noise(
                layer_indices=per_block["block5"],
                layer_name=self.layers_attribute,
            )
        if per_block["block4"]:
            self.handler.restore_layer_block4_noise(
                layer_indices=per_block["block4"],
                layer_name=self.layers_attribute,
            )
        if per_block["block3"]:
            self.handler.restore_layer_block3_noise(
                layer_indices=per_block["block3"],
                layer_name=self.layers_attribute,
            )
        if per_block["block2"]:
            self.handler.restore_layer_block2_noise(
                layer_indices=per_block["block2"],
                layer_name=self.layers_attribute,
            )
        if per_block["block1"]:
            self.handler.restore_layer_block1_noise(
                layer_indices=per_block["block1"],
                layer_name=self.layers_attribute,
            )
        if per_block["first_input"]:
            self.handler.restore_blb_first_input_noise(
                layer_indices=per_block["first_input"],
                layer_name=self.layers_attribute,
            )

        self._installed = {}

    # ------------------------------------------------------------------
    # 辅助：introspect
    # ------------------------------------------------------------------
    def installed_layers(self) -> Dict[int, set]:
        """返回当前桥接器装上去的 (layer_idx → {block_name, ...}) 拷贝。"""
        return {li: set(blks) for li, blks in self._installed.items()}

    # ------------------------------------------------------------------
    # 与 Rescale_optimizer 桥接的 reward 侧便利方法
    # ------------------------------------------------------------------
    def evaluate_with_rescale_optimizer(
            self,
            rescale_bridge,
            requests,
            *,
            extra_overrides=None,
            ):
        """已 apply 完毕后，把同一份 cfg 喂给 ``Rescale_optimizer`` 取奖励原料。

        本方法**不**调用 ``apply`` / ``clear``，纯粹是一个 reward 侧的语法糖：
        用户在 RL 一回合里
            ``bridge.apply(...)``  → forward → 通过本方法拿优化器原料 → ``bridge.clear()``。

        Args:
            rescale_bridge: ``RescaleOptimizerBridge`` 实例
            requests: ``{config_name: (block_name, cfg)}``，结构同
                      ``RescaleOptimizerBridge.evaluate_blocks``
            extra_overrides: 可选；同 ``RescaleOptimizerBridge.evaluate_blocks``

        Returns:
            (outputs, signals)，``outputs`` 是 ``{config_name: RescaleOptimizerOutput}``，
            ``signals`` 是 ``OptimizerRewardSignals``。
        """
        from rescale_optimizer_bridge import aggregate_optimizer_signals  # 局部导入避免循环

        outputs = rescale_bridge.evaluate_blocks(requests, extra_overrides=extra_overrides)
        signals = aggregate_optimizer_signals(outputs)
        return outputs, signals


# ===========================================================================
# 动作 → cfg 的几个便捷构造函数
#   * 这里只展示"per-noise-point 一个 scaling factor"的最直接映射；
#     业务侧可以基于它再加抽象（如 per-layer 共享 SF、per-block 共享 SF 等）。
# ===========================================================================

@dataclass
class Block1ActionSpec:
    """RL 动作 → Block 1 cfg 的字段映射。每个字段是一个 scaling_factor (int)；
    rescale_* 字段为 None 表示该处 rescale 不加。

    ``output_truncation_k``：Block 1 末尾 PPTI 截断位数；None ⇒ 不截断
    （**首层 Block 1 缺失**时直接传 None；其它层由 RL agent 选）。
    """
    gelu_out_sf: int
    wffn2_sf: int
    mean_inv_d_sf: int
    var_inv_d_sf: int
    wffn2_rescale_sf: Optional[int] = None
    mean_rescale_sf: Optional[int] = None
    square_rescale_sf: Optional[int] = None
    var_rescale_sf: Optional[int] = None
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"


def build_block1_cfg_from_action(
        action: Block1ActionSpec,
        N: int = 8192,
        ) -> Block1NoiseConfig:
    """``Block1ActionSpec`` → ``Block1NoiseConfig``。"""
    return make_block1_default_config(
        N=int(N),
        gelu_out_sf=int(action.gelu_out_sf),
        wffn2_sf=int(action.wffn2_sf),
        mean_inv_d_sf=int(action.mean_inv_d_sf),
        var_inv_d_sf=int(action.var_inv_d_sf),
        wffn2_rescale_sf=action.wffn2_rescale_sf,
        mean_rescale_sf=action.mean_rescale_sf,
        square_rescale_sf=action.square_rescale_sf,
        var_rescale_sf=action.var_rescale_sf,
        output_truncation_k=action.output_truncation_k,
        output_truncation_mode=action.output_truncation_mode,
    )


@dataclass
class Block2ActionSpec:
    """RL 动作 → Block 2 cfg。22 个噪声点全部 sf 字段化。"""
    inv_std_fresh_sf: int
    x_centered_fresh_sf: int
    gamma_sf: int
    wk_sf: int
    kt_mask1_sf: int
    kt_mask2_sf: int
    wq_sf: int
    q_mask1_sf: int
    q_mask2_sf: int
    wv_sf: int
    qkt_merge_mask_sf: int
    normalize_rescale_sf: Optional[int] = None
    gamma_rescale_sf: Optional[int] = None
    wk_rescale_sf: Optional[int] = None
    kt_mask1_rescale_sf: Optional[int] = None
    kt_mask2_rescale_sf: Optional[int] = None
    wq_rescale_sf: Optional[int] = None
    q_mask1_rescale_sf: Optional[int] = None
    q_mask2_rescale_sf: Optional[int] = None
    wv_rescale_sf: Optional[int] = None
    qkt_matmul_rescale_sf: Optional[int] = None
    qkt_merge_mask_rescale_sf: Optional[int] = None
    # PPTI Block 2 末尾 truncation；首层 Block 2 前半部分缺失但后半部分（Q·K^T）
    # 仍会执行，所以这里仍可加 truncation。
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"


def build_block2_cfg_from_action(
        action: Block2ActionSpec,
        N: int = 16384,
        ) -> Block2NoiseConfig:
    """``Block2ActionSpec`` → ``Block2NoiseConfig``。"""
    return make_block2_default_config(
        N=int(N),
        inv_std_fresh_sf=int(action.inv_std_fresh_sf),
        x_centered_fresh_sf=int(action.x_centered_fresh_sf),
        gamma_sf=int(action.gamma_sf),
        wk_sf=int(action.wk_sf),
        kt_mask1_sf=int(action.kt_mask1_sf),
        kt_mask2_sf=int(action.kt_mask2_sf),
        wq_sf=int(action.wq_sf),
        q_mask1_sf=int(action.q_mask1_sf),
        q_mask2_sf=int(action.q_mask2_sf),
        wv_sf=int(action.wv_sf),
        qkt_merge_mask_sf=int(action.qkt_merge_mask_sf),
        normalize_rescale_sf=action.normalize_rescale_sf,
        gamma_rescale_sf=action.gamma_rescale_sf,
        wk_rescale_sf=action.wk_rescale_sf,
        kt_mask1_rescale_sf=action.kt_mask1_rescale_sf,
        kt_mask2_rescale_sf=action.kt_mask2_rescale_sf,
        wq_rescale_sf=action.wq_rescale_sf,
        q_mask1_rescale_sf=action.q_mask1_rescale_sf,
        q_mask2_rescale_sf=action.q_mask2_rescale_sf,
        wv_rescale_sf=action.wv_rescale_sf,
        qkt_matmul_rescale_sf=action.qkt_matmul_rescale_sf,
        qkt_merge_mask_rescale_sf=action.qkt_merge_mask_rescale_sf,
        output_truncation_k=action.output_truncation_k,
        output_truncation_mode=action.output_truncation_mode,
    )


@dataclass
class Block3ActionSpec:
    """RL 动作 → Block 3 cfg。degree 决定 N 默认值与 square_rescales 长度。"""
    degree: int
    x_fresh_sf: int
    inv_2n_sf: int
    x_inv_2n_rescale_sf: Optional[int] = None
    # 长度必须 == degree
    square_rescale_sfs: Tuple[Optional[int], ...] = ()
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"


def build_block3_cfg_from_action(
        action: Block3ActionSpec,
        N: Optional[int] = None,
        ) -> Block3NoiseConfig:
    """``Block3ActionSpec`` → ``Block3NoiseConfig`` (N 默认按 degree 自动选)。"""
    return make_block3_default_config(
        degree=int(action.degree),
        N=N,
        x_fresh_sf=int(action.x_fresh_sf),
        inv_2n_sf=int(action.inv_2n_sf),
        x_inv_2n_rescale_sf=action.x_inv_2n_rescale_sf,
        square_rescale_sfs=action.square_rescale_sfs,
        output_truncation_k=action.output_truncation_k,
        output_truncation_mode=action.output_truncation_mode,
    )


@dataclass
class Block4ActionSpec:
    """RL 动作 → Block 4 cfg。"""
    softmax_out_fresh_sf: int
    softmax_out_mask_sf: int
    v_fresh_sf: int
    v_mask_sf: int
    softmax_v_mask_sf: int
    wo_sf: int
    ln_mean_inv_d_sf: int
    ln_var_inv_d_sf: int
    softmax_out_mask_rescale_sf: Optional[int] = None
    v_mask_rescale_sf: Optional[int] = None
    softmax_v_matmul_rescale_sf: Optional[int] = None
    softmax_v_mask_rescale_sf: Optional[int] = None
    wo_rescale_sf: Optional[int] = None
    ln_mean_rescale_sf: Optional[int] = None
    ln_square_rescale_sf: Optional[int] = None
    ln_var_rescale_sf: Optional[int] = None


def build_block4_cfg_from_action(
        action: Block4ActionSpec,
        N: int = 16384,
        ) -> Block4NoiseConfig:
    """``Block4ActionSpec`` → ``Block4NoiseConfig``。"""
    return make_block4_default_config(
        N=int(N),
        softmax_out_fresh_sf=int(action.softmax_out_fresh_sf),
        softmax_out_mask_sf=int(action.softmax_out_mask_sf),
        v_fresh_sf=int(action.v_fresh_sf),
        v_mask_sf=int(action.v_mask_sf),
        softmax_v_mask_sf=int(action.softmax_v_mask_sf),
        wo_sf=int(action.wo_sf),
        ln_mean_inv_d_sf=int(action.ln_mean_inv_d_sf),
        ln_var_inv_d_sf=int(action.ln_var_inv_d_sf),
        softmax_out_mask_rescale_sf=action.softmax_out_mask_rescale_sf,
        v_mask_rescale_sf=action.v_mask_rescale_sf,
        softmax_v_matmul_rescale_sf=action.softmax_v_matmul_rescale_sf,
        softmax_v_mask_rescale_sf=action.softmax_v_mask_rescale_sf,
        wo_rescale_sf=action.wo_rescale_sf,
        ln_mean_rescale_sf=action.ln_mean_rescale_sf,
        ln_square_rescale_sf=action.ln_square_rescale_sf,
        ln_var_rescale_sf=action.ln_var_rescale_sf,
    )


@dataclass
class Block5ActionSpec:
    """RL 动作 → Block 5 cfg。GELU degree 决定 N 默认与 power/coeff_mul rescales 长度。"""
    gelu_degree: int
    inv_std_fresh_sf: int
    x_centered_fresh_sf: int
    gamma_sf: int
    wffn1_sf: int
    gelu_coeff_sf: int
    normalize_rescale_sf: Optional[int] = None
    gamma_rescale_sf: Optional[int] = None
    wffn1_rescale_sf: Optional[int] = None
    # 长度 == gelu_degree-1
    gelu_power_rescale_sfs: Tuple[Optional[int], ...] = ()
    # 长度 == gelu_degree
    gelu_coeff_mul_rescale_sfs: Tuple[Optional[int], ...] = ()


def build_block5_cfg_from_action(
        action: Block5ActionSpec,
        N: Optional[int] = None,
        ) -> Block5NoiseConfig:
    """``Block5ActionSpec`` → ``Block5NoiseConfig`` (N 默认按 GELU degree 自动选)。"""
    return make_block5_default_config(
        gelu_degree=int(action.gelu_degree),
        N=N,
        inv_std_fresh_sf=int(action.inv_std_fresh_sf),
        x_centered_fresh_sf=int(action.x_centered_fresh_sf),
        gamma_sf=int(action.gamma_sf),
        wffn1_sf=int(action.wffn1_sf),
        gelu_coeff_sf=int(action.gelu_coeff_sf),
        normalize_rescale_sf=action.normalize_rescale_sf,
        gamma_rescale_sf=action.gamma_rescale_sf,
        wffn1_rescale_sf=action.wffn1_rescale_sf,
        gelu_power_rescale_sfs=action.gelu_power_rescale_sfs,
        gelu_coeff_mul_rescale_sfs=action.gelu_coeff_mul_rescale_sfs,
    )
