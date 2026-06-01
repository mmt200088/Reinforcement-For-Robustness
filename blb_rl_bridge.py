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
        first_input_sf=action.first_input_sf, first_input_N=8192,
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

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Dict, Mapping, Any

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
            first_input_N: int = 8192,
            first_input_layers: Sequence[int] = (0,),
            block1_cfgs: Optional[Dict[int, Block1NoiseConfig]] = None,
            block2_cfgs: Optional[Dict[int, Block2NoiseConfig]] = None,
            block3_cfgs: Optional[Dict[int, Block3NoiseConfig]] = None,
            block4_cfgs: Optional[Dict[int, Block4NoiseConfig]] = None,
            block5_cfgs: Optional[Dict[int, Block5NoiseConfig]] = None,
            ) -> None:
        """安装一次 RL 动作对应的所有 BLB 噪声。

        Args:
            first_input_sf:    **DEPRECATED**。语义上"第一个 HE 配置无损"，layer 0
                               input 端不再注入 fresh 噪声。保留参数仅为旧调用
                               站点兼容；任何非 None 取值会被忽略 + 警告。
            first_input_N:     与 ``first_input_sf`` 一同 deprecated。
            first_input_layers: 同上 deprecated。
            block1_cfgs:       {layer_idx: Block1NoiseConfig}；**layer 0 即便存在
                               也不会被安装**（用户语义：layer 0 没有上游 FFN2，
                               block1 噪声整体不加；与 Rescale_optimizer 对齐）。
                               ``action_vector_to_cfgs`` 已经不会把 layer 0 写进
                               这个 dict，这里的过滤是双保险。
            block2_cfgs..block5_cfgs 同上 layer 0 全部生效（block 2 完整存在）。

        每个 cfg 直接调用 ``handler.replace_layer_block*_noise`` 完成实际安装；
        Block 3 / Block 5 走 ``cfg_per_layer`` 路径以支持每层不同 degree。
        BLB / legacy 互斥校验由 handler 内部完成（残留 legacy 噪声会抛 RuntimeError）。
        """
        # ---------- 1) first-input fresh：deprecated，整体跳过 ----------
        # 旧调用站点可能仍在传 first_input_sf；为不破坏接口，悄悄忽略并提示。
        if first_input_sf is not None:
            import warnings
            warnings.warn(
                "BLBNoiseRLBridge.apply(first_input_sf=...) is deprecated and ignored: "
                "the first HE config is treated as lossless; no fresh noise is "
                "injected at layer-0 input.",
                DeprecationWarning,
                stacklevel=2,
            )

        # ---------- 2) Block 1 / 2 / 4：按 cfg 分组批量安装 ----------
        # 双保险：剔除 layer 0 的 block1 cfg（即使 caller 误传也安全）。
        if block1_cfgs:
            block1_cfgs = {
                li: cfg for li, cfg in block1_cfgs.items() if int(li) != 0
            }
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

        # ---------- 3) Block 5：cfg_per_layer 路径（degree-aware） ----------
        # C (2026-05-30): block 3 (softmax exp approx) noise is intentionally NOT
        # installed. Block 3 is removed from the RL search and frozen at the
        # static_skeletons baseline, so the model forward runs softmax WITHOUT any
        # block-3 CKKS noise. ``self.handler.replace_layer_block3_noise`` is kept
        # (it may be re-enabled later) but never called here; ``block3_cfgs`` is
        # accepted for API compatibility and deliberately ignored. Applying this
        # uniformly to baseline + candidates keeps the noise floor consistent.
        _ = block3_cfgs

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
    """RL 动作 → Block 1 cfg 的字段映射。

    2026-05-14 精简：删除了 ``wffn2_rescale_sf`` 和 ``square_rescale_sf`` 两个 RL
    动作槽（mrpc baseline skeleton 不在那两处下 rescale；Rescale_optimizer 也不
    会选用）。对应 cfg 的 ``wffn2_result_rescale`` / ``square_result_rescale``
    字段固定为 None（不安装这两处 rescale 噪声）。配套的 rotation flag
    （``rotation_after_wffn2_rescale_a/b`` / ``rotation_after_square_rescale``）
    也一并移除。

    ``output_truncation_k``：Block 1 末尾 PPTI 截断位数；None ⇒ 不截断
    （**首层 Block 1 缺失**时直接传 None；其它层由 RL agent 选）。
    """
    gelu_out_sf: int
    wffn2_sf: int
    mean_inv_d_sf: int
    var_inv_d_sf: int
    mean_rescale_sf: Optional[int] = None
    var_rescale_sf: Optional[int] = None
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    # Rotation 候选点（True ⇒ 在该位置加 rotation 噪声；SF 自动继承绑定源）
    # 仅保留 fresh 上的 rotation（gelu_out_fresh 之后的 rotation）。被删的
    # rescale 槽不再有对应 rotation flag。
    rotation_after_gelu_out_fresh: bool = False


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
        # 被删的 wffn2_rescale_sf / square_rescale_sf 固定 None（不安装该处噪声）
        wffn2_rescale_sf=None,
        mean_rescale_sf=action.mean_rescale_sf,
        square_rescale_sf=None,
        var_rescale_sf=action.var_rescale_sf,
        output_truncation_k=action.output_truncation_k,
        output_truncation_mode=action.output_truncation_mode,
        rotation_after_gelu_out_fresh=action.rotation_after_gelu_out_fresh,
        # 被删的 rotation flag 固定 False（cfg.rotation_after_* 默认 False）
        rotation_after_wffn2_rescale_a=False,
        rotation_after_wffn2_rescale_b=False,
        rotation_after_square_rescale=False,
    )


@dataclass
class Block2ActionSpec:
    """RL 动作 → Block 2 cfg。

    2026-05-14 精简：
    * Q 侧 3 个 encode 槽（``wq_sf`` / ``q_mask1_sf`` / ``q_mask2_sf``）与 K 侧
      绑定 —— RL 不再独立选择 Q 侧 SF，由 ``_build_block2_action`` 用 K 侧动作
      值同步填入。ActionSpec 保留这 3 个字段作为载体（仍然喂给 cfg）。
    * 删除 8 个 rescale 槽：``normalize_rescale_sf``、``wk_rescale_sf``、
      ``wq_rescale_sf``、``wv_rescale_sf``、``kt_mask1_rescale_sf``、
      ``q_mask1_rescale_sf``、``q_mask2_rescale_sf``、``qkt_matmul_rescale_sf``。
      它们对应的 cfg 字段固定 None。同步删除对应 rotation flags。
    * 保留 ``gamma_rescale_sf`` / ``kt_mask2_rescale_sf`` / ``qkt_merge_mask_rescale_sf``
      （这三个通过 t_new 进 optimizer cost）。
    * ``wv_sf`` 保留作模型噪声单独控制 Wv 的 SF（cfg 仍写入 wv_encode）。
    """
    inv_std_fresh_sf: int
    x_centered_fresh_sf: int
    gamma_sf: int
    wk_sf: int
    kt_mask1_sf: int
    kt_mask2_sf: int
    # Q 侧三个 encode：由 K 侧绑定填入，不是独立 RL 动作。
    wq_sf: int = 0
    q_mask1_sf: int = 0
    q_mask2_sf: int = 0
    wv_sf: int = 22
    qkt_merge_mask_sf: int = 22
    gamma_rescale_sf: Optional[int] = None
    kt_mask2_rescale_sf: Optional[int] = None
    qkt_merge_mask_rescale_sf: Optional[int] = None
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    # Rotation 候选点：仅保留与未删 rescale 绑定的两项。
    rotation_after_gamma_rescale: bool = False
    rotation_after_kt_mask2_rescale: bool = False


def build_block2_cfg_from_action(
        action: Block2ActionSpec,
        N: int = 16384,
        ) -> Block2NoiseConfig:
    """``Block2ActionSpec`` → ``Block2NoiseConfig``。

    Q/K 共享段绑定（2026-05-21 user spec）：
      * 三个 Q 侧 encode（wq / q_mask1 / q_mask2）已由 ``_build_block2_action``
        写为 K 侧同值。
      * ``q_mask2_r`` 也绑到 ``kt_mask2_r`` —— mrpc baseline 里
        ctpt_rotKT_mask2 sf_post 同时是 kt_mask2_r 和 q_mask2_r 的 baseline，
        model 必须在 V 侧装同样的 rescale 噪声才能和 optimizer 的链假设对齐。
        其它没有 baseline 的 rescale 槽（normalize / wk / wq / wv /
        kt_mask1 / q_mask1 / qkt_matmul）继续保持 None。
    """
    return make_block2_default_config(
        N=int(N),
        inv_std_fresh_sf=int(action.inv_std_fresh_sf),
        x_centered_fresh_sf=int(action.x_centered_fresh_sf),
        gamma_sf=int(action.gamma_sf),
        wk_sf=int(action.wk_sf),
        kt_mask1_sf=int(action.kt_mask1_sf),
        kt_mask2_sf=int(action.kt_mask2_sf),
        # Q/K 绑定：Q 侧三个 encode 与 K 侧同值（由 _build_block2_action 填入）
        wq_sf=int(action.wq_sf),
        q_mask1_sf=int(action.q_mask1_sf),
        q_mask2_sf=int(action.q_mask2_sf),
        wv_sf=int(action.wv_sf),
        qkt_merge_mask_sf=int(action.qkt_merge_mask_sf),
        normalize_rescale_sf=None,
        gamma_rescale_sf=action.gamma_rescale_sf,
        wk_rescale_sf=None,
        kt_mask1_rescale_sf=None,
        kt_mask2_rescale_sf=action.kt_mask2_rescale_sf,
        wq_rescale_sf=None,
        q_mask1_rescale_sf=None,
        # 2026-05-21: q_mask2_r 绑定到 kt_mask2_r（共享 graph 节点
        # ctpt_rotKT_mask2 sf_post）。
        q_mask2_rescale_sf=action.kt_mask2_rescale_sf,
        wv_rescale_sf=None,
        qkt_matmul_rescale_sf=None,
        qkt_merge_mask_rescale_sf=action.qkt_merge_mask_rescale_sf,
        output_truncation_k=action.output_truncation_k,
        output_truncation_mode=action.output_truncation_mode,
        rotation_after_gamma_rescale=action.rotation_after_gamma_rescale,
        rotation_after_wq_rescale=False,
        rotation_after_wk_rescale=False,
        rotation_after_wv_rescale=False,
        rotation_after_q_mask1_rescale=False,
        rotation_after_kt_mask1_rescale=False,
        # q_mask2_r 上的 rotation 跟随 kt_mask2_r（绑定对称）。
        rotation_after_q_mask2_rescale=action.rotation_after_kt_mask2_rescale,
        rotation_after_kt_mask2_rescale=action.rotation_after_kt_mask2_rescale,
        rotation_after_qkt_matmul_rescale=False,
    )


@dataclass
class Block3ActionSpec:
    """RL 动作 → Block 3 cfg。degree 决定 N 默认值与 square_rescales 长度。

    2026-05-14 精简：删除 ``x_inv_2n_rescale_sf`` —— mrpc baseline skeleton 不上
    ``ctct_x_inv_2n_rescale``，Rescale_optimizer 不会选用。cfg 上对应字段
    ``x_inv_2n_result_rescale`` 固定为 None。
    """
    degree: int
    x_fresh_sf: int
    inv_2n_sf: int
    # 长度必须 == degree
    square_rescale_sfs: Tuple[Optional[int], ...] = ()
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"


def build_block3_cfg_from_action(
        action: Block3ActionSpec,
        N: Optional[int] = None,
        ) -> Block3NoiseConfig:
    """``Block3ActionSpec`` → ``Block3NoiseConfig`` (N 默认按 degree 自动选)。

    被删的 ``x_inv_2n_rescale_sf`` 槽在 cfg 上固定为 None。
    """
    return make_block3_default_config(
        degree=int(action.degree),
        N=N,
        x_fresh_sf=int(action.x_fresh_sf),
        inv_2n_sf=int(action.inv_2n_sf),
        x_inv_2n_rescale_sf=None,
        square_rescale_sfs=action.square_rescale_sfs,
        output_truncation_k=action.output_truncation_k,
        output_truncation_mode=action.output_truncation_mode,
    )


@dataclass
class Block4ActionSpec:
    """RL 动作 → Block 4 cfg。

    2026-05-14 精简：删除 5 个 rescale 槽 ——
    ``softmax_out_mask_rescale_sf`` / ``v_mask_rescale_sf`` /
    ``softmax_v_mask_rescale_sf`` / ``wo_rescale_sf`` / ``ln_square_rescale_sf``
    及对应 5 个 rotation flag。它们对应的 cfg 字段固定 None。
    保留 V 侧 ``v_fresh_sf`` / ``v_mask_sf`` 控制模型 V 路径上的噪声（mrpc graph
    没有 V 节点，不入 optimizer cost）。
    """
    softmax_out_fresh_sf: int
    softmax_out_mask_sf: int
    v_fresh_sf: int
    v_mask_sf: int
    softmax_v_mask_sf: int
    wo_sf: int
    ln_mean_inv_d_sf: int
    ln_var_inv_d_sf: int
    softmax_v_matmul_rescale_sf: Optional[int] = None
    ln_mean_rescale_sf: Optional[int] = None
    ln_var_rescale_sf: Optional[int] = None
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    # Rotation 候选点：仅保留 softmax_v_matmul_rescale 后的 rotation
    rotation_after_softmax_v_matmul_rescale: bool = False


def build_block4_cfg_from_action(
        action: Block4ActionSpec,
        N: int = 16384,
        ) -> Block4NoiseConfig:
    """``Block4ActionSpec`` → ``Block4NoiseConfig``。

    被删的 5 个 rescale 槽 + 5 个 rotation flag 在 cfg 上固定 None / False。
    """
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
        softmax_out_mask_rescale_sf=None,
        v_mask_rescale_sf=None,
        softmax_v_matmul_rescale_sf=action.softmax_v_matmul_rescale_sf,
        softmax_v_mask_rescale_sf=None,
        wo_rescale_sf=None,
        ln_mean_rescale_sf=action.ln_mean_rescale_sf,
        ln_square_rescale_sf=None,
        ln_var_rescale_sf=action.ln_var_rescale_sf,
        output_truncation_k=action.output_truncation_k,
        output_truncation_mode=action.output_truncation_mode,
        rotation_after_softmax_out_mask_rescale=False,
        rotation_after_v_mask_rescale=False,
        rotation_after_softmax_v_matmul_rescale=action.rotation_after_softmax_v_matmul_rescale,
        rotation_after_softmax_v_mask_rescale=False,
        rotation_after_wo_rescale=False,
        rotation_after_ln_square_rescale=False,
    )


@dataclass
class Block5ActionSpec:
    """RL 动作 → Block 5 cfg。GELU degree 决定 N 默认与 power/coeff_mul rescales 长度。

    2026-05-14 精简：
    * ``gelu_power_rescale_sfs`` —— RL 仅控制 idx=0 (x²)；idx=1 (x³) / idx=2 (x⁴)
      在 mrpc graph 里被折掉，不上 skeleton，固定 None。``_build_block5_action``
      会用 ``(power_sf_0, None, ...)`` 构造正确长度的 tuple。
    * ``gelu_coeff_mul_rescale_sfs`` —— 整个 tuple 全部 RL 不再控制，由
      ``ctpt_gelu_coeff`` 单节点合并表达；``_build_block5_action`` 用
      ``(None, ..., None)`` 长度=degree 的 tuple。
    """
    gelu_degree: int
    inv_std_fresh_sf: int
    x_centered_fresh_sf: int
    gamma_sf: int
    wffn1_sf: int
    gelu_coeff_sf: int
    normalize_rescale_sf: Optional[int] = None
    gamma_rescale_sf: Optional[int] = None
    wffn1_rescale_sf: Optional[int] = None
    # 长度 == gelu_degree-1；仅 idx=0 (x²) 由 RL 控制，其它为 None
    gelu_power_rescale_sfs: Tuple[Optional[int], ...] = ()
    # 长度 == gelu_degree；全部 None（RL 不再控制 coeff_mul 链 rescale）
    gelu_coeff_mul_rescale_sfs: Tuple[Optional[int], ...] = ()
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    # Rotation 候选点（共 2 个）
    rotation_after_gamma_rescale: bool = False
    rotation_after_wffn1_rescale: bool = False


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
        output_truncation_k=action.output_truncation_k,
        output_truncation_mode=action.output_truncation_mode,
        rotation_after_gamma_rescale=action.rotation_after_gamma_rescale,
        rotation_after_wffn1_rescale=action.rotation_after_wffn1_rescale,
    )


# ===========================================================================
# Truncation reward 信号聚合
# ===========================================================================

@dataclass
class TruncationRewardSignals:
    """跨 (block, layer) 聚合的 PPTI truncation reward 原料。

    truncation k 是 RL 动作的一部分；reward 侧需要把 "整体上选了多大的 k /
    多少个 block 跳过 truncation / 平均 k 多少" 这类信号暴露给 reward 计算。
    本 dataclass 不规定 reward 公式 —— 业务侧用 ``per_block_total_k`` /
    ``avg_k_when_set`` 等字段自行组合。
    """
    total_k: int
    count_with_k: int
    count_skip: int
    avg_k_when_set: float
    per_block_total_k: Dict[str, int] = field(default_factory=dict)
    per_block_count_with_k: Dict[str, int] = field(default_factory=dict)
    per_block_count_skip: Dict[str, int] = field(default_factory=dict)


def aggregate_truncation_signals(
        cfg_dicts: Mapping[str, Mapping[int, Any]],
        ) -> TruncationRewardSignals:
    """跨 (block_name, layer_idx) 聚合每个 cfg 的 ``output_truncation_k``。

    Args:
        cfg_dicts: ``{block_name: {layer_idx: Block*NoiseConfig}}``，
                   block_name ∈ ``{"block1","block2","block3","block4","block5"}``
                   或自定义命名。

    Returns:
        ``TruncationRewardSignals``：
          * ``total_k``:           所有非 None k 的求和
          * ``count_with_k``:      非 None k 的 cfg 计数
          * ``count_skip``:        ``output_truncation_k=None`` 的 cfg 计数
          * ``avg_k_when_set``:    总 k / 非 None 计数（无非 None 时为 0）
          * ``per_block_*``:       每个 block 的对应分量
    """
    total_k = 0
    count_with_k = 0
    count_skip = 0
    per_block_total_k: Dict[str, int] = {}
    per_block_count_with_k: Dict[str, int] = {}
    per_block_count_skip: Dict[str, int] = {}

    for block_name, layer_cfgs in cfg_dicts.items():
        per_block_total_k.setdefault(block_name, 0)
        per_block_count_with_k.setdefault(block_name, 0)
        per_block_count_skip.setdefault(block_name, 0)
        if layer_cfgs is None:
            continue
        for _layer_idx, cfg in layer_cfgs.items():
            k = getattr(cfg, "output_truncation_k", None)
            if k is None:
                count_skip += 1
                per_block_count_skip[block_name] += 1
            else:
                total_k += int(k)
                count_with_k += 1
                per_block_total_k[block_name] += int(k)
                per_block_count_with_k[block_name] += 1

    avg = (total_k / count_with_k) if count_with_k > 0 else 0.0
    return TruncationRewardSignals(
        total_k=total_k,
        count_with_k=count_with_k,
        count_skip=count_skip,
        avg_k_when_set=float(avg),
        per_block_total_k=per_block_total_k,
        per_block_count_with_k=per_block_count_with_k,
        per_block_count_skip=per_block_count_skip,
    )


# ===========================================================================
# Rotation reward 信号聚合
# ===========================================================================
# Rotation 噪声已经在 cfg 里以 ``rotation_after_*: bool`` 的形式表达。reward
# 侧通常关心：每个 (block, layer) 选了多少个 rotation 上线。这里把所有 cfg
# 上以 ``rotation_after_*`` 开头的 bool 字段统计一下，给业务侧组合 reward。

@dataclass
class RotationRewardSignals:
    """跨 (block, layer) 聚合的 KS / rotation reward 原料。

    业务侧用 ``total_active`` / ``per_block_active`` 等字段自行组合 reward。
    """
    total_active: int
    total_slots: int
    per_block_active: Dict[str, int] = field(default_factory=dict)
    per_block_slots: Dict[str, int] = field(default_factory=dict)


def _count_rotations_on_cfg(cfg: Any) -> Tuple[int, int]:
    """返回 (active, total)：cfg 上以 rotation_after_ 开头的 bool 字段中开了多少个，总共多少个。"""
    fields = [name for name in vars(cfg).keys() if name.startswith("rotation_after_")]
    total = len(fields)
    active = sum(1 for name in fields if bool(getattr(cfg, name)))
    return active, total


def aggregate_rotation_signals(
        cfg_dicts: Mapping[str, Mapping[int, Any]],
        ) -> RotationRewardSignals:
    """跨 (block_name, layer_idx) 聚合 cfg 上 ``rotation_after_*`` 的开启计数。

    Args:
        cfg_dicts: ``{block_name: {layer_idx: Block*NoiseConfig}}``

    Returns:
        ``RotationRewardSignals``：
          * ``total_active``     所有 block / layer 上 True 的 rotation slot 总数
          * ``total_slots``      候选 slot 总数（用于计算激活率）
          * ``per_block_active`` 每 block 激活数
          * ``per_block_slots``  每 block 候选 slot 数
    """
    total_active = 0
    total_slots = 0
    per_block_active: Dict[str, int] = {}
    per_block_slots: Dict[str, int] = {}
    for block_name, layer_cfgs in cfg_dicts.items():
        per_block_active.setdefault(block_name, 0)
        per_block_slots.setdefault(block_name, 0)
        if layer_cfgs is None:
            continue
        for _layer_idx, cfg in layer_cfgs.items():
            active, total = _count_rotations_on_cfg(cfg)
            total_active += active
            total_slots += total
            per_block_active[block_name] += active
            per_block_slots[block_name] += total
    return RotationRewardSignals(
        total_active=total_active,
        total_slots=total_slots,
        per_block_active=per_block_active,
        per_block_slots=per_block_slots,
    )
