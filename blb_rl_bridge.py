"""Install and clear materialized BLB block configurations on a BERT model.

All noise variances come from the registered N-specific tables. Degree-aware
blocks are validated against the installed Stage-1 GELU and Softmax vectors.
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


def _default_allowed_sfs(distribution: str) -> Tuple[int, ...]:
    """Return the positive-variance scaling factors for one distribution."""

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
    """Decode one discrete action index to a scaling factor."""
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


class BLBNoiseRLBridge:
    """Install and restore materialized BLB configs through one handler."""

    def __init__(
            self,
            reversible_handler,
            layers_attribute: str = "model.bert.encoder.layer",
            ):
        self.handler = reversible_handler
        self.layers_attribute = str(layers_attribute)


        self._installed: Dict[int, set] = {}


    def apply(
            self,
            *,
            block1_cfgs: Optional[Dict[int, Block1NoiseConfig]] = None,
            block2_cfgs: Optional[Dict[int, Block2NoiseConfig]] = None,
            block3_cfgs: Optional[Dict[int, Block3NoiseConfig]] = None,
            block4_cfgs: Optional[Dict[int, Block4NoiseConfig]] = None,
            block5_cfgs: Optional[Dict[int, Block5NoiseConfig]] = None,
            ) -> None:
        """安装一次 RL 动作对应的所有 BLB 噪声。

        Args:
            block1_cfgs:       {layer_idx: Block1NoiseConfig}。layer 0 使用
                               ``noise_enabled=False`` 的 K-only 配置：不注入
                               Block 1 Gaussian/rotation 噪声，但会在 variance
                               进入 rsqrt 前执行与其它层相同的 truncation K。
            block2_cfgs..block5_cfgs 同上 layer 0 全部生效（block 2 完整存在）。

        每个 cfg 直接调用 ``handler.replace_layer_block*_noise`` 完成实际安装；
        Block 3 / Block 5 走 ``cfg_per_layer`` 路径以支持每层不同 degree。
        BLB 与单表噪声的互斥校验由 handler 完成。
        """


        for block_name, cfgs, install_method in (
                ("block1", block1_cfgs, self.handler.replace_layer_block1_noise),
                ("block2", block2_cfgs, self.handler.replace_layer_block2_noise),
                ("block4", block4_cfgs, self.handler.replace_layer_block4_noise),
                ):
            if not cfgs:
                continue

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


    def clear(self) -> None:
        """还原本次 RL 步骤装的所有 BLB 噪声，恢复到 apply 之前的状态。

        还原顺序按 BLB 数据流的"反向"做：
          block5 → block4 → block3 → block2 → block1 → first_input
        这样 LN 替身（NoisyBlock1/4LayerNorm）能被正确剥到原 LN。
        """
        if not self._installed:
            return


        per_block: Dict[str, list] = {
            "block5": [], "block4": [], "block3": [],
            "block2": [], "block1": [],
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
        self._installed = {}


    def installed_layers(self) -> Dict[int, set]:
        """返回当前桥接器装上去的 (layer_idx → {block_name, ...}) 拷贝。"""
        return {li: set(blks) for li, blks in self._installed.items()}


    def evaluate_with_rescale_optimizer(
            self,
            rescale_bridge,
            requests,
            *,
            extra_overrides=None,
            ):
        """Evaluate already-installed configs without changing model state."""
        from rfr.preparation.rescale.bridge import aggregate_optimizer_signals

        outputs = rescale_bridge.evaluate_blocks(requests, extra_overrides=extra_overrides)
        signals = aggregate_optimizer_signals(outputs)
        return outputs, signals


@dataclass
class Block1ActionSpec:
    """RL 动作 → Block 1 cfg 的字段映射。

    MRPC 的 optimizer skeleton 不使用 WFFN2 或 square rescale，因此对应 cfg
    字段固定为 None。

    ``output_truncation_k``：Block 1 末尾 PPTI 截断位数；None ⇒ 不截断。
    所有层（包括 layer 0）都可由 RL 选择；layer 0 通过 cfg 的
    ``noise_enabled=False`` 只执行截断而不注入 Block 1 噪声。
    """
    gelu_out_sf: int
    wffn2_sf: int
    mean_inv_d_sf: int
    var_inv_d_sf: int
    mean_rescale_sf: Optional[int] = None
    var_rescale_sf: Optional[int] = None
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"


    rotation_after_gelu_out_fresh: bool = False


def build_block1_cfg_from_action(
        action: Block1ActionSpec,
        N: int = 8192,
        *,
        noise_enabled: bool = True,
        ) -> Block1NoiseConfig:
    """``Block1ActionSpec`` → ``Block1NoiseConfig``。"""
    return make_block1_default_config(
        N=int(N),
        gelu_out_sf=int(action.gelu_out_sf),
        wffn2_sf=int(action.wffn2_sf),
        mean_inv_d_sf=int(action.mean_inv_d_sf),
        var_inv_d_sf=int(action.var_inv_d_sf),
        noise_enabled=bool(noise_enabled),

        wffn2_rescale_sf=None,
        mean_rescale_sf=action.mean_rescale_sf,
        square_rescale_sf=None,
        var_rescale_sf=action.var_rescale_sf,
        output_truncation_k=action.output_truncation_k,
        output_truncation_mode=action.output_truncation_mode,
        rotation_after_gelu_out_fresh=action.rotation_after_gelu_out_fresh,

        rotation_after_wffn2_rescale_a=False,
        rotation_after_wffn2_rescale_b=False,
        rotation_after_square_rescale=False,
    )


@dataclass
class Block2ActionSpec:
    """RL 动作 → Block 2 cfg。

    Q-side encode values mirror K-side values. Only the rescale fields that
    enter optimizer cost are active; Wv remains a model-noise setting.
    """
    inv_std_fresh_sf: int
    x_centered_fresh_sf: int
    gamma_sf: int
    wk_sf: int
    kt_mask1_sf: int
    kt_mask2_sf: int

    wq_sf: int = 0
    q_mask1_sf: int = 0
    q_mask2_sf: int = 0
    wv_sf: int = 22
    qkt_merge_mask_sf: int = 22
    gamma_rescale_sf: Optional[int] = None
    kt_mask2_rescale_sf: Optional[int] = None
    qkt_merge_mask_rescale_sf: Optional[int] = None


    normalize_rescale_sf: Optional[int] = None
    wk_rescale_sf: Optional[int] = None
    kt_mask1_rescale_sf: Optional[int] = None
    q_mask1_rescale_sf: Optional[int] = None
    q_mask2_rescale_sf: Optional[int] = None
    qkt_matmul_rescale_sf: Optional[int] = None
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"

    rotation_after_gamma_rescale: bool = False
    rotation_after_kt_mask2_rescale: bool = False


def build_block2_cfg_from_action(
        action: Block2ActionSpec,
        N: int = 16384,
        ) -> Block2NoiseConfig:
    """``Block2ActionSpec`` → ``Block2NoiseConfig``。

    Q/K 共享段绑定：
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
        wq_rescale_sf=None,
        q_mask1_rescale_sf=action.q_mask1_rescale_sf,
        q_mask2_rescale_sf=action.q_mask2_rescale_sf,
        wv_rescale_sf=None,
        qkt_matmul_rescale_sf=action.qkt_matmul_rescale_sf,
        qkt_merge_mask_rescale_sf=action.qkt_merge_mask_rescale_sf,
        output_truncation_k=action.output_truncation_k,
        output_truncation_mode=action.output_truncation_mode,
        rotation_after_gamma_rescale=action.rotation_after_gamma_rescale,
        rotation_after_wq_rescale=False,
        rotation_after_wk_rescale=False,
        rotation_after_wv_rescale=False,
        rotation_after_q_mask1_rescale=False,
        rotation_after_kt_mask1_rescale=False,

        rotation_after_q_mask2_rescale=action.rotation_after_kt_mask2_rescale,
        rotation_after_kt_mask2_rescale=action.rotation_after_kt_mask2_rescale,
        rotation_after_qkt_matmul_rescale=False,
    )


@dataclass
class Block3ActionSpec:
    """RL 动作 → Block 3 cfg。degree 决定 N 默认值与 square_rescales 长度。

    MRPC 的 optimizer skeleton 不使用 ``ctct_x_inv_2n_rescale``，因此对应
    cfg 字段固定为 None。
    """
    degree: int
    x_fresh_sf: int
    inv_2n_sf: int

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

    Rescale fields absent from the optimizer graph remain None. V-side fresh
    and mask values still control model noise without entering optimizer cost.
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

    ln_square_rescale_sf: Optional[int] = None
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"

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
        ln_square_rescale_sf=action.ln_square_rescale_sf,
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

    The optimizer graph exposes only x² power rescale. Higher powers remain
    None, and the coefficient products are represented by one merged node.
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

    gelu_power_rescale_sfs: Tuple[Optional[int], ...] = ()

    gelu_coeff_mul_rescale_sfs: Tuple[Optional[int], ...] = ()
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"

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
