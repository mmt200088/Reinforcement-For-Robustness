"""Install and clear materialized BLB block configurations on a BERT model.

All noise variances come from the registered N-specific tables. Degree-aware
blocks are validated against the installed Stage-1 GELU and Softmax vectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Dict, Mapping, Any

from rfr.search.runtime.model_handler import (
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
    """Decode a scaling-factor action with an explicit disabled level."""
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
        """Install decoded BLB noise configurations on the selected layers."""


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
        """Restore every BLB hook installed by this bridge."""
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
        """Return installed layer indices by BLB block."""
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
    """Discrete Block 1 action values."""
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
    """Discrete Block 2 action values."""
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
    """Materialize a Block 2 noise configuration from its action values."""
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
    """Discrete Block 3 action values."""
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
    """Materialize a Block 3 noise configuration from its action values."""
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
    """Discrete Block 4 action values."""
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
    """Materialize a Block 4 noise configuration from its action values."""
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
    """Discrete Block 5 action values."""
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
    """Materialize a Block 5 noise configuration from its action values."""
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
    """Communication-facing statistics for simulated truncation choices."""
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
    """Aggregate truncation statistics across materialized block configurations."""
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
    """Rotation counts derived from materialized block configurations."""
    total_active: int
    total_slots: int
    per_block_active: Dict[str, int] = field(default_factory=dict)
    per_block_slots: Dict[str, int] = field(default_factory=dict)


def _count_rotations_on_cfg(cfg: Any) -> Tuple[int, int]:
    """Count enabled rotation flags on one block configuration."""
    fields = [name for name in vars(cfg).keys() if name.startswith("rotation_after_")]
    total = len(fields)
    active = sum(1 for name in fields if bool(getattr(cfg, name)))
    return active, total


def aggregate_rotation_signals(
        cfg_dicts: Mapping[str, Mapping[int, Any]],
        ) -> RotationRewardSignals:
    """Aggregate enabled rotations across block configurations."""
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
