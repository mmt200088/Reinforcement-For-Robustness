"""Rescale_optimizer 桥接层（加强版 stage2 RL 奖励侧）。

把 BLB 噪声选择（Block 1-5 的 ``Block*NoiseConfig``）转成 ``Rescale_optimizer``
所需的 ``delta_overrides``，调用优化器，再从返回 JSON 抽出三个 RL 奖励信号：

    1) ``fusion_count``        ── 模数链 fusion 次数（越少越好）
    2) ``total_bits``          ── 模数链 total_bits（越小越好）
    3) ``invalid_chain``       ── 链合法性（None=合法；非 None=不合法 + 原因）

约定：``Rescale_optimizer`` 本体不在本仓库（用户在 ``Rescale_optimizer/`` 子目录
里维护），所以这里**不把它当 Python module 直接 import**，而是通过可插拔的
``RescaleOptimizerInvoker`` 间接调用。三种现成 invoker：

  * ``SubprocessInvoker``：Rescale_optimizer 是命令行；bridge fork subprocess。
  * ``CallableInvoker``：Rescale_optimizer 是可 import 的 Python callable。
  * ``StubInvoker``：测试 / mock；返回预设 JSON。

reward 部分**只给信号，不给最终公式** —— 用户明说了 reward 不止依赖这三项。

典型用法（接 RL stage 2）：

    bridge = BLBNoiseRLBridge(handler, ...)
    rescale = RescaleOptimizerBridge(invoker=SubprocessInvoker(
        optimizer_root="Rescale_optimizer",
        configs={"block1_mrpc": "Rescale_optimizer/configs/mrpc/block1_mrpc.json", ...},
    ))

    # 一回合：
    #   1) RL 出动作 → 5 个 Block*NoiseConfig
    #   2) bridge.apply(...)  把 cfg 装到模型
    #   3) rescale.evaluate_blocks({"block1_mrpc": cfg1, ...}) 跑优化器
    #   4) signals = aggregate_optimizer_signals(outputs)
    #      → 业务侧用 signals['total_fusion_count'] / ['total_bits_sum'] /
    #        ['any_invalid'] 加上其它项算 reward
    #   5) bridge.clear() 还原模型
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, Union, runtime_checkable

from function_handler import (
    Block1NoiseConfig,
    Block2NoiseConfig,
    Block3NoiseConfig,
    Block4NoiseConfig,
    Block5NoiseConfig,
)


# ---------------------------------------------------------------------------
# 解析后的优化器输出
# ---------------------------------------------------------------------------

@dataclass
class RescaleOptimizerOutput:
    """Rescale_optimizer 单个 block 的解析输出（RL reward 侧用）。

    三个核心字段直接来自原始 JSON：
      * ``fusion_count``   = raw["fusion_count"]
      * ``total_bits``     = raw["result"]["chain"]["total_bits"]（链合法时；
                             链不合法时取 raw["result"].get("invalid_chain",{}).get("total_bits", 0)
                             或 0，由 ``_parse`` 决定）
      * ``invalid_chain``  = raw["result"]["invalid_chain"]
                             None ⇒ 链合法；dict ⇒ 不合法 + 原因
      * ``valid``          = (invalid_chain is None) and raw.get("valid", True)

    ``raw`` 保留完整 JSON，便于业务侧拿其它字段做更复杂 reward。
    """
    config_name: str
    fusion_count: int
    total_bits: int
    invalid_chain: Optional[dict]
    valid: bool
    raw: dict

    def to_signal_dict(self) -> dict:
        """打包成"基础奖励信号 dict"，让业务侧组合成最终 reward。"""
        return {
            "config_name": self.config_name,
            "valid": self.valid,
            "fusion_count": int(self.fusion_count),
            "total_bits": int(self.total_bits),
            "invalid_chain": self.invalid_chain,  # None or dict
        }


def _parse_optimizer_raw(raw: dict, *, config_name: str) -> RescaleOptimizerOutput:
    """从原始 JSON 抽出 fusion_count / total_bits / invalid_chain。

    严格按照用户给的 JSON 结构解析；缺字段时 fall back 到合理默认（不抛异常，
    保持 RL loop 稳健性，但是 valid 会被设为 False 让 RL 视作失败动作）。
    """
    if not isinstance(raw, Mapping):
        return RescaleOptimizerOutput(
            config_name=config_name, fusion_count=0, total_bits=0,
            invalid_chain={"reason": "raw_not_dict", "raw_type": type(raw).__name__},
            valid=False, raw={"_invalid": True},
        )

    fusion_count = int(raw.get("fusion_count", 0))
    result = raw.get("result") or {}
    invalid_chain = result.get("invalid_chain")
    valid = (invalid_chain is None) and bool(raw.get("valid", True)) and bool(result.get("valid", True))

    chain = result.get("chain") or {}
    total_bits = chain.get("total_bits")
    if total_bits is None:
        # 链不合法 → 优先取 invalid_chain 内的 total_bits（如果有），否则 0
        if isinstance(invalid_chain, Mapping):
            total_bits = invalid_chain.get("total_bits", 0)
        else:
            total_bits = 0
    total_bits = int(total_bits)

    return RescaleOptimizerOutput(
        config_name=str(config_name),
        fusion_count=fusion_count,
        total_bits=total_bits,
        invalid_chain=invalid_chain if isinstance(invalid_chain, Mapping) or invalid_chain is None else {"raw": invalid_chain},
        valid=valid,
        raw=dict(raw),
    )


# ---------------------------------------------------------------------------
# Invoker 协议 + 三种现成实现
# ---------------------------------------------------------------------------

@runtime_checkable
class RescaleOptimizerInvoker(Protocol):
    """``invoker(config_name, delta_overrides) -> raw JSON dict``。

    实现方负责：
      * 把 ``delta_overrides`` 喂给 ``Rescale_optimizer``（命名规则同用户给的 JSON 例子）
      * 返回完整 JSON（dict），里面至少要包含 ``fusion_count`` 与 ``result.chain.total_bits``
        / ``result.invalid_chain``
    """
    def __call__(self, config_name: str, delta_overrides: dict) -> dict: ...


class StubInvoker:
    """测试 / mock invoker：按 ``config_name`` 返回预设 JSON。

    可以直接用于 sanity test、CI（不依赖实际 Rescale_optimizer 二进制）。
    """
    def __init__(self, canned: Dict[str, dict]):
        self._canned = {str(k): dict(v) for k, v in canned.items()}

    def __call__(self, config_name: str, delta_overrides: dict) -> dict:
        if config_name not in self._canned:
            raise KeyError(
                f"StubInvoker: 未注册 config_name={config_name!r}。"
                f"可用 keys = {sorted(self._canned.keys())}"
            )
        out = dict(self._canned[config_name])
        # 把 RL 这次的 delta_overrides 回写到 raw 里，方便上层 introspect
        out.setdefault("delta_overrides", dict(delta_overrides))
        return out


class CallableInvoker:
    """直接包一个 Python callable：当 Rescale_optimizer 是可 import 的本地模块时用。

    ``fn`` 必须满足：``fn(config_name: str, delta_overrides: dict) -> dict``。
    """
    def __init__(self, fn: Callable[[str, dict], dict]):
        self._fn = fn

    def __call__(self, config_name: str, delta_overrides: dict) -> dict:
        out = self._fn(config_name, delta_overrides)
        if not isinstance(out, Mapping):
            raise TypeError(
                f"CallableInvoker: 期望 dict 返回值，实际 {type(out).__name__}"
            )
        return dict(out)


class SubprocessInvoker:
    """走 subprocess 调用 Rescale_optimizer 的命令行。

    工作流（默认）：
      1. 把 ``delta_overrides`` 写到临时 JSON：``replan_actions_<config_name>.json``
      2. 调用：
            ``<python> -m <cli_module> --config <config_path>
              --actions <actions_path> --output <output_path>``
      3. 读取 ``output_path`` 里的 JSON 并返回。

    config_path 由 ``configs[config_name]`` 决定；用户自行准备。

    如果实际优化器入口 / 参数不一样，**请用户传 ``cli_argv_builder``** 自定义命令行；
    或干脆用 ``CallableInvoker`` 直接接 Python 函数。
    """

    def __init__(
            self,
            *,
            configs: Mapping[str, str],
            optimizer_root: Optional[str] = None,
            python_exe: Optional[str] = None,
            cli_module: str = "rescale_optimizer.replan",
            actions_dir: Optional[str] = None,
            output_dir: Optional[str] = None,
            cli_argv_builder: Optional[
                Callable[[str, str, str, str], List[str]]
            ] = None,
            timeout_sec: float = 60.0,
            extra_env: Optional[Mapping[str, str]] = None,
            ):
        self.configs = {str(k): str(v) for k, v in configs.items()}
        self.optimizer_root = str(optimizer_root) if optimizer_root else None
        self.python_exe = str(python_exe) if python_exe else sys.executable
        self.cli_module = str(cli_module)
        self.actions_dir = str(actions_dir) if actions_dir else None
        self.output_dir = str(output_dir) if output_dir else None
        self.cli_argv_builder = cli_argv_builder
        self.timeout_sec = float(timeout_sec)
        self.extra_env = dict(extra_env) if extra_env else {}

    def _default_argv(self, config_path: str, actions_path: str, output_path: str, config_name: str) -> List[str]:
        return [
            self.python_exe, "-m", self.cli_module,
            "--config", config_path,
            "--actions", actions_path,
            "--output", output_path,
        ]

    def __call__(self, config_name: str, delta_overrides: dict) -> dict:
        if config_name not in self.configs:
            raise KeyError(
                f"SubprocessInvoker: 未注册 config_name={config_name!r}。"
                f"可用 keys = {sorted(self.configs.keys())}"
            )
        config_path = self.configs[config_name]

        actions_dir = self.actions_dir or tempfile.gettempdir()
        output_dir = self.output_dir or tempfile.gettempdir()
        os.makedirs(actions_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        actions_path = os.path.join(actions_dir, f"replan_actions_{config_name}.json")
        output_path = os.path.join(output_dir, f"rescale_result_{config_name}.json")

        with open(actions_path, "w", encoding="utf-8") as f:
            json.dump({"delta_overrides": dict(delta_overrides)}, f, ensure_ascii=False)

        argv_builder = self.cli_argv_builder or self._default_argv
        argv = argv_builder(config_path, actions_path, output_path, config_name)

        env = os.environ.copy()
        env.update(self.extra_env)
        cwd = self.optimizer_root if self.optimizer_root else None

        completed = subprocess.run(
            argv, cwd=cwd, env=env,
            capture_output=True, text=True, timeout=self.timeout_sec,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Rescale_optimizer 子进程退出码 {completed.returncode}，"
                f"stderr={completed.stderr[-500:]!r}"
            )
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"Rescale_optimizer 没产出 {output_path}（stdout={completed.stdout[-500:]!r}）"
            )
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# BLB cfg → delta_overrides 映射（per-block 默认 + 可注册覆盖）
# ---------------------------------------------------------------------------
# 用户给的 JSON 例子展示了 Block 1 的节点命名约定：
#   ctpt_ffn2 / ctpt_inv_d_1 / ctpt_inv_d_2 (CTPT_MUL)
#   ctct_ext_square (CTCT_MUL，delta = "x2")
#
# 这里给一个**默认映射**，实际优化器节点名以用户最终使用的 config 为准。
# 如果业务侧需要不同的命名 / delta 计算规则，请通过
# ``RescaleOptimizerBridge.register_cfg_to_delta_overrides(block_name, fn)``
# 注册自己的转换函数。

CfgToDeltaFn = Callable[[Any], Dict[str, Union[int, str]]]


def default_block1_cfg_to_delta(cfg: Block1NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 1 默认映射（基于用户提供的 JSON 例子）。

    delta_overrides 规则：
      * CTPT_MUL 节点 → 用 cfg 里对应的 ``encode.scaling_factor``（int）
      * CTCT_MUL 节点（squaring） → 固定 "x2" 标记
    rescale 字段不进 delta_overrides（用户的例子里没有暴露 rescale 节点）。
    """
    return {
        "ctpt_ffn2": int(cfg.wffn2_encode.scaling_factor),
        "ctpt_inv_d_1": int(cfg.mean_inv_d_encode.scaling_factor),
        "ctpt_inv_d_2": int(cfg.var_inv_d_encode.scaling_factor),
        "ctct_ext_square": "x2",
    }


def default_block2_cfg_to_delta(cfg: Block2NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 2 默认映射（占位；实际节点名以优化器配置为准）。"""
    return {
        "ctct_normalize": "x2",            # (1/std)·(X−μ) ct·ct
        "ctpt_gamma": int(cfg.gamma_encode.scaling_factor),
        "ctpt_wq": int(cfg.wq_encode.scaling_factor),
        "ctpt_wk": int(cfg.wk_encode.scaling_factor),
        "ctpt_wv": int(cfg.wv_encode.scaling_factor),
        "ctpt_kt_mask1": int(cfg.kt_mask1_encode.scaling_factor),
        "ctpt_kt_mask2": int(cfg.kt_mask2_encode.scaling_factor),
        "ctpt_q_mask1": int(cfg.q_mask1_encode.scaling_factor),
        "ctpt_q_mask2": int(cfg.q_mask2_encode.scaling_factor),
        "ctct_qk_matmul": "x2",            # Q·K^T ct·ct
        "ctpt_qkt_merge_mask": int(cfg.qkt_merge_mask_encode.scaling_factor),
    }


def default_block3_cfg_to_delta(cfg: Block3NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 3 默认映射（占位；实际节点名以优化器配置为准）。

    softmax exp 多项式：scalar_div + degree 次自乘。
    """
    deltas: Dict[str, Union[int, str]] = {
        "ctpt_softmax_inv_2n": int(cfg.inv_2n_encode.scaling_factor),
    }
    for k in range(int(cfg.degree)):
        deltas[f"ctct_softmax_pow_s{k+1}"] = "x2"
    return deltas


def default_block4_cfg_to_delta(cfg: Block4NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 4 默认映射（占位；实际节点名以优化器配置为准）。"""
    return {
        "ctpt_softmax_out_mask": int(cfg.softmax_out_mask_encode.scaling_factor),
        "ctpt_v_mask": int(cfg.v_mask_encode.scaling_factor),
        "ctct_softmax_v": "x2",            # softmax×V ct·ct matmul
        "ctpt_softmax_v_mask": int(cfg.softmax_v_mask_encode.scaling_factor),
        "ctpt_wo": int(cfg.wo_encode.scaling_factor),
        "ctpt_inv_d_attn_mean": int(cfg.ln_mean_inv_d_encode.scaling_factor),
        "ctct_attn_square": "x2",           # post-attn LN (X−μ)²
        "ctpt_inv_d_attn_var": int(cfg.ln_var_inv_d_encode.scaling_factor),
    }


def default_block5_cfg_to_delta(cfg: Block5NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 5 默认映射（占位；实际节点名以优化器配置为准）。

    含 LN tail (×1/std, ×γ) + Wffn1 + GELU 多项式（degree 决定 power 数）。
    """
    deltas: Dict[str, Union[int, str]] = {
        "ctct_normalize_attn": "x2",       # (1/std)·(X−μ)
        "ctpt_gamma_attn": int(cfg.gamma_encode.scaling_factor),
        "ctpt_wffn1": int(cfg.wffn1_encode.scaling_factor),
        "ctpt_gelu_coeff": int(cfg.gelu_coeff_encode.scaling_factor),
    }
    if cfg.gelu_degree >= 2:
        deltas["ctct_gelu_x2"] = "x2"
    if cfg.gelu_degree >= 4:
        deltas["ctct_gelu_x3"] = "x2"
        deltas["ctct_gelu_x4"] = "x2"
    return deltas


_DEFAULT_BLOCK_MAPPERS: Dict[str, CfgToDeltaFn] = {
    "block1": default_block1_cfg_to_delta,
    "block2": default_block2_cfg_to_delta,
    "block3": default_block3_cfg_to_delta,
    "block4": default_block4_cfg_to_delta,
    "block5": default_block5_cfg_to_delta,
}


# ---------------------------------------------------------------------------
# 主桥接
# ---------------------------------------------------------------------------

@dataclass
class _BlockRequest:
    config_name: str
    cfg: Any
    block_name: str  # "block1" .. "block5"


class RescaleOptimizerBridge:
    """一站式：BLB cfg → delta_overrides → 调用 Rescale_optimizer → 解析输出。

    设计取舍：
      * 每个 (block, layer) 不一定对应一个独立的 ``Rescale_optimizer`` config；
        典型场景下用户的 ``Rescale_optimizer`` 是按 (block, dataset) 切的，比如
        ``block1_mrpc``、``block1_qqp``。RL 一回合可能跑多个 (block, dataset) 组合。
      * 因此 ``evaluate_blocks(...)`` 接受一个 ``{config_name: (block_name, cfg)}``
        映射，每条记录被独立 invoke 一次。

    cfg → delta_overrides：先看用户在构造时是否传了 ``cfg_to_delta_overrides``
    覆盖；否则用 ``_DEFAULT_BLOCK_MAPPERS[block_name]``。
    """

    def __init__(
            self,
            invoker: RescaleOptimizerInvoker,
            *,
            cfg_to_delta_overrides: Optional[Mapping[str, CfgToDeltaFn]] = None,
            ):
        self.invoker = invoker
        # 先深复制默认映射，再用业务侧覆盖
        self._cfg_mappers: Dict[str, CfgToDeltaFn] = dict(_DEFAULT_BLOCK_MAPPERS)
        if cfg_to_delta_overrides:
            for k, fn in cfg_to_delta_overrides.items():
                self._cfg_mappers[str(k)] = fn

    def register_cfg_to_delta_overrides(self, block_name: str, fn: CfgToDeltaFn) -> None:
        """业务侧动态注册 / 覆盖某个 block 的 cfg → delta 转换函数。"""
        self._cfg_mappers[str(block_name)] = fn

    def cfg_to_delta_overrides(self, block_name: str, cfg: Any) -> Dict[str, Union[int, str]]:
        """单独把 cfg 翻译成 delta_overrides；不调用优化器，仅做翻译。"""
        block_name = str(block_name)
        if block_name not in self._cfg_mappers:
            raise KeyError(
                f"未注册 block_name={block_name!r} 的 cfg→delta 映射；"
                f"已知 = {sorted(self._cfg_mappers.keys())}"
            )
        return dict(self._cfg_mappers[block_name](cfg))

    def evaluate(
            self,
            *,
            config_name: str,
            block_name: str,
            cfg: Any,
            extra_overrides: Optional[Mapping[str, Union[int, str]]] = None,
            ) -> RescaleOptimizerOutput:
        """对单个 (config, block, cfg) 三元组跑一次优化器。

        ``extra_overrides`` 会覆盖默认 cfg→delta 翻译的相同 key（用于业务侧
        对个别节点强行指定 delta，比如固定 ctct_ext_square="x2"）。
        """
        deltas = self.cfg_to_delta_overrides(block_name, cfg)
        if extra_overrides:
            deltas.update({str(k): v for k, v in extra_overrides.items()})
        raw = self.invoker(config_name, deltas)
        return _parse_optimizer_raw(raw, config_name=config_name)

    def evaluate_blocks(
            self,
            requests: Mapping[str, Tuple[str, Any]],
            *,
            extra_overrides: Optional[Mapping[str, Mapping[str, Union[int, str]]]] = None,
            ) -> Dict[str, RescaleOptimizerOutput]:
        """一次跑多个 config。

        Args:
            requests: ``{config_name: (block_name, cfg)}``，比如
                      ``{"block1_mrpc": ("block1", block1_cfg), ...}``
            extra_overrides: ``{config_name: {node: delta, ...}}``，可选

        Returns:
            ``{config_name: RescaleOptimizerOutput}``
        """
        outputs: Dict[str, RescaleOptimizerOutput] = {}
        for config_name, (block_name, cfg) in requests.items():
            xtra = (extra_overrides or {}).get(config_name)
            outputs[config_name] = self.evaluate(
                config_name=config_name,
                block_name=block_name,
                cfg=cfg,
                extra_overrides=xtra,
            )
        return outputs


# ---------------------------------------------------------------------------
# 奖励信号聚合
# ---------------------------------------------------------------------------

@dataclass
class OptimizerRewardSignals:
    """跨 block 聚合后的 RL 奖励原料。

    业务侧把这些字段组合成最终 reward；本 bridge 不做最终公式（用户明说了
    reward 不止依赖这三项）。
    """
    total_fusion_count: int
    total_bits_sum: int
    any_invalid: bool
    valid_block_count: int
    invalid_block_count: int
    per_config: Dict[str, dict] = field(default_factory=dict)
    invalid_chains: Dict[str, dict] = field(default_factory=dict)


def apply_rotation_flags_to_cfg(cfg: Any, rotation_flag_names) -> None:
    """把"开启的 rotation 候选点列表"应用到 cfg 上。

    cfg 上所有 ``rotation_after_*`` bool 字段：在 ``rotation_flag_names`` 里出现
    的置 True，其余统一置 False。这是和 Rescale_optimizer 输出对接的最小钩子 ──
    业务侧需要先把优化器输出的 ``effective_rotations`` 转换成 BLB 命名空间的
    flag 名（per-block 名字表，本仓库不写死）。

    Args:
        cfg:                Block*NoiseConfig 实例（任意 block）
        rotation_flag_names: iterable[str]，要置 True 的 ``rotation_after_*`` 字段名
    """
    enable = {str(n) for n in rotation_flag_names}
    for name in vars(cfg).keys():
        if name.startswith("rotation_after_"):
            setattr(cfg, name, name in enable)


def aggregate_optimizer_signals(
        outputs: Mapping[str, RescaleOptimizerOutput],
        ) -> OptimizerRewardSignals:
    """跨 block 聚合 fusion_count / total_bits / invalid_chain。"""
    fusion_total = 0
    bits_total = 0
    valid_n = 0
    invalid_n = 0
    per_config: Dict[str, dict] = {}
    invalid_chains: Dict[str, dict] = {}
    for cname, out in outputs.items():
        fusion_total += int(out.fusion_count)
        bits_total += int(out.total_bits)
        if out.valid:
            valid_n += 1
        else:
            invalid_n += 1
            if out.invalid_chain is not None:
                invalid_chains[cname] = dict(out.invalid_chain)
        per_config[cname] = out.to_signal_dict()

    return OptimizerRewardSignals(
        total_fusion_count=fusion_total,
        total_bits_sum=bits_total,
        any_invalid=(invalid_n > 0),
        valid_block_count=valid_n,
        invalid_block_count=invalid_n,
        per_config=per_config,
        invalid_chains=invalid_chains,
    )
