"""Rescale_optimizer 桥接层（加强版 stage2 RL 奖励侧）。

把 BLB 噪声选择（Block 1-5 的 ``Block*NoiseConfig``）转成 ``Rescale_optimizer``
的 ``replan`` 输入（``t_new`` + ``delta_overrides``），调用优化器，再从返回
JSON 抽出三个 RL 奖励信号：

    1) ``fusion_count``        ── 模数链 fusion 次数（越少越好）
    2) ``total_bits``          ── 模数链 total_bits（越小越好）
    3) ``invalid_chain``       ── 链合法性（None=合法；非 None=不合法 + 原因）

可用的 invoker 实现（4 种）：

  * ``InProcessInvoker`` ── **推荐**。直接 ``import rescale_optimizer`` 调
    ``replan_with_user_actions``；预加载图与 baseline，单步开销 ms 级。
  * ``SubprocessInvoker`` ── fork ``python scripts/replan_what_if.py``，开销大但
    完全隔离；适合 debug 时用。
  * ``CallableInvoker`` ── 包装任意 ``(config_name, payload) -> dict`` callable。
  * ``StubInvoker`` ── 单元测试用 canned 响应。注意：BLB Stage-2 RL 训练 / 最终
    评估路径只接受 ``InProcessInvoker`` / ``SubprocessInvoker`` 这两种"真正过
    Rescale_optimizer 计算"的 invoker；之前的 ``HeuristicStubInvoker`` 已删除。

invoker 调用签名：``invoker(config_name, payload)``，``payload`` 支持两种形态：

  (1) **bare dict**（向后兼容）：``{"node_name": delta, ...}``，被当作
      ``delta_overrides``，``t_new`` 默认取 baseline t_baseline。
  (2) **rich dict**：``{"t_new": [int, ...], "delta_overrides": {...}}``，``t_new``
      长度必须 == baseline skeleton 的 R+1。

reward 部分**只给信号，不给最终公式** —— 用户明说了 reward 不止依赖这三项。

典型用法：

    from rescale_optimizer_bridge import (
        InProcessInvoker, RescaleOptimizerBridge,
        aggregate_optimizer_signals,
    )
    inv = InProcessInvoker.from_profile(
        rescale_optimizer_root="Rescale_optimizer",
        profile="mrpc",
        configs_dir="Rescale_optimizer/configs/mrpc",
    )
    bridge = RescaleOptimizerBridge(invoker=inv)
    out = bridge.evaluate(
        config_name="block1_mrpc",
        block_name="block1",
        cfg=block1_cfg,                # Block1NoiseConfig
        t_new=[30, 34, 34],            # length R+1; None ⇒ 用 baseline
    )
    print(out.fusion_count, out.total_bits, out.invalid_chain)
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


def _split_payload(payload: Any) -> Tuple[Optional[List[int]], Dict[str, Union[int, str]]]:
    """从 invoker 调用方接到的 ``payload`` 中拆出 (t_new, delta_overrides)。

    向后兼容：
      * 如果 payload 是 bare dict（不含 ``"t_new"`` 也不含 ``"delta_overrides"``），
        把它整体当作 ``delta_overrides``，``t_new=None``（调用侧使用 baseline）。
      * 如果 payload 是 rich dict（含 ``"t_new"`` 和/或 ``"delta_overrides"``），
        分别取出。
    """
    if not isinstance(payload, Mapping):
        return None, {}
    keys = set(payload.keys())
    if keys & {"t_new", "delta_overrides"}:
        t_new_raw = payload.get("t_new")
        t_new = [int(x) for x in t_new_raw] if t_new_raw is not None else None
        deltas_raw = payload.get("delta_overrides") or {}
        deltas = {str(k): v for k, v in deltas_raw.items()}
        return t_new, deltas
    # bare dict
    return None, {str(k): v for k, v in payload.items()}


def load_baseline_archive(path: str) -> Dict[str, Tuple[List[int], List[int], List[int]]]:
    """读 ``static_skeletons_<profile>.json`` → ``{config_name: (skeleton, t_baseline, q_bits_baseline)}``。

    schema v2 (``cut_point_sf`` / ``modulus_chain.drop_order``)：
      * skeleton: ``entry["skeleton"]``
      * t_baseline: 按 skeleton 取 cut_point_sf 的 ``sf_post`` (rescale 点) 或 ``sf`` (source/普通点)
      * q_bits_baseline: ``modulus_chain.drop_order[1:-1]``（去掉首尾 head/tail prime）
    """
    with open(path, "r", encoding="utf-8") as f:
        archive = json.load(f)
    out: Dict[str, Tuple[List[int], List[int], List[int]]] = {}
    for entry in archive.get("results", []):
        if not entry.get("success"):
            continue
        cname = str(entry["config_name"])
        skel = [int(x) for x in entry.get("skeleton", [])]
        # t_baseline
        t_for_idx: Dict[int, int] = {}
        for row in entry.get("cut_point_sf", []):
            i = int(row["i"])
            if "sf_post" in row:
                t_for_idx[i] = int(row["sf_post"])
            elif "sf" in row:
                t_for_idx[i] = int(row["sf"])
        t_base = [t_for_idx[i] for i in skel if i in t_for_idx]
        # q_bits
        mc = entry.get("modulus_chain", {}) or {}
        drop_order = list(mc.get("drop_order", []))
        q_base = [int(x) for x in drop_order[1:-1]] if len(drop_order) >= 2 else []
        out[cname] = (skel, t_base, q_base)
    return out


class InProcessInvoker:
    """Real in-process Rescale_optimizer invoker.

    Thin adapter around ``rescale_optimizer.ReplanSession``: preserves this
    bridge's invoker surface (``__call__(graph_key, payload) -> dict``,
    ``baselines`` property in tuple form, ``from_profile`` classmethod) while
    delegating all graph / baseline / replan work to ``ReplanSession``.

    Usage:

        inv = InProcessInvoker.from_profile(
            rescale_optimizer_root="Rescale_optimizer",
            profile="mrpc",
        )
        bridge = RescaleOptimizerBridge(invoker=inv)

    Manual config map:

        inv = InProcessInvoker(
            configs={
                "block1_mrpc": "Rescale_optimizer/configs/mrpc/block1_mrpc.json",
                ...,
            },
            baseline_archive="Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json",
            rescale_optimizer_root="Rescale_optimizer",
        )
    """

    def __init__(
            self,
            *,
            configs: Mapping[str, str],
            baseline_archive: str,
            rescale_optimizer_root: Optional[str] = None,
            ):
        if rescale_optimizer_root:
            root = os.path.abspath(rescale_optimizer_root)
            if root not in sys.path:
                sys.path.insert(0, root)

        # Lazy import keeps this module usable when Rescale_optimizer
        # isn't installed (e.g. unit tests that only need the bridge skeleton).
        from rescale_optimizer import ReplanSession

        self._session = ReplanSession(
            configs={str(k): str(v) for k, v in configs.items()},
            baseline_archive=str(baseline_archive),
        )

    @classmethod
    def from_profile(
            cls,
            *,
            rescale_optimizer_root: str,
            profile: str,
            configs_dir: Optional[str] = None,
            baseline_archive: Optional[str] = None,
            include: Optional[List[str]] = None,
            ) -> "InProcessInvoker":
        """Scan ``configs/<profile>/*.json`` and register every graph with a
        successful baseline entry."""
        root = os.path.abspath(str(rescale_optimizer_root))
        if root not in sys.path:
            sys.path.insert(0, root)
        from rescale_optimizer import ReplanSession

        inv = cls.__new__(cls)
        inv._session = ReplanSession.from_profile(
            profile=str(profile),
            root=root,
            configs_dir=configs_dir,
            baseline_archive=baseline_archive,
            include=include,
        )
        return inv

    @property
    def session(self):
        """Underlying ``ReplanSession`` — exposed for tools that need richer API."""
        return self._session

    @property
    def baselines(self) -> Dict[str, Tuple[List[int], List[int], List[int]]]:
        """``{graph_key: (skeleton, t_baseline, q_bits_baseline)}`` — tuple form
        for backward compatibility with bridge code that iterates baselines."""
        return {
            k: (list(rec.skeleton), list(rec.t_baseline), list(rec.q_bits_baseline))
            for k, rec in self._session.baselines.items()
        }

    def __call__(self, config_name: str, payload: Any) -> dict:
        return self._session(str(config_name), payload)


class SubprocessInvoker:
    """fork ``python scripts/replan_what_if.py`` 调用 Rescale_optimizer。

    单步开销大（每次 ~几百 ms），主要给 debug / 隔离场景用。RL 训练推荐
    ``InProcessInvoker``。

    工作流：
      1. 把 ``{"t_new": [...], "delta_overrides": {...}}`` 写到临时 JSON
         ``replan_actions_<config_name>.json``；
      2. 调用：
            ``<python> <root>/scripts/replan_what_if.py
                --config <root>/configs/<profile>/<cname>.json
                --baseline-from <static_skeletons_path>
                --actions-file <tmp_actions>
                --out <tmp_out>``
      3. 读取 ``<tmp_out>`` 里的 JSON。
    """

    def __init__(
            self,
            *,
            rescale_optimizer_root: str,
            configs: Mapping[str, str],
            baseline_archive: str,
            python_exe: Optional[str] = None,
            actions_dir: Optional[str] = None,
            output_dir: Optional[str] = None,
            cli_script: str = "scripts/replan_what_if.py",
            timeout_sec: float = 60.0,
            extra_env: Optional[Mapping[str, str]] = None,
            ):
        self.rescale_optimizer_root = os.path.abspath(rescale_optimizer_root)
        self.configs = {str(k): str(v) for k, v in configs.items()}
        self.baseline_archive = str(baseline_archive)
        self.python_exe = str(python_exe) if python_exe else sys.executable
        self.actions_dir = str(actions_dir) if actions_dir else None
        self.output_dir = str(output_dir) if output_dir else None
        self.cli_script = str(cli_script)
        self.timeout_sec = float(timeout_sec)
        self.extra_env = dict(extra_env) if extra_env else {}

    def __call__(self, config_name: str, payload: Any) -> dict:
        cname = str(config_name)
        if cname not in self.configs:
            raise KeyError(
                f"SubprocessInvoker: 未注册 config_name={cname!r}。"
                f"可用 keys = {sorted(self.configs.keys())}"
            )
        config_path = self.configs[cname]
        actions_dir = self.actions_dir or tempfile.gettempdir()
        output_dir = self.output_dir or tempfile.gettempdir()
        os.makedirs(actions_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        actions_path = os.path.join(actions_dir, f"replan_actions_{cname}.json")
        output_path = os.path.join(output_dir, f"rescale_result_{cname}.json")

        t_new, delta_overrides = _split_payload(payload)
        actions_doc: Dict[str, Any] = {"config_name": cname}
        if t_new is not None:
            actions_doc["t_new"] = list(t_new)
        if delta_overrides:
            actions_doc["delta_overrides"] = dict(delta_overrides)
        with open(actions_path, "w", encoding="utf-8") as f:
            json.dump(actions_doc, f, ensure_ascii=False)

        argv = [
            self.python_exe,
            os.path.join(self.rescale_optimizer_root, self.cli_script),
            "--config", config_path,
            "--baseline-from", self.baseline_archive,
            "--config-name", cname,
            "--actions-file", actions_path,
            "--out", output_path,
        ]

        env = os.environ.copy()
        env.update(self.extra_env)
        completed = subprocess.run(
            argv, cwd=self.rescale_optimizer_root, env=env,
            capture_output=True, text=True, timeout=self.timeout_sec,
        )
        # 注意 replan_what_if.py 在 valid=False 时 returncode=3，但仍然写出 JSON
        if completed.returncode not in (0, 3):
            raise RuntimeError(
                f"replan_what_if.py 子进程退出码 {completed.returncode}，"
                f"stderr={completed.stderr[-500:]!r}"
            )
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"replan_what_if.py 没产出 {output_path}（stdout={completed.stdout[-500:]!r}）"
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
    """Block 1 默认映射 —— 与 ``configs/<profile>/block1_<profile>.json`` 节点对齐。

    实际节点（schema v2）：
      * ctpt_ffn2 (CTPT_MUL)        ← Wffn2·X
      * ctpt_inv_d_1 (CTPT_MUL)     ← μ 的 1/D
      * ctct_ext_square (CTCT_MUL)  ← (X−μ)²，固定 "x2"
      * ctpt_inv_d_2 (CTPT_MUL)     ← var 的 1/D
    """
    return {
        "ctpt_ffn2":      int(cfg.wffn2_encode.scaling_factor),
        "ctpt_inv_d_1":   int(cfg.mean_inv_d_encode.scaling_factor),
        "ctct_ext_square": "x2",
        "ctpt_inv_d_2":   int(cfg.var_inv_d_encode.scaling_factor),
    }


def default_block2_cfg_to_delta(cfg: Block2NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 2 默认映射 —— 与 ``block2_<profile>.json`` 节点对齐。

    实际节点：
      * ctct_x_mean_over_std (CTCT_MUL) ← (X−μ)·(1/std)，固定 "x2"
      * ctpt_gama1 (CTPT_MUL)           ← γ
      * ctpt_wq_wk (CTPT_MUL)           ← **Q/K 共享**一个节点
      * ctpt_rotKT_mask1 (CTPT_MUL)     ← K^T BSGS mask 1
      * ctpt_rotKT_mask2 (CTPT_MUL)     ← K^T BSGS mask 2
      * ctct_preprocess_qkt (CTCT_MUL)  ← Q·K^T，固定 "x2"
      * ctpt_mask (CTPT_MUL)            ← 合并 Q,K mask

    2026-05-14 起 BLB Stage-2 RL 的 Q 侧动作（wq_sf / q_mask1_sf / q_mask2_sf）
    与 K 侧绑定 —— ``_build_block2_action`` 用 K 侧的 SF 同时填 cfg 的 Q/K 字段，
    所以这里 ``ctpt_wq_wk`` 直接读 ``cfg.wk_encode.scaling_factor``（语义上"由 K
    侧控制"）。``wk_encode == wq_encode`` 始终成立。BLB cfg 的 ``wv_encode``
    在该 graph 没有对应节点，被丢弃（仅影响模型噪声）。
    """
    return {
        "ctct_x_mean_over_std": "x2",
        "ctpt_gama1":          int(cfg.gamma_encode.scaling_factor),
        "ctpt_wq_wk":          int(cfg.wk_encode.scaling_factor),
        "ctpt_rotKT_mask1":    int(cfg.kt_mask1_encode.scaling_factor),
        "ctpt_rotKT_mask2":    int(cfg.kt_mask2_encode.scaling_factor),
        "ctct_preprocess_qkt": "x2",
        "ctpt_mask":           int(cfg.qkt_merge_mask_encode.scaling_factor),
    }


def default_block3_cfg_to_delta(cfg: Block3NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 3 默认映射 —— 与 ``block3_exp_n<degree>.json`` 节点对齐。

    实际节点：
      * ctpt_inv_2n (CTPT_MUL)             ← 1/2^n
      * ctct_square_1 .. ctct_square_<n> (CTCT_MUL) ← 迭代平方，固定 "x2"
    """
    deltas: Dict[str, Union[int, str]] = {
        "ctpt_inv_2n": int(cfg.inv_2n_encode.scaling_factor),
    }
    for k in range(int(cfg.degree)):
        deltas[f"ctct_square_{k+1}"] = "x2"
    return deltas


def default_block4_cfg_to_delta(cfg: Block4NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 4 默认映射 —— 与 ``block4.json`` 节点对齐。

    实际节点（注意只有 2 个 mask 节点；BLB cfg 的 3 个 mask encode 中
    ``softmax_out_mask`` / ``v_mask`` 被 graph 合成一个 ``ctpt_mask2``，
    我们取 ``softmax_out_mask`` 的 SF）：
      * ctpt_mask2 (CTPT_MUL)               ← softmax 输出 / V 路径上的 mask
      * ctct_rot_softmax_mul_v (CTCT_MUL)   ← softmax×V matmul，MRPC baseline delta=39
      * ctpt_mask (CTPT_MUL)                ← 合并 softmax×V 后的 mask
      * ctpt_wo_attnout (CTPT_MUL)          ← Wo
      * ctpt_inv_d_1 (CTPT_MUL)             ← post-attn LN μ 的 1/D
      * ctct_square (CTCT_MUL)              ← post-attn LN (X−μ)²，固定 "x2"
      * ctpt_inv_d_2 (CTPT_MUL)             ← post-attn LN var 的 1/D
    """
    return {
        "ctpt_mask2":             int(cfg.softmax_out_mask_encode.scaling_factor),
        "ctct_rot_softmax_mul_v": 39,
        "ctpt_mask":              int(cfg.softmax_v_mask_encode.scaling_factor),
        "ctpt_wo_attnout":        int(cfg.wo_encode.scaling_factor),
        "ctpt_inv_d_1":           int(cfg.ln_mean_inv_d_encode.scaling_factor),
        "ctct_square":            "x2",
        "ctpt_inv_d_2":           int(cfg.ln_var_inv_d_encode.scaling_factor),
    }


def default_block5_cfg_to_delta(cfg: Block5NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 5 默认映射 —— 与 ``block5_n<degree>.json`` 节点对齐。

    实际节点（按 GELU degree 不同，graph 含的 ctct_gelu_* 也不同）：
      * ctct_xmean_over_std (CTCT_MUL) ← post-attn (X−μ)·(1/std)
      * ctpt_gamal (CTPT_MUL)          ← γ
      * ctpt_wffn1 (CTPT_MUL)          ← W_ffn1·X
      * (degree==1：无 ctct_gelu_*)
      * (degree==2：ctct_gelu_x2)
      * (degree==4：ctct_gelu_x2 + ctct_gelu_x4；**不含 ctct_gelu_x3**，
        因为 graph 把 x^3 直接折进 x^4)
      * ctpt_gelu_coeff (CTPT_MUL)     ← 多项式系数
    """
    deltas: Dict[str, Union[int, str]] = {
        "ctct_xmean_over_std": "x2",
        "ctpt_gamal":          int(cfg.gamma_encode.scaling_factor),
        "ctpt_wffn1":          int(cfg.wffn1_encode.scaling_factor),
    }
    if cfg.gelu_degree >= 2:
        deltas["ctct_gelu_x2"] = "x2"
    if cfg.gelu_degree >= 4:
        deltas["ctct_gelu_x4"] = "x2"
    deltas["ctpt_gelu_coeff"] = int(cfg.gelu_coeff_encode.scaling_factor)
    return deltas


# ---------------------------------------------------------------------------
# Inverse map: graph node name -> (cfg attribute path that drives this node).
#
# Stays in sync with default_block{N}_cfg_to_delta above. Each entry maps a
# graph node (the key the bridge sends in delta_overrides) back to the cfg
# attribute the bridge read to produce that value. Literal-valued entries
# (e.g. "ctct_ext_square" -> "x2") have no cfg origin and are absent here.
#
# `block5` entries gated on cfg.gelu_degree are listed unconditionally; the
# caller must skip those that don't apply for the active degree.
GRAPH_NODE_TO_CFG_ATTR: Dict[int, Dict[str, str]] = {
    1: {
        "ctpt_ffn2":    "wffn2_encode",
        "ctpt_inv_d_1": "mean_inv_d_encode",
        "ctpt_inv_d_2": "var_inv_d_encode",
    },
    2: {
        "ctpt_gama1":       "gamma_encode",
        # ``ctpt_wq_wk`` 现在从 ``cfg.wk_encode`` 读（Q/K 绑定，K 侧为主）
        "ctpt_wq_wk":       "wk_encode",
        "ctpt_rotKT_mask1": "kt_mask1_encode",
        "ctpt_rotKT_mask2": "kt_mask2_encode",
        "ctpt_mask":        "qkt_merge_mask_encode",
    },
    3: {
        "ctpt_inv_2n": "inv_2n_encode",
    },
    4: {
        "ctpt_mask2":      "softmax_out_mask_encode",
        "ctpt_mask":       "softmax_v_mask_encode",
        "ctpt_wo_attnout": "wo_encode",
        "ctpt_inv_d_1":    "ln_mean_inv_d_encode",
        "ctpt_inv_d_2":    "ln_var_inv_d_encode",
    },
    5: {
        "ctpt_gamal":      "gamma_encode",
        "ctpt_wffn1":      "wffn1_encode",
        "ctpt_gelu_coeff": "gelu_coeff_encode",
    },
}


_DEFAULT_BLOCK_MAPPERS: Dict[str, CfgToDeltaFn] = {
    "block1": default_block1_cfg_to_delta,
    "block2": default_block2_cfg_to_delta,
    "block3": default_block3_cfg_to_delta,
    "block4": default_block4_cfg_to_delta,
    "block5": default_block5_cfg_to_delta,
}


# ---------------------------------------------------------------------------
# BLB cfg → t_new 映射（per-(block, profile) 默认 + 可注册覆盖）
# ---------------------------------------------------------------------------
# 为什么需要这张表：
#   ``Rescale_optimizer`` 的 ``replan_with_user_actions`` 接受的 ``t_new`` 是
#   "skeleton 上每个 stage 的目标 SF"。BLB 的 cfg 里每个 rescale 字段（如
#   ``Block1NoiseConfig.mean_result_rescale``）的 ``scaling_factor`` 就是该
#   rescale 点的目标 SF —— 只是哪些 rescale 点真正落到 skeleton 上由 graph
#   决定。所以需要一张 (config_name) → (skeleton 顺序的 cfg 字段路径) 的表。
#
#   skeleton 第 r=0 个 stage 是 source，对应 cfg 的 fresh 字段；
#   skeleton 第 r>=1 个 stage 是某个 multiplication 节点之后的 rescale 点，
#   对应 cfg 的 *_result_rescale 字段。
#
# 当 RL 没启用某个 rescale 字段 (cfg.field is None) 时，``cfg_to_t_new_from_table``
# 会把该位置 fallback 到 baseline t (如果调用方提供 baseline_t_new)，否则该
# (config_name) 整体放弃自动派生 t_new，让 invoker 走 baseline。

@dataclass(frozen=True)
class _SkelEntry:
    """skeleton 上某个 stage 对应到 cfg 哪个字段。

    Args:
        cfg_field:   cfg 上的属性名（必须是 NoisePoint 或 Optional[NoisePoint]
                     或 Tuple[Optional[NoisePoint]]）
        tuple_index: 若 cfg_field 是 tuple，取第 N 项；None=该字段是单一 NoisePoint。
                     支持负索引（如 -1 表示 last 项）。
    """
    cfg_field: str
    tuple_index: Optional[int] = None


# 默认表：覆盖 ``Rescale_optimizer/configs/mrpc/`` 下 11 个 baseline。
# 其它 profile（wnli 等）需要自行扩展或通过 ``RescaleOptimizerBridge(cfg_to_t_new_overrides=...)``
# 注册。
DEFAULT_CFG_TO_T_NEW_MAP: Dict[str, Tuple[_SkelEntry, ...]] = {
    # --- block 1 (mrpc): skeleton=[0, 2, 4, 5]
    #   i=0 (gelu_out)     → fresh
    #   i=2 (ctpt_inv_d_1) → mean (μ) 的 rescale
    #   i=4 (ctpt_inv_d_2) → var 的 rescale
    "block1_mrpc": (
        _SkelEntry("gelu_out_fresh"),
        _SkelEntry("mean_result_rescale"),
        _SkelEntry("var_result_rescale"),
    ),
    # --- block 2 (mrpc): skeleton=[0, 2, 5, 7, 8]
    #   i=0 (inv_std)         → cfg.inv_std_fresh
    #   i=2 (ctpt_gama1)      → γ rescale
    #   i=5 (ctpt_rotKT_mask2)→ kt_mask2 rescale
    #   i=7 (ctpt_mask)       → qkt_merge_mask rescale
    "block2_mrpc": (
        _SkelEntry("inv_std_fresh"),
        _SkelEntry("gamma_result_rescale"),
        _SkelEntry("kt_mask2_result_rescale"),
        _SkelEntry("qkt_merge_mask_result_rescale"),
    ),
    # --- block 3 (mrpc, exp_n<degree>): skeleton=[0, 2..(2+deg-1), dummy]
    #   i=0          → cfg.x_fresh
    #   i=2..        → cfg.square_rescales[k-1]
    "block3_exp_n2": (
        _SkelEntry("x_fresh"),
        _SkelEntry("square_rescales", 0),
        _SkelEntry("square_rescales", 1),
    ),
    "block3_exp_n3": (
        _SkelEntry("x_fresh"),
        _SkelEntry("square_rescales", 0),
        _SkelEntry("square_rescales", 1),
        _SkelEntry("square_rescales", 2),
    ),
    "block3_exp_n4": (
        _SkelEntry("x_fresh"),
        _SkelEntry("square_rescales", 0),
        _SkelEntry("square_rescales", 1),
        _SkelEntry("square_rescales", 2),
        _SkelEntry("square_rescales", 3),
    ),
    "block3_exp_n5": (
        _SkelEntry("x_fresh"),
        _SkelEntry("square_rescales", 0),
        _SkelEntry("square_rescales", 1),
        _SkelEntry("square_rescales", 2),
        _SkelEntry("square_rescales", 3),
        # cfg 只有 max degree-1 的 square_rescales（受 RL action 维度限制 max=4）；
        # 第 5 次平方就重用最后一个 entry 作为 fallback。
        _SkelEntry("square_rescales", 3),
    ),
    "block3_exp_n6": (
        _SkelEntry("x_fresh"),
        _SkelEntry("square_rescales", 0),
        _SkelEntry("square_rescales", 1),
        _SkelEntry("square_rescales", 2),
        _SkelEntry("square_rescales", 3),
        _SkelEntry("square_rescales", 3),
        _SkelEntry("square_rescales", 3),
    ),
    # --- block 4: skeleton=[0, 2, 5, 7, 8]
    #   i=0  (rot_softmax)             → softmax_out_fresh
    #   i=2  (ctct_rot_softmax_mul_v)  → softmax_v_matmul_rescale
    #   i=5  (ctpt_inv_d_1)            → ln_mean_result_rescale
    #   i=7  (ctpt_inv_d_2)            → ln_var_result_rescale
    "block4": (
        _SkelEntry("softmax_out_fresh"),
        _SkelEntry("softmax_v_matmul_rescale"),
        _SkelEntry("ln_mean_result_rescale"),
        _SkelEntry("ln_var_result_rescale"),
    ),
    # --- block 5 (n=1): skeleton=[0, 2, 4, 5]
    #   i=0 (x_mean)            → x_centered_fresh
    #   i=2 (ctpt_gamal)        → gamma_result_rescale
    #   i=4 (ctpt_gelu_coeff)   → gelu_coeff_mul_rescales[-1]（深度最大的那个 coeff·x^k）
    "block5_n1": (
        _SkelEntry("x_centered_fresh"),
        _SkelEntry("gamma_result_rescale"),
        _SkelEntry("gelu_coeff_mul_rescales", -1),
    ),
    # --- block 5 (n=2): skeleton=[0, 1, 3, 5, 6]
    #   i=0 (x_mean)              → x_centered_fresh
    #   i=1 (ctct_xmean_over_std) → normalize_result_rescale
    #   i=3 (ctpt_wffn1)          → wffn1_result_rescale
    #   i=5 (ctpt_gelu_coeff)     → gelu_coeff_mul_rescales[-1]
    "block5_n2": (
        _SkelEntry("x_centered_fresh"),
        _SkelEntry("normalize_result_rescale"),
        _SkelEntry("wffn1_result_rescale"),
        _SkelEntry("gelu_coeff_mul_rescales", -1),
    ),
    # --- block 5 (n=4): skeleton=[0, 1, 3, 4, 6, 7]
    #   i=4 (ctct_gelu_x2) → gelu_power_rescales[0]（x²）
    #     注：graph 把 x³ 折进 x⁴，所以只看 x² 这一个 power rescale 进 t_new
    "block5_n4": (
        _SkelEntry("x_centered_fresh"),
        _SkelEntry("normalize_result_rescale"),
        _SkelEntry("wffn1_result_rescale"),
        _SkelEntry("gelu_power_rescales", 0),
        _SkelEntry("gelu_coeff_mul_rescales", -1),
    ),
}


def _strip_layer_suffix(config_name: str) -> Tuple[str, Optional[int]]:
    """``"block1_mrpc_L0"`` → ``("block1_mrpc", 0)``；``"block1_mrpc"`` → ``("block1_mrpc", None)``。

    BLB Stage 2 RL 的 env 端会把每层独立编号成 ``"<graph_key>_L<i>"``；invoker
    端只关心 graph_key（每个 (block, profile) 共享一份 graph + baseline）。
    """
    name = str(config_name)
    if "_L" in name:
        head, _, tail = name.rpartition("_L")
        try:
            return head, int(tail)
        except ValueError:
            return name, None
    return name, None


def _extract_sf_from_cfg(cfg: Any, entry: _SkelEntry) -> Optional[int]:
    """按 ``_SkelEntry`` 从 cfg 里抽出 NoisePoint 的 ``scaling_factor``。

    None ⇒ 字段不存在 / 值为 None / tuple 越界。
    """
    attr = getattr(cfg, entry.cfg_field, None)
    if attr is None:
        return None
    if entry.tuple_index is not None:
        try:
            attr = attr[entry.tuple_index]
        except (IndexError, TypeError, KeyError):
            return None
        if attr is None:
            return None
    sf = getattr(attr, "scaling_factor", None)
    if sf is None:
        return None
    try:
        return int(sf)
    except (TypeError, ValueError):
        return None


def cfg_to_t_new_from_table(
        graph_key: str,
        cfg: Any,
        *,
        baseline_t_new: Optional[Sequence[int]] = None,
        table: Optional[Mapping[str, Sequence[_SkelEntry]]] = None,
        ) -> Optional[List[int]]:
    """从 cfg 自动派生 ``t_new``。

    返回 None ⇒ 当前 ``graph_key`` 在表里没有映射（caller 应让 invoker fallback
    到 baseline）。

    返回 list[int] ⇒ 全部 stage 都成功取到 SF（或 baseline_t_new 提供了缺位的
    fallback）。

    规则：
      * 如果某 stage 的 cfg 字段是 None（RL 没启用该 rescale）且 baseline_t_new
        里有对应位置 ⇒ 用 baseline_t_new[r] 填位。
      * 如果某 stage 的 cfg 字段是 None 且没有 baseline ⇒ 整体放弃，返回 None
        （让 invoker 用 baseline）。
    """
    tbl = table or DEFAULT_CFG_TO_T_NEW_MAP
    entries = tbl.get(str(graph_key))
    if not entries:
        return None
    out: List[int] = []
    for r, ent in enumerate(entries):
        sf = _extract_sf_from_cfg(cfg, ent)
        if sf is None:
            if baseline_t_new is not None and r < len(baseline_t_new):
                out.append(int(baseline_t_new[r]))
            else:
                return None
        else:
            out.append(int(sf))
    return out


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
            cfg_to_t_new_overrides: Optional[Mapping[str, Sequence[_SkelEntry]]] = None,
            auto_t_new_from_cfg: bool = True,
            ):
        """构造 bridge。

        Args:
            invoker:                    ``RescaleOptimizerInvoker``（InProcess / Subprocess / Stub / Heuristic）
            cfg_to_delta_overrides:     ``{block_name: fn(cfg) -> delta_overrides_dict}``，
                                        覆盖默认 ``default_block{1..5}_cfg_to_delta``
            cfg_to_t_new_overrides:     ``{graph_key: tuple[_SkelEntry, ...]}``，扩展或覆盖
                                        ``DEFAULT_CFG_TO_T_NEW_MAP``。可用于支持 mrpc 之外的
                                        profile（wnli 等）。
            auto_t_new_from_cfg:        默认 ``True`` ⇒ 当 ``evaluate(t_new=None)`` 时
                                        自动从 cfg 派生 t_new；False ⇒ 保持旧行为
                                        （t_new=None ⇒ invoker fallback 到 baseline）。
        """
        self.invoker = invoker
        # cfg → delta_overrides 映射
        self._cfg_mappers: Dict[str, CfgToDeltaFn] = dict(_DEFAULT_BLOCK_MAPPERS)
        if cfg_to_delta_overrides:
            for k, fn in cfg_to_delta_overrides.items():
                self._cfg_mappers[str(k)] = fn
        # cfg → t_new 映射（per graph_key）
        self._cfg_to_t_new_table: Dict[str, Tuple[_SkelEntry, ...]] = {
            k: tuple(v) for k, v in DEFAULT_CFG_TO_T_NEW_MAP.items()
        }
        if cfg_to_t_new_overrides:
            for k, v in cfg_to_t_new_overrides.items():
                self._cfg_to_t_new_table[str(k)] = tuple(v)
        self.auto_t_new_from_cfg = bool(auto_t_new_from_cfg)

    def register_cfg_to_delta_overrides(self, block_name: str, fn: CfgToDeltaFn) -> None:
        """业务侧动态注册 / 覆盖某个 block 的 cfg → delta 转换函数。"""
        self._cfg_mappers[str(block_name)] = fn

    def register_cfg_to_t_new(self, graph_key: str, entries: Sequence[_SkelEntry]) -> None:
        """业务侧动态注册 / 覆盖某个 ``graph_key`` 的 skeleton→cfg 字段映射。"""
        self._cfg_to_t_new_table[str(graph_key)] = tuple(entries)

    def _lookup_baseline_t_new(self, graph_key: str) -> Optional[List[int]]:
        """如果 invoker 暴露 ``baselines`` 属性（``InProcessInvoker`` 有），就读
        baseline t_new 用于 cfg-derived t_new 的 fallback；否则返回 None。"""
        baselines = getattr(self.invoker, "baselines", None)
        if not isinstance(baselines, Mapping):
            return None
        entry = baselines.get(str(graph_key))
        if not entry:
            return None
        # entry 形如 (skeleton, t_baseline, q_bits_baseline)
        try:
            return list(entry[1]) if entry[1] is not None else None
        except (IndexError, TypeError):
            return None

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
            t_new: Optional[List[int]] = None,
            extra_overrides: Optional[Mapping[str, Union[int, str]]] = None,
            ) -> RescaleOptimizerOutput:
        """对单个 (config, block, cfg) 三元组跑一次优化器。

        Args:
            config_name: 配置名。可以是 baseline 的原始名（``"block1_mrpc"``、
                         ``"block3_exp_n4"``）或 RL 端按层加 ``_L<i>`` 后缀的形式
                         （``"block1_mrpc_L0"``）。后缀会被自动剥掉用作 graph key。
            block_name:  ``"block1"`` … ``"block5"``，决定 cfg→delta 用哪个 mapper。
            cfg:         ``Block{N}NoiseConfig`` 实例。
            t_new:       per-stage 新 SF（length = R+1，与 baseline skeleton 对齐）。
                         * **None + auto_t_new_from_cfg=True**（默认）⇒ 从 cfg 自动
                           派生 t_new（按 ``DEFAULT_CFG_TO_T_NEW_MAP``）。
                         * **None + auto_t_new_from_cfg=False** 或表里没有该 graph_key
                           ⇒ invoker 内部用 baseline ``t_baseline``。
            extra_overrides: 在默认 cfg→delta 翻译之上叠加 / 覆盖的节点 deltas。

        Returns:
            ``RescaleOptimizerOutput``（``config_name`` 字段保留 RL 端的原始值，
            含 ``_L<i>`` 后缀）。
        """
        # ---- (Bug A 修复) 标准化 config_name：剥 _L<i> 后缀作为 graph key ----
        graph_key, _layer_idx = _strip_layer_suffix(config_name)

        # ---- delta_overrides ----
        deltas = self.cfg_to_delta_overrides(block_name, cfg)
        if extra_overrides:
            deltas.update({str(k): v for k, v in extra_overrides.items()})

        # ---- (Bug B 修复) 如果调用方没传 t_new，从 cfg 自动派生 ----
        effective_t_new: Optional[List[int]] = None
        if t_new is not None:
            effective_t_new = [int(x) for x in t_new]
        elif self.auto_t_new_from_cfg:
            baseline_t = self._lookup_baseline_t_new(graph_key)
            effective_t_new = cfg_to_t_new_from_table(
                graph_key, cfg,
                baseline_t_new=baseline_t,
                table=self._cfg_to_t_new_table,
            )

        # ---- 构造 invoker payload ----
        if effective_t_new is not None:
            payload: Any = {
                "t_new": list(effective_t_new),
                "delta_overrides": deltas,
            }
        else:
            payload = deltas

        # ---- 调 invoker（用 graph_key，不带 _L<i> 后缀） ----
        raw = self.invoker(graph_key, payload)

        # 把派生出的 t_new 回写到 raw（便于上层 introspect / debug）；不破坏原结构
        try:
            if isinstance(raw, dict) and effective_t_new is not None and "_t_new_used" not in raw:
                raw["_t_new_used"] = list(effective_t_new)
                raw["_t_new_source"] = (
                    "user_provided" if t_new is not None
                    else ("cfg_derived" if graph_key in self._cfg_to_t_new_table else "baseline")
                )
                raw["_graph_key"] = graph_key
        except Exception:
            pass

        # 解析时保留原始 config_name（带 _L<i>），让上层能按层归并
        return _parse_optimizer_raw(raw, config_name=config_name)

    def evaluate_baseline(
            self,
            *,
            config_name: str,
            ) -> RescaleOptimizerOutput:
        graph_key, _layer_idx = _strip_layer_suffix(config_name)
        raw = self.invoker(graph_key, {})
        if isinstance(raw, dict):
            raw = dict(raw)
            raw.setdefault("_t_new_source", "optimizer_baseline")
            raw.setdefault("_graph_key", graph_key)
        return _parse_optimizer_raw(raw, config_name=config_name)

    def evaluate_baseline_blocks(
            self,
            requests: Mapping[str, Tuple[str, Any]],
            ) -> Dict[str, RescaleOptimizerOutput]:
        """Evaluate an all-max baseline through the cfg-derived action path.

        The optimizer's empty-payload baseline can differ from the BLB action
        decoder's all-max cfg.  Candidate ranking and reward calibration must
        not mix those conventions, so this batch helper now delegates to the
        same ``evaluate_blocks`` path used by ordinary actions.  The lower
        level ``evaluate_baseline`` method remains available for explicit
        diagnostic comparisons against the optimizer-native baseline.
        """
        return self.evaluate_blocks(requests)

    def evaluate_blocks(
            self,
            requests: Mapping[str, Tuple[str, Any]],
            *,
            t_new_per_config: Optional[Mapping[str, List[int]]] = None,
            extra_overrides: Optional[Mapping[str, Mapping[str, Union[int, str]]]] = None,
            ) -> Dict[str, RescaleOptimizerOutput]:
        """一次跑多个 config。

        Args:
            requests: ``{config_name: (block_name, cfg)}``，比如
                      ``{"block1_mrpc": ("block1", block1_cfg), ...}``
            t_new_per_config: ``{config_name: [int, ...]}``，可选；不传则对应
                              config 走 baseline t_new。
            extra_overrides: ``{config_name: {node: delta, ...}}``，可选

        Returns:
            ``{config_name: RescaleOptimizerOutput}``
        """
        outputs: Dict[str, RescaleOptimizerOutput] = {}
        for config_name, (block_name, cfg) in requests.items():
            xtra = (extra_overrides or {}).get(config_name)
            t_new = (t_new_per_config or {}).get(config_name)
            outputs[config_name] = self.evaluate(
                config_name=config_name,
                block_name=block_name,
                cfg=cfg,
                t_new=t_new,
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


# Block 2 Q/K binding: action_space._build_block2_action sets ``wq_sf == wk_sf``
# (and equivalently for ``q_mask{1,2}`` ↔ ``kt_mask{1,2}``). After the optimizer
# rewrites cfg via ``apply_optimizer_output_to_cfg``, only ``wk_encode`` /
# ``kt_mask{1,2}_encode`` get refreshed (those are the names in
# ``GRAPH_NODE_TO_CFG_ATTR[2]``); the bound Q-side fields stay at the pre-override
# value. function_handler.handler_block2 reads ``wq_encode`` / ``q_mask*_encode``
# directly when installing model noise, so without this sync the Q channel ends
# up with the *RL-decoded* SF while the K channel uses the *optimizer-snapped*
# SF — silently breaking the Q/K binding invariant.
_BLOCK2_QK_BINDING_PAIRS = (
    ("wq_encode", "wk_encode"),
    ("q_mask1_encode", "kt_mask1_encode"),
    ("q_mask2_encode", "kt_mask2_encode"),
)


def sync_block2_qk_binding(cfg: Any) -> List[CfgOverrideEntry]:
    """Mirror K-side encode SFs onto their bound Q-side counterparts.

    Call this on a Block2NoiseConfig immediately after every cfg mutation that
    might have touched ``wk_encode`` / ``kt_mask{1,2}_encode`` (typically right
    after :func:`apply_optimizer_output_to_cfg`). Returns the override entries
    actually applied so callers can include them in the same diagnostic record
    as the optimizer-driven overrides.
    """
    overrides: List[CfgOverrideEntry] = []
    for q_name, k_name in _BLOCK2_QK_BINDING_PAIRS:
        q_point = getattr(cfg, q_name, None)
        k_point = getattr(cfg, k_name, None)
        if q_point is None or k_point is None:
            continue
        new_sf = int(getattr(k_point, "scaling_factor", 0))
        old_sf = int(getattr(q_point, "scaling_factor", 0))
        if old_sf == new_sf:
            continue
        q_point.scaling_factor = new_sf
        overrides.append(CfgOverrideEntry(
            cfg_attr=f"{q_name}.scaling_factor",
            graph_node=None,
            source="qk_binding_sync",
            old_value=old_sf,
            new_value=new_sf,
        ))
    return overrides


# ---------------------------------------------------------------------------
# Optimizer return -> cfg overrides
# ---------------------------------------------------------------------------
#
# Given a Rescale_optimizer replan output for a single (block, layer) config,
# rewrite the action-decoded cfg in place so the noise actually installed in
# the model matches what the optimizer settled on:
#
#   * Source / first cut point  -> cfg.<*_fresh>.scaling_factor = sf
#   * Surviving rescale points   -> cfg.<*_result_rescale>.scaling_factor = sf_post
#   * Rescale points fused away  -> cfg.<*_result_rescale> = None
#   * CTPT_MUL propagation deltas -> cfg.<*_encode>.scaling_factor = delta
#   * Effective rotations        -> cfg.rotation_after_* flags set per rotation_name_map
#
# Cfg fields that are pure model-side (no optimizer counterpart) are left
# untouched and continue to carry the RL-chosen SF. The audit at
# `reports/blb_opt/orphan_slots/audit_<profile>.md` enumerates those.

@dataclass(frozen=True)
class CfgOverrideEntry:
    """Single change applied to one cfg attribute, for diagnostics / HTML report."""
    cfg_attr: str               # e.g. "wffn2_encode.scaling_factor" or "square_rescales[0]"
    graph_node: Optional[str]   # source graph node name, when known
    source: str                 # "fresh" | "rescale_post" | "rescale_fused_away" | "propagation_delta" | "rotation_flag"
    old_value: Any
    new_value: Any


def _get_cfg_field(cfg: Any, name: str) -> Any:
    return getattr(cfg, name, None)


def _set_noise_point_sf(cfg: Any, field_name: str, new_sf: int) -> Tuple[Any, Any]:
    """Set ``cfg.<field_name>.scaling_factor = new_sf``. Returns (old_sf, new_sf)."""
    point = getattr(cfg, field_name, None)
    if point is None:
        return (None, None)
    old = int(getattr(point, "scaling_factor", 0))
    point.scaling_factor = int(new_sf)
    return (old, int(new_sf))


def _set_noise_point_tuple_sf(
        cfg: Any, field_name: str, tuple_index: int, new_sf: Optional[int],
        ) -> Tuple[Any, Any]:
    """For a tuple-typed cfg field (e.g. ``square_rescales``) update one slot."""
    seq = getattr(cfg, field_name, None)
    if seq is None:
        return (None, None)
    items = list(seq)
    idx = int(tuple_index)
    if idx < 0:
        idx = len(items) + idx
    if not (0 <= idx < len(items)):
        return (None, None)
    old_point = items[idx]
    old_sf = int(getattr(old_point, "scaling_factor", 0)) if old_point is not None else None
    if new_sf is None:
        items[idx] = None
    else:
        if old_point is None:
            # No template NoisePoint to copy N from — skip rather than guess.
            return (old_sf, None)
        from function_handler import NoisePoint
        items[idx] = NoisePoint(
            distribution=str(old_point.distribution),
            scaling_factor=int(new_sf),
            N=int(old_point.N),
        )
    setattr(cfg, field_name, tuple(items))
    return (old_sf, new_sf)


def apply_optimizer_output_to_cfg(
        cfg: Any,
        *,
        output_raw: Mapping[str, Any],
        block_idx: int,
        graph_key: str,
        baseline_skeleton: Sequence[int],
        cfg_to_t_new_table: Optional[Mapping[str, Sequence[_SkelEntry]]] = None,
        rotation_name_map: Optional[Mapping[str, str]] = None,
        ) -> List[CfgOverrideEntry]:
    """Rewrite ``cfg`` in place to match the optimizer's replan output.

    Args:
        cfg:                 action-decoded ``Block{N}NoiseConfig`` instance.
        output_raw:          raw output dict from one invoker call (the dict
                             ReplanSession returns; equivalently
                             ``RescaleOptimizerOutput.raw``).
        block_idx:           1..5.
        graph_key:           e.g. "block1_mrpc" / "block3_exp_n4".
        baseline_skeleton:   baseline skeleton list for this graph (as found
                             in static_skeletons_<profile>.json; supplies the
                             baseline node-id sequence so we can detect fused
                             positions).
        cfg_to_t_new_table:  per-graph skel-position -> _SkelEntry mapping;
                             defaults to ``DEFAULT_CFG_TO_T_NEW_MAP``.
        rotation_name_map:   ``{src_rotation_node_name: cfg_rotation_flag_name}``
                             — typically pulled from
                             ``BLBStage2EnvConfig.rotation_name_map[(block, profile)]``.

    Returns:
        Ordered list of ``CfgOverrideEntry`` describing every change made.
        Empty list if the result was invalid or no compact config is present.
    """
    overrides: List[CfgOverrideEntry] = []

    compact = output_raw.get("new_compact_config") if isinstance(output_raw, Mapping) else None
    result = output_raw.get("result") if isinstance(output_raw, Mapping) else None
    if not compact or not isinstance(compact, Mapping):
        return overrides
    if isinstance(result, Mapping) and not bool(result.get("valid", True)):
        return overrides

    table = cfg_to_t_new_table if cfg_to_t_new_table is not None else DEFAULT_CFG_TO_T_NEW_MAP
    skel_entries = list(table.get(str(graph_key), ()))
    if not skel_entries:
        # No t_new mapping registered — fall back to just propagation_deltas + rotations.
        skel_entries = []

    # cut_point_sf entries are keyed by graph node index `i`. Build node-id -> entry.
    cut_points: Dict[int, Dict[str, Any]] = {}
    for entry in compact.get("cut_point_sf", []) or []:
        if not isinstance(entry, Mapping) or "i" not in entry:
            continue
        cut_points[int(entry["i"])] = dict(entry)

    # baseline_skeleton[r] = graph node index at baseline position r.
    # If position 0 has an `sf` entry in cut_points -> apply to source fresh field.
    # If position r>=1 has an `sf_post` entry -> apply to rescale field.
    # If position r>=1 baseline node id NOT in cut_points (fused away) -> set rescale field to None.
    for r, skel_entry in enumerate(skel_entries):
        if r >= len(baseline_skeleton):
            break
        node_id = int(baseline_skeleton[r])
        cpt = cut_points.get(node_id)
        cfg_field = skel_entry.cfg_field
        tuple_index = skel_entry.tuple_index

        if r == 0:
            if cpt and "sf" in cpt:
                if tuple_index is None:
                    old, new = _set_noise_point_sf(cfg, cfg_field, int(cpt["sf"]))
                    if new is not None and old != new:
                        overrides.append(CfgOverrideEntry(
                            cfg_attr=f"{cfg_field}.scaling_factor",
                            graph_node=str(cpt.get("name", "")),
                            source="fresh",
                            old_value=old, new_value=new,
                        ))
            continue

        if cpt is None:
            # Baseline rescale position fused away — disable cfg rescale field.
            if tuple_index is None:
                old_val = getattr(cfg, cfg_field, None)
                if old_val is not None:
                    setattr(cfg, cfg_field, None)
                    overrides.append(CfgOverrideEntry(
                        cfg_attr=cfg_field,
                        graph_node=None,
                        source="rescale_fused_away",
                        old_value=getattr(old_val, "scaling_factor", None),
                        new_value=None,
                    ))
            else:
                old, _ = _set_noise_point_tuple_sf(cfg, cfg_field, tuple_index, None)
                if old is not None:
                    overrides.append(CfgOverrideEntry(
                        cfg_attr=f"{cfg_field}[{tuple_index}]",
                        graph_node=None,
                        source="rescale_fused_away",
                        old_value=old, new_value=None,
                    ))
            continue

        if "sf_post" in cpt:
            new_sf = int(cpt["sf_post"])
            if tuple_index is None:
                old, new = _set_noise_point_sf(cfg, cfg_field, new_sf)
                if new is not None and old != new:
                    overrides.append(CfgOverrideEntry(
                        cfg_attr=f"{cfg_field}.scaling_factor",
                        graph_node=str(cpt.get("name", "")),
                        source="rescale_post",
                        old_value=old, new_value=new,
                    ))
            else:
                old, new = _set_noise_point_tuple_sf(cfg, cfg_field, tuple_index, new_sf)
                if new is not None and old != new:
                    overrides.append(CfgOverrideEntry(
                        cfg_attr=f"{cfg_field}[{tuple_index}].scaling_factor",
                        graph_node=str(cpt.get("name", "")),
                        source="rescale_post",
                        old_value=old, new_value=new,
                    ))

    # Propagation deltas: per-CTPT_MUL applied delta after replan.
    node_to_attr = GRAPH_NODE_TO_CFG_ATTR.get(int(block_idx), {})
    for entry in compact.get("propagation_deltas", []) or []:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", ""))
        delta = entry.get("delta")
        if not isinstance(delta, int):
            # "x2" or non-integer -> no cfg correspondence
            continue
        cfg_field = node_to_attr.get(name)
        if cfg_field is None:
            continue
        old, new = _set_noise_point_sf(cfg, cfg_field, int(delta))
        if new is not None and old != new:
            overrides.append(CfgOverrideEntry(
                cfg_attr=f"{cfg_field}.scaling_factor",
                graph_node=name,
                source="propagation_delta",
                old_value=old, new_value=new,
            ))

    # Effective rotations: reset all rotation_after_* flags then enable those
    # the optimizer chose. rotation_name_map translates graph rotation node
    # names to cfg flag names.
    eff_rotations = compact.get("effective_rotations", []) or []
    enabled_flags: List[str] = []
    if rotation_name_map:
        for entry in eff_rotations:
            if not isinstance(entry, Mapping):
                continue
            src = str(entry.get("name", ""))
            flag = rotation_name_map.get(src)
            if flag:
                enabled_flags.append(flag)

    # Capture state before reset so we can report which flags changed.
    pre_flags = {
        n: bool(getattr(cfg, n))
        for n in vars(cfg).keys()
        if n.startswith("rotation_after_")
    }
    apply_rotation_flags_to_cfg(cfg, enabled_flags)
    post_flags = {
        n: bool(getattr(cfg, n))
        for n in vars(cfg).keys()
        if n.startswith("rotation_after_")
    }
    for flag_name in sorted(set(pre_flags) | set(post_flags)):
        before = pre_flags.get(flag_name, False)
        after = post_flags.get(flag_name, False)
        if before != after:
            overrides.append(CfgOverrideEntry(
                cfg_attr=flag_name,
                graph_node=None,
                source="rotation_flag",
                old_value=before, new_value=after,
            ))

    return overrides


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
