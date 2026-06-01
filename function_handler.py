import math
import os as _os
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention
try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D as _GPT2Conv1D
except Exception:  # pragma: no cover - transformers always ships this, but be defensive
    _GPT2Conv1D = None
try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as _GPT2Attention
except Exception:  # pragma: no cover
    _GPT2Attention = None
import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple
from torch import Tensor


_BLB_INSTALL_LOG_ENV = "BLB_NOISE_INSTALL_LOGS"


def _print_blb_install(message: str) -> None:
    raw = str(_os.environ.get(_BLB_INSTALL_LOG_ENV, "1")).strip().lower()
    if raw in ("0", "false", "no", "off", "quiet"):
        return
    print(message)


# ---------------------------------------------------------------------------
# 独立噪声 RNG（与 torch 全局 RNG 完全隔离）
# ---------------------------------------------------------------------------
# 设计目标：用户可能在外部脚本（如 layer_importance_evaluator.py）里调
# ``torch.manual_seed(...)`` 来做"可复现的搜索"。如果我们的噪声采样走 PyTorch
# 全局 RNG（``torch.randn_like`` 默认就是），就会被外部 seed 污染——同样的
# scaling factor 配置每次都得到一模一样的噪声样本，违背"加 Gaussian 噪声"的语义。
#
# 解决：所有噪声采样统一走一个/一组独立的 ``torch.Generator``（按 device 分组），
# 进程启动时用 OS 熵 (``os.urandom``) 填充种子。外部 ``torch.manual_seed``
# 只动 ``torch.default_generator``，影响不到这里的 generator。
#
# 后门：``reseed_noise_rng(seed)`` 允许用户临时把噪声 RNG 也固定下来（仅当
# 真的需要复现某次实验时用），用 ``reseed_noise_rng(None)`` 恢复真随机模式。
# ---------------------------------------------------------------------------

_NOISE_GENERATORS: dict = {}     # str(device) -> torch.Generator
_NOISE_RNG_SEED_MODE: str = "os"  # "os" / "fixed"
_NOISE_RNG_FIXED_SEED: Optional[int] = None


def _fresh_os_seed() -> int:
    """返回 64-bit 整数，从 OS 熵源派生（每次都不一样）。"""
    return int.from_bytes(_os.urandom(8), "little")


def _get_noise_generator(device) -> torch.Generator:
    """返回一个针对 ``device`` 的独立 ``torch.Generator``。

    - 第一次访问某 device 时新建并用 OS 熵 seed（``_NOISE_RNG_SEED_MODE='os'``）
      或固定 seed（``='fixed'``）填充。
    - 后续直接复用已有 generator。
    - 与 ``torch.default_generator`` 完全隔离，不被 ``torch.manual_seed`` 影响。
    """
    key = str(device)
    g = _NOISE_GENERATORS.get(key)
    if g is None:
        g = torch.Generator(device=device)
        if _NOISE_RNG_SEED_MODE == "fixed" and _NOISE_RNG_FIXED_SEED is not None:
            g.manual_seed(int(_NOISE_RNG_FIXED_SEED))
        else:
            g.manual_seed(_fresh_os_seed())
        _NOISE_GENERATORS[key] = g
    return g


def _sample_independent_gaussian(reference: Tensor, std: float) -> Tensor:
    """从独立噪声 RNG 采样与 ``reference`` 同形状的 N(0, std²) 张量。

    生成的噪声 device/dtype 与 reference 一致；不消耗 torch 全局 RNG 状态。
    """
    if std <= 0.0:
        return torch.zeros_like(reference)
    gen = _get_noise_generator(reference.device)
    return torch.empty_like(reference).normal_(0.0, float(std), generator=gen)


# ---------------------------------------------------------------------------
# Truncation (PPTI 模拟：MPC ↔ HE 转换时的小数截断)
# ---------------------------------------------------------------------------
# 用法：在每个 BLB Block 的"最终输出"上调用 ``_apply_truncation(x, k, mode)``
# 模拟 MPC/HE 互转之前对结果保留 k 位小数（默认按二进制位，符合 CKKS scaling
# factor 语义）。数学：
#   binary：trunc(x · 2^k) / 2^k         （保留 k 位二进制小数；CKKS 默认）
#   decimal：trunc(x · 10^k) / 10^k      （保留 k 位十进制小数）
# k=None  → 不截断（用于"首层 Block 1 不存在"等需要跳过的位置）。

def _apply_truncation(
        x: Tensor,
        k: Optional[int],
        mode: str = "binary",
        ) -> Tensor:
    """Truncate ``x`` to ``k`` fractional bits (binary) or digits (decimal)。

    - ``k is None``：no-op，原样返回。
    - mode="binary"：``trunc(x · 2^k) / 2^k``（PPTI / CKKS 默认）
    - mode="decimal"：``trunc(x · 10^k) / 10^k``（普通"保留 k 位小数"）

    使用 ``torch.trunc``（朝零取整）保证正负对称。
    """
    if k is None:
        return x
    base = 2.0 if str(mode).lower() == "binary" else 10.0
    scale = base ** int(k)
    return torch.trunc(x * scale) / scale


def reseed_noise_rng(seed: Optional[int] = None) -> None:
    """手动控制噪声 RNG 的种子模式。

    - ``seed=None``（默认 / 推荐）：所有 device 的噪声 generator 用 OS 熵
      重新 seed，回到"真随机"模式；新建 device 的 generator 也走 OS 熵。
    - ``seed=<int>``：所有 device 的噪声 generator 用 ``int(seed)`` 重新 seed；
      新建 device 也走这个固定 seed —— 仅当你确需复现某一次特定实验时使用。
      复现完后请立刻 ``reseed_noise_rng(None)``。

    与外部 ``torch.manual_seed`` 仍然完全独立。
    """
    global _NOISE_RNG_SEED_MODE, _NOISE_RNG_FIXED_SEED
    if seed is None:
        _NOISE_RNG_SEED_MODE = "os"
        _NOISE_RNG_FIXED_SEED = None
        for g in _NOISE_GENERATORS.values():
            g.manual_seed(_fresh_os_seed())
    else:
        _NOISE_RNG_SEED_MODE = "fixed"
        _NOISE_RNG_FIXED_SEED = int(seed)
        for g in _NOISE_GENERATORS.values():
            g.manual_seed(int(seed))


# ---------------------------------------------------------------------------
# Helpers shared by BERT and GPT-2 code paths.
# ---------------------------------------------------------------------------
def _get_attr_path(obj, path):
    """Resolve a dotted attribute path starting from ``obj``."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_attr_path(obj, path, value):
    """Set a dotted attribute path starting from ``obj``."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _is_gpt2_conv1d(module) -> bool:
    return _GPT2Conv1D is not None and isinstance(module, _GPT2Conv1D)


# GELU approximation coeff
# GELU_COEEF[i][0]-positive; GELU_COEEF[i][1]-negative (-2.7, 2.7)
GELU_COEEF = {
            # degree 0: same coefficients as degree 1, but applied without piecewise comparison
            0: [[-0.20266642, 1.07484643], [-0.20266642, -0.57484643+0.5]],
            # todo: change the pivot point of degree 1
            # pivot point: use SEAF- -2.5, -0.75, 0 , 0.5, 2.5?
            1: [[-0.20266642, 1.07484643], [-0.20266642, -0.57484643+0.5]],
            2: [[-0.12136484, 0.94386247, 0.04261206],[-0.12136484, -0.44386247+0.5, 0.04261206]],
            # relative error: -0.75 pivot point
            # 1: [[5.9839183235390844e-05, 0.6170698026386807], [-0.2052886977665538, -0.0759420475301809]],
            # 2: [[4.680008304412681e-06, 0.4740042074483325, 0.29206518457930236],[-0.3773006655410396, -0.25069817033674346, -0.04239126463806122]],
            
            
            3: [[-0.01524885, 0.57426473, 0.35500657, -0.07415983], [-0.01524885, -0.07426473+0.5, 0.35500657, 0.07415983]],
            4: [[0.00746413, -0.07087454+0.5, 0.58960402, -0.20949432, 0.02540485], [ 0.00746413, 0.07087454+0.5, 0.58960402, 0.20949432, 0.02540485]]
            # 4: [[0.00162080853184154, -0.03798164612714154+0.5, 0.5410550166368381, -0.18352506127082727, 0.020848611754127593], [0.00162080853184154, 0.03798164612714154+0.5, 0.5410550166368381, 0.18352506127082727, 0.020848611754127593]]
}

# SiLU approximation coeff (Bumblebee piecewise)
SiLU_COEEF = {
#             1: [[0.7618737346704126, 0.5000893434026534],[-0.10118073891975127,-0.013543261873265973]],
            1: [[0.14238437510901367, 0.5000053621970405, 0.12920887677506931],[-0.10118073891975127,-0.013543261873265973]],
            2: [[0.14238437510901367, 0.5000053621970405, 0.12920887677506931],[-0.2932427892002413,-0.07801652478737445,-0.005269243960262952]],
            3: [[0.14241236482342567, 0.4999863582405589, 0.12920235286785606, 0],[-0.4233567569791515,-0.14755599495248886,-0.017365847597972207,-0.0006859293250386277]],
            4: [[0.03284668051202981,0.5000000914210826,0.19746490458050728,0,-0.005281681095454781],[-0.49057828462086733,-0.02757518199120323,0.05336178194846048,0.011409101768158705,0.0006606624719387583]]
}


# Exponential approximation Taylor bound
Exp_bound = {
    1:-2,
    2:-4,
    3:-8,
    4:-12,
    5:-13,
    6:-13
}

# Transformer-layer input noise table.
# Values are variances sigma^2 for N(0, sigma^2).
# Current x-noise injection uses the "fresh" column.
INPUT_NOISE_VARIANCE_TABLE = {
    10: {"encoding": 6.510416e-04, "fresh": 1.310800e+03, "rescale": 5.333984e+00},
    12: {"encoding": 4.069010e-05, "fresh": 8.192500e+01, "rescale": 3.333740e-01},
    14: {"encoding": 2.543131e-06, "fresh": 5.120312e+00, "rescale": 2.083588e-02},
    16: {"encoding": 1.589457e-07, "fresh": 3.200195e-01, "rescale": 1.302242e-03},
    18: {"encoding": 9.934107e-09, "fresh": 2.000122e-02, "rescale": 8.139014e-05},
    20: {"encoding": 6.208817e-10, "fresh": 1.250076e-03, "rescale": 5.086884e-06},
    22: {"encoding": 3.880511e-11, "fresh": 7.812977e-05, "rescale": 3.179302e-07},
    24: {"encoding": 2.425319e-12, "fresh": 4.883110e-06, "rescale": 1.987064e-08},
    26: {"encoding": 1.515824e-13, "fresh": 3.051944e-07, "rescale": 1.241915e-09},
    28: {"encoding": 9.473903e-15, "fresh": 1.907465e-08, "rescale": 7.761969e-11},
    30: {"encoding": 5.921189e-16, "fresh": 1.192166e-09, "rescale": 4.851231e-12},
    32: {"encoding": 3.700743e-17, "fresh": 7.451035e-11, "rescale": 3.032019e-13},
    34: {"encoding": 2.312965e-18, "fresh": 4.656897e-12, "rescale": 1.895012e-14},
    36: {"encoding": 1.445603e-19, "fresh": 2.910561e-13, "rescale": 1.184382e-15},
    38: {"encoding": 9.035018e-21, "fresh": 1.819100e-14, "rescale": 7.402390e-17},
    40: {"encoding": 5.646886e-22, "fresh": 1.136938e-15, "rescale": 4.626494e-18},
    42: {"encoding": 3.529304e-23, "fresh": 7.105861e-17, "rescale": 2.891559e-19},
    44: {"encoding": 2.205815e-24, "fresh": 4.441163e-18, "rescale": 1.807224e-20},
    46: {"encoding": 1.378634e-25, "fresh": 2.775727e-19, "rescale": 1.129515e-21},
    48: {"encoding": 8.616464e-27, "fresh": 1.734829e-20, "rescale": 7.059470e-23},
}

INPUT_NOISE_ALLOWED_SCALING_FACTORS = (22, 24, 26, 28, 30)
INPUT_NOISE_DEFAULT_SCALING_FACTOR = 30
WEIGHT_NOISE_ALLOWED_SCALING_FACTORS = (14, 16, 18, 20, 22)
WEIGHT_NOISE_DEFAULT_SCALING_FACTOR = 22
WFFN1_NOISE_ALLOWED_SCALING_FACTORS = (16, 18, 20, 22, 24)
WFFN1_NOISE_DEFAULT_SCALING_FACTOR = 24
SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS = tuple(sorted(INPUT_NOISE_VARIANCE_TABLE))
SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR = max(SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS)


def get_input_noise_variance(scaling_factor: int, distribution: str = "fresh") -> float:
    if scaling_factor not in INPUT_NOISE_VARIANCE_TABLE:
        raise ValueError(
            f"Unsupported scaling factor {scaling_factor}. "
            f"Supported values: {sorted(INPUT_NOISE_VARIANCE_TABLE)}"
        )
    distribution_key = str(distribution).lower()
    if distribution_key not in INPUT_NOISE_VARIANCE_TABLE[scaling_factor]:
        raise ValueError(
            f"Unsupported input-noise distribution '{distribution}'. "
            "Use one of: encoding, fresh, rescale."
        )
    return float(INPUT_NOISE_VARIANCE_TABLE[scaling_factor][distribution_key])


# ---------------------------------------------------------------------------
# BLB-aware multi-N noise variance table.
# Source: noise_std_table.csv —— 三列分别是 (σ_enc, σ_fresh, σ_rs) 标准差。
# 这里存的是方差 σ² = std² (用于 N(0, σ²))。
# 不同 BLB block 用不同 N (8192 / 16384) 对应的表，详见 docs/noise_targets_BLB_mapping.md。
# ---------------------------------------------------------------------------

# scale_bits → (σ_enc, σ_fresh, σ_rs)；从 noise_std_table.csv 直接抄录的标准差
_NOISE_STD_RAW = {
    8192: {
        10: (2.551552e-02, 1.337809e+00, 1.333455e+00),
        11: (1.275776e-02, 6.689046e-01, 6.667277e-01),
        12: (6.378880e-03, 3.344523e-01, 3.333638e-01),
        13: (3.189440e-03, 1.672262e-01, 1.666819e-01),
        14: (1.594720e-03, 8.361308e-02, 8.334096e-02),
        15: (7.973599e-04, 4.180654e-02, 4.167048e-02),
        16: (3.986800e-04, 2.090327e-02, 2.083524e-02),
        17: (1.993400e-04, 1.045163e-02, 1.041762e-02),
        18: (9.966999e-05, 5.225817e-03, 5.208810e-03),
        19: (4.983500e-05, 2.612909e-03, 2.604405e-03),
        20: (2.491750e-05, 1.306454e-03, 1.302203e-03),
        21: (1.245875e-05, 6.532272e-04, 6.511013e-04),
        22: (6.229375e-06, 3.266136e-04, 3.255506e-04),
        23: (3.114687e-06, 1.633068e-04, 1.627753e-04),
        24: (1.557344e-06, 8.165340e-05, 8.138766e-05),
        25: (7.786718e-07, 4.082670e-05, 4.069383e-05),
        26: (3.893359e-07, 2.041335e-05, 2.034691e-05),
        27: (1.946680e-07, 1.020667e-05, 1.017346e-05),
        28: (9.733398e-08, 5.103337e-06, 5.086729e-06),
        29: (4.866699e-08, 2.551669e-06, 2.543364e-06),
        30: (2.433349e-08, 1.275834e-06, 1.271682e-06),
        31: (1.216675e-08, 6.379172e-07, 6.358411e-07),
        32: (6.083374e-09, 3.189586e-07, 3.179205e-07),
        33: (3.041687e-09, 1.594793e-07, 1.589603e-07),
        34: (1.520843e-09, 7.973964e-08, 7.948014e-08),
        35: (7.604217e-10, 3.986982e-08, 3.974007e-08),
        36: (3.802108e-10, 1.993491e-08, 1.987003e-08),
        37: (1.901054e-10, 9.967456e-09, 9.935017e-09),
        38: (9.505271e-11, 4.983728e-09, 4.967508e-09),
        39: (4.752636e-11, 2.491864e-09, 2.483754e-09),
        40: (2.376318e-11, 1.245932e-09, 1.241877e-09),
        41: (1.188159e-11, 6.229660e-10, 6.209386e-10),
        42: (5.940795e-12, 3.114830e-10, 3.104693e-10),
        43: (2.970397e-12, 1.557415e-10, 1.552346e-10),
        44: (1.485199e-12, 7.787075e-11, 7.761732e-11),
        45: (7.425993e-13, 3.893537e-11, 3.880866e-11),
        46: (3.712997e-13, 1.946769e-11, 1.940433e-11),
    },
    16384: {
        10: (3.608439e-02, 2.675557e+00, 2.666789e+00),
        11: (1.804220e-02, 1.337779e+00, 1.333394e+00),
        12: (9.021098e-03, 6.688893e-01, 6.666972e-01),
        13: (4.510549e-03, 3.344447e-01, 3.333486e-01),
        14: (2.255274e-03, 1.672223e-01, 1.666743e-01),
        15: (1.127637e-03, 8.361116e-02, 8.333715e-02),
        16: (5.638186e-04, 4.180558e-02, 4.166857e-02),
        17: (2.819093e-04, 2.090279e-02, 2.083429e-02),
        18: (1.409547e-04, 1.045140e-02, 1.041714e-02),
        19: (7.047733e-05, 5.225698e-03, 5.208572e-03),
        20: (3.523866e-05, 2.612849e-03, 2.604286e-03),
        21: (1.761933e-05, 1.306424e-03, 1.302143e-03),
        22: (8.809666e-06, 6.532122e-04, 6.510715e-04),
        23: (4.404833e-06, 3.266061e-04, 3.255357e-04),
        24: (2.202416e-06, 1.633031e-04, 1.627679e-04),
        25: (1.101208e-06, 8.165153e-05, 8.138393e-05),
        26: (5.506041e-07, 4.082576e-05, 4.069197e-05),
        27: (2.753021e-07, 2.041288e-05, 2.034598e-05),
        28: (1.376510e-07, 1.020644e-05, 1.017299e-05),
        29: (6.882552e-08, 5.103220e-06, 5.086496e-06),
        30: (3.441276e-08, 2.551610e-06, 2.543248e-06),
        31: (1.720638e-08, 1.275805e-06, 1.271624e-06),
        32: (8.603189e-09, 6.379026e-07, 6.358120e-07),
        33: (4.301595e-09, 3.189513e-07, 3.179060e-07),
        34: (2.150797e-09, 1.594756e-07, 1.589530e-07),
        35: (1.075399e-09, 7.973782e-08, 7.947650e-08),
        36: (5.376993e-10, 3.986891e-08, 3.973825e-08),
        37: (2.688497e-10, 1.993445e-08, 1.986912e-08),
        38: (1.344248e-10, 9.967227e-09, 9.934562e-09),
        39: (6.721242e-11, 4.983614e-09, 4.967281e-09),
        40: (3.360621e-11, 2.491807e-09, 2.483641e-09),
        41: (1.680310e-11, 1.245903e-09, 1.241820e-09),
        42: (8.401552e-12, 6.229517e-10, 6.209101e-10),
        43: (4.200776e-12, 3.114759e-10, 3.104551e-10),
        44: (2.100388e-12, 1.557379e-10, 1.552275e-10),
        45: (1.050194e-12, 7.786896e-11, 7.761377e-11),
        46: (5.250970e-13, 3.893448e-11, 3.880688e-11),
    },
}

# 方差表 σ² = std²；外部直接消费这个表
NOISE_VARIANCE_TABLE_BY_N = {
    _N: {
        _sb: {
            "encoding": _stds[0] ** 2,
            "fresh":    _stds[1] ** 2,
            "rescale":  _stds[2] ** 2,
            # ``rotation`` (KS / galois automorphism 引入的噪声) 当前直接复用
            # rescale 列的方差。后续如果有更精细的 KS 噪声实测表，可以拆开。
            "rotation": _stds[2] ** 2,
        }
        for _sb, _stds in _scales.items()
    }
    for _N, _scales in _NOISE_STD_RAW.items()
}

NOISE_TABLE_ALLOWED_N = tuple(sorted(NOISE_VARIANCE_TABLE_BY_N))
NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N = {
    _N: tuple(sorted(_t)) for _N, _t in NOISE_VARIANCE_TABLE_BY_N.items()
}


def get_input_noise_variance_by_N(
        scaling_factor: int,
        distribution: str,
        N: int,
        ) -> float:
    """σ² 查表（BLB-aware 多 N 版本）。

    数据来源：``noise_std_table.csv``；表里存的是方差 σ²（= std²）。
    用法（推荐：直接用 ``add_gaussian_noise_by_N`` 或 ``_sample_independent_gaussian``，
    它们走独立的噪声 RNG，不会被外部 ``torch.manual_seed`` 污染）：
        noisy_x = add_gaussian_noise_by_N(x, scale, "fresh", N=16384)

    Args:
        scaling_factor: int in [10, 46]。
        distribution: ``"encoding"`` / ``"fresh"`` / ``"rescale"``。
        N: CKKS 多项式阶数；当前支持 8192 / 16384。

    Returns:
        float: σ² for N(0, σ²)。
    """
    if N not in NOISE_VARIANCE_TABLE_BY_N:
        raise ValueError(
            f"Unsupported N={N}. Supported: {NOISE_TABLE_ALLOWED_N}"
        )
    table = NOISE_VARIANCE_TABLE_BY_N[N]
    if scaling_factor not in table:
        raise ValueError(
            f"Unsupported scaling_factor={scaling_factor} for N={N}. "
            f"Supported: {NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N[N]}"
        )
    distribution_key = str(distribution).lower()
    if distribution_key not in table[scaling_factor]:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. "
            "Use one of: encoding, fresh, rescale."
        )
    return float(table[scaling_factor][distribution_key])


def add_gaussian_noise_by_N(
        tensor: Tensor,
        scaling_factor: int,
        distribution: str,
        N: int,
        ) -> Tensor:
    """用 BLB 多 N 表给 ``tensor`` 加 Gaussian 噪声 N(0, σ²)。"""
    variance = get_input_noise_variance_by_N(
        scaling_factor=scaling_factor,
        distribution=distribution,
        N=N,
    )
    if variance <= 0.0:
        return tensor
    std = math.sqrt(variance)
    noise = _sample_independent_gaussian(tensor, std)
    return tensor + noise


def add_gaussian_input_noise(
        hidden_states: Tensor,
        scaling_factor: int,
        distribution: str = "fresh"
        ) -> Tensor:
    variance = get_input_noise_variance(scaling_factor, distribution=distribution)
    if variance <= 0.0:
        return hidden_states
    std = math.sqrt(variance)
    noise = _sample_independent_gaussian(hidden_states, std)
    return hidden_states + noise


def _format_noise_distribution_label(distribution: str) -> str:
    distribution_key = str(distribution).lower()
    distribution_labels = {
        "fresh": "新采样（fresh）",
        "encoding": "编码分布（encoding）",
        "rescale": "重缩放（rescale）",
    }
    return distribution_labels.get(distribution_key, str(distribution))


def _format_noise_target_label(target_key: str) -> str:
    target_labels = {
        "input": "输入噪声（Input noise）",
        "query": "查询投影噪声（query noise）",
        "key": "键投影噪声（key noise）",
        "value": "值投影噪声（value noise）",
        "wo": "注意力输出投影噪声（wo noise）",
        "wffn1": "前馈网络第一层噪声（wffn1 noise）",
        "wffn2": "前馈网络第二层噪声（wffn2 noise）",
    }
    return target_labels.get(str(target_key).lower(), f"{target_key} 噪声")


def _format_noise_enable_message(
        target_key: str,
        layer_count: int,
        scaling_factor: int,
        distribution: str
        ) -> str:
    return (
        f"已为 {int(layer_count)} 层启用{_format_noise_target_label(target_key)}，"
        f"缩放因子（scaling_factor）={int(scaling_factor)}，"
        f"分布（distribution）={_format_noise_distribution_label(distribution)}"
    )


def _make_input_noise_forward(original_forward, scaling_factor: int, distribution: str = "fresh"):
    def noisy_forward(hidden_states, *args, **kwargs):
        if hidden_states is None:
            return original_forward(hidden_states, *args, **kwargs)
        noisy_hidden_states = add_gaussian_input_noise(
            hidden_states,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )
        return original_forward(noisy_hidden_states, *args, **kwargs)
    return noisy_forward


def add_gaussian_weight_noise(
        weight: Tensor,
        scaling_factor: int,
        distribution: str = "encoding"
        ) -> Tensor:
    variance = get_input_noise_variance(scaling_factor, distribution=distribution)
    if variance <= 0.0:
        return weight
    std = math.sqrt(variance)
    noise = _sample_independent_gaussian(weight, std)
    return weight + noise


def _apply_softmax_value_noise(attention_probs: Tensor, value_layer: Tensor, owner) -> tuple:
    """Apply fresh tensor noise to attention_probs and value_layer before attention matmul."""
    state = getattr(owner, "_softmax_value_noise_state", None)
    if not state:
        return attention_probs, value_layer

    distribution = str(state.get("distribution", "fresh")).lower()
    softmax_scaling_factor = state.get("softmax_scaling_factor")
    value_scaling_factor = state.get("value_scaling_factor")

    noisy_attention_probs = attention_probs
    noisy_value_layer = value_layer
    if softmax_scaling_factor is not None:
        noisy_attention_probs = add_gaussian_input_noise(
            attention_probs,
            scaling_factor=int(softmax_scaling_factor),
            distribution=distribution,
        )
    if value_scaling_factor is not None:
        noisy_value_layer = add_gaussian_input_noise(
            value_layer,
            scaling_factor=int(value_scaling_factor),
            distribution=distribution,
        )
    return noisy_attention_probs, noisy_value_layer


def _make_noisy_linear_forward(linear_module: nn.Linear, scaling_factor: int, distribution: str = "encoding"):
    def noisy_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        noisy_weight = add_gaussian_weight_noise(
            linear_module.weight,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )
        noisy_weight = noisy_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = linear_module.bias
        if bias is not None:
            bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return nn.functional.linear(hidden_states, noisy_weight, bias)
    return noisy_forward


def _make_noisy_conv1d_forward(conv1d, scaling_factor: int, distribution: str = "encoding"):
    """Weight-noise forward for HuggingFace GPT-2 ``Conv1D`` (weight shape ``[in, out]``)."""
    def noisy_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        noisy_weight = add_gaussian_weight_noise(
            conv1d.weight,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )
        noisy_weight = noisy_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = conv1d.bias
        if bias is not None:
            bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        size_out = hidden_states.size()[:-1] + (conv1d.nf,)
        out = torch.addmm(
            bias,
            hidden_states.view(-1, hidden_states.size(-1)),
            noisy_weight,
        )
        return out.view(size_out)
    return noisy_forward


def _make_noisy_projection_forward(module, scaling_factor: int, distribution: str = "encoding"):
    """Dispatch to the right noisy-forward builder depending on module type."""
    if _is_gpt2_conv1d(module):
        return _make_noisy_conv1d_forward(module, scaling_factor, distribution)
    return _make_noisy_linear_forward(module, scaling_factor, distribution)


# ============================================================================
# BLB Block 1 噪声注入 ── 范围：从前一层 GELU 输出到本层 post-FFN LayerNorm
# 中 rsqrt 之前。共 8 个噪声注入点（4 必选 + 4 可选 rescale）。
#
# 对应 noise_targets_registry.py 里的：
#   ffn.output_proj
#   ffn.layernorm.head.{mean_smul, square_ctct, var_smul}
# 以及 Gelu_out (Block 1 入口) 对应的 input-on-tensor 噪声。
#
# 默认 N=8192（按 BLB Figure 10 / 用户手绘图标注），但 N、scaling factor 都
# 走 NOISE_VARIANCE_TABLE_BY_N 查表，**不写死**，后续可动态调整。
# ============================================================================

@dataclass
class NoisePoint:
    """单个噪声注入点的参数三元组：(distribution, scaling_factor, N)。

    实际 σ² 由 ``get_input_noise_variance_by_N(scaling_factor, distribution, N)``
    查 ``NOISE_VARIANCE_TABLE_BY_N`` 得到，不写死。
    """
    distribution: str       # "encoding" / "fresh" / "rescale"
    scaling_factor: int     # 10..46，对应 NOISE_VARIANCE_TABLE_BY_N 的 key
    N: int = 8192           # 8192 / 16384


@dataclass
class Block1NoiseConfig:
    """BLB Block 1 噪声配置。

    Block 1 范围：GELU 输出 → Wffn2 → 残差 → post-FFN LayerNorm 的 mean / variance
    （直到 rsqrt 之前）。共 8 个噪声注入点：

    必选 (1 fresh + 3 encode)：
        gelu_out_fresh:    fresh   on Gelu_out (Block 1 入口张量)
        wffn2_encode:      encode  on W_ffn2 (与现有 wffn2 噪声方式一致)
        mean_inv_d_encode: encode  on 1/D (求 μ 的乘法操作数)
        var_inv_d_encode:  encode  on 1/D (求 variance 的乘法操作数)

    可选 (4 个 rescale; None = 不加该处)：
        wffn2_result_rescale:  rescale on Wffn2 乘法结果
        mean_result_rescale:   rescale on μ
        square_result_rescale: rescale on (x − μ)²
        var_result_rescale:    rescale on variance
    """
    gelu_out_fresh: NoisePoint
    wffn2_encode: NoisePoint
    mean_inv_d_encode: NoisePoint
    var_inv_d_encode: NoisePoint
    wffn2_result_rescale: Optional[NoisePoint] = None
    mean_result_rescale: Optional[NoisePoint] = None
    square_result_rescale: Optional[NoisePoint] = None
    var_result_rescale: Optional[NoisePoint] = None
    # PPTI MPC↔HE 转换的小数截断：施加在 Block 1 末尾（var, rsqrt 之前）。
    # k=None ⇒ 不截断（可用于"首层 Block 1 缺失"的语义）。
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    # Rotation (KS / galois automorphism) 噪声候选点：
    #   #1 紧跟 gelu_out fresh 之后（绑定 fresh 的 SF）
    #   #2 紧跟 W_ffn2·X 的 rescale 之后（绑定 wffn2_result_rescale 的 SF）
    #   #3 紧跟 #2 之后（也绑定 wffn2_result_rescale 的 SF，连续两次）
    #   #4 紧跟 (X−μ)² 的 rescale 之后（绑定 square_result_rescale 的 SF）
    # 取值：True ⇒ 在该位置加 rotation 噪声，SF/N 自动继承绑定源；
    #       False ⇒ 不加。绑定源为 None 时即便置 True 也不加（无 SF 可继承）。
    rotation_after_gelu_out_fresh: bool = False
    rotation_after_wffn2_rescale_a: bool = False
    rotation_after_wffn2_rescale_b: bool = False
    rotation_after_square_rescale: bool = False  # "binary" / "decimal"


def make_block1_default_config(
        *,
        N: int = 8192,
        gelu_out_sf: int = 30,
        wffn2_sf: int = 22,
        mean_inv_d_sf: int = 22,
        var_inv_d_sf: int = 22,
        wffn2_rescale_sf: Optional[int] = None,
        mean_rescale_sf: Optional[int] = None,
        square_rescale_sf: Optional[int] = None,
        var_rescale_sf: Optional[int] = None,
        output_truncation_k: Optional[int] = None,
        output_truncation_mode: str = "binary",
        rotation_after_gelu_out_fresh: bool = False,
        rotation_after_wffn2_rescale_a: bool = False,
        rotation_after_wffn2_rescale_b: bool = False,
        rotation_after_square_rescale: bool = False,
        ) -> "Block1NoiseConfig":
    """构建 Block 1 噪声配置。

    每个 ``*_sf`` 都是 ``NOISE_VARIANCE_TABLE_BY_N`` 的 key（即 scale_bits）。
    rescale_sf=None 表示**不加**这一处的 rescale 噪声。

    默认 N=8192（BLB Block 1 推荐表）；也可以传 N=16384 等动态调整。

    ``output_truncation_k``：Block 1 末尾（var, rsqrt 之前）的 PPTI 截断位数。
    None ⇒ 不截断（用于"首层 Block 1 缺失"的语义）。
    """
    cfg = Block1NoiseConfig(
        gelu_out_fresh=NoisePoint("fresh", int(gelu_out_sf), int(N)),
        wffn2_encode=NoisePoint("encoding", int(wffn2_sf), int(N)),
        mean_inv_d_encode=NoisePoint("encoding", int(mean_inv_d_sf), int(N)),
        var_inv_d_encode=NoisePoint("encoding", int(var_inv_d_sf), int(N)),
        output_truncation_k=(int(output_truncation_k) if output_truncation_k is not None else None),
        output_truncation_mode=str(output_truncation_mode),
        rotation_after_gelu_out_fresh=bool(rotation_after_gelu_out_fresh),
        rotation_after_wffn2_rescale_a=bool(rotation_after_wffn2_rescale_a),
        rotation_after_wffn2_rescale_b=bool(rotation_after_wffn2_rescale_b),
        rotation_after_square_rescale=bool(rotation_after_square_rescale),
    )
    if wffn2_rescale_sf is not None:
        cfg.wffn2_result_rescale = NoisePoint("rescale", int(wffn2_rescale_sf), int(N))
    if mean_rescale_sf is not None:
        cfg.mean_result_rescale = NoisePoint("rescale", int(mean_rescale_sf), int(N))
    if square_rescale_sf is not None:
        cfg.square_result_rescale = NoisePoint("rescale", int(square_rescale_sf), int(N))
    if var_rescale_sf is not None:
        cfg.var_result_rescale = NoisePoint("rescale", int(var_rescale_sf), int(N))
    return cfg


# ---------------------------------------------------------------------------
# Rotation (KS / galois automorphism) 噪声辅助
# ---------------------------------------------------------------------------
# rotation 噪声的 scaling factor 不由它本身决定，而是绑定到它前面紧接着的
# fresh / rescale 噪声的 SF。``_make_rotation_point`` 把一个 fresh/rescale
# 的 NoisePoint 转成同 SF / 同 N、distribution="rotation" 的 NoisePoint。
# 实际查表落在 ``NOISE_VARIANCE_TABLE_BY_N[N][SF]["rotation"]`` 那一列上。

def _make_rotation_point(source: Optional[NoisePoint]) -> Optional[NoisePoint]:
    """把绑定的 fresh/rescale NoisePoint 转成 rotation NoisePoint。

    - source=None → None（前置 rescale 没启用，rotation 也无 SF 可继承）
    - 否则返回 NoisePoint("rotation", source.scaling_factor, source.N)
    """
    if source is None:
        return None
    return NoisePoint("rotation", int(source.scaling_factor), int(source.N))


def _sample_gaussian_for_point(reference: Tensor, point: Optional[NoisePoint]) -> Tensor:
    """根据 NoisePoint 的 (distribution, scaling_factor, N) 三元组，
    返回与 ``reference`` 同形状（同 device/dtype）的 N(0, σ²) 噪声张量。

    - ``point=None``：返回 0（用于 rescale 关闭时的统一处理）。
    - 走的是 ``NOISE_VARIANCE_TABLE_BY_N`` 多 N 表；σ² 严禁写死。
    """
    if point is None:
        return torch.zeros_like(reference)
    variance = get_input_noise_variance_by_N(
        scaling_factor=int(point.scaling_factor),
        distribution=str(point.distribution).lower(),
        N=int(point.N),
    )
    if variance <= 0.0:
        return torch.zeros_like(reference)
    std = math.sqrt(variance)
    return _sample_independent_gaussian(reference, std)


# ============================================================================
# BLB Block 2 噪声注入 ── 范围：post-FFN LN tail (rsqrt → normalize → γ scale)
#                          + Wq / Wk 投影 + Q·K^T 之前的 BSGS mask
#
# 与 Block 1 相同的设计原则：
#   * 所有 σ² 走 NOISE_VARIANCE_TABLE_BY_N[N][scale_bits][dist] 查表，**不写死**。
#   * 默认 N=16384（按 BLB Figure 10），可由 cfg 动态调整。
#   * encode / fresh 必加；rescale 全部可选 (None = 该处不加 rescale)。
#
# 与 Block 1 / 现有 input-X 噪声的关系：
#   * 旧的 ``replace_layer_input_noise`` 给 attention 入口 X 加 fresh 噪声 ──
#     在 BLB Block 2 视角下不需要（X 的 PPTI 噪声来源于 LN tail γ 乘法的 rescale）。
#     legacy 代码保留（stage2 RL 还要用），但 Block 2 install 不会主动激活它。
#   * Block 1 head 与 Block 2 tail 共用同一个 ``layer.output.LayerNorm``，
#     因此 ``NoisyBlock1LayerNorm`` 现在接受 ``block2_cfg`` 字段。
# ============================================================================

@dataclass
class Block2NoiseConfig:
    """BLB Block 2 噪声配置。

    Block 2 范围：post-FFN LN tail (rsqrt 之后 → γ 标量乘法 → +β)
                  + 同层 attention.self.{query, key} 的投影
                  + K^T / Q 在 Q·K^T 之前的两步 BSGS-style mask 乘法。

    必选 (9 encode + 2 fresh)：
        inv_std_fresh:           fresh   on 1/std (Block 1→2 边界 ct)
        x_centered_fresh:        fresh   on (X − μ) (Block 1→2 边界 ct)
        gamma_encode:            encode  on γ (broadcast 到 [B, S, H] 后每 slot 独立)
        wk_encode:               encode  on W_k (与 wffn2_encode 同方式)
        kt_mask1_encode:         encode  on K^T BSGS 第 1 步 ones-mask
        kt_mask2_encode:         encode  on K^T BSGS 第 2 步 ones-mask
        wq_encode:               encode  on W_q
        q_mask1_encode:          encode  on Q BSGS 第 1 步 ones-mask
        q_mask2_encode:          encode  on Q BSGS 第 2 步 ones-mask
        wv_encode:               encode  on W_v
        qkt_merge_mask_encode:   encode  on Q·K^T 之后合并步骤的 ones-mask

    可选 (11 个 rescale；None = 不加该处)：
        normalize_result_rescale:        rescale on (1/std)·(X−μ) 乘法结果
        gamma_result_rescale:            rescale on γ·normalize 乘法结果（β 之前）
        wk_result_rescale:               rescale on X·W_k 乘法结果 (= K)
        kt_mask1_result_rescale:         rescale on K^T·mask1 结果
        kt_mask2_result_rescale:         rescale on (K^T·mask1)·mask2 结果
        wq_result_rescale:               rescale on X·W_q 乘法结果 (= Q)
        q_mask1_result_rescale:          rescale on Q·mask1 结果
        q_mask2_result_rescale:          rescale on (Q·mask1)·mask2 结果
        wv_result_rescale:               rescale on X·W_v 乘法结果 (= V)
        qkt_matmul_result_rescale:       rescale on Q·K^T matmul 结果
        qkt_merge_mask_result_rescale:   rescale on Q·K^T·ones 合并 mask 乘法结果

    BLB 共享约束：q_proj / k_proj 共享 scaling factor（动作选择必须一致）。
    本 cfg 把它们做成独立字段以保证可观察性，调用方需自行保证 wq_encode == wk_encode。
    """
    # ---- LN tail: rsqrt → normalize ----
    inv_std_fresh: NoisePoint
    x_centered_fresh: NoisePoint
    # ---- LN tail: γ scale ----
    gamma_encode: NoisePoint
    # ---- Wk branch ----
    wk_encode: NoisePoint
    kt_mask1_encode: NoisePoint
    kt_mask2_encode: NoisePoint
    # ---- Wq branch ----
    wq_encode: NoisePoint
    q_mask1_encode: NoisePoint
    q_mask2_encode: NoisePoint
    # ---- Wv branch ----
    wv_encode: NoisePoint
    # ---- 合并 Q,K 过程 (Q·K^T 之后的 ones-mask) ----
    qkt_merge_mask_encode: NoisePoint

    # ---- Optional rescale ----
    normalize_result_rescale: Optional[NoisePoint] = None
    gamma_result_rescale: Optional[NoisePoint] = None
    wk_result_rescale: Optional[NoisePoint] = None
    kt_mask1_result_rescale: Optional[NoisePoint] = None
    kt_mask2_result_rescale: Optional[NoisePoint] = None
    wq_result_rescale: Optional[NoisePoint] = None
    q_mask1_result_rescale: Optional[NoisePoint] = None
    q_mask2_result_rescale: Optional[NoisePoint] = None
    wv_result_rescale: Optional[NoisePoint] = None
    qkt_matmul_result_rescale: Optional[NoisePoint] = None
    qkt_merge_mask_result_rescale: Optional[NoisePoint] = None
    # PPTI MPC↔HE 截断：Block 2 末尾（合并 Q,K mask 之后的 attention_scores）。
    # 即便首层 Block 2 前半部分缺失，本截断仍照常应用（Q·K^T 输出存在）。
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    # Rotation 候选点（共 5 个位置 / 9 个 sub-slot）：
    #   #1 γ·((X−μ)/std) rescale 之后                         （绑定 gamma_result_rescale）
    #   #2 Wq/Wk/Wv·X rescale 之后（3 个独立分支）             （绑定各自的 *_result_rescale）
    #   #3 第 1 个 mask·Q/K^T rescale 之后（仅 Q/K 分支，2 个）（绑定 q/kt_mask1_result_rescale）
    #   #4 第 2 个 mask·Q/K^T rescale 之后（仅 Q/K 分支，2 个）（绑定 q/kt_mask2_result_rescale）
    #   #5 Q·K^T matmul rescale 之后                          （绑定 qkt_matmul_result_rescale）
    rotation_after_gamma_rescale: bool = False
    rotation_after_wq_rescale: bool = False
    rotation_after_wk_rescale: bool = False
    rotation_after_wv_rescale: bool = False
    rotation_after_q_mask1_rescale: bool = False
    rotation_after_kt_mask1_rescale: bool = False
    rotation_after_q_mask2_rescale: bool = False
    rotation_after_kt_mask2_rescale: bool = False
    rotation_after_qkt_matmul_rescale: bool = False


def make_block2_default_config(
        *,
        N: int = 16384,
        inv_std_fresh_sf: int = 30,
        x_centered_fresh_sf: int = 30,
        gamma_sf: int = 22,
        wk_sf: int = 22,
        kt_mask1_sf: int = 22,
        kt_mask2_sf: int = 22,
        wq_sf: int = 22,
        q_mask1_sf: int = 22,
        q_mask2_sf: int = 22,
        wv_sf: int = 22,
        qkt_merge_mask_sf: int = 22,
        # rescale 全部可选；传 None = 不加
        normalize_rescale_sf: Optional[int] = None,
        gamma_rescale_sf: Optional[int] = None,
        wk_rescale_sf: Optional[int] = None,
        kt_mask1_rescale_sf: Optional[int] = None,
        kt_mask2_rescale_sf: Optional[int] = None,
        wq_rescale_sf: Optional[int] = None,
        q_mask1_rescale_sf: Optional[int] = None,
        q_mask2_rescale_sf: Optional[int] = None,
        wv_rescale_sf: Optional[int] = None,
        qkt_matmul_rescale_sf: Optional[int] = None,
        qkt_merge_mask_rescale_sf: Optional[int] = None,
        output_truncation_k: Optional[int] = None,
        output_truncation_mode: str = "binary",
        rotation_after_gamma_rescale: bool = False,
        rotation_after_wq_rescale: bool = False,
        rotation_after_wk_rescale: bool = False,
        rotation_after_wv_rescale: bool = False,
        rotation_after_q_mask1_rescale: bool = False,
        rotation_after_kt_mask1_rescale: bool = False,
        rotation_after_q_mask2_rescale: bool = False,
        rotation_after_kt_mask2_rescale: bool = False,
        rotation_after_qkt_matmul_rescale: bool = False,
        ) -> "Block2NoiseConfig":
    """构建 Block 2 噪声配置。

    每个 ``*_sf`` 都是 ``NOISE_VARIANCE_TABLE_BY_N`` 的 key（即 scale_bits）；
    rescale_sf=None 表示**不加**这一处的 rescale 噪声。

    默认 N=16384（BLB Block 2 推荐表）；也可以传 N=8192 等动态调整。

    BLB 约束（用户应自行保证）：q_proj / k_proj 共享 scaling factor，
    建议 ``wq_sf == wk_sf``。
    """
    cfg = Block2NoiseConfig(
        inv_std_fresh=NoisePoint("fresh", int(inv_std_fresh_sf), int(N)),
        x_centered_fresh=NoisePoint("fresh", int(x_centered_fresh_sf), int(N)),
        gamma_encode=NoisePoint("encoding", int(gamma_sf), int(N)),
        wk_encode=NoisePoint("encoding", int(wk_sf), int(N)),
        kt_mask1_encode=NoisePoint("encoding", int(kt_mask1_sf), int(N)),
        kt_mask2_encode=NoisePoint("encoding", int(kt_mask2_sf), int(N)),
        wq_encode=NoisePoint("encoding", int(wq_sf), int(N)),
        q_mask1_encode=NoisePoint("encoding", int(q_mask1_sf), int(N)),
        q_mask2_encode=NoisePoint("encoding", int(q_mask2_sf), int(N)),
        wv_encode=NoisePoint("encoding", int(wv_sf), int(N)),
        qkt_merge_mask_encode=NoisePoint("encoding", int(qkt_merge_mask_sf), int(N)),
        output_truncation_k=(int(output_truncation_k) if output_truncation_k is not None else None),
        output_truncation_mode=str(output_truncation_mode),
        rotation_after_gamma_rescale=bool(rotation_after_gamma_rescale),
        rotation_after_wq_rescale=bool(rotation_after_wq_rescale),
        rotation_after_wk_rescale=bool(rotation_after_wk_rescale),
        rotation_after_wv_rescale=bool(rotation_after_wv_rescale),
        rotation_after_q_mask1_rescale=bool(rotation_after_q_mask1_rescale),
        rotation_after_kt_mask1_rescale=bool(rotation_after_kt_mask1_rescale),
        rotation_after_q_mask2_rescale=bool(rotation_after_q_mask2_rescale),
        rotation_after_kt_mask2_rescale=bool(rotation_after_kt_mask2_rescale),
        rotation_after_qkt_matmul_rescale=bool(rotation_after_qkt_matmul_rescale),
    )
    if normalize_rescale_sf is not None:
        cfg.normalize_result_rescale = NoisePoint("rescale", int(normalize_rescale_sf), int(N))
    if gamma_rescale_sf is not None:
        cfg.gamma_result_rescale = NoisePoint("rescale", int(gamma_rescale_sf), int(N))
    if wk_rescale_sf is not None:
        cfg.wk_result_rescale = NoisePoint("rescale", int(wk_rescale_sf), int(N))
    if kt_mask1_rescale_sf is not None:
        cfg.kt_mask1_result_rescale = NoisePoint("rescale", int(kt_mask1_rescale_sf), int(N))
    if kt_mask2_rescale_sf is not None:
        cfg.kt_mask2_result_rescale = NoisePoint("rescale", int(kt_mask2_rescale_sf), int(N))
    if wq_rescale_sf is not None:
        cfg.wq_result_rescale = NoisePoint("rescale", int(wq_rescale_sf), int(N))
    if q_mask1_rescale_sf is not None:
        cfg.q_mask1_result_rescale = NoisePoint("rescale", int(q_mask1_rescale_sf), int(N))
    if q_mask2_rescale_sf is not None:
        cfg.q_mask2_result_rescale = NoisePoint("rescale", int(q_mask2_rescale_sf), int(N))
    if wv_rescale_sf is not None:
        cfg.wv_result_rescale = NoisePoint("rescale", int(wv_rescale_sf), int(N))
    if qkt_matmul_rescale_sf is not None:
        cfg.qkt_matmul_result_rescale = NoisePoint("rescale", int(qkt_matmul_rescale_sf), int(N))
    if qkt_merge_mask_rescale_sf is not None:
        cfg.qkt_merge_mask_result_rescale = NoisePoint("rescale", int(qkt_merge_mask_rescale_sf), int(N))
    return cfg


def _make_block1_ffn2_forward(linear_module: nn.Linear, cfg: Block1NoiseConfig):
    """包装 ``layer.output.dense.forward`` (Wffn2 投影) 注入 Block 1 前段噪声。

    顺序：
      1. fresh   on Gelu_out (input)              ── 必加
      2. encode  on W_ffn2 (operand-side)         ── 必加，方式同现有 W 噪声
      3. linear: x · noisy_W + b
      4. rescale on output (optional)             ── cfg.wffn2_result_rescale 决定是否加
    """
    def block1_ffn2_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        # 1. fresh on Gelu_out
        x = hidden_states + _sample_gaussian_for_point(hidden_states, cfg.gelu_out_fresh)
        # 1b. rotation #1：紧跟 gelu_out fresh 之后；SF 继承自 gelu_out_fresh
        if cfg.rotation_after_gelu_out_fresh:
            x = x + _sample_gaussian_for_point(x, _make_rotation_point(cfg.gelu_out_fresh))
        # 2. encode on W_ffn2
        weight = linear_module.weight
        noisy_weight = weight + _sample_gaussian_for_point(weight, cfg.wffn2_encode)
        noisy_weight = noisy_weight.to(device=x.device, dtype=x.dtype)
        bias = linear_module.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        # 3. linear
        out = nn.functional.linear(x, noisy_weight, bias)
        # 4. rescale on result (optional)
        if cfg.wffn2_result_rescale is not None:
            out = out + _sample_gaussian_for_point(out, cfg.wffn2_result_rescale)
            # 4b. rotation #2：紧跟 W_ffn2 rescale 之后；SF 继承自 wffn2_result_rescale
            if cfg.rotation_after_wffn2_rescale_a:
                out = out + _sample_gaussian_for_point(out, _make_rotation_point(cfg.wffn2_result_rescale))
            # 4c. rotation #3：紧跟 #2 之后；SF 同样来自 wffn2_result_rescale
            if cfg.rotation_after_wffn2_rescale_b:
                out = out + _sample_gaussian_for_point(out, _make_rotation_point(cfg.wffn2_result_rescale))
        return out
    return block1_ffn2_forward


class NoisyBlock1LayerNorm(nn.Module):
    """LayerNorm 替身：把 LN 拆解开，按 BLB Block 1（head: mean/square/var）
    与 BLB Block 2（tail: rsqrt 之后 normalize + γ scale）分别加噪声。

    拆解的算子序列（与 ``noise_targets_registry`` 的 ``ffn.layernorm.head.*``
    及 ``ffn.layernorm.tail.*`` 对应）：

        sum_x = Σ_d x                                # reduction (no mul)
        μ   = sum_x · (1/D)                          # Block 1 ─ encode on 1/D；rescale on μ (opt)
        x_c = x − μ                                  # subtraction (no mul)
        sq  = x_c · x_c                              # Block 1 ─ rescale on sq (opt)
        sum_sq = Σ_d sq                              # reduction
        var = sum_sq · (1/D)                         # Block 1 ─ encode on 1/D；rescale on var (opt)
        --- Block 1 / Block 2 边界 ---
        inv_std    = 1 / sqrt(var + ε)               # rsqrt 非线性 (无噪)
        normalized = x_c · inv_std                   # Block 2 ─ fresh on 1/std；fresh on x_c；rescale on result (opt)
        γ_full     = γ.broadcast_to([B,S,H]) + ε_enc # Block 2 ─ encode on γ (per-slot)
        γ_mul      = normalized · γ_full             # Block 2 ─ rescale on γ_mul (opt)
        out        = γ_mul + β                       # +β 非乘法

    ``cfg``  (= ``cfg1``)：Block 1 配置；可为 None（仅装 Block 2 时用）。
    ``cfg2``：Block 2 配置；可为 None（仅装 Block 1 时用）。
    """

    def __init__(
            self,
            original_ln: nn.LayerNorm,
            cfg: Optional[Block1NoiseConfig] = None,
            cfg2: Optional["Block2NoiseConfig"] = None,
            ):
        super().__init__()
        # 直接复用原 LN 的 Parameter，保持训练状态、device、dtype
        self.weight = original_ln.weight
        self.bias = original_ln.bias
        self.eps = float(original_ln.eps)
        self.normalized_shape = tuple(original_ln.normalized_shape)
        self.cfg = cfg          # Block 1 cfg (head)
        self.cfg2 = cfg2        # Block 2 cfg (tail)

    def set_block2_cfg(self, cfg2: Optional["Block2NoiseConfig"]) -> None:
        """安装 / 覆盖 / 关闭（None）Block 2 LN-tail 噪声。"""
        self.cfg2 = cfg2

    def forward(self, x: Tensor) -> Tensor:
        D = int(x.shape[-1])
        cfg = self.cfg
        cfg2 = self.cfg2

        # ===================== Block 1: head =====================
        # ===== mean = sum_x · (1/D) =====
        sum_x = x.sum(dim=-1, keepdim=True)                            # [B, S, 1]
        if cfg is not None:
            # encode on 1/D：模拟 CKKS 真实密文情况——
            # plaintext 1/D 被显式广播到与操作数矩阵 x 同形 [B, S, H]，
            # 然后每个 slot 加独立 encode 噪声 ε_{b,s,h}（不再是同一个标量 ε）。
            # 之后与 sum_x ([B, S, 1]) 按位乘（自动 broadcast），得到 [B, S, H] 的 noisy μ。
            inv_d_broadcast = torch.full_like(x, 1.0 / D)              # [B, S, H]
            noisy_inv_d = inv_d_broadcast + _sample_gaussian_for_point(inv_d_broadcast, cfg.mean_inv_d_encode)
            mean = sum_x * noisy_inv_d                                 # [B, S, H]，每 slot 独立噪声
            if cfg.mean_result_rescale is not None:
                mean = mean + _sample_gaussian_for_point(mean, cfg.mean_result_rescale)
        else:
            # Block 1 未启用：clean LN head
            mean = sum_x / float(D)

        # ===== (x − μ) =====
        x_centered = x - mean                                          # [B, S, H] or [B, S, 1] mean

        # ===== squaring =====
        sq = x_centered * x_centered                                   # ct*ct: (x − μ)²，[B, S, H]
        if cfg is not None and cfg.square_result_rescale is not None:
            sq = sq + _sample_gaussian_for_point(sq, cfg.square_result_rescale)
            # Block 1 rotation #4：紧跟 (X−μ)² rescale 之后；SF 继承自 square_result_rescale
            if cfg.rotation_after_square_rescale:
                sq = sq + _sample_gaussian_for_point(sq, _make_rotation_point(cfg.square_result_rescale))

        # ===== variance = sum_sq · (1/D) =====
        sum_sq = sq.sum(dim=-1, keepdim=True)                          # [B, S, 1]
        if cfg is not None:
            inv_d_var_broadcast = torch.full_like(sq, 1.0 / D)         # [B, S, H]
            noisy_inv_d_var = inv_d_var_broadcast + _sample_gaussian_for_point(inv_d_var_broadcast, cfg.var_inv_d_encode)
            var = sum_sq * noisy_inv_d_var                             # [B, S, H]，每 slot 独立噪声
            if cfg.var_result_rescale is not None:
                var = var + _sample_gaussian_for_point(var, cfg.var_result_rescale)
        else:
            var = sum_sq / float(D)

        # Block 1 末尾：PPTI MPC↔HE 截断（var = Block 1 输出，rsqrt 之前）
        if cfg is not None and cfg.output_truncation_k is not None:
            var = _apply_truncation(var, cfg.output_truncation_k, cfg.output_truncation_mode)

        # ===== Block 1 / Block 2 边界：rsqrt 非线性，无噪 =====
        # 若 Block 1 已启用 → var 是 [B, S, H]，inv_std 也 [B, S, H]
        # 若仅 Block 2 启用 → var 是 [B, S, 1]，inv_std 也 [B, S, 1]
        inv_std = torch.rsqrt(var + self.eps)

        # ===================== Block 2: tail =====================
        if cfg2 is not None:
            # ----- (1) 1/std 与 X-μ 的 ct·ct 乘法（normalize 步） -----
            # 模拟 CKKS：两个操作数都是 ciphertext，分别加 fresh 噪声后做 ewmulcc。
            #   * x_centered 永远是 [B, S, H]（x 形状）。
            #   * inv_std 在 Block 1 启用时为 [B, S, H]；Block 1 关闭时为 [B, S, 1]，
            #     需要先 expand 到 [B, S, H] 才能保证每 slot 独立 fresh 噪声。
            if inv_std.shape != x.shape:
                inv_std = inv_std.expand_as(x).contiguous()
            noisy_inv_std = inv_std + _sample_gaussian_for_point(inv_std, cfg2.inv_std_fresh)
            noisy_x_centered = x_centered + _sample_gaussian_for_point(x_centered, cfg2.x_centered_fresh)
            normalized = noisy_x_centered * noisy_inv_std
            if cfg2.normalize_result_rescale is not None:
                normalized = normalized + _sample_gaussian_for_point(normalized, cfg2.normalize_result_rescale)

            # ----- (2) γ 标量乘法（CKKS smulcp，按 1/D 一样的 broadcast 加噪方式） -----
            # γ 形状 [H] → broadcast 到 [B, S, H]，每 slot 独立 encode 噪声后做 ewmulcp。
            gamma_broadcast = self.weight.expand_as(normalized)        # [B, S, H]，view-only
            noisy_gamma = gamma_broadcast + _sample_gaussian_for_point(gamma_broadcast, cfg2.gamma_encode)
            gamma_mul = normalized * noisy_gamma
            if cfg2.gamma_result_rescale is not None:
                gamma_mul = gamma_mul + _sample_gaussian_for_point(gamma_mul, cfg2.gamma_result_rescale)
                # Block 2 rotation #1：紧跟 γ rescale 之后；SF 继承自 gamma_result_rescale
                if cfg2.rotation_after_gamma_rescale:
                    gamma_mul = gamma_mul + _sample_gaussian_for_point(gamma_mul, _make_rotation_point(cfg2.gamma_result_rescale))
            # +β 是 ctpt 加法，非乘法 → 不加噪
            out = gamma_mul + self.bias
        else:
            # Block 2 未启用：clean LN tail
            normalized = x_centered * inv_std
            out = normalized * self.weight + self.bias
        return out


def _make_block2_qk_proj_forward(
        linear_module: nn.Linear,
        encode_point: NoisePoint,
        rescale_point: Optional[NoisePoint],
        rotation_after_rescale: bool = False,
        ):
    """Wq / Wk / Wv 投影包装：encode on W (matmulcp 操作数侧) + 可选 rescale on result
    + 可选 rotation 噪声（紧跟 rescale 之后；SF 继承自 rescale_point）。

    与 Block 1 的 ``_make_block1_ffn2_forward`` 同方式（与现有 ``replace_layer_*_noise``
    通过 ``_make_noisy_linear_forward`` 加 W 噪声的 PPTI 语义一致），但额外支持
    在 ``X · W`` 之后加 rescale 噪声（cfg.*_result_rescale 控制是否加）。

    注意：这里**不**对输入 X 加 fresh 噪声 —— 按用户 Block 2 设定，X 的 fresh
    噪声在 LN tail γ 乘法的 rescale 那里就够了，旧的 ``replace_layer_input_noise``
    的 fresh-on-X 不再纳入 Block 2 框架（legacy 保留供 stage2 RL 使用）。
    """
    def block2_qk_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        weight = linear_module.weight
        noisy_weight = weight + _sample_gaussian_for_point(weight, encode_point)
        noisy_weight = noisy_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = linear_module.bias
        if bias is not None:
            bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        out = nn.functional.linear(hidden_states, noisy_weight, bias)
        if rescale_point is not None:
            out = out + _sample_gaussian_for_point(out, rescale_point)
            # rotation：紧跟 rescale 之后；SF 继承自 rescale_point
            if rotation_after_rescale:
                out = out + _sample_gaussian_for_point(out, _make_rotation_point(rescale_point))
        return out
    return block2_qk_forward


def _make_block2_qkt_merge_hook(
        qkt_matmul_rescale: Optional[NoisePoint],
        merge_mask_encode: NoisePoint,
        merge_mask_rescale: Optional[NoisePoint],
        output_truncation_k: Optional[int] = None,
        output_truncation_mode: str = "binary",
        rotation_after_qkt_matmul_rescale: bool = False,
        ):
    """构造 Q·K^T matmul **之后**、softmax **之前**的 "合并 Q,K" 噪声 hook。

    顺序：
        1. rescale on Q·K^T matmul 结果        (qkt_matmul_rescale, 可选)
        1b. rotation 紧跟 #1 之后                (rotation_after_qkt_matmul_rescale, 可选；
                                                  SF 继承自 qkt_matmul_rescale)
        2. ⊙ ones-mask: noisy_ones = 1 + ε_enc; out = qkt_result · noisy_ones
        3. rescale on mask 乘法结果              (merge_mask_rescale, 可选)
        4. PPTI MPC↔HE 截断 (output_truncation_k, 可选)：Block 2 输出末尾

    返回 ``hook(attention_scores) -> attention_scores`` 形状 [B, A, S, S]。
    """
    def hook(qkt_result: Tensor) -> Tensor:
        # 1. rescale on Q·K^T matmul 结果（可选）
        if qkt_matmul_rescale is not None:
            qkt_result = qkt_result + _sample_gaussian_for_point(qkt_result, qkt_matmul_rescale)
            # Block 2 rotation #5：紧跟 Q·K^T matmul rescale 之后
            if rotation_after_qkt_matmul_rescale:
                qkt_result = qkt_result + _sample_gaussian_for_point(qkt_result, _make_rotation_point(qkt_matmul_rescale))
        # 2. ⊙ ones-mask（CKKS ewmulcp）
        ones = torch.ones_like(qkt_result)
        noisy_mask = ones + _sample_gaussian_for_point(ones, merge_mask_encode)
        out = qkt_result * noisy_mask
        # 3. rescale on mask 乘法结果（可选）
        if merge_mask_rescale is not None:
            out = out + _sample_gaussian_for_point(out, merge_mask_rescale)
        # 4. Block 2 末尾 truncation
        if output_truncation_k is not None:
            out = _apply_truncation(out, output_truncation_k, output_truncation_mode)
        return out
    return hook


# ============================================================================
# BLB Block 3 噪声注入 ── 范围：softmax exp 多项式近似 (1 + x/2^n)^(2^n)
#
# 与 Block 1/2 相同的设计原则：
#   * 所有 σ² 走 NOISE_VARIANCE_TABLE_BY_N 查表，**不写死**。
#   * encode / fresh 必加；rescale 全部可选。
#   * N 默认按 degree 自动选 ── degree=2 → N=8192；degree∈{1,3,4,5,6} → N=16384。
#     可由 cfg 显式覆盖（保持动态可调整）。
#
# Block 3 噪声点（degree=n 时共 1 fresh + 1 encode + (n+1) rescale）：
#   1) fresh   on x_softmax （softmax 的输入 x，方式同 input X）
#   2) encode  on 1/2^n     （broadcast 到 [B,A,S,S] 每 slot 独立，方式同 1/D / γ）
#   3) rescale on x · (1/2^n) 乘法结果         （可选）
#   4) iterative squaring n 次：每次 y = y² 之后加一个 rescale（可选）
#
# 注：1 + x·(1/2^n) 的 +1 是 ctpt 加法，不计乘法，无噪声。
#     Block 3 末尾的 mask（x < lower_bound → 0）和 norm_div（rec + smulcc）
#     都是非线性 / MPC 路径，不在 Block 3 的多项式噪声范围内。
# ============================================================================

@dataclass
class Block3NoiseConfig:
    """BLB Block 3 噪声配置：softmax exp 多项式近似 (1 + x/2^n)^(2^n)"""
    degree: int                         # softmax 近似度 n ∈ {1, 2, 3, 4, 5, 6}
    x_fresh: NoisePoint                 # 必加：fresh on softmax 输入 x
    inv_2n_encode: NoisePoint           # 必加：encode on 1/2^n broadcast
    # (n+1) 个可选 rescale；None = 不加该处
    x_inv_2n_result_rescale: Optional[NoisePoint] = None  # rescale on x · (1/2^n)
    # 长度必须 == degree；每个元素 None 表示该次平方不加 rescale
    square_rescales: Tuple[Optional[NoisePoint], ...] = field(default_factory=tuple)
    # PPTI MPC↔HE 截断：Block 3 末尾（最后一次 squaring 之后，softmax mask/norm_div 之前）
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"


def make_block3_default_config(
        *,
        degree: int,
        N: Optional[int] = None,
        x_fresh_sf: int = 30,
        inv_2n_sf: int = 22,
        x_inv_2n_rescale_sf: Optional[int] = None,
        square_rescale_sfs: Sequence[Optional[int]] = (),
        output_truncation_k: Optional[int] = None,
        output_truncation_mode: str = "binary",
        ) -> "Block3NoiseConfig":
    """构建 Block 3 噪声配置。

    Args:
        degree: softmax 近似度 n ∈ {1, 2, 3, 4, 5, 6}。决定迭代平方次数。
        N: CKKS 多项式阶。None = 按 degree 自动选（degree==2 → 8192，否则 16384）。
        x_fresh_sf:        scale_bits for fresh on softmax 输入 x
        inv_2n_sf:         scale_bits for encode on 1/2^n
        x_inv_2n_rescale_sf: scale_bits for rescale on x·(1/2^n)；None=不加
        square_rescale_sfs:  长度 == degree 的序列；每元素 int 或 None；
                            None=该次平方不加 rescale。空序列 () = 全部不加。

    每个 ``*_sf`` 都是 ``NOISE_VARIANCE_TABLE_BY_N`` 的 key（即 scale_bits）；
    σ² 严禁写死。
    """
    deg = int(degree)
    if deg < 1 or deg > 6:
        raise ValueError(f"Block 3 degree 必须在 1..6 之间，实际 {deg}")
    if N is None:
        N = 8192 if deg == 2 else 16384

    cfg = Block3NoiseConfig(
        degree=deg,
        x_fresh=NoisePoint("fresh", int(x_fresh_sf), int(N)),
        inv_2n_encode=NoisePoint("encoding", int(inv_2n_sf), int(N)),
        output_truncation_k=(int(output_truncation_k) if output_truncation_k is not None else None),
        output_truncation_mode=str(output_truncation_mode),
    )
    if x_inv_2n_rescale_sf is not None:
        cfg.x_inv_2n_result_rescale = NoisePoint("rescale", int(x_inv_2n_rescale_sf), int(N))

    if not square_rescale_sfs:
        cfg.square_rescales = tuple(None for _ in range(deg))
    else:
        if len(square_rescale_sfs) != deg:
            raise ValueError(
                f"square_rescale_sfs 长度必须 == degree={deg}, 实际 {len(square_rescale_sfs)}"
            )
        cfg.square_rescales = tuple(
            (NoisePoint("rescale", int(sf), int(N)) if sf is not None else None)
            for sf in square_rescale_sfs
        )
    return cfg


def _make_block3_approximation_exponential(cfg: Block3NoiseConfig):
    """构造 BLB Block 3 噪声版的 ``approximation_exponential``。

    替换 ``BertSelfAttentionWithAproximation.approximation_exponential`` 的实例方法。
    顺序：
        1. fresh on x_softmax（输入 x，已在 approximation_softmax 里做完 max 移位）
        2. encode on 1/2^n broadcast 到 x 同形 [B, A, S, S]（每 slot 独立）
        3. ewmulcp: x · noisy(1/2^n)；可选 rescale on 乘法结果
        4. y = 1 + x_scaled  （ctpt 加法，无噪声）
        5. for k in range(degree): y = y · y  （ewmulcc 自乘）；可选 rescale
    """
    degree = int(cfg.degree)
    inv_2n_value = 1.0 / float(2 ** degree)
    sq_rescales = cfg.square_rescales  # tuple len == degree

    def block3_approx_exp(x: Tensor) -> Tensor:
        # 1. fresh on softmax 输入 x
        x = x + _sample_gaussian_for_point(x, cfg.x_fresh)
        # 2. encode on 1/2^n（CKKS smulcp 的 plaintext-side 噪声；按 1/D / γ 同方式 per-slot）
        inv_2n_broadcast = torch.full_like(x, inv_2n_value)
        noisy_inv_2n = inv_2n_broadcast + _sample_gaussian_for_point(inv_2n_broadcast, cfg.inv_2n_encode)
        # 3. ewmulcp: x · (1/2^n)
        x_scaled = x * noisy_inv_2n
        if cfg.x_inv_2n_result_rescale is not None:
            x_scaled = x_scaled + _sample_gaussian_for_point(x_scaled, cfg.x_inv_2n_result_rescale)
        # 4. 1 + x · (1/2^n)：ctpt 加法（不加噪）
        y = 1.0 + x_scaled
        # 5. iterative squaring degree 次
        for k in range(degree):
            y = y * y                                      # ewmulcc 自乘
            rs = sq_rescales[k] if k < len(sq_rescales) else None
            if rs is not None:
                y = y + _sample_gaussian_for_point(y, rs)
        # 6. Block 3 末尾 truncation
        if cfg.output_truncation_k is not None:
            y = _apply_truncation(y, cfg.output_truncation_k, cfg.output_truncation_mode)
        return y

    return block3_approx_exp


def _make_block2_bsgs_mask_hook(
        mask1_encode: NoisePoint,
        mask1_rescale: Optional[NoisePoint],
        mask2_encode: NoisePoint,
        mask2_rescale: Optional[NoisePoint],
        rotation_after_mask1_rescale: bool = False,
        rotation_after_mask2_rescale: bool = False,
        ):
    """构造 K^T / Q 在 Q·K^T 之前的两步 "BSGS mask 模拟" hook。

    在密文 BSGS 转置 / 重排里，每一步会做 ewmulcp(ct, ones-mask) ── 全 1 plaintext
    与 ciphertext 按位乘。明文模拟版本：
        step1: noisy_ones_1 = 1 + ε_enc1; out = tensor · noisy_ones_1; (+ ε_resc1?) (+ ε_rot1?)
        step2: noisy_ones_2 = 1 + ε_enc2; out = out · noisy_ones_2;    (+ ε_resc2?) (+ ε_rot2?)

    每步可选 rotation 噪声（紧跟 rescale 之后），SF 继承自该步的 rescale_point。

    返回 ``hook(tensor) -> tensor``：tensor 形状任意（K^T 是 [B,A,Dh,S]，Q 是 [B,A,S,Dh]），
    全 1 mask 沿 tensor 形状广播。
    """
    def hook(tensor: Tensor) -> Tensor:
        # ----- 第 1 步：tensor ⊙ (ones + ε_enc1) -----
        ones1 = torch.ones_like(tensor)
        noisy_mask1 = ones1 + _sample_gaussian_for_point(ones1, mask1_encode)
        out = tensor * noisy_mask1
        if mask1_rescale is not None:
            out = out + _sample_gaussian_for_point(out, mask1_rescale)
            if rotation_after_mask1_rescale:
                out = out + _sample_gaussian_for_point(out, _make_rotation_point(mask1_rescale))
        # ----- 第 2 步：out ⊙ (ones + ε_enc2) -----
        ones2 = torch.ones_like(out)
        noisy_mask2 = ones2 + _sample_gaussian_for_point(ones2, mask2_encode)
        out = out * noisy_mask2
        if mask2_rescale is not None:
            out = out + _sample_gaussian_for_point(out, mask2_rescale)
            if rotation_after_mask2_rescale:
                out = out + _sample_gaussian_for_point(out, _make_rotation_point(mask2_rescale))
        return out
    return hook


# ============================================================================
# BLB Block 4 噪声注入 ── 范围：softmax 输出 → softmax×V → Wo → post-attn LN head
#                          （rsqrt 之前为止；rsqrt + tail 留给 Block 5）
#
# 与 Block 1/2/3 相同的设计原则：
#   * 所有 σ² 走 NOISE_VARIANCE_TABLE_BY_N 查表，**不写死**。
#   * encode / fresh 必加；rescale 全部可选。
#   * 默认 N=16384（按用户 BLB 推荐），可由 cfg 动态调整。
#
# Block 4 噪声点（共 16 个 = 2 fresh + 6 encode + 8 rescale）：
#
#   (a) softmax 输出 mask 步：
#         1) fresh   on softmax 输出 P
#         2) encode  on ones-mask
#         3) rescale on P · ones-mask 结果（可选）
#
#   (b) V mask 步：
#         4) fresh   on V
#         5) encode  on ones-mask
#         6) rescale on V · ones-mask 结果（可选）
#
#   (c) softmax × V matmul + 合并 mask 步：
#         7)  rescale on (P_masked · V_masked) matmul 结果（可选）
#         8)  encode  on ones-mask
#         9)  rescale on (softmax×V) · ones-mask 结果（可选）
#
#   (d) Wo 投影：
#         10) encode  on W_o
#         11) rescale on X · W_o 结果（可选；输出即 Att）
#
#   (e) post-attn LN head（与 Block 1 head 同结构，运算同 1/D 广播）：
#         12) encode  on 1/D for μ
#         13) rescale on μ（可选）
#         14) rescale on (X − μ)²（可选）
#         15) encode  on 1/D for variance
#         16) rescale on variance（可选）
#
# 与 legacy ``_apply_softmax_value_noise`` 的关系：
#   * Block 4 的 softmax_out 与 V hooks 接管 softmax×V 之前的噪声路径；
#     install Block 4 时 ``_apply_softmax_value_noise`` 路径会被 short-circuit。
#   * legacy 仍保留供 stage2 RL 使用。
# ============================================================================

@dataclass
class Block4NoiseConfig:
    """BLB Block 4 噪声配置（16 个注入点）。

    必选 (2 fresh + 6 encode)：
        softmax_out_fresh:       fresh   on softmax 输出 P
        softmax_out_mask_encode: encode  on softmax 输出之后的 ones-mask
        v_fresh:                 fresh   on V (Block 2 的 wv 投影输出)
        v_mask_encode:           encode  on V 之后的 ones-mask
        softmax_v_mask_encode:   encode  on softmax×V 之后的 ones-mask
        wo_encode:               encode  on W_o
        ln_mean_inv_d_encode:    encode  on post-attn LN head 的 μ-1/D
        ln_var_inv_d_encode:     encode  on post-attn LN head 的 var-1/D

    可选 (8 个 rescale；None = 不加该处)：
        softmax_out_mask_rescale:    rescale on P · ones-mask
        v_mask_rescale:              rescale on V · ones-mask
        softmax_v_matmul_rescale:    rescale on softmax×V matmul 结果
        softmax_v_mask_rescale:      rescale on (softmax×V) · ones-mask
        wo_result_rescale:           rescale on Att = X · W_o
        ln_mean_result_rescale:      rescale on post-attn LN μ
        ln_square_result_rescale:    rescale on post-attn LN (X−μ)²
        ln_var_result_rescale:       rescale on post-attn LN variance
    """
    # ---- (a) softmax 输出 mask ----
    softmax_out_fresh: NoisePoint
    softmax_out_mask_encode: NoisePoint
    # ---- (b) V mask ----
    v_fresh: NoisePoint
    v_mask_encode: NoisePoint
    # ---- (c) softmax × V mask ----
    softmax_v_mask_encode: NoisePoint
    # ---- (d) Wo ----
    wo_encode: NoisePoint
    # ---- (e) post-attn LN head ----
    ln_mean_inv_d_encode: NoisePoint
    ln_var_inv_d_encode: NoisePoint

    # ---- Optional rescale ----
    softmax_out_mask_rescale: Optional[NoisePoint] = None
    v_mask_rescale: Optional[NoisePoint] = None
    softmax_v_matmul_rescale: Optional[NoisePoint] = None
    softmax_v_mask_rescale: Optional[NoisePoint] = None
    wo_result_rescale: Optional[NoisePoint] = None
    ln_mean_result_rescale: Optional[NoisePoint] = None
    ln_square_result_rescale: Optional[NoisePoint] = None
    ln_var_result_rescale: Optional[NoisePoint] = None
    # PPTI MPC↔HE 截断：Block 4 末尾（post-attn LN var, rsqrt 之前）
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    # Rotation 候选点（共 6 个）：
    #   #1 softmax 输出·mask rescale 后  （绑定 softmax_out_mask_rescale）
    #   #2 V·mask rescale 后             （绑定 v_mask_rescale）
    #   #3 (P_masked·V_masked) matmul rescale 后 （绑定 softmax_v_matmul_rescale）
    #   #4 (softmax×V)·ones-mask rescale 后      （绑定 softmax_v_mask_rescale）
    #   #5 Att·Wo rescale 后             （绑定 wo_result_rescale）
    #   #6 post-attn LN (X−μ)² rescale 后 （绑定 ln_square_result_rescale）
    rotation_after_softmax_out_mask_rescale: bool = False
    rotation_after_v_mask_rescale: bool = False
    rotation_after_softmax_v_matmul_rescale: bool = False
    rotation_after_softmax_v_mask_rescale: bool = False
    rotation_after_wo_rescale: bool = False
    rotation_after_ln_square_rescale: bool = False


def make_block4_default_config(
        *,
        N: int = 16384,
        softmax_out_fresh_sf: int = 30,
        softmax_out_mask_sf: int = 22,
        v_fresh_sf: int = 30,
        v_mask_sf: int = 22,
        softmax_v_mask_sf: int = 22,
        wo_sf: int = 22,
        ln_mean_inv_d_sf: int = 22,
        ln_var_inv_d_sf: int = 22,
        # rescale 全部可选；传 None = 不加
        softmax_out_mask_rescale_sf: Optional[int] = None,
        v_mask_rescale_sf: Optional[int] = None,
        softmax_v_matmul_rescale_sf: Optional[int] = None,
        softmax_v_mask_rescale_sf: Optional[int] = None,
        wo_rescale_sf: Optional[int] = None,
        ln_mean_rescale_sf: Optional[int] = None,
        ln_square_rescale_sf: Optional[int] = None,
        ln_var_rescale_sf: Optional[int] = None,
        output_truncation_k: Optional[int] = None,
        output_truncation_mode: str = "binary",
        rotation_after_softmax_out_mask_rescale: bool = False,
        rotation_after_v_mask_rescale: bool = False,
        rotation_after_softmax_v_matmul_rescale: bool = False,
        rotation_after_softmax_v_mask_rescale: bool = False,
        rotation_after_wo_rescale: bool = False,
        rotation_after_ln_square_rescale: bool = False,
        ) -> "Block4NoiseConfig":
    """构建 Block 4 噪声配置。

    每个 ``*_sf`` 都是 ``NOISE_VARIANCE_TABLE_BY_N`` 的 key（即 scale_bits）；
    rescale_sf=None 表示**不加**这一处的 rescale 噪声。

    默认 N=16384（BLB Block 4 推荐表）；也可以传 N=8192 等动态调整。
    """
    cfg = Block4NoiseConfig(
        softmax_out_fresh=NoisePoint("fresh", int(softmax_out_fresh_sf), int(N)),
        softmax_out_mask_encode=NoisePoint("encoding", int(softmax_out_mask_sf), int(N)),
        v_fresh=NoisePoint("fresh", int(v_fresh_sf), int(N)),
        v_mask_encode=NoisePoint("encoding", int(v_mask_sf), int(N)),
        softmax_v_mask_encode=NoisePoint("encoding", int(softmax_v_mask_sf), int(N)),
        wo_encode=NoisePoint("encoding", int(wo_sf), int(N)),
        ln_mean_inv_d_encode=NoisePoint("encoding", int(ln_mean_inv_d_sf), int(N)),
        ln_var_inv_d_encode=NoisePoint("encoding", int(ln_var_inv_d_sf), int(N)),
        output_truncation_k=(int(output_truncation_k) if output_truncation_k is not None else None),
        output_truncation_mode=str(output_truncation_mode),
        rotation_after_softmax_out_mask_rescale=bool(rotation_after_softmax_out_mask_rescale),
        rotation_after_v_mask_rescale=bool(rotation_after_v_mask_rescale),
        rotation_after_softmax_v_matmul_rescale=bool(rotation_after_softmax_v_matmul_rescale),
        rotation_after_softmax_v_mask_rescale=bool(rotation_after_softmax_v_mask_rescale),
        rotation_after_wo_rescale=bool(rotation_after_wo_rescale),
        rotation_after_ln_square_rescale=bool(rotation_after_ln_square_rescale),
    )
    if softmax_out_mask_rescale_sf is not None:
        cfg.softmax_out_mask_rescale = NoisePoint("rescale", int(softmax_out_mask_rescale_sf), int(N))
    if v_mask_rescale_sf is not None:
        cfg.v_mask_rescale = NoisePoint("rescale", int(v_mask_rescale_sf), int(N))
    if softmax_v_matmul_rescale_sf is not None:
        cfg.softmax_v_matmul_rescale = NoisePoint("rescale", int(softmax_v_matmul_rescale_sf), int(N))
    if softmax_v_mask_rescale_sf is not None:
        cfg.softmax_v_mask_rescale = NoisePoint("rescale", int(softmax_v_mask_rescale_sf), int(N))
    if wo_rescale_sf is not None:
        cfg.wo_result_rescale = NoisePoint("rescale", int(wo_rescale_sf), int(N))
    if ln_mean_rescale_sf is not None:
        cfg.ln_mean_result_rescale = NoisePoint("rescale", int(ln_mean_rescale_sf), int(N))
    if ln_square_rescale_sf is not None:
        cfg.ln_square_result_rescale = NoisePoint("rescale", int(ln_square_rescale_sf), int(N))
    if ln_var_rescale_sf is not None:
        cfg.ln_var_result_rescale = NoisePoint("rescale", int(ln_var_rescale_sf), int(N))
    return cfg


def _make_block4_input_mask_hook(
        fresh_point: NoisePoint,
        mask_encode_point: NoisePoint,
        mask_rescale_point: Optional[NoisePoint],
        rotation_after_mask_rescale: bool = False,
        ):
    """softmax 输出 / V 共用：fresh on tensor → ⊙ ones-mask (encode) → optional rescale。

    可选 rotation 紧跟 mask_rescale 之后；SF 继承自 mask_rescale_point。
    """
    def hook(tensor: Tensor) -> Tensor:
        # 1. fresh on tensor
        out = tensor + _sample_gaussian_for_point(tensor, fresh_point)
        # 2. ⊙ ones-mask (CKKS ewmulcp)
        ones = torch.ones_like(out)
        noisy_mask = ones + _sample_gaussian_for_point(ones, mask_encode_point)
        out = out * noisy_mask
        # 3. optional rescale + optional rotation
        if mask_rescale_point is not None:
            out = out + _sample_gaussian_for_point(out, mask_rescale_point)
            if rotation_after_mask_rescale:
                out = out + _sample_gaussian_for_point(out, _make_rotation_point(mask_rescale_point))
        return out
    return hook


def _make_block4_softmax_v_hook(
        matmul_rescale: Optional[NoisePoint],
        mask_encode: NoisePoint,
        mask_rescale: Optional[NoisePoint],
        rotation_after_matmul_rescale: bool = False,
        rotation_after_mask_rescale: bool = False,
        ):
    """softmax×V matmul 之后：optional rescale on matmul → ⊙ ones-mask (encode) → optional rescale。

    matmul rescale 与 mask rescale 各支持一个独立 rotation 选项，SF 继承自各自的 rescale。
    """
    def hook(tensor: Tensor) -> Tensor:
        # 1. optional rescale on matmul 结果 + optional rotation
        if matmul_rescale is not None:
            tensor = tensor + _sample_gaussian_for_point(tensor, matmul_rescale)
            if rotation_after_matmul_rescale:
                tensor = tensor + _sample_gaussian_for_point(tensor, _make_rotation_point(matmul_rescale))
        # 2. ⊙ ones-mask
        ones = torch.ones_like(tensor)
        noisy_mask = ones + _sample_gaussian_for_point(ones, mask_encode)
        out = tensor * noisy_mask
        # 3. optional rescale + optional rotation
        if mask_rescale is not None:
            out = out + _sample_gaussian_for_point(out, mask_rescale)
            if rotation_after_mask_rescale:
                out = out + _sample_gaussian_for_point(out, _make_rotation_point(mask_rescale))
        return out
    return hook


def _make_block4_wo_forward(
        linear_module: nn.Linear,
        encode_point: NoisePoint,
        rescale_point: Optional[NoisePoint],
        rotation_after_rescale: bool = False,
        ):
    """Wo 投影包装：encode on W_o + 可选 rescale on Att = X·W_o 结果。

    可选 rotation 紧跟 rescale 之后；SF 继承自 rescale_point。
    """
    def block4_wo_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        weight = linear_module.weight
        noisy_weight = weight + _sample_gaussian_for_point(weight, encode_point)
        noisy_weight = noisy_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = linear_module.bias
        if bias is not None:
            bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        out = nn.functional.linear(hidden_states, noisy_weight, bias)
        if rescale_point is not None:
            out = out + _sample_gaussian_for_point(out, rescale_point)
            if rotation_after_rescale:
                out = out + _sample_gaussian_for_point(out, _make_rotation_point(rescale_point))
        return out
    return block4_wo_forward


class NoisyBlock4LayerNorm(nn.Module):
    """post-attn LayerNorm (``layer.attention.output.LayerNorm``) 替身：
    BLB Block 4 head + （Block 5 tail 预留 cfg5 接口，TBD）。

    Head 部分与 ``NoisyBlock1LayerNorm`` 结构相同，但驱动字段改为 Block 4 命名空间：

        sum_x = Σ_d x                                # reduction (no mul)
        μ   = sum_x · (1/D)                          # encode on 1/D；rescale on μ (opt)
        x_c = x − μ                                  # subtraction (no mul)
        sq  = x_c · x_c                              # rescale on sq (opt)
        sum_sq = Σ_d sq                              # reduction
        var = sum_sq · (1/D)                         # encode on 1/D；rescale on var (opt)
        --- Block 4 / Block 5 边界 ---
        inv_std    = 1 / sqrt(var + ε)               # rsqrt 非线性 (Block 5 起点)
        normalized = x_c · inv_std                   # Block 5 (TBD)
        out        = normalized · γ + β              # Block 5 (TBD)
    """

    def __init__(
            self,
            original_ln: nn.LayerNorm,
            cfg4: Optional[Block4NoiseConfig] = None,
            cfg5=None,
            ):
        super().__init__()
        self.weight = original_ln.weight
        self.bias = original_ln.bias
        self.eps = float(original_ln.eps)
        self.normalized_shape = tuple(original_ln.normalized_shape)
        self.cfg4 = cfg4
        self.cfg5 = cfg5

    def set_block4_cfg(self, cfg4: Optional[Block4NoiseConfig]) -> None:
        self.cfg4 = cfg4

    def set_block5_cfg(self, cfg5) -> None:
        """Block 5 cfg 占位接口；TBD 等用户给完 Block 5 spec 再激活。"""
        self.cfg5 = cfg5

    def forward(self, x: Tensor) -> Tensor:
        D = int(x.shape[-1])
        cfg4 = self.cfg4

        # ===================== Block 4: head =====================
        sum_x = x.sum(dim=-1, keepdim=True)                            # [B, S, 1]
        if cfg4 is not None:
            inv_d_broadcast = torch.full_like(x, 1.0 / D)              # [B, S, H]
            noisy_inv_d = inv_d_broadcast + _sample_gaussian_for_point(inv_d_broadcast, cfg4.ln_mean_inv_d_encode)
            mean = sum_x * noisy_inv_d                                 # [B, S, H]
            if cfg4.ln_mean_result_rescale is not None:
                mean = mean + _sample_gaussian_for_point(mean, cfg4.ln_mean_result_rescale)
        else:
            mean = sum_x / float(D)

        x_centered = x - mean

        sq = x_centered * x_centered
        if cfg4 is not None and cfg4.ln_square_result_rescale is not None:
            sq = sq + _sample_gaussian_for_point(sq, cfg4.ln_square_result_rescale)
            # Block 4 rotation #6：紧跟 (X−μ)² rescale 之后；SF 继承自 ln_square_result_rescale
            if cfg4.rotation_after_ln_square_rescale:
                sq = sq + _sample_gaussian_for_point(sq, _make_rotation_point(cfg4.ln_square_result_rescale))

        sum_sq = sq.sum(dim=-1, keepdim=True)
        if cfg4 is not None:
            inv_d_var_broadcast = torch.full_like(sq, 1.0 / D)
            noisy_inv_d_var = inv_d_var_broadcast + _sample_gaussian_for_point(inv_d_var_broadcast, cfg4.ln_var_inv_d_encode)
            var = sum_sq * noisy_inv_d_var
            if cfg4.ln_var_result_rescale is not None:
                var = var + _sample_gaussian_for_point(var, cfg4.ln_var_result_rescale)
        else:
            var = sum_sq / float(D)

        # Block 4 末尾：PPTI MPC↔HE 截断（var = Block 4 输出，rsqrt 之前）
        if cfg4 is not None and cfg4.output_truncation_k is not None:
            var = _apply_truncation(var, cfg4.output_truncation_k, cfg4.output_truncation_mode)

        # ===== Block 4 / Block 5 边界：rsqrt 非线性，无噪 =====
        inv_std = torch.rsqrt(var + self.eps)

        # ===================== Block 5: tail =====================
        # cfg5 提供则按 Block 5 LN tail 加噪：
        #   1) inv_std / X−μ 各自 fresh + ewmulcc → optional rescale
        #   2) γ broadcast 到 [B,S,H] + per-slot encode + ewmulcp → optional rescale
        #   3) +β（ctpt 加法，无噪）
        cfg5 = self.cfg5
        if cfg5 is not None:
            if inv_std.shape != x.shape:
                inv_std = inv_std.expand_as(x).contiguous()
            noisy_inv_std = inv_std + _sample_gaussian_for_point(inv_std, cfg5.inv_std_fresh)
            noisy_x_centered = x_centered + _sample_gaussian_for_point(x_centered, cfg5.x_centered_fresh)
            normalized = noisy_x_centered * noisy_inv_std
            if cfg5.normalize_result_rescale is not None:
                normalized = normalized + _sample_gaussian_for_point(normalized, cfg5.normalize_result_rescale)
            gamma_broadcast = self.weight.expand_as(normalized)
            noisy_gamma = gamma_broadcast + _sample_gaussian_for_point(gamma_broadcast, cfg5.gamma_encode)
            gamma_mul = normalized * noisy_gamma
            if cfg5.gamma_result_rescale is not None:
                gamma_mul = gamma_mul + _sample_gaussian_for_point(gamma_mul, cfg5.gamma_result_rescale)
                # Block 5 rotation #1：紧跟 γ rescale 之后；SF 继承自 gamma_result_rescale
                if cfg5.rotation_after_gamma_rescale:
                    gamma_mul = gamma_mul + _sample_gaussian_for_point(gamma_mul, _make_rotation_point(cfg5.gamma_result_rescale))
            out = gamma_mul + self.bias
        else:
            normalized = x_centered * inv_std
            out = normalized * self.weight + self.bias
        return out


# ============================================================================
# BLB Block 5 噪声注入 ── 范围：post-attn LN tail (rsqrt → normalize → γ scale)
#                           + Wffn1 + GELU 多项式近似（degree-aware）
#
# 与 Block 1/2/3/4 相同的设计原则：
#   * 所有 σ² 走 NOISE_VARIANCE_TABLE_BY_N 查表，**不写死**。
#   * encode / fresh 必加；rescale 全部可选。
#   * N 默认按 GELU degree 自动选 ── degree=1 → 8192；degree∈{2,4} → 16384。
#     可由 cfg 显式覆盖（保持动态可调整）。
#
# Block 5 噪声点（抛开 GELU 部分共 2 fresh + 2 encode + 3 rescale）：
#   (a) post-attn LN tail（结构与 Block 2 LN tail 完全一致）：
#         1) fresh   on 1/std (Block 4→5 边界 ct)
#         2) fresh   on (X − μ) (Block 4→5 边界 ct)
#         3) encode  on γ (broadcast 到 [B, S, H] 后每 slot 独立)
#         4) rescale on (1/std)·(X−μ) 乘法结果（可选）
#         5) rescale on γ·normalize 乘法结果（可选）
#
#   (b) Wffn1 投影：
#         6) encode  on W_ffn1
#         7) rescale on X · W_ffn1 结果（可选；输出即 GELU 输入 x）
#
#   (c) GELU 多项式近似（degree-aware；用户已给定 degree ∈ {1, 2, 4}）：
#         8) encode  on **所有多项式系数**（共享同一个 SF；每个系数独立采样噪声）
#         9..) rescale on 每个 power 计算结果（x², x³, x⁴；degree-1 个）
#         ..)  rescale on 每个系数乘以 power 的乘法结果（共 degree 个）
#
#       degree=1：power=0 个，coeff_mul=1 个 → +0 +1 = 1 rescale
#       degree=2：power=1 个 (x²)，coeff_mul=2 个 (b·x, c·x²) → +1 +2 = 3 rescale
#       degree=4：power=3 个 (x², x³=x²·x, x⁴=x²·x²)，coeff_mul=4 个 → +3 +4 = 7 rescale
#
# Block 5 GELU 系数 encode 共享 SF：cfg.gelu_coeff_encode 对每个系数独立采噪
# （不是同一份噪声！相同分布参数、独立样本），与"每个 slot 独立"语义一致。
# ============================================================================

@dataclass
class Block5NoiseConfig:
    """BLB Block 5 噪声配置。

    Block 5 范围：post-attn LN tail (rsqrt 之后) → Wffn1 → GELU 多项式近似。
    GELU 部分仅支持 degree ∈ {1, 2, 4}（按 BLB Figure 10 / 用户 spec）。

    必选 (2 fresh + 2 encode + 1 GELU coeff encode)：
        inv_std_fresh:        fresh   on 1/std (Block 4→5 边界 ct)
        x_centered_fresh:     fresh   on (X − μ) (Block 4→5 边界 ct)
        gamma_encode:         encode  on γ (per-slot, broadcast 到 [B, S, H])
        wffn1_encode:         encode  on W_ffn1
        gelu_coeff_encode:    encode  on **所有 GELU 多项式系数**（共享 SF；
                              对每个系数独立采样噪声）

    可选：
        normalize_result_rescale:    rescale on (1/std)·(X−μ) 结果
        gamma_result_rescale:        rescale on γ·normalize 结果
        wffn1_result_rescale:        rescale on X·W_ffn1 结果
        gelu_power_rescales:         tuple，长度 == degree-1；
                                     degree=1: ()；degree=2: (x²,)；
                                     degree=4: (x², x³, x⁴)
        gelu_coeff_mul_rescales:     tuple，长度 == degree；
                                     degree=1: (b·x,)；degree=2: (b·x, c·x²)；
                                     degree=4: (b·x, c·x², d·x³, e·x⁴)

    GELU 多项式约定（与 ``polynomial(x, coeff, sign)`` 实现一致）：
        coeffs = [c_0, c_1, ..., c_n]，n=degree
        result = c_0 + c_1·x + c_2·x² + ... + c_n·x^n
        c_0（常数项）只 encode，不参与乘法 → 不加 rescale。
    """
    inv_std_fresh: NoisePoint
    x_centered_fresh: NoisePoint
    gamma_encode: NoisePoint
    wffn1_encode: NoisePoint
    gelu_degree: int
    gelu_coeff_encode: NoisePoint

    normalize_result_rescale: Optional[NoisePoint] = None
    gamma_result_rescale: Optional[NoisePoint] = None
    wffn1_result_rescale: Optional[NoisePoint] = None
    # GELU 部分：长度由 degree 决定
    gelu_power_rescales: Tuple[Optional[NoisePoint], ...] = field(default_factory=tuple)
    gelu_coeff_mul_rescales: Tuple[Optional[NoisePoint], ...] = field(default_factory=tuple)
    # PPTI MPC↔HE 截断：Block 5 末尾（GELU 多项式输出之后）
    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    # Rotation 候选点（共 2 个）：
    #   #1 γ·((X−μ)/std) rescale 之后  （绑定 gamma_result_rescale）
    #   #2 W_ffn1·X rescale 之后        （绑定 wffn1_result_rescale）
    rotation_after_gamma_rescale: bool = False
    rotation_after_wffn1_rescale: bool = False


def make_block5_default_config(
        *,
        gelu_degree: int,
        N: Optional[int] = None,
        inv_std_fresh_sf: int = 30,
        x_centered_fresh_sf: int = 30,
        gamma_sf: int = 22,
        wffn1_sf: int = 22,
        gelu_coeff_sf: int = 22,
        normalize_rescale_sf: Optional[int] = None,
        gamma_rescale_sf: Optional[int] = None,
        wffn1_rescale_sf: Optional[int] = None,
        gelu_power_rescale_sfs: Sequence[Optional[int]] = (),
        gelu_coeff_mul_rescale_sfs: Sequence[Optional[int]] = (),
        output_truncation_k: Optional[int] = None,
        output_truncation_mode: str = "binary",
        rotation_after_gamma_rescale: bool = False,
        rotation_after_wffn1_rescale: bool = False,
        ) -> "Block5NoiseConfig":
    """构建 Block 5 噪声配置。

    Args:
        gelu_degree:  GELU 多项式 degree ∈ {1, 2, 4}
        N:            CKKS 多项式阶。None = 按 degree 自动选
                      （degree=1 → 8192，degree∈{2,4} → 16384）
        gelu_power_rescale_sfs:    长度 == degree-1；
                                   degree=1: ()；degree=2: (x²,)；degree=4: (x²,x³,x⁴)
        gelu_coeff_mul_rescale_sfs: 长度 == degree；按 c_1·x, c_2·x², ... 顺序
        其它 *_sf 含义同 Block 1/2 同名参数。

    每个 ``*_sf`` 都是 ``NOISE_VARIANCE_TABLE_BY_N`` 的 key（即 scale_bits）；
    σ² 严禁写死。
    """
    deg = int(gelu_degree)
    if deg not in (1, 2, 4):
        raise ValueError(f"Block 5 GELU degree 必须 ∈ {{1, 2, 4}}, got {deg}")
    if N is None:
        N = 8192 if deg == 1 else 16384

    cfg = Block5NoiseConfig(
        inv_std_fresh=NoisePoint("fresh", int(inv_std_fresh_sf), int(N)),
        x_centered_fresh=NoisePoint("fresh", int(x_centered_fresh_sf), int(N)),
        gamma_encode=NoisePoint("encoding", int(gamma_sf), int(N)),
        wffn1_encode=NoisePoint("encoding", int(wffn1_sf), int(N)),
        gelu_degree=deg,
        gelu_coeff_encode=NoisePoint("encoding", int(gelu_coeff_sf), int(N)),
        output_truncation_k=(int(output_truncation_k) if output_truncation_k is not None else None),
        output_truncation_mode=str(output_truncation_mode),
        rotation_after_gamma_rescale=bool(rotation_after_gamma_rescale),
        rotation_after_wffn1_rescale=bool(rotation_after_wffn1_rescale),
    )
    if normalize_rescale_sf is not None:
        cfg.normalize_result_rescale = NoisePoint("rescale", int(normalize_rescale_sf), int(N))
    if gamma_rescale_sf is not None:
        cfg.gamma_result_rescale = NoisePoint("rescale", int(gamma_rescale_sf), int(N))
    if wffn1_rescale_sf is not None:
        cfg.wffn1_result_rescale = NoisePoint("rescale", int(wffn1_rescale_sf), int(N))

    expected_power_len = deg - 1
    if not gelu_power_rescale_sfs:
        cfg.gelu_power_rescales = tuple(None for _ in range(expected_power_len))
    else:
        if len(gelu_power_rescale_sfs) != expected_power_len:
            raise ValueError(
                f"gelu_power_rescale_sfs 长度必须 == degree-1 = {expected_power_len}, "
                f"实际 {len(gelu_power_rescale_sfs)}"
            )
        cfg.gelu_power_rescales = tuple(
            (NoisePoint("rescale", int(sf), int(N)) if sf is not None else None)
            for sf in gelu_power_rescale_sfs
        )

    if not gelu_coeff_mul_rescale_sfs:
        cfg.gelu_coeff_mul_rescales = tuple(None for _ in range(deg))
    else:
        if len(gelu_coeff_mul_rescale_sfs) != deg:
            raise ValueError(
                f"gelu_coeff_mul_rescale_sfs 长度必须 == degree = {deg}, "
                f"实际 {len(gelu_coeff_mul_rescale_sfs)}"
            )
        cfg.gelu_coeff_mul_rescales = tuple(
            (NoisePoint("rescale", int(sf), int(N)) if sf is not None else None)
            for sf in gelu_coeff_mul_rescale_sfs
        )
    return cfg


# ============================================================================
# BLB 首次输入 X 的 fresh 噪声
#
# Block 2-5 是 transformer 各层间循环时的 block；它们覆盖了"前一层 LN tail
# 输出 → Wq/Wk/Wv"这条路径上的噪声。但 transformer 第一次接受输入时，X
# 是直接进入 layer 0 的 Wq/Wk/Wv 的（没有上一层 LN tail），所以缺一个对
# 应位置的 fresh 噪声。这里专门补上：在指定层（默认 layer 0）的 forward
# 入口加 fresh 噪声，与 legacy ``replace_layer_input_noise`` 同语义但走
# ``NOISE_VARIANCE_TABLE_BY_N`` 多 N 表（保持与其它 BLB 噪声口径一致）。
# ============================================================================

def _make_blb_first_input_noise_forward(
        original_forward,
        point: NoisePoint,
        ):
    """构造在 ``layer.forward`` 入口对 hidden_states 加 fresh 噪声的包装。

    与 legacy ``_make_input_noise_forward`` 同结构，但走 BLB 多 N 表。
    """
    def noisy_forward(hidden_states, *args, **kwargs):
        if hidden_states is None:
            return original_forward(hidden_states, *args, **kwargs)
        noisy_hidden_states = hidden_states + _sample_gaussian_for_point(hidden_states, point)
        return original_forward(noisy_hidden_states, *args, **kwargs)
    return noisy_forward


def _make_block5_wffn1_forward(
        linear_module: nn.Linear,
        encode_point: NoisePoint,
        rescale_point: Optional[NoisePoint],
        rotation_after_rescale: bool = False,
        ):
    """Wffn1 投影包装：encode on W_ffn1 + 可选 rescale on result（GELU 输入 x）
    + 可选 rotation 紧跟 rescale 之后（SF 继承自 rescale_point）。
    """
    def block5_wffn1_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        weight = linear_module.weight
        noisy_weight = weight + _sample_gaussian_for_point(weight, encode_point)
        noisy_weight = noisy_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = linear_module.bias
        if bias is not None:
            bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        out = nn.functional.linear(hidden_states, noisy_weight, bias)
        if rescale_point is not None:
            out = out + _sample_gaussian_for_point(out, rescale_point)
            # Block 5 rotation #2：紧跟 W_ffn1·X rescale 之后
            if rotation_after_rescale:
                out = out + _sample_gaussian_for_point(out, _make_rotation_point(rescale_point))
        return out
    return block5_wffn1_forward


def _make_block5_gelu_forward(original_gelu, cfg5: Block5NoiseConfig):
    """构造 BLB Block 5 噪声版的 ``PolynomialGELU.forward``。

    替换 ``layer.intermediate.intermediate_act_fn.forward``。

    工作流（与原 PolynomialGELU 等价但带噪）：
      1. 计算 x 的幂 x², x³, x⁴（按 degree 决定哪些）：每个 power 之后加可选 rescale。
         共享 power 用于 piecewise 两段多项式，避免重复加噪。
      2. 对负段 (x ∈ [-2.7, 0)) 和正段 (x ∈ [0, 2.7]) 分别用各自 ``coeff[sign]``：
         a) 每个系数 c_k 广播到 x 同形 [B, S, H] 后加 encode 噪声（per-slot 独立）；
         b) 常数项 c_0 直接累加（无乘法 → 无 rescale）；
         c) 非常数项 c_k * x^k：乘法后加可选 rescale（按 cfg5.gelu_coeff_mul_rescales[k-1]）。
      3. 用 mask 选段，与原 PolynomialGELU.forward 一致。
    """
    coeff_dict = original_gelu.coeff   # {0: pos_coeffs, 1: neg_coeffs}（与 GELU_COEEF[degree] 同型）
    degree = int(original_gelu.degree)
    cfg_degree = int(cfg5.gelu_degree)
    if degree != cfg_degree:
        raise ValueError(
            f"PolynomialGELU.degree={degree} 与 cfg5.gelu_degree={cfg_degree} 不匹配"
        )
    if degree not in (1, 2, 4):
        raise ValueError(f"Block 5 仅支持 GELU degree ∈ {{1, 2, 4}}, got {degree}")

    pwr_rs = cfg5.gelu_power_rescales
    coeff_rs = cfg5.gelu_coeff_mul_rescales

    def _compute_powers(x: Tensor):
        """返回 [x^0, x^1, ..., x^degree]，按 degree 决定中间 rescale。"""
        powers = [None] * (degree + 1)
        powers[0] = torch.ones_like(x)   # x^0：仅作为 c_0·1 的占位（实际不会乘 powers[0]）
        powers[1] = x
        if degree >= 2:
            x2 = x * x
            rs = pwr_rs[0] if len(pwr_rs) > 0 else None
            if rs is not None:
                x2 = x2 + _sample_gaussian_for_point(x2, rs)
            powers[2] = x2
        if degree >= 4:
            x3 = powers[2] * x          # x^3 = x² · x
            rs = pwr_rs[1] if len(pwr_rs) > 1 else None
            if rs is not None:
                x3 = x3 + _sample_gaussian_for_point(x3, rs)
            powers[3] = x3
            x4 = powers[2] * powers[2]  # x^4 = x² · x²
            rs = pwr_rs[2] if len(pwr_rs) > 2 else None
            if rs is not None:
                x4 = x4 + _sample_gaussian_for_point(x4, rs)
            powers[4] = x4
        return powers

    def _compute_polynomial(powers, coeffs_for_piece, x_ref: Tensor) -> Tensor:
        """c_0 + c_1·x + c_2·x² + ... + c_n·x^n, 每个系数 encode + 每个乘法 rescale。"""
        if len(coeffs_for_piece) != degree + 1:
            # GELU_COEEF[degree] 期望长度 = degree + 1
            raise RuntimeError(
                f"coeff piece 长度 {len(coeffs_for_piece)} != degree+1 = {degree+1}"
            )
        result = None
        for k in range(degree + 1):
            coeff_value = float(coeffs_for_piece[k])
            coeff_broadcast = torch.full_like(x_ref, coeff_value)
            noisy_coeff = coeff_broadcast + _sample_gaussian_for_point(
                coeff_broadcast, cfg5.gelu_coeff_encode
            )
            if k == 0:
                # 常数项：仅 encode，不乘 power（c_0 · 1 = c_0），无 rescale
                term = noisy_coeff
            else:
                term = powers[k] * noisy_coeff
                rs = coeff_rs[k - 1] if (k - 1) < len(coeff_rs) else None
                if rs is not None:
                    term = term + _sample_gaussian_for_point(term, rs)
            result = term if result is None else result + term
        return result

    def block5_gelu_forward(x: Tensor) -> Tensor:
        powers = _compute_powers(x)
        # 两段多项式（共享 powers）：负段 sign=1，正段 sign=0
        y0 = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        y1 = _compute_polynomial(powers, coeff_dict[1], x)   # [-2.7, 0)
        y2 = _compute_polynomial(powers, coeff_dict[0], x)   # [0, 2.7]
        y3 = x                                                # > 2.7（GELU 大正值近似 x）

        mask_low = x < -2.7
        mask_neg = (x >= -2.7) & (x < 0)
        mask_pos = (x >= 0) & (x <= 2.7)
        mask_high = x > 2.7

        out = torch.where(mask_low, y0, torch.zeros_like(x))
        out = torch.where(mask_neg, y1, out)
        out = torch.where(mask_pos, y2, out)
        out = torch.where(mask_high, y3, out)
        # Block 5 末尾 truncation（GELU 输出之后）
        if cfg5.output_truncation_k is not None:
            out = _apply_truncation(out, cfg5.output_truncation_k, cfg5.output_truncation_mode)
        return out

    return block5_gelu_forward


# tensor polynomial approximation
def polynomial(x, coeff, sign):
    # x: Tensor, 可能在 cuda:0 或 cpu
    device = x.device
    dtype  = x.dtype

    # 1. 生成 x 的幂
    powers = torch.stack([x.pow(i) for i in range(len(coeff[sign]))], dim=-1)

    # 2. 在同一设备上创建系数 Tensor
    coeff_tensor = torch.tensor(
        coeff[sign],
        device=device,
        dtype=dtype
    )

    # 3. 按维度相乘求和
    return (powers * coeff_tensor).sum(dim=-1)

class PolynomialGELU(nn.Module):
    """可逆的三次多项式GELU近似"""
    def __init__(self, degree=4):
        super().__init__()
        self.coeff = GELU_COEEF[degree]  # 正向系数
        self.degree = degree
        # Lazily-built {(sign, device, dtype): coeff Tensor}. Plain dict (not a
        # registered buffer) so it stays out of state_dict; keyed by device so a
        # module moved to a new device rebuilds correctly.
        self._coeff_cache = {}

    def _coeff_tensor(self, sign: int, device, dtype) -> Tensor:
        key = (sign, device, dtype)
        t = self._coeff_cache.get(key)
        if t is None:
            t = torch.tensor(self.coeff[sign], device=device, dtype=dtype)
            self._coeff_cache[key] = t
        return t

    def _poly(self, x: Tensor, sign: int) -> Tensor:
        # Bit-identical to the module-level ``polynomial(x, self.coeff, sign)``
        # but the coeff tensor is cached per (sign, device, dtype) instead of
        # being rebuilt (host->device copy) on every forward call.
        coeff_tensor = self._coeff_tensor(sign, x.device, x.dtype)
        powers = torch.stack([x.pow(i) for i in range(len(self.coeff[sign]))], dim=-1)
        return (powers * coeff_tensor).sum(dim=-1)

    def forward(self, x: Tensor) -> Tensor:

        if self.degree == 0:
            # Degree 0: skip piecewise comparison, directly use [-2.7, 0] interval polynomial
            return self._poly(x, 1)

        y0 = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        y1 = self._poly(x, 1)
        y2 = self._poly(x, 0)
        y3 = x
        
        # 创建与x相同设备和类型的输出张量
        
        if(self.degree == 1 or self.degree == 2):
            # degree 1, use the Bumblebee piecewise
            mask_low = x < -2.7
            mask_neg = (x >= -2.7) & (x < 0)
            mask_pos = (x >= 0) & (x <= 2.7)
            mask_high = x > 2.7
        else:
            mask_low = x < -2.7
            mask_neg = (x >= -2.7) & (x < 0)
            mask_pos = (x >= 0) & (x <= 2.7)
            mask_high = x > 2.7
        
        # 分段处理
        # print(f"y0 : {y0}, y1 : {y1}, y2 : {y2}, y3 : {y3}")
        out = torch.where(mask_low, y0, torch.zeros_like(x))
        out = torch.where(mask_neg, y1, out)
        out = torch.where(mask_pos, y2, out)
        out = torch.where(mask_high, y3, out)

        # print(f"X : {x}, Y : {out}, OriginGelu: {origin}")
        return out
    
# change BertsdpaAttention to normal self attention and change its softmax
class BertSelfAttentionWithAproximation(BertSelfAttention):
    """BertSelfAttention with softmax approximation"""
    def __init__(self, config, degree, lower_bound, position_embedding_type=None, layer_idx=None):
        try:
            super().__init__(
                config,
                position_embedding_type=position_embedding_type,
                layer_idx=layer_idx,
            )
        except TypeError:
            try:
                super().__init__(
                    config,
                    position_embedding_type=position_embedding_type,
                )
            except TypeError:
                super().__init__(config)
        if position_embedding_type is not None:
            self.position_embedding_type = position_embedding_type
        self.layer_idx = layer_idx
        self.degree = degree 
        self.lower_bound = lower_bound
        self._softmax_value_noise_state = None

    def approximation_exponential(self, x: torch.Tensor) -> torch.Tensor:
        """近似计算指数函数""" # degree = 1,2,3,4,5,6 
        x = torch.pow(1 + x / (2 ** self.degree), 2 ** self.degree) 
        return x


    # do approximation softmax
    def approximation_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """使用指数近似计算softmax"""
        # print("do approximation softmax")
        # 计算指数近似,  < lower bound的exp值为0 
        # need to be optimized
        # (degree, lower_bound) -> (1, -2), (2, -4), (3, -10), (4, -13), (5, -13), (6, -13)
        x = x - x.max(dim=-1, keepdim=True)[0] + 1e-9  # 数值稳定处理
        # print(f"This is x: {x}")
        # print(torch.isnan(x).any(), torch.isinf(x).any())  # 检测异常值
        # print(x.abs().max())  # 确认数值量级

        exp_approx = self.approximation_exponential(x)
        exp_out = torch.where(x < self.lower_bound, torch.zeros_like(x), exp_approx)
        sum_exp = torch.sum(exp_out, dim=-1, keepdim=True) + 1e-9
        # print(f"this is exp_out: {exp_out}; this is sum_exp: {sum_exp}")
        return exp_out / sum_exp  # 统一使用掩码后结果

    # error construction
    # def error_construction(self, scales: torch.Tensor) -> torch.Tensor:
    #     absolute_error = torch.
        
    
    def _looks_like_attention_mask(self, value) -> bool:
        return torch.is_tensor(value) and value.dim() >= 2

    def _looks_like_cache_position(self, value) -> bool:
        return (
            torch.is_tensor(value)
            and value.dim() <= 1
            and value.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.long,
            )
        )

    def _looks_like_cache(self, value) -> bool:
        if value is None:
            return True
        if isinstance(value, (tuple, list)):
            return True
        return any(
            hasattr(value, attr)
            for attr in ("update", "is_updated", "layers", "self_attention_cache", "cross_attention_cache")
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        *args,
        **kwargs,
    ):
       # Follow the current transformers BERT attention flow and replace
       # only the softmax step with the approximation variant.
        encoder_attention_mask = kwargs.pop("encoder_attention_mask", None)
        past_key_value = kwargs.pop("past_key_value", None)
        past_key_values = kwargs.pop("past_key_values", None)
        output_attentions = kwargs.pop("output_attentions", False)
        cache_position = kwargs.pop("cache_position", None)

        tail = list(args)

        if isinstance(past_key_value, bool):
            if output_attentions in (False, None):
                output_attentions = past_key_value
            past_key_value = None
        if isinstance(past_key_values, bool):
            if output_attentions in (False, None):
                output_attentions = past_key_values
            past_key_values = None

        if tail and self._looks_like_cache_position(tail[-1]) and cache_position is None:
            cache_position = tail.pop()

        if tail and isinstance(tail[-1], bool):
            output_attentions = tail.pop()

        if encoder_hidden_states is not None and tail:
            first = tail[0]
            if encoder_attention_mask is None and (first is None or self._looks_like_attention_mask(first)):
                encoder_attention_mask = tail.pop(0)

        if past_key_value is None and past_key_values is None and tail:
            candidate = tail.pop(0)
            if isinstance(candidate, bool):
                if output_attentions in (False, None):
                    output_attentions = candidate
                candidate = None
            elif (
                encoder_hidden_states is None
                and encoder_attention_mask is None
                and self._looks_like_attention_mask(candidate)
            ):
                # Some legacy positional paths may still include a placeholder
                # encoder-attention mask slot even for encoder-only BERT.
                encoder_attention_mask = candidate
                candidate = tail.pop(0) if tail else None
            past_key_value = candidate

        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values
        elif past_key_values is None and past_key_value is not None:
            past_key_values = past_key_value

        if isinstance(past_key_value, bool):
            if output_attentions in (False, None):
                output_attentions = past_key_value
            past_key_value = None
            past_key_values = None

        batch_size, _, _ = hidden_states.shape
        query_layer = self.query(hidden_states)
        query_layer = query_layer.view(
            batch_size, -1, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

        # BLB Block 2: Q BSGS-style mask 模拟（在 Q·K^T 之前对 Q 做两步 ewmulcp）
        block2_q_hook = getattr(self, "_block2_q_bsgs_hook", None)
        if block2_q_hook is not None:
            query_layer = block2_q_hook(query_layer)

        is_updated = False
        is_cross_attention = encoder_hidden_states is not None
        curr_past_key_value = None
        if past_key_value is not None:
            if hasattr(past_key_value, "is_updated"):
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and encoder_attention_mask is not None:
            attention_mask = encoder_attention_mask

        if is_cross_attention and curr_past_key_value is not None and is_updated:
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_layer = self.key(current_states)
            key_layer = key_layer.view(
                batch_size, -1, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)
            value_layer = self.value(current_states)
            value_layer = value_layer.view(
                batch_size, -1, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)

            if curr_past_key_value is not None:
                if hasattr(curr_past_key_value, "update"):
                    cache_position = cache_position if not is_cross_attention else None
                    key_layer, value_layer = curr_past_key_value.update(
                        key_layer,
                        value_layer,
                        self.layer_idx,
                        {"cache_position": cache_position},
                    )
                    if is_cross_attention and hasattr(past_key_value, "is_updated"):
                        past_key_value.is_updated[self.layer_idx] = True
                elif self._looks_like_cache(curr_past_key_value):
                    key_layer = torch.cat([curr_past_key_value[0], key_layer], dim=2)
                    value_layer = torch.cat([curr_past_key_value[1], value_layer], dim=2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # BLB Block 2: K^T BSGS-style mask 模拟（在 Q·K^T 之前对 K^T 做两步 ewmulcp）
        kt = key_layer.transpose(-1, -2)
        block2_kt_hook = getattr(self, "_block2_kt_bsgs_hook", None)
        if block2_kt_hook is not None:
            kt = block2_kt_hook(kt)
        attention_scores = torch.matmul(query_layer, kt)

        # BLB Block 2: 合并 Q,K 过程
        # （Q·K^T matmul 结果上加 rescale + 一次 ones-mask ewmulcp + 结果 rescale）
        block2_qkt_merge_hook = getattr(self, "_block2_qkt_merge_hook", None)
        if block2_qkt_merge_hook is not None:
            attention_scores = block2_qkt_merge_hook(attention_scores)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if past_key_value is not None:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Changed Softmax approximation
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.approximation_softmax(attention_scores)

        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # BLB Block 4: softmax 输出 / V 路径噪声。
        # 装了 Block 4 hook 时 short-circuit 掉 legacy ``_apply_softmax_value_noise``
        # （fresh-on-attn_probs / fresh-on-V）。legacy 仍保留供 stage2 RL 使用。
        block4_softmax_out_hook = getattr(self, "_block4_softmax_out_hook", None)
        block4_v_hook = getattr(self, "_block4_v_hook", None)
        if block4_softmax_out_hook is not None or block4_v_hook is not None:
            context_attention_probs = (
                block4_softmax_out_hook(attention_probs)
                if block4_softmax_out_hook is not None else attention_probs
            )
            context_value_layer = (
                block4_v_hook(value_layer)
                if block4_v_hook is not None else value_layer
            )
        else:
            context_attention_probs, context_value_layer = _apply_softmax_value_noise(
                attention_probs,
                value_layer,
                self,
            )
        context_layer = torch.matmul(context_attention_probs, context_value_layer)

        # BLB Block 4: softmax×V matmul 之后的 rescale + ones-mask + rescale
        block4_softmax_v_hook = getattr(self, "_block4_softmax_v_hook", None)
        if block4_softmax_v_hook is not None:
            context_layer = block4_softmax_v_hook(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

# ---------------------------------------------------------------------------
# GPT-2 Softmax 近似: 通过 monkey-patch eager_attention_forward 实现
# ---------------------------------------------------------------------------

def _approx_exponential(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Taylor 展开近似 exp(x), degree 控制精度."""
    return torch.pow(1 + x / (2 ** degree), 2 ** degree)


def _approx_softmax(x: torch.Tensor, degree: int, lower_bound: float) -> torch.Tensor:
    """使用指数近似计算 softmax, 与 BertSelfAttentionWithAproximation 保持一致."""
    x = x - x.max(dim=-1, keepdim=True)[0] + 1e-9
    exp_approx = _approx_exponential(x, degree)
    exp_out = torch.where(x < lower_bound, torch.zeros_like(x), exp_approx)
    sum_exp = torch.sum(exp_out, dim=-1, keepdim=True) + 1e-9
    return exp_out / sum_exp


def _make_gpt2_approx_attn_forward(attn_module, degree: int, lower_bound: float):
    """构造一个替代 GPT2Attention.forward 的函数, 将 softmax 替换为近似版本.

    该函数完整复制 HuggingFace eager_attention_forward 的计算逻辑,
    唯一区别是把 ``nn.functional.softmax(attn_weights, dim=-1)`` 换成
    ``_approx_softmax(attn_weights, degree, lower_bound)``.
    """
    original_forward = attn_module.forward

    def _approx_eager_attention(module, query, key, value,
                                attention_mask, head_mask=None, **kwargs):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if module.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5,
                dtype=attn_weights.dtype, device=attn_weights.device,
            )
        if getattr(module, "scale_attn_by_inverse_layer_idx", False):
            attn_weights = attn_weights / float(module.layer_idx + 1)
        if not module.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = module.bias[:, :, key_length - query_length:key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full(
                [], mask_value, dtype=attn_weights.dtype, device=attn_weights.device,
            )
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key.shape[-2]]
            attn_weights = attn_weights + causal_mask
        # ----- 核心替换: 使用近似 softmax -----
        attn_weights = _approx_softmax(attn_weights, degree, lower_bound)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = module.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        context_attn_weights, context_value = _apply_softmax_value_noise(
            attn_weights,
            value,
            module,
        )
        attn_output = torch.matmul(context_attn_weights, context_value)
        attn_output = attn_output.transpose(1, 2)
        return attn_output, attn_weights

    def patched_forward(hidden_states, *args, **kwargs):
        """替换 GPT2Attention.forward, 强制使用带近似 softmax 的 eager attention."""
        # 保存原始 _attn_implementation, 临时强制 eager 模式
        orig_impl = attn_module.config._attn_implementation
        orig_reorder = attn_module.reorder_and_upcast_attn
        attn_module.config._attn_implementation = "eager"
        attn_module.reorder_and_upcast_attn = False
        # 注入自定义 attention 函数
        import transformers.models.gpt2.modeling_gpt2 as _gpt2_mod
        _saved_fn = _gpt2_mod.eager_attention_forward
        _gpt2_mod.eager_attention_forward = _approx_eager_attention
        try:
            result = original_forward(hidden_states, *args, **kwargs)
        finally:
            _gpt2_mod.eager_attention_forward = _saved_fn
            attn_module.config._attn_implementation = orig_impl
            attn_module.reorder_and_upcast_attn = orig_reorder
        return result

    return patched_forward


class ReversibleLayerHandler:
    """管理GELU/Softmax/噪声替换与恢复的工具类.

    支持两类 backbone:
      * BERT 家族 (bert-base / bert-large, roberta): 依赖 ``attention.self.{query,key,value}``
        / ``attention.output.dense`` / ``intermediate.dense`` / ``output.dense``
        / ``intermediate.intermediate_act_fn`` 的模块路径.
      * GPT-2 家族 (openai-community/gpt2): 使用融合的 ``attn.c_attn`` (Conv1D) +
        ``attn.c_proj`` / ``mlp.c_fc`` / ``mlp.c_proj`` / ``mlp.act``.
        由于 c_attn 把 Q/K/V 融合成一个 Conv1D, 这里通过一次性包装 c_attn.forward,
        在单层上按需累加 q/k/v 各自的权重噪声.
    """

    # Layer-local path tables (relative to a single transformer block).
    _BERT_PATHS = {
        "gelu_act": "intermediate.intermediate_act_fn",
        "wo_dense": "attention.output.dense",
        "wffn1_dense": "intermediate.dense",
        "wffn2_dense": "output.dense",
    }
    _GPT2_PATHS = {
        "gelu_act": "mlp.act",
        "wo_dense": "attn.c_proj",
        "wffn1_dense": "mlp.c_fc",
        "wffn2_dense": "mlp.c_proj",
    }

    @staticmethod
    def _detect_arch(model) -> str:
        """Return ``'gpt2'`` or ``'bert'`` based on top-level module layout."""
        # GPT-2 style: has .transformer.h (list of GPT2Block)
        transformer = getattr(model, "transformer", None)
        if transformer is not None and hasattr(transformer, "h"):
            return "gpt2"
        # BERT / RoBERTa style
        if hasattr(model, "bert") or hasattr(model, "roberta"):
            return "bert"
        # Fallback: inspect first layer attributes
        for attr in ("bert", "roberta", "transformer"):
            sub = getattr(model, attr, None)
            if sub is None:
                continue
            layers = getattr(sub, "h", None) or getattr(getattr(sub, "encoder", None), "layer", None)
            if layers is None or len(layers) == 0:
                continue
            first = layers[0]
            if hasattr(first, "attn") and hasattr(first.attn, "c_attn"):
                return "gpt2"
            if hasattr(first, "attention") and hasattr(first.attention, "self"):
                return "bert"
        return "bert"

    def __init__(self, model):
        self.model = model
        self._arch = self._detect_arch(model)
        self._paths = self._GPT2_PATHS if self._arch == "gpt2" else self._BERT_PATHS
        self.original_gelu = {}
        self.original_attention = {}
        self.original_input_noise = {}
        self.original_projection_noise = {
            "query": {},
            "key": {},
            "value": {},
            "wo": {},
            "wffn1": {},
            "wffn2": {},
        }
        self.original_softmax_value_noise = {}
        # --- Approx-module reuse caches (Stage-1 GELU/Softmax degree search) ---
        # During the Stage-1 degree search the BERT weights are frozen (only the
        # GTrXL policy is trained) and the policy can only pick approximation
        # degrees (GELU_MAP/SOFTMAX_MAP never expose the "original" sentinel -1),
        # so replace_layer_softmax / replace_layer_gelu run every episode but
        # restore_* never does. Rebuilding a fresh BertSelfAttentionWithAproximation
        # per layer per episode (CPU kaiming-init of Q/K/V Linears + state_dict
        # copy + device transfer) is pure overhead that cannot change the forward
        # result, because a module that copied the frozen weights once stays
        # bit-identical to one rebuilt every call. Cache the modules and only
        # update the degree in place. Reuse is gated on the cached module still
        # being "fresh-equivalent" (see _approx_attn_is_fresh_equivalent) so the
        # BLB Stage-2 path — which installs extra per-instance hooks after
        # replace_* — always falls back to the original reconstruct path.
        self.reuse_approx_modules = True
        self._approx_softmax_cache = {}   # {layer_idx: BertSelfAttentionWithAproximation}
        self._approx_gelu_cache = {}      # {(layer_idx, degree): PolynomialGELU}
        self._approx_softmax_rebuilds = 0  # diagnostics: full reconstruct count
        self._approx_gelu_rebuilds = 0     # diagnostics: full PolynomialGELU build count
        # GPT-2 fused Q/K/V state: {layer_idx: {"query"/"key"/"value": (sf, distribution)}}
        self._gpt2_qkv_state = {}
        # Wrapped c_attn registry so we install the proxy forward only once per layer.
        self._gpt2_qkv_wrapped = {}
        # BLB Block 1 噪声安装状态：保存原始 forward / LayerNorm 模块 + 当前 cfg
        self.original_block1_ffn2 = {}        # {layer_idx: original ffn2.forward (bound method)}
        self.original_block1_layernorm = {}   # {layer_idx: original output.LayerNorm module}
        self.block1_cfg_per_layer = {}        # {layer_idx: Block1NoiseConfig}
        # BLB Block 2 噪声安装状态
        self.original_block2_qproj = {}       # {layer_idx: original attention.self.query.forward}
        self.original_block2_kproj = {}       # {layer_idx: original attention.self.key.forward}
        self.original_block2_vproj = {}       # {layer_idx: original attention.self.value.forward}
        self.block2_cfg_per_layer = {}        # {layer_idx: Block2NoiseConfig}
        # BLB Block 3 噪声安装状态：approximation_exponential 是用 instance attribute
        # 覆盖 class method 实现的；restore 时 delattr 即可。这里只记录哪些层装过，
        # 以及当前 cfg 供 introspect 用。
        self.block3_installed_layers = set()  # set of layer_idx
        self.block3_cfg_per_layer = {}        # {layer_idx: Block3NoiseConfig}
        # BLB Block 4 噪声安装状态
        self.original_block4_wo = {}          # {layer_idx: original attention.output.dense.forward}
        self.original_block4_post_attn_ln = {}  # {layer_idx: original attention.output.LayerNorm}
        self.block4_cfg_per_layer = {}        # {layer_idx: Block4NoiseConfig}
        # BLB Block 5 噪声安装状态
        self.original_block5_wffn1 = {}       # {layer_idx: original intermediate.dense.forward}
        self.original_block5_gelu = {}        # {layer_idx: original intermediate.intermediate_act_fn.forward}
        self.block5_cfg_per_layer = {}        # {layer_idx: Block5NoiseConfig}
        # BLB 首次输入 X 的 fresh 噪声（layer 0 进入 Wq/Wk/Wv 之前的 X，
        # 因为没有上一层 LN tail，BLB 缺一个对应位置的 fresh）。
        # {layer_idx: {"forward": original_layer_forward, "point": NoisePoint}}
        self.blb_first_input_noise_state = {}
        self.backup_model = copy.deepcopy(model)  # 完整模型备份
    
    @staticmethod
    def _approx_attn_is_fresh_equivalent(module) -> bool:
        """True iff ``module`` is bit-identical to a freshly constructed
        ``BertSelfAttentionWithAproximation``: no block-3 instance override of
        ``approximation_exponential``, no softmax/value noise state, and no BLB
        per-instance hooks. Only then is "reuse the cached module + update the
        degree" equivalent to reconstructing it from scratch."""
        if "approximation_exponential" in vars(module):
            return False  # block-3 installed an instance-level exp override
        if getattr(module, "_softmax_value_noise_state", None) is not None:
            return False
        for hook in ("_block2_q_bsgs_hook", "_block2_kt_bsgs_hook",
                     "_block2_qkt_merge_hook", "_block4_softmax_out_hook",
                     "_block4_v_hook", "_block4_softmax_v_hook"):
            if getattr(module, hook, None) is not None:
                return False
        return True

    def replace_layer_gelu(self, layer_indices=None, layer_name="model.model.layers", degree=1):
        """替换指定层的GELU函数 (BERT: intermediate.intermediate_act_fn; GPT-2: mlp.act)"""
        act_path = self._paths["gelu_act"]
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices:
                if i not in self.original_gelu:
                    self.original_gelu[i] = {
                        "act_fn": _get_attr_path(layer, act_path),
                    }
                orig_act = _get_attr_path(layer, act_path)
                orig_training = getattr(orig_act, "training", layer.training)
                # Reuse a cached PolynomialGELU for this (layer, degree). The
                # module is stateless apart from a lazily-built coeff-tensor
                # cache, so reusing it across episodes is bit-identical and skips
                # rebuilding the coeff tensor on every forward. Keyed per
                # (layer, degree) — not per degree — so modules stay un-shared
                # across layers and the BLB block-5 per-instance forward wrap
                # cannot leak between layers. Skip the cache if a wrap was
                # installed (instance-level ``forward``).
                cached = (self._approx_gelu_cache.get((i, degree))
                          if self.reuse_approx_modules else None)
                if cached is not None and "forward" not in vars(cached):
                    new_act = cached
                else:
                    # degree 0 = ReLU（用 ReLU 替换 GELU）；其余 degree 用多项式 GELU。
                    new_act = nn.ReLU() if int(degree) == 0 else PolynomialGELU(degree=degree)
                    self._approx_gelu_rebuilds += 1
                    if self.reuse_approx_modules:
                        self._approx_gelu_cache[(i, degree)] = new_act
                new_act.train(bool(orig_training))
                _set_attr_path(layer, act_path, new_act)

        print(f"已替换 {len(layer_indices)} 层的GELU函数（GELU function）")
    
    def replace_layer_softmax(self, layer_indices=None, layer_name="model.model.layers", attention_name = "attention", degree=1):
        """替换指定层的Softmax函数 (BERT: 替换 BertSelfAttention; GPT-2: monkey-patch forward)"""
        if self._arch == "gpt2":
            lb = Exp_bound.get(degree)
            if lb is None:
                print(f"[ReversibleLayerHandler] 警告: degree={degree} 没有对应的 Exp_bound, 跳过 softmax 近似.")
                return
            for i, layer in enumerate(eval("self." + layer_name)):
                if i in layer_indices:
                    if i not in self.original_attention:
                        self.original_attention[i] = {
                            'attention_forward': layer.attn.forward,
                        }
                    layer.attn.forward = _make_gpt2_approx_attn_forward(
                        layer.attn, degree=degree, lower_bound=lb,
                    )
            print(f"已替换 {len(layer_indices)} 层的Softmax函数（GPT-2 approximate softmax, degree={degree}）")
            return
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices:
                # 保存原始函数引用
                if i not in self.original_attention:
                    self.original_attention[i] = {
                        'attention': eval("layer."+ attention_name)
                    }

                # Fast path: reuse the cached approx-attention module for this
                # layer and only update its degree. Bit-identical to a fresh
                # reconstruct because the BERT weights are frozen during the
                # degree search, so the cached module (which copied them once)
                # carries the same weight bits a reconstruct would re-copy.
                # Gated on the cached module still being installed AND
                # fresh-equivalent, so non-Stage-1 callers (BLB Stage-2 installs
                # per-instance hooks after this) fall back to reconstruct.
                cached = self._approx_softmax_cache.get(i) if self.reuse_approx_modules else None
                if (cached is not None
                        and layer.attention.self is cached
                        and self._approx_attn_is_fresh_equivalent(cached)):
                    cached.degree = degree
                    cached.lower_bound = Exp_bound[degree]
                    continue

                # 应用新函数 (full reconstruct — original behavior)
                orig_self = layer.attention.self
                orig_sd = orig_self.state_dict()
                new_attn = BertSelfAttentionWithAproximation(
                    self.model.config,
                    degree=degree,
                    lower_bound=Exp_bound[degree],
                    position_embedding_type=getattr(orig_self, "position_embedding_type", None),
                    layer_idx=getattr(orig_self, "layer_idx", None),
                )
                new_attn.load_state_dict(orig_sd, strict=False)
                new_attn = new_attn.to(
                    device=orig_self.query.weight.device,
                    dtype=orig_self.query.weight.dtype,
                )
                new_attn.train(orig_self.training)
                layer.attention.self = new_attn
                self._approx_softmax_rebuilds += 1
                if self.reuse_approx_modules:
                    self._approx_softmax_cache[i] = new_attn

        print(f"已替换 {len(layer_indices)} 层的Softmax函数（Softmax function）")
    
    def replace_layer_input_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=INPUT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="fresh"
            ):
        """Inject x-noise on transformer-layer inputs: x + N(0, sigma^2)."""
        _ = get_input_noise_variance(int(scaling_factor), distribution=distribution)

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        # legacy / BLB 互斥校验
        self._check_blb_legacy_conflict(selected, installing="legacy")

        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            stored_forward = self.original_input_noise.get(i, {}).get("forward")
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not layer:
                self.original_input_noise[i] = {
                    "forward": layer.forward,
                }

            original_forward = self.original_input_noise[i]["forward"]
            layer.forward = _make_input_noise_forward(
                original_forward,
                scaling_factor=int(scaling_factor),
                distribution=distribution,
            )
            self.original_input_noise[i]["scaling_factor"] = int(scaling_factor)
            self.original_input_noise[i]["distribution"] = str(distribution).lower()

        print(_format_noise_enable_message("input", len(selected), scaling_factor, distribution))

    def _get_attention_core_module(self, layer):
        if self._arch == "gpt2":
            return layer.attn
        return layer.attention.self

    def replace_layer_softmax_value_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            softmax_scaling_factor=SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR,
            value_scaling_factor=SOFTMAX_VALUE_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="fresh",
            ):
        """Inject fresh noise as (softmax + e1) @ (V + e2) in attention."""
        _ = get_input_noise_variance(int(softmax_scaling_factor), distribution=distribution)
        _ = get_input_noise_variance(int(value_scaling_factor), distribution=distribution)

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        # legacy / BLB 互斥校验
        self._check_blb_legacy_conflict(selected, installing="legacy")

        state = {
            "softmax_scaling_factor": int(softmax_scaling_factor),
            "value_scaling_factor": int(value_scaling_factor),
            "distribution": str(distribution).lower(),
        }
        for i, layer in enumerate(layers):
            if i not in selected:
                continue
            attn_module = self._get_attention_core_module(layer)
            self.original_softmax_value_noise.setdefault(
                i,
                getattr(attn_module, "_softmax_value_noise_state", None),
            )
            setattr(attn_module, "_softmax_value_noise_state", dict(state))

        print(
            "Enabled softmax/V attention-product noise for "
            f"{len(selected)} layers "
            f"(softmax sf={int(softmax_scaling_factor)}, "
            f"V sf={int(value_scaling_factor)}, distribution={str(distribution).lower()})"
        )

    def _ensure_gpt2_qkv_wrapper(self, layer_idx, layer):
        """Install a single proxy forward on this layer's ``attn.c_attn``.

        The proxy reads ``self._gpt2_qkv_state[layer_idx]`` each forward pass and
        adds per-slice weight noise (query / key / value) on top of the untouched
        base Conv1D output. This keeps the three projection noises independent
        even though GPT-2 stores them as a single fused Conv1D.
        """
        if layer_idx in self._gpt2_qkv_wrapped:
            return
        c_attn = layer.attn.c_attn  # HuggingFace Conv1D
        original_forward = c_attn.forward
        handler = self
        hidden_size = c_attn.nf // 3

        def proxy_forward(hidden_states, *args, **kwargs):
            base = original_forward(hidden_states, *args, **kwargs)
            state = handler._gpt2_qkv_state.get(layer_idx)
            if not state:
                return base
            result = base.clone()
            in_dim = hidden_states.size(-1)
            for slot_name, slot_idx in (("query", 0), ("key", 1), ("value", 2)):
                params = state.get(slot_name)
                if params is None:
                    continue
                sf, dist = params
                variance = get_input_noise_variance(int(sf), distribution=dist)
                if variance <= 0.0:
                    continue
                std = math.sqrt(variance)
                # 走独立噪声 RNG，不被外部 torch.manual_seed 污染
                _gen = _get_noise_generator(hidden_states.device)
                noise_w = torch.empty(
                    in_dim, hidden_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ).normal_(0.0, std, generator=_gen)
                noise_out = torch.matmul(hidden_states, noise_w)
                start = slot_idx * hidden_size
                end = start + hidden_size
                result[..., start:end] = result[..., start:end] + noise_out
            return result

        c_attn.forward = proxy_forward
        self._gpt2_qkv_wrapped[layer_idx] = {
            "c_attn": c_attn,
            "forward": original_forward,
        }

    def _replace_attention_projection_noise(
            self,
            projection_name,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        """Temporarily use (W + We) inside Q/K/V projection without mutating the stored weight."""
        _ = get_input_noise_variance(int(scaling_factor), distribution=distribution)

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        # legacy / BLB 互斥校验
        self._check_blb_legacy_conflict(selected, installing="legacy")

        # GPT-2 fused c_attn path: use a per-layer proxy and accumulate state.
        if self._arch == "gpt2":
            for i, layer in enumerate(layers):
                if i not in selected:
                    continue
                self._ensure_gpt2_qkv_wrapper(i, layer)
                state = self._gpt2_qkv_state.setdefault(i, {})
                state[projection_name] = (int(scaling_factor), str(distribution).lower())
            print(_format_noise_enable_message(projection_name, len(selected), scaling_factor, distribution))
            return

        projection_store = self.original_projection_noise.setdefault(projection_name, {})
        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            projection_module = getattr(layer.attention.self, projection_name)
            stored_forward = projection_store.get(i, {}).get("forward")
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not projection_module:
                projection_store[i] = {
                    "forward": projection_module.forward,
                }

            projection_module.forward = _make_noisy_linear_forward(
                projection_module,
                scaling_factor=int(scaling_factor),
                distribution=distribution,
            )
            projection_store[i]["scaling_factor"] = int(scaling_factor)
            projection_store[i]["distribution"] = str(distribution).lower()

        print(_format_noise_enable_message(projection_name, len(selected), scaling_factor, distribution))

    def _replace_layer_linear_module_noise(
            self,
            store_key,
            module_path,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        """Temporarily use (W + We) inside a layer Linear module without mutating the stored weight."""
        _ = get_input_noise_variance(int(scaling_factor), distribution=distribution)

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        # legacy / BLB 互斥校验
        self._check_blb_legacy_conflict(selected, installing="legacy")

        projection_store = self.original_projection_noise.setdefault(store_key, {})
        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            linear_module = eval("layer." + module_path)
            stored_forward = projection_store.get(i, {}).get("forward")
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not linear_module:
                projection_store[i] = {
                    "forward": linear_module.forward,
                }

            linear_module.forward = _make_noisy_projection_forward(
                linear_module,
                scaling_factor=int(scaling_factor),
                distribution=distribution,
            )
            projection_store[i]["scaling_factor"] = int(scaling_factor)
            projection_store[i]["distribution"] = str(distribution).lower()

        print(_format_noise_enable_message(store_key, len(selected), scaling_factor, distribution))

    def replace_layer_query_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_attention_projection_noise(
            "query",
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_key_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_attention_projection_noise(
            "key",
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_value_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_attention_projection_noise(
            "value",
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_attention_output_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_layer_linear_module_noise(
            "wo",
            self._paths["wo_dense"],
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_ffn1_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WFFN1_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_layer_linear_module_noise(
            "wffn1",
            self._paths["wffn1_dense"],
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def replace_layer_ffn2_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            scaling_factor=WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            distribution="encoding"
            ):
        self._replace_layer_linear_module_noise(
            "wffn2",
            self._paths["wffn2_dense"],
            layer_indices=layer_indices,
            layer_name=layer_name,
            scaling_factor=scaling_factor,
            distribution=distribution,
        )

    def restore_layer_gelu(self, layer_indices=None, layer_name="model.model.layers"):
        """恢复指定层的原始GELU函数"""
        act_path = self._paths["gelu_act"]
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices and i in self.original_gelu:
                _set_attr_path(layer, act_path, self.original_gelu[i]["act_fn"])

        print(f"已恢复 {len(layer_indices)} 层的原始GELU函数（original GELU function）")
    
    def restore_layer_softmax(self, layer_indices=None, layer_name="model.model.layers", attention_name = "attention"):
        """恢复指定层的原始Softmax函数"""
        if self._arch == "gpt2":
            for i, layer in enumerate(eval("self." + layer_name)):
                if i in layer_indices and i in self.original_attention:
                    original_fwd = self.original_attention[i].get('attention_forward')
                    if original_fwd is not None:
                        layer.attn.forward = original_fwd
                    del self.original_attention[i]
            return
        for i, layer in enumerate(eval("self." + layer_name)):
            if i in layer_indices and i in self.original_attention:
                current_training = layer.attention.self.training
                restored_attention = self.original_attention[i]['attention']
                restored_attention.train(bool(current_training))
                layer.attention.self = restored_attention

   
    def restore_layer_input_noise(self, layer_indices=None, layer_name="model.model.layers"):
        """Restore original transformer-layer inputs for selected layers."""
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        for i, layer in enumerate(layers):
            if i in selected and i in self.original_input_noise:
                original_forward = self.original_input_noise[i]["forward"]
                if getattr(original_forward, "__self__", None) is layer:
                    layer.forward = original_forward
                del self.original_input_noise[i]

    def restore_layer_softmax_value_noise(self, layer_indices=None, layer_name="model.model.layers"):
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        for i, layer in enumerate(layers):
            if i not in selected:
                continue
            attn_module = self._get_attention_core_module(layer)
            previous_state = self.original_softmax_value_noise.pop(i, None)
            setattr(attn_module, "_softmax_value_noise_state", previous_state)

    def _restore_attention_projection_noise(
            self,
            projection_name,
            layer_indices=None,
            layer_name="model.model.layers"
            ):
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        if self._arch == "gpt2":
            for i in list(selected):
                state = self._gpt2_qkv_state.get(i)
                if state is None:
                    continue
                state.pop(projection_name, None)
                if not state:
                    self._gpt2_qkv_state.pop(i, None)
                    # All three slots cleared — restore the base Conv1D forward
                    wrapped = self._gpt2_qkv_wrapped.pop(i, None)
                    if wrapped is not None:
                        wrapped["c_attn"].forward = wrapped["forward"]
            return

        projection_store = self.original_projection_noise.get(projection_name, {})
        for i, layer in enumerate(layers):
            if i in selected and i in projection_store:
                projection_module = getattr(layer.attention.self, projection_name)
                original_forward = projection_store[i]["forward"]
                if getattr(original_forward, "__self__", None) is projection_module:
                    projection_module.forward = original_forward
                del projection_store[i]

    def _restore_layer_linear_module_noise(
            self,
            store_key,
            module_path,
            layer_indices=None,
            layer_name="model.model.layers"
            ):
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        projection_store = self.original_projection_noise.get(store_key, {})
        for i, layer in enumerate(layers):
            if i in selected and i in projection_store:
                linear_module = eval("layer." + module_path)
                original_forward = projection_store[i]["forward"]
                if getattr(original_forward, "__self__", None) is linear_module:
                    linear_module.forward = original_forward
                del projection_store[i]

    def restore_layer_query_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_attention_projection_noise(
            "query",
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_key_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_attention_projection_noise(
            "key",
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_value_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_attention_projection_noise(
            "value",
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_attention_output_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_layer_linear_module_noise(
            "wo",
            self._paths["wo_dense"],
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_ffn1_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_layer_linear_module_noise(
            "wffn1",
            self._paths["wffn1_dense"],
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    def restore_layer_ffn2_noise(self, layer_indices=None, layer_name="model.model.layers"):
        self._restore_layer_linear_module_noise(
            "wffn2",
            self._paths["wffn2_dense"],
            layer_indices=layer_indices,
            layer_name=layer_name,
        )

    # ------------------------------------------------------------------
    # BLB Block 1：Wffn2 + post-FFN LN head 的完整噪声安装 / 恢复
    # ------------------------------------------------------------------
    def replace_layer_block1_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            cfg: Optional[Block1NoiseConfig] = None,
            ):
        """安装 BLB Block 1 (Wffn2 + post-FFN LN head) 噪声。

        Block 1 范围：从前一层 GELU 输出，到本层 post-FFN LayerNorm 中
        rsqrt 之前为止。共 8 个噪声注入点：

          1. fresh   on Gelu_out             (必加)
          2. encode  on W_ffn2               (必加；与现有 wffn2 噪声方式一致)
          3. rescale on Wffn2 result         (cfg.wffn2_result_rescale 决定，可选)
          4. encode  on 1/D for μ            (必加)
          5. rescale on μ                    (cfg.mean_result_rescale 决定，可选)
          6. rescale on (x − μ)²             (cfg.square_result_rescale 决定，可选)
          7. encode  on 1/D for variance     (必加)
          8. rescale on variance             (cfg.var_result_rescale 决定，可选)

        所有 σ² 都通过 ``NOISE_VARIANCE_TABLE_BY_N[N][scale_bits][dist]`` 查表，
        不写死。默认 N=8192（按 BLB Figure 10 推荐），可由 cfg 动态调整。

        Args:
            layer_indices: 要安装的层索引；None = 全部层
            layer_name: encoder layer list 的属性路径（默认与其它 replace_* 一致）
            cfg: ``Block1NoiseConfig``；None = 用 ``make_block1_default_config()``

        注：本方法**会覆盖**之前 ``replace_layer_ffn2_noise`` 对 Wffn2 forward
        的包装（因为 Block 1 是 ffn2 噪声的严格扩展）。如果想回到 legacy 模式，
        请先调用 ``restore_layer_block1_noise``。
        """
        if self._arch != "bert":
            raise NotImplementedError(
                "BLB Block 1 噪声目前仅支持 BERT 家族 (post-LN, layer.output.{dense,LayerNorm})。"
            )

        if cfg is None:
            cfg = make_block1_default_config()

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        # BLB / legacy 互斥校验
        self._check_blb_legacy_conflict(selected, installing="blb")

        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            # ---- 1. 包 layer.output.dense.forward (Wffn2 投影段) ----
            ffn2_module = layer.output.dense
            stored_forward = self.original_block1_ffn2.get(i)
            # 仅在第一次（或重新挂载到不同模块）时记录 original
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not ffn2_module:
                self.original_block1_ffn2[i] = ffn2_module.forward
            ffn2_module.forward = _make_block1_ffn2_forward(ffn2_module, cfg)

            # ---- 2. 替换 layer.output.LayerNorm 为 NoisyBlock1LayerNorm ----
            current_ln = layer.output.LayerNorm
            if i not in self.original_block1_layernorm:
                # 仅在第一次时把 *真正原始的* LN 存下来
                self.original_block1_layernorm[i] = current_ln
            source_ln = self.original_block1_layernorm[i]
            new_ln = NoisyBlock1LayerNorm(source_ln, cfg)
            new_ln.train(source_ln.training)
            # 跟随原 LN 的 device / dtype
            try:
                ref_param = source_ln.weight
                new_ln = new_ln.to(device=ref_param.device, dtype=ref_param.dtype)
            except Exception:
                pass
            layer.output.LayerNorm = new_ln

            # ---- 3. 记录 cfg ----
            self.block1_cfg_per_layer[i] = cfg

        rescale_summary = (
            f"wffn2_result={cfg.wffn2_result_rescale.scaling_factor if cfg.wffn2_result_rescale else 'off'}, "
            f"mean={cfg.mean_result_rescale.scaling_factor if cfg.mean_result_rescale else 'off'}, "
            f"square={cfg.square_result_rescale.scaling_factor if cfg.square_result_rescale else 'off'}, "
            f"var={cfg.var_result_rescale.scaling_factor if cfg.var_result_rescale else 'off'}"
        )
        _print_blb_install(
            f"已为 {len(selected)} 层启用 BLB Block 1 噪声 "
            f"(N={cfg.gelu_out_fresh.N}, "
            f"fresh_gelu_out={cfg.gelu_out_fresh.scaling_factor}, "
            f"encode_wffn2={cfg.wffn2_encode.scaling_factor}, "
            f"encode_inv_d={cfg.mean_inv_d_encode.scaling_factor}/{cfg.var_inv_d_encode.scaling_factor}, "
            f"rescale=[{rescale_summary}])"
        )

    def restore_layer_block1_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            ):
        """恢复 Block 1 噪声安装前的 ``layer.output.dense.forward`` 与
        ``layer.output.LayerNorm`` 状态。"""
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        for i, layer in enumerate(layers):
            if i not in selected:
                continue
            if i in self.original_block1_ffn2:
                ffn2 = layer.output.dense
                original_forward = self.original_block1_ffn2[i]
                if getattr(original_forward, "__self__", None) is ffn2:
                    ffn2.forward = original_forward
                del self.original_block1_ffn2[i]
            if i in self.original_block1_layernorm:
                layer.output.LayerNorm = self.original_block1_layernorm[i]
                del self.original_block1_layernorm[i]
            self.block1_cfg_per_layer.pop(i, None)

    # ------------------------------------------------------------------
    # BLB Block 2：post-FFN LN tail + Wq/Wk/Wv + Q/K^T BSGS mask + 合并 Q,K
    # ------------------------------------------------------------------
    def replace_layer_block2_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            cfg: Optional[Block2NoiseConfig] = None,
            ):
        """安装 BLB Block 2 噪声（22 个注入点）。

        Block 2 范围（按 BLB Figure 10 / 用户手绘图）：
          (a) post-FFN LN tail：rsqrt 之后的 (1/std)·(X−μ) → γ scale
          (b) Wq / Wk / Wv 三路投影
          (c) Q / K^T 在 Q·K^T **之前**的两步 BSGS-style ones-mask ewmulcp
          (d) Q·K^T 之后的"合并 Q,K"步骤：rescale + 一次 ones-mask ewmulcp + rescale

        必选 (9 encode + 2 fresh)：
          1) fresh on 1/std (Block 1→2 边界 ct)
          2) fresh on (X − μ) (Block 1→2 边界 ct)
          3) encode on γ (per-slot, broadcast 到 [B, S, H])
          4) encode on W_k
          5) encode on K^T BSGS step 1 ones-mask
          6) encode on K^T BSGS step 2 ones-mask
          7) encode on W_q
          8) encode on Q BSGS step 1 ones-mask
          9) encode on Q BSGS step 2 ones-mask
         10) encode on W_v
         11) encode on Q·K^T merge ones-mask

        可选 (11 个 rescale)：每路乘法结果上的 rescale，cfg.*_result_rescale=None
        表示该处不加 rescale。

        所有 σ² 通过 ``NOISE_VARIANCE_TABLE_BY_N[N][scale_bits][dist]`` 查表，
        不写死。默认 N=16384（按 BLB Figure 10），可由 cfg 动态调整。

        BLB 共享约束：Q/K 投影必须共享 scaling factor —— 调用方需自行保证
        ``cfg.wq_encode == cfg.wk_encode``（本函数不强制校验，便于做 ablation）。

        前置条件：
          * 仅支持 BERT 家族（attention.self.{query,key,value} + output.LayerNorm）。
          * BSGS / merge hook 通过 ``BertSelfAttentionWithAproximation`` 的
            ``_block2_*_hook`` 机制注入，因此本方法**要求 attention 已经被
            ``replace_attention`` 替换为 ``BertSelfAttentionWithAproximation``**
            （否则 self-attention forward 不会调用我们的 hook）。

        Args:
            layer_indices: 要安装的层索引；None = 全部层
            layer_name: encoder layer list 的属性路径（与其他 replace_* 一致）
            cfg: ``Block2NoiseConfig``；None = 用 ``make_block2_default_config()``

        与 Block 1 的关系：
          * Block 1 / Block 2 共用同一个 ``layer.output.LayerNorm`` ──
            head 由 Block 1 cfg 控制，tail 由 Block 2 cfg 控制。
          * 若 Block 1 已先安装：复用现有 ``NoisyBlock1LayerNorm``，仅设置 ``cfg2``。
          * 若 Block 1 未安装：把原始 LayerNorm 包成 ``NoisyBlock1LayerNorm(cfg1=None, cfg2=cfg)``，
            等价于 LN head 用 clean 计算、tail 加 Block 2 噪声。

        与 legacy ``replace_layer_input_noise`` 的关系：
          * Block 2 视角下，X 进 Wq/Wk/Wv 之前**不再加 fresh**（X 的 PPTI 噪声
            来自 LN tail γ 乘法的 rescale）。本方法**不会**触发 legacy 的
            input-X fresh 噪声；如果你之前装过 ``replace_layer_input_noise``，
            建议先 ``restore_*`` 再 install Block 2，避免双重加噪。
        """
        if self._arch != "bert":
            raise NotImplementedError(
                "BLB Block 2 噪声目前仅支持 BERT 家族 (attention.self.{q,k,v} + output.LayerNorm)。"
            )

        if cfg is None:
            cfg = make_block2_default_config()

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        # BLB / legacy 互斥校验
        self._check_blb_legacy_conflict(selected, installing="blb")

        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            # ---- 1. LN tail：把 cfg2 设到 NoisyBlock1LayerNorm 上 ----
            current_ln = layer.output.LayerNorm
            if isinstance(current_ln, NoisyBlock1LayerNorm):
                # Block 1 已安装：直接 set cfg2
                current_ln.set_block2_cfg(cfg)
            else:
                # Block 1 未安装：把原 LN 包装为 NoisyBlock1LayerNorm(cfg=None, cfg2=cfg)
                if i not in self.original_block1_layernorm:
                    self.original_block1_layernorm[i] = current_ln
                source_ln = self.original_block1_layernorm[i]
                new_ln = NoisyBlock1LayerNorm(source_ln, cfg=None, cfg2=cfg)
                new_ln.train(source_ln.training)
                try:
                    ref_param = source_ln.weight
                    new_ln = new_ln.to(device=ref_param.device, dtype=ref_param.dtype)
                except Exception:
                    pass
                layer.output.LayerNorm = new_ln

            # ---- 2. Wq / Wk / Wv 投影：encode on W + 可选 rescale on result ----
            attn_self = layer.attention.self

            # 检查 attention.self 是否已被替换为带 hook 支持的近似版
            if not isinstance(attn_self, BertSelfAttentionWithAproximation):
                raise RuntimeError(
                    f"layer {i} 的 attention.self 不是 BertSelfAttentionWithAproximation，"
                    f"无法安装 Block 2 BSGS / merge hook。"
                    f"请先调用 replace_attention 把 self-attention 替换为近似版本。"
                )

            q_module = attn_self.query
            if i not in self.original_block2_qproj:
                self.original_block2_qproj[i] = q_module.forward
            q_module.forward = _make_block2_qk_proj_forward(
                q_module, cfg.wq_encode, cfg.wq_result_rescale,
                rotation_after_rescale=cfg.rotation_after_wq_rescale,
            )

            k_module = attn_self.key
            if i not in self.original_block2_kproj:
                self.original_block2_kproj[i] = k_module.forward
            k_module.forward = _make_block2_qk_proj_forward(
                k_module, cfg.wk_encode, cfg.wk_result_rescale,
                rotation_after_rescale=cfg.rotation_after_wk_rescale,
            )

            v_module = attn_self.value
            if i not in self.original_block2_vproj:
                self.original_block2_vproj[i] = v_module.forward
            v_module.forward = _make_block2_qk_proj_forward(
                v_module, cfg.wv_encode, cfg.wv_result_rescale,
                rotation_after_rescale=cfg.rotation_after_wv_rescale,
            )

            # ---- 3. Q / K^T BSGS hooks ----
            attn_self._block2_q_bsgs_hook = _make_block2_bsgs_mask_hook(
                cfg.q_mask1_encode, cfg.q_mask1_result_rescale,
                cfg.q_mask2_encode, cfg.q_mask2_result_rescale,
                rotation_after_mask1_rescale=cfg.rotation_after_q_mask1_rescale,
                rotation_after_mask2_rescale=cfg.rotation_after_q_mask2_rescale,
            )
            attn_self._block2_kt_bsgs_hook = _make_block2_bsgs_mask_hook(
                cfg.kt_mask1_encode, cfg.kt_mask1_result_rescale,
                cfg.kt_mask2_encode, cfg.kt_mask2_result_rescale,
                rotation_after_mask1_rescale=cfg.rotation_after_kt_mask1_rescale,
                rotation_after_mask2_rescale=cfg.rotation_after_kt_mask2_rescale,
            )

            # ---- 4. Q·K^T merge hook（含 Block 2 末尾 truncation 与 rotation #5） ----
            attn_self._block2_qkt_merge_hook = _make_block2_qkt_merge_hook(
                cfg.qkt_matmul_result_rescale,
                cfg.qkt_merge_mask_encode, cfg.qkt_merge_mask_result_rescale,
                output_truncation_k=cfg.output_truncation_k,
                output_truncation_mode=cfg.output_truncation_mode,
                rotation_after_qkt_matmul_rescale=cfg.rotation_after_qkt_matmul_rescale,
            )

            # ---- 5. 记录 cfg ----
            self.block2_cfg_per_layer[i] = cfg

        rescale_summary = (
            f"normalize={cfg.normalize_result_rescale.scaling_factor if cfg.normalize_result_rescale else 'off'}, "
            f"gamma={cfg.gamma_result_rescale.scaling_factor if cfg.gamma_result_rescale else 'off'}, "
            f"wk={cfg.wk_result_rescale.scaling_factor if cfg.wk_result_rescale else 'off'}, "
            f"wq={cfg.wq_result_rescale.scaling_factor if cfg.wq_result_rescale else 'off'}, "
            f"wv={cfg.wv_result_rescale.scaling_factor if cfg.wv_result_rescale else 'off'}, "
            f"qkt={cfg.qkt_matmul_result_rescale.scaling_factor if cfg.qkt_matmul_result_rescale else 'off'}, "
            f"merge_mask={cfg.qkt_merge_mask_result_rescale.scaling_factor if cfg.qkt_merge_mask_result_rescale else 'off'}"
        )
        _print_blb_install(
            f"已为 {len(selected)} 层启用 BLB Block 2 噪声 "
            f"(N={cfg.gamma_encode.N}, "
            f"fresh_inv_std={cfg.inv_std_fresh.scaling_factor}, "
            f"fresh_x_centered={cfg.x_centered_fresh.scaling_factor}, "
            f"encode_gamma={cfg.gamma_encode.scaling_factor}, "
            f"encode_wq/wk/wv={cfg.wq_encode.scaling_factor}/{cfg.wk_encode.scaling_factor}/{cfg.wv_encode.scaling_factor}, "
            f"encode_qkt_merge_mask={cfg.qkt_merge_mask_encode.scaling_factor}, "
            f"rescale=[{rescale_summary}])"
        )

    def restore_layer_block2_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            ):
        """恢复 Block 2 噪声安装前的 attention.self.{q,k,v}.forward / hook /
        LayerNorm.cfg2 状态。

        如果 Block 1 也安装了：LayerNorm 仍保持 NoisyBlock1LayerNorm（仅 cfg2=None）。
        如果 Block 1 没装：把 LayerNorm 还原为最初的 nn.LayerNorm。
        """
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            # ---- 1. 还原 attention.self.{q,k,v}.forward ----
            attn_self = layer.attention.self
            if i in self.original_block2_qproj:
                q_module = attn_self.query
                original_forward = self.original_block2_qproj[i]
                if getattr(original_forward, "__self__", None) is q_module:
                    q_module.forward = original_forward
                del self.original_block2_qproj[i]
            if i in self.original_block2_kproj:
                k_module = attn_self.key
                original_forward = self.original_block2_kproj[i]
                if getattr(original_forward, "__self__", None) is k_module:
                    k_module.forward = original_forward
                del self.original_block2_kproj[i]
            if i in self.original_block2_vproj:
                v_module = attn_self.value
                original_forward = self.original_block2_vproj[i]
                if getattr(original_forward, "__self__", None) is v_module:
                    v_module.forward = original_forward
                del self.original_block2_vproj[i]

            # ---- 2. 清除 hooks ----
            for hook_attr in (
                    "_block2_q_bsgs_hook",
                    "_block2_kt_bsgs_hook",
                    "_block2_qkt_merge_hook",
                    ):
                if hasattr(attn_self, hook_attr):
                    try:
                        delattr(attn_self, hook_attr)
                    except AttributeError:
                        setattr(attn_self, hook_attr, None)

            # ---- 3. LayerNorm tail：清除 cfg2 ----
            current_ln = layer.output.LayerNorm
            if isinstance(current_ln, NoisyBlock1LayerNorm):
                if i in self.block1_cfg_per_layer:
                    # Block 1 仍激活：保留 NoisyBlock1LayerNorm，只关闭 cfg2
                    current_ln.set_block2_cfg(None)
                else:
                    # Block 1 没装：把原 LN 还原回去
                    if i in self.original_block1_layernorm:
                        layer.output.LayerNorm = self.original_block1_layernorm[i]
                        del self.original_block1_layernorm[i]
                    else:
                        # 异常状态，至少把 cfg2 关掉
                        current_ln.set_block2_cfg(None)

            self.block2_cfg_per_layer.pop(i, None)

    # ------------------------------------------------------------------
    # BLB Block 3：softmax exp 多项式近似 (1 + x/2^n)^(2^n)
    # ------------------------------------------------------------------
    def replace_layer_block3_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            cfg: Optional[Block3NoiseConfig] = None,
            cfg_per_layer: Optional[dict] = None,
            ):
        """安装 BLB Block 3 (softmax exp 多项式) 噪声。

        Block 3 范围：``approximation_exponential`` 中的 (1 + x/2^n)^(2^n)
        多项式部分；不含 max-shift / lower_bound mask / norm_div（这些是非线性 / MPC）。

        Block 3 的 degree 是 per-attention 决定的（每层 attention 可能不同 degree），
        因此 cfg 的 degree 必须和该层 attention.degree 相符。提供两种调用：
          * ``cfg=Block3NoiseConfig(...)``：所有 selected 层共用同一个 cfg；
            要求每层 attention.degree == cfg.degree（不一致会报错）。
          * ``cfg_per_layer={i: Block3NoiseConfig, ...}``：逐层指定 cfg。
          * 都不传 → 用 ``make_block3_default_config(degree=attn.degree)`` 自动生成。

        Block 3 噪声点（degree=n 时）：
            1 fresh   on softmax 输入 x
            1 encode  on 1/2^n broadcast
            (n+1) optional rescale (1 on x·(1/2^n) + n on 每次平方)

        前置条件：每层 attention.self 必须已是 ``BertSelfAttentionWithAproximation``
        （需先调用 ``replace_layer_softmax`` 安装 softmax 近似）。
        """
        if self._arch != "bert":
            raise NotImplementedError(
                "BLB Block 3 噪声目前仅支持 BERT 家族（依赖 BertSelfAttentionWithAproximation.approximation_exponential）。"
            )
        if cfg is not None and cfg_per_layer is not None:
            raise ValueError("cfg 与 cfg_per_layer 互斥，二选一。")

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        # BLB / legacy 互斥校验
        self._check_blb_legacy_conflict(selected, installing="blb")

        installed_summary = []
        for i, layer in enumerate(layers):
            if i not in selected:
                continue
            attn_self = layer.attention.self
            if not isinstance(attn_self, BertSelfAttentionWithAproximation):
                raise RuntimeError(
                    f"layer {i} 的 attention.self 不是 BertSelfAttentionWithAproximation，"
                    f"无法安装 Block 3 噪声。请先 replace_layer_softmax 安装 softmax 近似。"
                )

            layer_degree = int(attn_self.degree)
            if cfg_per_layer is not None:
                if i not in cfg_per_layer:
                    raise ValueError(f"cfg_per_layer 缺少 layer {i} 的配置")
                this_cfg = cfg_per_layer[i]
            elif cfg is not None:
                this_cfg = cfg
            else:
                this_cfg = make_block3_default_config(degree=layer_degree)

            if int(this_cfg.degree) != layer_degree:
                raise ValueError(
                    f"layer {i} attention.degree={layer_degree}, "
                    f"但 Block3NoiseConfig.degree={this_cfg.degree} 不匹配"
                )

            # 用实例属性覆盖 class method approximation_exponential
            attn_self.approximation_exponential = _make_block3_approximation_exponential(this_cfg)
            self.block3_installed_layers.add(i)
            self.block3_cfg_per_layer[i] = this_cfg
            installed_summary.append((i, layer_degree))

        if installed_summary:
            sample_cfg = self.block3_cfg_per_layer[installed_summary[0][0]]
            sq_rs_summary = ",".join(
                str(rs.scaling_factor) if rs is not None else "off"
                for rs in sample_cfg.square_rescales
            )
            _print_blb_install(
                f"已为 {len(installed_summary)} 层启用 BLB Block 3 噪声 "
                f"(degree∈{{{','.join(str(d) for _, d in installed_summary)}}}, "
                f"N={sample_cfg.x_fresh.N}, "
                f"fresh_x={sample_cfg.x_fresh.scaling_factor}, "
                f"encode_inv2n={sample_cfg.inv_2n_encode.scaling_factor}, "
                f"rescale_x_inv2n={sample_cfg.x_inv_2n_result_rescale.scaling_factor if sample_cfg.x_inv_2n_result_rescale else 'off'}, "
                f"square_rescales=[{sq_rs_summary}])"
            )

    def restore_layer_block3_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            ):
        """恢复 ``approximation_exponential`` 为类的原始方法（删除实例属性覆盖）。"""
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        for i, layer in enumerate(layers):
            if i not in selected:
                continue
            if i in self.block3_installed_layers:
                attn_self = layer.attention.self
                # 删掉实例属性，让 class method 重新可见
                if "approximation_exponential" in attn_self.__dict__:
                    del attn_self.__dict__["approximation_exponential"]
                self.block3_installed_layers.discard(i)
            self.block3_cfg_per_layer.pop(i, None)

    # ------------------------------------------------------------------
    # BLB Block 4：softmax 输出 → softmax×V → Wo → post-attn LN head
    # ------------------------------------------------------------------
    def replace_layer_block4_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            cfg: Optional[Block4NoiseConfig] = None,
            ):
        """安装 BLB Block 4 噪声（16 个注入点）。

        Block 4 范围：
          (a) softmax 输出 P 上 fresh + ⊙ ones-mask + 可选 rescale
          (b) V 上 fresh + ⊙ ones-mask + 可选 rescale
          (c) softmax×V matmul + 合并 mask（rescale + ⊙ ones-mask + rescale）
          (d) Wo 投影：encode on W_o + 可选 rescale on Att
          (e) post-attn LN head（encode on 1/D ×2，rescale on μ/sq/var）

        共 2 fresh + 6 encode + 8 rescale。所有 σ² 走查表，N 默认 16384。

        前置条件：
          * 仅支持 BERT 家族；
          * attention.self 必须已是 ``BertSelfAttentionWithAproximation``
            （Block 4 hook 通过 ``_block4_*_hook`` 实例属性激活）。

        与 legacy ``replace_layer_softmax_value_noise`` 的关系：
          * 装 Block 4 后，BertSelfAttentionWithAproximation.forward 会 short-circuit
            掉 legacy ``_apply_softmax_value_noise``。restore Block 4 后回到 legacy 路径。

        与 legacy ``replace_layer_attention_output_noise`` 的关系：
          * Block 4 的 Wo wrap 是 legacy wo wrap 的严格扩展（多了 result rescale）。
            install Block 4 时会**覆盖**之前的 wo wrap；restore 时回到原始 forward。
        """
        if self._arch != "bert":
            raise NotImplementedError(
                "BLB Block 4 噪声目前仅支持 BERT 家族。"
            )
        if cfg is None:
            cfg = make_block4_default_config()

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        # BLB / legacy 互斥校验
        self._check_blb_legacy_conflict(selected, installing="blb")

        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            attn_self = layer.attention.self
            if not isinstance(attn_self, BertSelfAttentionWithAproximation):
                raise RuntimeError(
                    f"layer {i} 的 attention.self 不是 BertSelfAttentionWithAproximation，"
                    f"无法安装 Block 4 hook。请先 replace_layer_softmax 安装 softmax 近似。"
                )

            # ---- 1. softmax 输出 / V / softmax×V hooks ----
            attn_self._block4_softmax_out_hook = _make_block4_input_mask_hook(
                cfg.softmax_out_fresh, cfg.softmax_out_mask_encode, cfg.softmax_out_mask_rescale,
                rotation_after_mask_rescale=cfg.rotation_after_softmax_out_mask_rescale,
            )
            attn_self._block4_v_hook = _make_block4_input_mask_hook(
                cfg.v_fresh, cfg.v_mask_encode, cfg.v_mask_rescale,
                rotation_after_mask_rescale=cfg.rotation_after_v_mask_rescale,
            )
            attn_self._block4_softmax_v_hook = _make_block4_softmax_v_hook(
                cfg.softmax_v_matmul_rescale, cfg.softmax_v_mask_encode, cfg.softmax_v_mask_rescale,
                rotation_after_matmul_rescale=cfg.rotation_after_softmax_v_matmul_rescale,
                rotation_after_mask_rescale=cfg.rotation_after_softmax_v_mask_rescale,
            )

            # ---- 2. Wo 投影包装：encode on W_o + 可选 rescale on Att ----
            wo_module = layer.attention.output.dense
            stored_forward = self.original_block4_wo.get(i)
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not wo_module:
                self.original_block4_wo[i] = wo_module.forward
            wo_module.forward = _make_block4_wo_forward(
                wo_module, cfg.wo_encode, cfg.wo_result_rescale,
                rotation_after_rescale=cfg.rotation_after_wo_rescale,
            )

            # ---- 3. post-attn LN：替换为 NoisyBlock4LayerNorm ----
            current_ln = layer.attention.output.LayerNorm
            if isinstance(current_ln, NoisyBlock4LayerNorm):
                # 已经是 NoisyBlock4LayerNorm（重复装 / Block 5 已先装）：仅更新 cfg4
                current_ln.set_block4_cfg(cfg)
            else:
                if i not in self.original_block4_post_attn_ln:
                    self.original_block4_post_attn_ln[i] = current_ln
                source_ln = self.original_block4_post_attn_ln[i]
                new_ln = NoisyBlock4LayerNorm(source_ln, cfg4=cfg, cfg5=None)
                new_ln.train(source_ln.training)
                try:
                    ref_param = source_ln.weight
                    new_ln = new_ln.to(device=ref_param.device, dtype=ref_param.dtype)
                except Exception:
                    pass
                layer.attention.output.LayerNorm = new_ln

            self.block4_cfg_per_layer[i] = cfg

        rescale_summary = (
            f"sm_mask={cfg.softmax_out_mask_rescale.scaling_factor if cfg.softmax_out_mask_rescale else 'off'}, "
            f"v_mask={cfg.v_mask_rescale.scaling_factor if cfg.v_mask_rescale else 'off'}, "
            f"sm_v_matmul={cfg.softmax_v_matmul_rescale.scaling_factor if cfg.softmax_v_matmul_rescale else 'off'}, "
            f"sm_v_mask={cfg.softmax_v_mask_rescale.scaling_factor if cfg.softmax_v_mask_rescale else 'off'}, "
            f"wo={cfg.wo_result_rescale.scaling_factor if cfg.wo_result_rescale else 'off'}, "
            f"ln_mean={cfg.ln_mean_result_rescale.scaling_factor if cfg.ln_mean_result_rescale else 'off'}, "
            f"ln_sq={cfg.ln_square_result_rescale.scaling_factor if cfg.ln_square_result_rescale else 'off'}, "
            f"ln_var={cfg.ln_var_result_rescale.scaling_factor if cfg.ln_var_result_rescale else 'off'}"
        )
        _print_blb_install(
            f"已为 {len(selected)} 层启用 BLB Block 4 噪声 "
            f"(N={cfg.softmax_out_fresh.N}, "
            f"fresh_softmax/V={cfg.softmax_out_fresh.scaling_factor}/{cfg.v_fresh.scaling_factor}, "
            f"encode_masks(sm/V/sm_v)={cfg.softmax_out_mask_encode.scaling_factor}/{cfg.v_mask_encode.scaling_factor}/{cfg.softmax_v_mask_encode.scaling_factor}, "
            f"encode_wo={cfg.wo_encode.scaling_factor}, "
            f"encode_ln_inv_d={cfg.ln_mean_inv_d_encode.scaling_factor}/{cfg.ln_var_inv_d_encode.scaling_factor}, "
            f"rescale=[{rescale_summary}])"
        )

    def restore_layer_block4_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            ):
        """恢复 Block 4 安装前的 attention.self hook / Wo forward / post-attn LN 状态。

        如果 Block 5 也安装了：post-attn LN 仍保持 NoisyBlock4LayerNorm（仅 cfg4=None）。
        如果 Block 5 没装：把 LN 还原为最初的 nn.LayerNorm。
        """
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            # ---- 1. 清除 attention.self 上的 hooks ----
            attn_self = layer.attention.self
            for hook_attr in (
                    "_block4_softmax_out_hook",
                    "_block4_v_hook",
                    "_block4_softmax_v_hook",
                    ):
                if hasattr(attn_self, hook_attr):
                    try:
                        delattr(attn_self, hook_attr)
                    except AttributeError:
                        setattr(attn_self, hook_attr, None)

            # ---- 2. 还原 Wo forward ----
            if i in self.original_block4_wo:
                wo_module = layer.attention.output.dense
                original_forward = self.original_block4_wo[i]
                if getattr(original_forward, "__self__", None) is wo_module:
                    wo_module.forward = original_forward
                del self.original_block4_wo[i]

            # ---- 3. post-attn LN：清 cfg4 / 还原 ----
            current_ln = layer.attention.output.LayerNorm
            if isinstance(current_ln, NoisyBlock4LayerNorm):
                if current_ln.cfg5 is not None:
                    # Block 5 仍激活：保留 NoisyBlock4LayerNorm，只关 cfg4
                    current_ln.set_block4_cfg(None)
                else:
                    # Block 5 没装：把原 LN 还原回去
                    if i in self.original_block4_post_attn_ln:
                        layer.attention.output.LayerNorm = self.original_block4_post_attn_ln[i]
                        del self.original_block4_post_attn_ln[i]
                    else:
                        current_ln.set_block4_cfg(None)

            self.block4_cfg_per_layer.pop(i, None)

    # ------------------------------------------------------------------
    # BLB Block 5：post-attn LN tail + Wffn1 + GELU 多项式近似
    # ------------------------------------------------------------------
    def replace_layer_block5_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            cfg: Optional[Block5NoiseConfig] = None,
            cfg_per_layer: Optional[dict] = None,
            ):
        """安装 BLB Block 5 噪声。

        Block 5 范围：
          (a) post-attn LN tail：rsqrt 之后的 (1/std)·(X−μ) → γ scale
          (b) Wffn1：encode on W_ffn1 + 可选 rescale on result
          (c) GELU 多项式近似：power 计算 + 系数 encode + 系数乘法 rescale

        共 2 fresh + 2 encode + 3 rescale (LN tail + Wffn1) +
        1 encode + (degree-1) rescale (powers) + (degree) rescale (coeff muls) (GELU)。

        所有 σ² 走查表。N 默认按 GELU degree 选：degree=1 → 8192；degree∈{2,4} → 16384。

        前置条件：
          * 仅支持 BERT 家族；
          * GELU 必须已是 ``PolynomialGELU``（先调用 ``replace_layer_gelu`` 安装）；
            cfg.gelu_degree 必须 == 该层 PolynomialGELU.degree。
          * 与 Block 4 共享 ``attention.output.LayerNorm``：若 Block 4 已装，复用现有
            ``NoisyBlock4LayerNorm`` 仅设置 cfg5；否则把原 LN 包成 NoisyBlock4LayerNorm
            (cfg4=None, cfg5=cfg)。

        Args:
            layer_indices: 要安装的层索引；None = 全部层
            layer_name: encoder layer list 的属性路径
            cfg: 全部层共用一个 ``Block5NoiseConfig``；要求每层 GELU degree 相同
            cfg_per_layer: ``{layer_idx: Block5NoiseConfig}``；与 cfg 互斥
            （都不传 → 用 ``make_block5_default_config(gelu_degree=PolynomialGELU.degree)``）
        """
        if self._arch != "bert":
            raise NotImplementedError(
                "BLB Block 5 噪声目前仅支持 BERT 家族。"
            )
        if cfg is not None and cfg_per_layer is not None:
            raise ValueError("cfg 与 cfg_per_layer 互斥，二选一。")

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        # BLB / legacy 互斥校验
        self._check_blb_legacy_conflict(selected, installing="blb")

        installed_summary = []
        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            gelu_module = layer.intermediate.intermediate_act_fn
            if not isinstance(gelu_module, PolynomialGELU):
                raise RuntimeError(
                    f"layer {i} 的 intermediate.intermediate_act_fn 不是 PolynomialGELU，"
                    f"无法安装 Block 5 GELU 噪声。请先 replace_layer_gelu 安装多项式 GELU。"
                )
            layer_degree = int(gelu_module.degree)

            if cfg_per_layer is not None:
                if i not in cfg_per_layer:
                    raise ValueError(f"cfg_per_layer 缺少 layer {i} 的配置")
                this_cfg = cfg_per_layer[i]
            elif cfg is not None:
                this_cfg = cfg
            else:
                this_cfg = make_block5_default_config(gelu_degree=layer_degree)

            if int(this_cfg.gelu_degree) != layer_degree:
                raise ValueError(
                    f"layer {i} PolynomialGELU.degree={layer_degree}, "
                    f"但 Block5NoiseConfig.gelu_degree={this_cfg.gelu_degree} 不匹配"
                )

            # ---- 1. post-attn LN tail：把 cfg5 设到 NoisyBlock4LayerNorm 上 ----
            current_ln = layer.attention.output.LayerNorm
            if isinstance(current_ln, NoisyBlock4LayerNorm):
                current_ln.set_block5_cfg(this_cfg)
            else:
                if i not in self.original_block4_post_attn_ln:
                    self.original_block4_post_attn_ln[i] = current_ln
                source_ln = self.original_block4_post_attn_ln[i]
                new_ln = NoisyBlock4LayerNorm(source_ln, cfg4=None, cfg5=this_cfg)
                new_ln.train(source_ln.training)
                try:
                    ref_param = source_ln.weight
                    new_ln = new_ln.to(device=ref_param.device, dtype=ref_param.dtype)
                except Exception:
                    pass
                layer.attention.output.LayerNorm = new_ln

            # ---- 2. Wffn1 投影包装 ----
            wffn1_module = layer.intermediate.dense
            stored_forward = self.original_block5_wffn1.get(i)
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not wffn1_module:
                self.original_block5_wffn1[i] = wffn1_module.forward
            wffn1_module.forward = _make_block5_wffn1_forward(
                wffn1_module, this_cfg.wffn1_encode, this_cfg.wffn1_result_rescale,
                rotation_after_rescale=this_cfg.rotation_after_wffn1_rescale,
            )

            # ---- 3. GELU 多项式：替换 forward ----
            stored_gelu_forward = self.original_block5_gelu.get(i)
            if stored_gelu_forward is None or getattr(stored_gelu_forward, "__self__", None) is not gelu_module:
                self.original_block5_gelu[i] = gelu_module.forward
            gelu_module.forward = _make_block5_gelu_forward(gelu_module, this_cfg)

            self.block5_cfg_per_layer[i] = this_cfg
            installed_summary.append((i, layer_degree))

        if installed_summary:
            sample_cfg = self.block5_cfg_per_layer[installed_summary[0][0]]
            pwr_rs_summary = ",".join(
                str(rs.scaling_factor) if rs is not None else "off"
                for rs in sample_cfg.gelu_power_rescales
            ) or "(none)"
            coeff_rs_summary = ",".join(
                str(rs.scaling_factor) if rs is not None else "off"
                for rs in sample_cfg.gelu_coeff_mul_rescales
            )
            rescale_summary = (
                f"normalize={sample_cfg.normalize_result_rescale.scaling_factor if sample_cfg.normalize_result_rescale else 'off'}, "
                f"gamma={sample_cfg.gamma_result_rescale.scaling_factor if sample_cfg.gamma_result_rescale else 'off'}, "
                f"wffn1={sample_cfg.wffn1_result_rescale.scaling_factor if sample_cfg.wffn1_result_rescale else 'off'}, "
                f"gelu_powers=[{pwr_rs_summary}], "
                f"gelu_coeff_muls=[{coeff_rs_summary}]"
            )
            _print_blb_install(
                f"已为 {len(installed_summary)} 层启用 BLB Block 5 噪声 "
                f"(gelu_degree∈{{{','.join(str(d) for _, d in installed_summary)}}}, "
                f"N={sample_cfg.inv_std_fresh.N}, "
                f"fresh_inv_std/x_centered={sample_cfg.inv_std_fresh.scaling_factor}/{sample_cfg.x_centered_fresh.scaling_factor}, "
                f"encode_gamma/wffn1/gelu_coeff={sample_cfg.gamma_encode.scaling_factor}/{sample_cfg.wffn1_encode.scaling_factor}/{sample_cfg.gelu_coeff_encode.scaling_factor}, "
                f"rescale=[{rescale_summary}])"
            )

    def restore_layer_block5_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            ):
        """恢复 Block 5 安装前的 post-attn LN.cfg5 / Wffn1 forward / GELU forward 状态。

        如果 Block 4 也安装了：post-attn LN 仍保持 NoisyBlock4LayerNorm（仅 cfg5=None）。
        如果 Block 4 没装：把 LN 还原为最初的 nn.LayerNorm。
        """
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return

        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            # ---- 1. 还原 GELU forward ----
            if i in self.original_block5_gelu:
                gelu_module = layer.intermediate.intermediate_act_fn
                original_forward = self.original_block5_gelu[i]
                if getattr(original_forward, "__self__", None) is gelu_module:
                    gelu_module.forward = original_forward
                del self.original_block5_gelu[i]

            # ---- 2. 还原 Wffn1 forward ----
            if i in self.original_block5_wffn1:
                wffn1_module = layer.intermediate.dense
                original_forward = self.original_block5_wffn1[i]
                if getattr(original_forward, "__self__", None) is wffn1_module:
                    wffn1_module.forward = original_forward
                del self.original_block5_wffn1[i]

            # ---- 3. post-attn LN：清 cfg5 / 还原 ----
            current_ln = layer.attention.output.LayerNorm
            if isinstance(current_ln, NoisyBlock4LayerNorm):
                if current_ln.cfg4 is not None:
                    # Block 4 仍激活：保留 NoisyBlock4LayerNorm，只关 cfg5
                    current_ln.set_block5_cfg(None)
                else:
                    # Block 4 没装：把原 LN 还原回去
                    if i in self.original_block4_post_attn_ln:
                        layer.attention.output.LayerNorm = self.original_block4_post_attn_ln[i]
                        del self.original_block4_post_attn_ln[i]
                    else:
                        current_ln.set_block5_cfg(None)

            self.block5_cfg_per_layer.pop(i, None)

    # ------------------------------------------------------------------
    # BLB 首次输入 X 的 fresh 噪声安装 / 还原（**DEPRECATED**）
    # ------------------------------------------------------------------
    # 语义更新：BLB Stage-2 现在认为"第一个 HE 配置是无损的"——即 layer-0
    # input 端**不**注入 fresh 噪声。``BLBNoiseRLBridge.apply()`` 已经不再
    # 调用本方法；保留实现仅用于（1）旧 checkpoint resume 时清理残留 hook，
    # （2）实验性手工干预。新代码不要调用。
    def replace_blb_first_input_noise(
            self,
            scaling_factor: int,
            N: int = 8192,
            layer_indices=None,
            layer_name="model.model.layers",
            ):
        """[DEPRECATED] 在指定层（默认 layer 0）的 forward 入口注入 BLB-style fresh 噪声。

        BLB Block 2-5 是 transformer 各层之间循环的 block，覆盖"上一层 LN tail
        → Wq/Wk/Wv"路径噪声。但 layer 0 的 X 直接来自 embedding（没有上一层
        LN tail），所以缺一个对应位置的 fresh 噪声。本方法补上这块。

        Args:
            scaling_factor: scale_bits（``NOISE_VARIANCE_TABLE_BY_N`` 的 key）
            N: CKKS 多项式阶（默认 8192）
            layer_indices: 要安装的层；None 默认只装 layer 0
            layer_name: encoder layer 列表的属性路径

        与 legacy ``replace_layer_input_noise`` 的关系：
          * 二者**互斥**：legacy 用 ``INPUT_NOISE_VARIANCE_TABLE`` 单 N 表，
            BLB 这边用 ``NOISE_VARIANCE_TABLE_BY_N`` 多 N 表。
          * 同时安装时本方法会报错（参见 ``_check_blb_legacy_conflict``）。
        """
        if self._arch != "bert":
            raise NotImplementedError(
                "BLB first-input fresh 噪声目前仅支持 BERT 家族。"
            )
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = {0}  # 默认只装 layer 0
        else:
            selected = set(layer_indices)
            if not selected:
                return

        # 冲突检查：legacy input noise 与 BLB first-input 不可同时装
        for i in selected:
            if i in self.original_input_noise:
                raise RuntimeError(
                    f"layer {i} 已装 legacy ``replace_layer_input_noise``；"
                    f"BLB first-input 与 legacy input 互斥，请先 restore_layer_input_noise。"
                )

        point = NoisePoint("fresh", int(scaling_factor), int(N))
        for i, layer in enumerate(layers):
            if i not in selected:
                continue
            stored = self.blb_first_input_noise_state.get(i)
            if stored is None or getattr(stored.get("forward"), "__self__", None) is not layer:
                self.blb_first_input_noise_state[i] = {
                    "forward": layer.forward,
                    "point": point,
                }
            else:
                self.blb_first_input_noise_state[i]["point"] = point
            original_forward = self.blb_first_input_noise_state[i]["forward"]
            layer.forward = _make_blb_first_input_noise_forward(original_forward, point)

        _print_blb_install(
            f"已为 {len(selected)} 层启用 BLB first-input fresh 噪声 "
            f"(layers={sorted(selected)}, N={N}, scaling_factor={scaling_factor})"
        )

    def restore_blb_first_input_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            ):
        """恢复 BLB first-input fresh 噪声安装前的 layer.forward。"""
        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(self.blb_first_input_noise_state.keys())
        else:
            selected = set(layer_indices)
            if not selected:
                return

        for i, layer in enumerate(layers):
            if i not in selected:
                continue
            if i in self.blb_first_input_noise_state:
                original_forward = self.blb_first_input_noise_state[i]["forward"]
                if getattr(original_forward, "__self__", None) is layer:
                    layer.forward = original_forward
                del self.blb_first_input_noise_state[i]

    # ------------------------------------------------------------------
    # BLB / legacy 噪声冲突检查
    # ------------------------------------------------------------------
    def get_active_legacy_noise_layers(self) -> dict:
        """返回每种 legacy 噪声类型当前安装到了哪些层。

        返回 ``{noise_type: set(layer_idx)}``，noise_type ∈
        {input, query, key, value, wo, wffn1, wffn2, softmax_value}。
        """
        active = {
            "input": set(self.original_input_noise.keys()),
            "softmax_value": set(self.original_softmax_value_noise.keys()),
        }
        for proj_name, proj_dict in self.original_projection_noise.items():
            active[proj_name] = set(proj_dict.keys())
        return active

    def get_active_blb_noise_layers(self) -> dict:
        """返回每个 BLB block 当前安装到了哪些层。

        返回 ``{block_name: set(layer_idx)}``，block_name ∈
        {block1, block2, block3, block4, block5, first_input}。
        """
        return {
            "block1": set(self.block1_cfg_per_layer.keys()),
            "block2": set(self.block2_cfg_per_layer.keys()),
            "block3": set(self.block3_cfg_per_layer.keys()),
            "block4": set(self.block4_cfg_per_layer.keys()),
            "block5": set(self.block5_cfg_per_layer.keys()),
            "first_input": set(self.blb_first_input_noise_state.keys()),
        }

    def _check_blb_legacy_conflict(self, target_layers, *, installing: str):
        """安装 BLB 噪声前，确认目标层没有任何 legacy 噪声残留；反之亦然。

        Args:
            target_layers: 即将操作的 layer 索引集合
            installing: "blb" 表示在装 BLB（检查是否有 legacy 残留）；
                        "legacy" 表示在装 legacy（检查是否有 BLB 残留）。
                        二者互斥时抛 RuntimeError。
        """
        target = set(int(i) for i in target_layers)
        if not target:
            return

        if installing == "blb":
            legacy_active = self.get_active_legacy_noise_layers()
            conflicts = []
            for noise_type, layer_set in legacy_active.items():
                inter = layer_set & target
                if inter:
                    conflicts.append(f"legacy {noise_type}: layers {sorted(inter)}")
            if conflicts:
                raise RuntimeError(
                    "BLB 噪声与 legacy 噪声互斥，检测到 legacy 残留：\n  - "
                    + "\n  - ".join(conflicts)
                    + "\n请先调用对应的 restore_layer_*_noise 还原 legacy 噪声后再装 BLB。"
                )
        elif installing == "legacy":
            blb_active = self.get_active_blb_noise_layers()
            conflicts = []
            for block_name, layer_set in blb_active.items():
                inter = layer_set & target
                if inter:
                    conflicts.append(f"BLB {block_name}: layers {sorted(inter)}")
            if conflicts:
                raise RuntimeError(
                    "legacy 噪声与 BLB 噪声互斥，检测到 BLB 残留：\n  - "
                    + "\n  - ".join(conflicts)
                    + "\n请先调用对应的 restore_layer_block*_noise 还原 BLB 噪声后再装 legacy。"
                )
        else:
            raise ValueError(f"installing 必须是 'blb' 或 'legacy'，不能是 {installing!r}")

    def restore_all(self):
        """完全恢复原始模型状态"""
        self.model = copy.deepcopy(self.backup_model)
        self.original_gelu = {}
        self.original_attention = {}
        self.original_input_noise = {}
        self.original_projection_noise = {
            "query": {},
            "key": {},
            "value": {},
            "wo": {},
            "wffn1": {},
            "wffn2": {},
        }
        self.original_softmax_value_noise = {}
        self._gpt2_qkv_state = {}
        self._gpt2_qkv_wrapped = {}
        # Block 1 状态：deepcopy 已经把改动覆盖回原模型，这里只清状态字典
        self.original_block1_ffn2 = {}
        self.original_block1_layernorm = {}
        self.block1_cfg_per_layer = {}
        # Block 2 状态：同上
        self.original_block2_qproj = {}
        self.original_block2_kproj = {}
        self.original_block2_vproj = {}
        self.block2_cfg_per_layer = {}
        # Block 3 状态：同上
        self.block3_installed_layers = set()
        self.block3_cfg_per_layer = {}
        # Block 4 状态：同上
        self.original_block4_wo = {}
        self.original_block4_post_attn_ln = {}
        self.block4_cfg_per_layer = {}
        # Block 5 状态：同上
        self.original_block5_wffn1 = {}
        self.original_block5_gelu = {}
        self.block5_cfg_per_layer = {}
        # BLB first-input fresh 噪声状态：同上
        self.blb_first_input_noise_state = {}
        print("已完全恢复原始模型状态")
