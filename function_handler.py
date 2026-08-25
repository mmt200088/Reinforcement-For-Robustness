import math
import os as _os
import hashlib as _hashlib
import threading as _threading
from contextlib import contextmanager
from functools import lru_cache
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention
import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple
from torch import Tensor


_BLB_INSTALL_LOG_ENV = "BLB_NOISE_INSTALL_LOGS"


def _print_blb_install(message: str) -> None:
    raw = str(_os.environ.get(_BLB_INSTALL_LOG_ENV, "0")).strip().lower()
    if raw in ("0", "false", "no", "off", "quiet"):
        return
    print(message)


_NOISE_GENERATORS: dict = {}
_TRUNCATION_GENERATORS: dict = {}
_NOISE_RNG_SEED_MODE: str = "os"
_NOISE_RNG_FIXED_SEED: Optional[int] = None
_NOISE_RNG_LOCAL = _threading.local()


def _fresh_os_seed() -> int:
    """返回 64-bit 整数，从 OS 熵源派生（每次都不一样）。"""
    return int.from_bytes(_os.urandom(8), "little")


def _noise_generator_key(device, scope: Optional[str] = None) -> str:
    key = str(device)
    if scope is None:
        scope = getattr(_NOISE_RNG_LOCAL, "scope", None)
    if scope is not None:
        key = f"{key}|scope={str(scope)}"
    return key


@contextmanager
def noise_rng_scope(scope: Optional[str]):
    """Temporarily route noise samples on this thread to a scoped generator.

    The generated values are still controlled only by the manual seed. The
    scope just gives same-device workers separate ``torch.Generator`` objects,
    so one worker can reseed its deterministic probe stream without racing a
    sibling on the same CUDA device.
    """
    if scope is None:
        yield
        return
    sentinel = object()
    previous = getattr(_NOISE_RNG_LOCAL, "scope", sentinel)
    _NOISE_RNG_LOCAL.scope = str(scope)
    try:
        yield
    finally:
        if previous is sentinel:
            try:
                delattr(_NOISE_RNG_LOCAL, "scope")
            except AttributeError:
                pass
        else:
            _NOISE_RNG_LOCAL.scope = previous


def _get_noise_generator(device) -> torch.Generator:
    """返回一个针对 ``device`` 的独立 ``torch.Generator``。

    - 第一次访问某 device 时新建并用 OS 熵 seed（``_NOISE_RNG_SEED_MODE='os'``）
      或固定 seed（``='fixed'``）填充。
    - 后续直接复用已有 generator。
    - 与 ``torch.default_generator`` 完全隔离，不被 ``torch.manual_seed`` 影响。
    """
    key = _noise_generator_key(device)
    g = _NOISE_GENERATORS.get(key)
    if g is None:
        g = torch.Generator(device=device)
        if _NOISE_RNG_SEED_MODE == "fixed" and _NOISE_RNG_FIXED_SEED is not None:
            g.manual_seed(int(_NOISE_RNG_FIXED_SEED))
        else:
            g.manual_seed(_fresh_os_seed())
        _NOISE_GENERATORS[key] = g
    return g


def _derive_truncation_seed(seed: int) -> int:
    payload = f"blb-truncation-v1:{int(seed)}".encode("ascii")
    return int.from_bytes(_hashlib.sha256(payload).digest()[:8], "little")


def _get_truncation_generator(device) -> torch.Generator:
    """Return the truncation RNG without consuming Gaussian noise state."""
    key = _noise_generator_key(device)
    generator = _TRUNCATION_GENERATORS.get(key)
    if generator is None:
        generator = torch.Generator(device=device)
        if _NOISE_RNG_SEED_MODE == "fixed" and _NOISE_RNG_FIXED_SEED is not None:
            generator.manual_seed(_derive_truncation_seed(_NOISE_RNG_FIXED_SEED))
        else:
            generator.manual_seed(_fresh_os_seed())
        _TRUNCATION_GENERATORS[key] = generator
    return generator


def _sample_independent_gaussian(reference: Tensor, std: float) -> Tensor:
    """从独立噪声 RNG 采样与 ``reference`` 同形状的 N(0, std²) 张量。

    生成的噪声 device/dtype 与 reference 一致；不消耗 torch 全局 RNG 状态。
    """
    if std <= 0.0:
        return torch.zeros_like(reference)
    gen = _get_noise_generator(reference.device)
    return torch.empty_like(reference).normal_(0.0, float(std), generator=gen)


_BINARY_TRUNCATION_FUSED_CUDA_ENABLED = str(
    _os.environ.get("BLB_STAGE2_TRUNCATION_FUSED_CUDA", "1")
).strip().lower() not in {"0", "false", "no", "off"}
_BINARY_TRUNCATION_FUSED_CUDA_IMPL = None
_BINARY_TRUNCATION_FUSED_CUDA_RESOLVED = False


def _resolve_binary_truncation_fused_cuda_impl():
    global _BINARY_TRUNCATION_FUSED_CUDA_IMPL
    global _BINARY_TRUNCATION_FUSED_CUDA_RESOLVED
    if not _BINARY_TRUNCATION_FUSED_CUDA_RESOLVED:
        _BINARY_TRUNCATION_FUSED_CUDA_RESOLVED = True
        try:
            from blb_stage2_rl.truncation_fused_cuda import (
                binary_truncation_cuda,
                is_available,
            )

            if is_available():
                _BINARY_TRUNCATION_FUSED_CUDA_IMPL = binary_truncation_cuda
        except (ImportError, ModuleNotFoundError):
            _BINARY_TRUNCATION_FUSED_CUDA_IMPL = None
    return _BINARY_TRUNCATION_FUSED_CUDA_IMPL


def _binary_truncation_fused_cuda_target_k(
        x: Tensor,
        k: int,
        ) -> Optional[int]:
    """Return the supported exact-CUDA K, otherwise ``None``."""
    target_k = int(k)
    if (
            not _BINARY_TRUNCATION_FUSED_CUDA_ENABLED
            or target_k < 6
            or target_k > 13
            or not x.is_cuda
            or x.dtype != torch.float32
            or x.requires_grad
            or not x.is_contiguous()
            or int(x.numel()) == 0
    ):
        return None
    return target_k


def _try_binary_truncation_fused_cuda(
        x: Tensor,
        k: int,
        ) -> Optional[Tensor]:
    """Run the exact CUDA specialization for the active K domain."""
    target_k = _binary_truncation_fused_cuda_target_k(x, k)
    if target_k is None:
        return None
    implementation = _resolve_binary_truncation_fused_cuda_impl()
    if implementation is None:
        return None
    return implementation(x, target_k)


def _configured_binary_truncation_fused_cuda_scale(
        x: Tensor,
        cfg,
        ) -> Optional[float]:
    """Return a fusion scale only when the materialized cfg is eligible."""
    if cfg is None:
        return None
    mode = str(getattr(cfg, "output_truncation_mode", "binary")).strip().lower()
    k = getattr(cfg, "output_truncation_k", None)
    if mode != "binary" or k is None:
        return None
    target_k = _binary_truncation_fused_cuda_target_k(x, int(k))
    if target_k is None:
        return None
    return float(2 ** target_k)


def _apply_truncation(
        x: Tensor,
        k: Optional[int],
        mode: str = "binary",
        *,
        ring_bits: int = 43,
        source_fractional_bits: int = 24,
        ) -> Tensor:
    """Apply the selected plaintext truncation simulation to ``x``.

    - ``k is None``：no-op，原样返回。
    - mode="binary"：``trunc(x · 2^k) / 2^k``（PPTI / CKKS 默认）
    - mode="decimal"：``trunc(x · 10^k) / 10^k``（普通"保留 k 位小数"）
    - mode="stochastic_ring"：signed ring encode + probabilistic right shift；
      仅作协议数值语义近似，不实现秘密共享或安全通信。
    """
    if k is None:
        return x
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "binary":
        fused = _try_binary_truncation_fused_cuda(x, int(k))
        if fused is not None:
            return fused
        scale = 2.0 ** int(k)
        return torch.trunc(x * scale) / scale
    if normalized_mode == "decimal":
        scale = 10.0 ** int(k)
        return torch.trunc(x * scale) / scale
    if normalized_mode != "stochastic_ring":
        raise ValueError(f"unsupported truncation mode: {mode!r}")

    ring = int(ring_bits)
    source_bits = int(source_fractional_bits)
    target_bits = int(k)
    if not 2 <= ring <= 62:
        raise ValueError("ring_bits must be in [2, 62]")
    if source_bits < 0 or source_bits >= ring:
        raise ValueError(
            "source_fractional_bits must be non-negative and smaller than ring_bits"
        )
    if target_bits < 0 or target_bits > source_bits:
        raise ValueError(
            "source_fractional_bits must be non-negative and >= target k"
        )

    source_scale = 1 << source_bits
    modulus = 1 << ring
    half_modulus = 1 << (ring - 1)
    encoded = torch.round(x.to(dtype=torch.float64) * float(source_scale)).to(torch.int64)
    wrapped = torch.remainder(encoded, modulus)
    signed = torch.where(wrapped >= half_modulus, wrapped - modulus, wrapped)

    shift = source_bits - target_bits
    if shift == 0:
        rounded = signed
    else:
        divisor = 1 << shift
        quotient = torch.div(signed, divisor, rounding_mode="floor")
        remainder = signed - quotient * divisor
        probability = remainder.to(torch.float64) / float(divisor)
        draws = torch.empty_like(probability).uniform_(
            0.0, 1.0, generator=_get_truncation_generator(x.device),
        )
        rounded = quotient + (draws < probability).to(torch.int64)
    decoded = rounded.to(torch.float64) / float(1 << target_bits)
    return decoded.to(dtype=x.dtype)


def _apply_configured_truncation(
        x: Tensor,
        cfg,
        *,
        binary_already_applied: bool = False,
        ) -> Tensor:
    """Apply the backend carried by one final, materialized block config."""
    if binary_already_applied:
        return x
    if cfg is None:
        return x
    return _apply_truncation(
        x,
        getattr(cfg, "output_truncation_k", None),
        getattr(cfg, "output_truncation_mode", "binary"),
        ring_bits=int(getattr(cfg, "output_truncation_ring_bits", 43)),
        source_fractional_bits=int(
            getattr(cfg, "output_truncation_source_fractional_bits", 24)
        ),
    )


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
        for g in _TRUNCATION_GENERATORS.values():
            g.manual_seed(_fresh_os_seed())
    else:
        _NOISE_RNG_SEED_MODE = "fixed"
        _NOISE_RNG_FIXED_SEED = int(seed)
        for g in _NOISE_GENERATORS.values():
            g.manual_seed(int(seed))
        truncation_seed = _derive_truncation_seed(int(seed))
        for g in _TRUNCATION_GENERATORS.values():
            g.manual_seed(truncation_seed)


def reseed_noise_rng_for_device(
        device,
        seed: int,
        scope: Optional[str] = None,
        ) -> None:
    """只重播种 ``device`` 自己的噪声 generator（不动其它 device、不动全局模式）。

    Stage-2 确定性 probe 在每个 trial 开始前把本卡的噪声流定到
    ``(run_seed, global_episode, trial)`` 派生的种子上：
    CUDA Philox 与设备无关，同一种子在任何卡上产生同一噪声序列，因此 1 卡
    与 N 卡（以及任何 trial→卡 的调度）逐位一致。worker 线程各自持有不同
    device，只触碰自己的 generator —— 与并发的其它 worker 无竞态。
    """
    key = _noise_generator_key(device, scope)
    g = _NOISE_GENERATORS.get(key)
    if g is None:
        g = torch.Generator(device=device)
        _NOISE_GENERATORS[key] = g
    g.manual_seed(int(seed))
    truncation_generator = _TRUNCATION_GENERATORS.get(key)
    if truncation_generator is None:
        truncation_generator = torch.Generator(device=device)
        _TRUNCATION_GENERATORS[key] = truncation_generator
    truncation_generator.manual_seed(_derive_truncation_seed(int(seed)))


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


GELU_COEEF = {

            0: [[-0.20266642, 1.07484643], [-0.20266642, -0.57484643+0.5]],


            1: [[-0.20266642, 1.07484643], [-0.20266642, -0.57484643+0.5]],
            2: [[-0.12136484, 0.94386247, 0.04261206],[-0.12136484, -0.44386247+0.5, 0.04261206]],


            3: [[-0.01524885, 0.57426473, 0.35500657, -0.07415983], [-0.01524885, -0.07426473+0.5, 0.35500657, 0.07415983]],
            4: [[0.00746413, -0.07087454+0.5, 0.58960402, -0.20949432, 0.02540485], [ 0.00746413, 0.07087454+0.5, 0.58960402, 0.20949432, 0.02540485]]

}


SiLU_COEEF = {

            1: [[0.14238437510901367, 0.5000053621970405, 0.12920887677506931],[-0.10118073891975127,-0.013543261873265973]],
            2: [[0.14238437510901367, 0.5000053621970405, 0.12920887677506931],[-0.2932427892002413,-0.07801652478737445,-0.005269243960262952]],
            3: [[0.14241236482342567, 0.4999863582405589, 0.12920235286785606, 0],[-0.4233567569791515,-0.14755599495248886,-0.017365847597972207,-0.0006859293250386277]],
            4: [[0.03284668051202981,0.5000000914210826,0.19746490458050728,0,-0.005281681095454781],[-0.49057828462086733,-0.02757518199120323,0.05336178194846048,0.011409101768158705,0.0006606624719387583]]
}


Exp_bound = {
    1:-2,
    2:-4,
    3:-8,
    4:-12,
    5:-13,
    6:-13
}


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


NOISE_VARIANCE_TABLE_BY_N = {
    _N: {
        _sb: {
            "encoding": _stds[0] ** 2,
            "fresh":    _stds[1] ** 2,
            "rescale":  _stds[2] ** 2,


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


        if scaling_factor > max(table):
            return 0.0
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


@dataclass
class NoisePoint:
    """单个噪声注入点的参数三元组：(distribution, scaling_factor, N)。

    实际 σ² 由 ``get_input_noise_variance_by_N(scaling_factor, distribution, N)``
    查 ``NOISE_VARIANCE_TABLE_BY_N`` 得到，不写死。
    """
    distribution: str
    scaling_factor: int
    N: int = 8192


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


    noise_enabled: bool = True


    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    output_truncation_ring_bits: int = 43
    output_truncation_source_fractional_bits: int = 24


    rotation_after_gelu_out_fresh: bool = False
    rotation_after_wffn2_rescale_a: bool = False
    rotation_after_wffn2_rescale_b: bool = False
    rotation_after_square_rescale: bool = False
    rotation_repeat_counts: dict = field(default_factory=dict)


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
        noise_enabled: bool = True,
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
    None ⇒ 不截断。``noise_enabled=False`` 保留此截断但关闭全部 Block 1
    Gaussian/rotation 噪声，用于 layer 0 的 K-only 配置。
    """
    cfg = Block1NoiseConfig(
        gelu_out_fresh=NoisePoint("fresh", int(gelu_out_sf), int(N)),
        wffn2_encode=NoisePoint("encoding", int(wffn2_sf), int(N)),
        mean_inv_d_encode=NoisePoint("encoding", int(mean_inv_d_sf), int(N)),
        var_inv_d_encode=NoisePoint("encoding", int(var_inv_d_sf), int(N)),
        noise_enabled=bool(noise_enabled),
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


def _make_rotation_point(source: Optional[NoisePoint]) -> Optional[NoisePoint]:
    """把绑定的 fresh/rescale NoisePoint 转成 rotation NoisePoint。

    - source=None → None（前置 rescale 没启用，rotation 也无 SF 可继承）
    - 否则返回 NoisePoint("rotation", source.scaling_factor, source.N)
    """
    if source is None:
        return None
    return NoisePoint("rotation", int(source.scaling_factor), int(source.N))


@lru_cache(maxsize=256)
def _noise_std_for_values(
        distribution: str,
        scaling_factor: int,
        N: int,
        ) -> float:
    """Resolve one immutable noise-table standard deviation."""
    variance = get_input_noise_variance_by_N(
        scaling_factor=int(scaling_factor),
        distribution=str(distribution).lower(),
        N=int(N),
    )
    return math.sqrt(variance) if variance > 0.0 else 0.0


def _sample_gaussian_for_point(reference: Tensor, point: Optional[NoisePoint]) -> Tensor:
    """根据 NoisePoint 的 (distribution, scaling_factor, N) 三元组，
    返回与 ``reference`` 同形状（同 device/dtype）的 N(0, σ²) 噪声张量。

    - ``point=None``：返回 0（用于 rescale 关闭时的统一处理）。
    - 走的是 ``NOISE_VARIANCE_TABLE_BY_N`` 多 N 表；σ² 严禁写死。
    """
    if point is None:
        return torch.zeros_like(reference)
    std = _noise_std_for_values(
        str(point.distribution).lower(),
        int(point.scaling_factor),
        int(point.N),
    )
    if std <= 0.0:
        return torch.zeros_like(reference)
    return _sample_independent_gaussian(reference, std)


_BLB_INFERENCE_NOISE_ADD_ENABLED = str(
    _os.environ.get("BLB_STAGE2_INFERENCE_NOISE_ADD", "1")
).strip().lower() not in {"0", "false", "no", "off"}


def _sample_and_add_gaussian_for_point(
        reference: Tensor,
        point: Optional[NoisePoint],
        ) -> Tensor:
    """Add one noise sample while reusing its storage during inference."""
    noise = _sample_gaussian_for_point(reference, point)
    if (
            _BLB_INFERENCE_NOISE_ADD_ENABLED
            and torch.is_inference_mode_enabled()
    ):
        torch.add(reference, noise, out=noise)
        return noise
    return reference + noise


def _rotation_repeat_count(cfg, flag_name: str) -> int:
    """Resolve one installed rotation flag to its optimizer-provided count."""
    if cfg is None or not bool(getattr(cfg, flag_name, False)):
        return 0
    counts = getattr(cfg, "rotation_repeat_counts", {}) or {}
    raw_count = counts.get(flag_name, 1) if hasattr(counts, "get") else 1
    if isinstance(raw_count, bool):
        raw_count = int(raw_count)
    count = int(raw_count)
    if count <= 0:
        raise ValueError(
            f"enabled rotation flag {flag_name!r} has invalid count {raw_count!r}"
        )
    return count


def _apply_rotation_noise(
        value: Tensor,
        source: Optional[NoisePoint],
        repeat_count: int = 1,
        ) -> Tensor:
    """Apply one independent Gaussian draw for every effective rotation."""
    count = int(repeat_count)
    if count < 0:
        raise ValueError(f"rotation repeat_count must be non-negative, got {count}")
    if source is None or count == 0:
        return value
    point = _make_rotation_point(source)
    for _ in range(count):
        value = _sample_and_add_gaussian_for_point(value, point)
    return value


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

    inv_std_fresh: NoisePoint
    x_centered_fresh: NoisePoint

    gamma_encode: NoisePoint

    wk_encode: NoisePoint
    kt_mask1_encode: NoisePoint
    kt_mask2_encode: NoisePoint

    wq_encode: NoisePoint
    q_mask1_encode: NoisePoint
    q_mask2_encode: NoisePoint

    wv_encode: NoisePoint

    qkt_merge_mask_encode: NoisePoint


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


    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    output_truncation_ring_bits: int = 43
    output_truncation_source_fractional_bits: int = 24


    rotation_after_gamma_rescale: bool = False
    rotation_after_wq_rescale: bool = False
    rotation_after_wk_rescale: bool = False
    rotation_after_wv_rescale: bool = False
    rotation_after_q_mask1_rescale: bool = False
    rotation_after_kt_mask1_rescale: bool = False
    rotation_after_q_mask2_rescale: bool = False
    rotation_after_kt_mask2_rescale: bool = False
    rotation_after_qkt_matmul_rescale: bool = False
    rotation_repeat_counts: dict = field(default_factory=dict)


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

    BLB invariant: q_proj and k_proj share one scaling factor,
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
        if not bool(getattr(cfg, "noise_enabled", True)):
            return nn.functional.linear(
                hidden_states,
                linear_module.weight,
                linear_module.bias,
            )

        x = _sample_and_add_gaussian_for_point(
            hidden_states, cfg.gelu_out_fresh,
        )

        x = _apply_rotation_noise(
            x,
            cfg.gelu_out_fresh,
            _rotation_repeat_count(cfg, "rotation_after_gelu_out_fresh"),
        )

        weight = linear_module.weight
        noisy_weight = _sample_and_add_gaussian_for_point(
            weight, cfg.wffn2_encode,
        )
        noisy_weight = noisy_weight.to(device=x.device, dtype=x.dtype)
        bias = linear_module.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)

        out = nn.functional.linear(x, noisy_weight, bias)

        if cfg.wffn2_result_rescale is not None:
            out = _sample_and_add_gaussian_for_point(
                out, cfg.wffn2_result_rescale,
            )

            out = _apply_rotation_noise(
                out,
                cfg.wffn2_result_rescale,
                _rotation_repeat_count(cfg, "rotation_after_wffn2_rescale_a"),
            )

            out = _apply_rotation_noise(
                out,
                cfg.wffn2_result_rescale,
                _rotation_repeat_count(cfg, "rotation_after_wffn2_rescale_b"),
            )
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

        self.weight = original_ln.weight
        self.bias = original_ln.bias
        self.eps = float(original_ln.eps)
        self.normalized_shape = tuple(original_ln.normalized_shape)
        self.cfg = cfg
        self.cfg2 = cfg2

    def set_block2_cfg(self, cfg2: Optional["Block2NoiseConfig"]) -> None:
        """安装 / 覆盖 / 关闭（None）Block 2 LN-tail 噪声。"""
        self.cfg2 = cfg2

    def forward(self, x: Tensor) -> Tensor:
        D = int(x.shape[-1])
        cfg = self.cfg
        cfg2 = self.cfg2
        noise_enabled = (
            cfg is not None and bool(getattr(cfg, "noise_enabled", True))
        )


        sum_x = x.sum(dim=-1, keepdim=True)
        if noise_enabled:


            noisy_inv_d = _sample_gaussian_for_point(x, cfg.mean_inv_d_encode)
            noisy_inv_d.add_(1.0 / D)
            mean = sum_x * noisy_inv_d
            if cfg.mean_result_rescale is not None:
                mean = _sample_and_add_gaussian_for_point(
                    mean, cfg.mean_result_rescale,
                )
        else:

            mean = sum_x / float(D)


        x_centered = x - mean


        sq = x_centered * x_centered
        if noise_enabled and cfg.square_result_rescale is not None:
            sq = _sample_and_add_gaussian_for_point(
                sq, cfg.square_result_rescale,
            )

            sq = _apply_rotation_noise(
                sq,
                cfg.square_result_rescale,
                _rotation_repeat_count(cfg, "rotation_after_square_rescale"),
            )


        sum_sq = sq.sum(dim=-1, keepdim=True)
        if noise_enabled:
            noisy_inv_d_var = _sample_gaussian_for_point(sq, cfg.var_inv_d_encode)
            noisy_inv_d_var.add_(1.0 / D)
            var = sum_sq * noisy_inv_d_var
            if cfg.var_result_rescale is not None:
                var = _sample_and_add_gaussian_for_point(
                    var, cfg.var_result_rescale,
                )
        else:
            var = sum_sq / float(D)


        if cfg is not None:
            var = _apply_configured_truncation(var, cfg)


        inv_std = torch.rsqrt(var + self.eps)


        if cfg2 is not None:


            if inv_std.shape != x.shape:
                inv_std = inv_std.expand_as(x).contiguous()
            noisy_inv_std = _sample_and_add_gaussian_for_point(
                inv_std, cfg2.inv_std_fresh,
            )
            noisy_x_centered = _sample_and_add_gaussian_for_point(
                x_centered, cfg2.x_centered_fresh,
            )
            normalized = noisy_x_centered * noisy_inv_std
            if cfg2.normalize_result_rescale is not None:
                normalized = _sample_and_add_gaussian_for_point(
                    normalized, cfg2.normalize_result_rescale,
                )


            gamma_broadcast = self.weight.expand_as(normalized)
            noisy_gamma = _sample_and_add_gaussian_for_point(
                gamma_broadcast, cfg2.gamma_encode,
            )
            gamma_mul = normalized * noisy_gamma
            if cfg2.gamma_result_rescale is not None:
                gamma_mul = _sample_and_add_gaussian_for_point(
                    gamma_mul, cfg2.gamma_result_rescale,
                )

                gamma_mul = _apply_rotation_noise(
                    gamma_mul,
                    cfg2.gamma_result_rescale,
                    _rotation_repeat_count(cfg2, "rotation_after_gamma_rescale"),
                )

            out = gamma_mul + self.bias
        else:

            normalized = x_centered * inv_std
            out = normalized * self.weight + self.bias
        return out


def _make_block2_qk_proj_forward(
        linear_module: nn.Linear,
        encode_point: NoisePoint,
        rescale_point: Optional[NoisePoint],
        rotation_after_rescale: int = 0,
        ):
    """Wq / Wk / Wv 投影包装：encode on W (matmulcp 操作数侧) + 可选 rescale on result
    + 可选 rotation 噪声（紧跟 rescale 之后；SF 继承自 rescale_point）。

    与 Block 1 的 ``_make_block1_ffn2_forward`` 同方式（与现有 ``replace_layer_*_noise``
    通过 ``_make_noisy_linear_forward`` 加 W 噪声的 PPTI 语义一致），但额外支持
    在 ``X · W`` 之后加 rescale 噪声（cfg.*_result_rescale 控制是否加）。

    This path does not add fresh noise to X. Block 2 receives X noise
    噪声在 LN tail γ 乘法的 rescale 处注入；Block 2 不再额外添加 input-X
    fresh noise。
    """
    def block2_qk_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        weight = linear_module.weight
        noisy_weight = _sample_and_add_gaussian_for_point(
            weight, encode_point,
        )
        noisy_weight = noisy_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = linear_module.bias
        if bias is not None:
            bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        out = nn.functional.linear(hidden_states, noisy_weight, bias)
        if rescale_point is not None:
            out = _sample_and_add_gaussian_for_point(out, rescale_point)

            out = _apply_rotation_noise(
                out, rescale_point, rotation_after_rescale,
            )
        return out
    return block2_qk_forward


def _make_block2_qkt_merge_hook(
        qkt_matmul_rescale: Optional[NoisePoint],
        merge_mask_encode: NoisePoint,
        merge_mask_rescale: Optional[NoisePoint],
        truncation_cfg: Optional[Block2NoiseConfig] = None,
        rotation_after_qkt_matmul_rescale: int = 0,
        ):
    """构造 Q·K^T matmul **之后**、softmax **之前**的 "合并 Q,K" 噪声 hook。

    顺序：
        1. rescale on Q·K^T matmul 结果        (qkt_matmul_rescale, 可选)
        1b. rotation 紧跟 #1 之后                (rotation_after_qkt_matmul_rescale, 可选；
                                                  SF 继承自 qkt_matmul_rescale)
        2. ⊙ ones-mask: noisy_ones = 1 + ε_enc; out = qkt_result · noisy_ones
        3. rescale on mask 乘法结果              (merge_mask_rescale, 可选)
        4. PPTI MPC↔HE 截断（由 materialized Block2 cfg 选择后端与 K）

    返回 ``hook(attention_scores) -> attention_scores`` 形状 [B, A, S, S]。
    """
    def hook(qkt_result: Tensor) -> Tensor:

        if qkt_matmul_rescale is not None:
            qkt_result = _sample_and_add_gaussian_for_point(
                qkt_result, qkt_matmul_rescale,
            )

            qkt_result = _apply_rotation_noise(
                qkt_result,
                qkt_matmul_rescale,
                rotation_after_qkt_matmul_rescale,
            )

        noisy_mask = _sample_gaussian_for_point(qkt_result, merge_mask_encode)
        noisy_mask.add_(1.0)
        out = qkt_result * noisy_mask

        if merge_mask_rescale is not None:
            out = _sample_and_add_gaussian_for_point(
                out, merge_mask_rescale,
            )

        out = _apply_configured_truncation(out, truncation_cfg)
        return out
    return hook


@dataclass
class Block3NoiseConfig:
    """BLB Block 3 噪声配置：softmax exp 多项式近似 (1 + x/2^n)^(2^n)"""
    degree: int
    x_fresh: NoisePoint
    inv_2n_encode: NoisePoint

    x_inv_2n_result_rescale: Optional[NoisePoint] = None

    square_rescales: Tuple[Optional[NoisePoint], ...] = field(default_factory=tuple)

    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    output_truncation_ring_bits: int = 43
    output_truncation_source_fractional_bits: int = 24


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


_BLOCK3_FUSED_CUDA_ENABLED = str(
    _os.environ.get("BLB_STAGE2_BLOCK3_FUSED_CUDA", "1")
).strip().lower() not in {"0", "false", "no", "off"}
_BLOCK3_FUSED_CUDA_IMPLS = {}
_BLOCK3_FUSED_CUDA_RESOLVED = False
_BLOCK3_FUSED_CUDA_WORKSPACES = {}


def _get_block3_fused_cuda_noise_workspace(
        x: Tensor,
        point_count: int,
        ) -> Tensor:
    """Reuse noise storage for sequential Block3 calls on one thread/stream."""
    count = int(point_count)
    if count <= 0:
        raise ValueError(f"point_count must be positive, got {count}")
    stream = torch.cuda.current_stream(x.device)
    key = (
        int(_threading.get_ident()),
        str(x.device),
        int(stream.cuda_stream),
        x.dtype,
    )
    required = count * int(x.numel())
    workspace = _BLOCK3_FUSED_CUDA_WORKSPACES.get(key)
    if workspace is None or int(workspace.numel()) < required:
        workspace = torch.empty(
            required,
            device=x.device,
            dtype=x.dtype,
        )
        _BLOCK3_FUSED_CUDA_WORKSPACES[key] = workspace
    return workspace[:required].view(count, *x.shape)


def _resolve_block3_fused_cuda_impl(degree: int):
    global _BLOCK3_FUSED_CUDA_IMPLS, _BLOCK3_FUSED_CUDA_RESOLVED
    if not _BLOCK3_FUSED_CUDA_RESOLVED:
        _BLOCK3_FUSED_CUDA_RESOLVED = True
        try:
            from blb_stage2_rl.block3_fused_cuda import (
                block3_degree4_cuda,
                block3_degree6_cuda,
                is_available,
            )

            if is_available():
                _BLOCK3_FUSED_CUDA_IMPLS = {
                    4: block3_degree4_cuda,
                    6: block3_degree6_cuda,
                }
        except (ImportError, ModuleNotFoundError):
            _BLOCK3_FUSED_CUDA_IMPLS = {}
    return _BLOCK3_FUSED_CUDA_IMPLS.get(int(degree))


def _try_block3_fused_cuda(
        x: Tensor,
        cfg: Block3NoiseConfig,
        *,
        truncation_scale: Optional[float] = None,
        ) -> Optional[Tensor]:
    """Run an exact degree-4/6 CUDA specialization, or return ``None``."""
    degree = int(getattr(cfg, "degree", 0))
    square_rescales = tuple(getattr(cfg, "square_rescales", ()) or ())
    if (
            not _BLOCK3_FUSED_CUDA_ENABLED
            or degree not in (4, 6)
            or getattr(cfg, "x_inv_2n_result_rescale", None) is not None
            or len(square_rescales) != degree
            or any(point is None for point in square_rescales)
            or not x.is_cuda
            or x.dtype != torch.float32
            or x.requires_grad
            or not x.is_contiguous()
    ):
        return None

    implementation = _resolve_block3_fused_cuda_impl(degree)
    if implementation is None:
        return None

    points = (cfg.x_fresh, cfg.inv_2n_encode, *square_rescales)
    stds = []
    for point in points:
        std = _noise_std_for_values(
            str(point.distribution).lower(),
            int(point.scaling_factor),
            int(point.N),
        )
        if std <= 0.0:
            return None
        stds.append(std)

    try:
        noise_slab = _get_block3_fused_cuda_noise_workspace(
            x,
            len(points),
        )
    except torch.cuda.OutOfMemoryError:
        return None
    generator = _get_noise_generator(x.device)
    noises = []
    for index, std in enumerate(stds):
        noise = noise_slab[index]
        noise.normal_(0.0, float(std), generator=generator)
        noises.append(noise)
    return implementation(
        x,
        noises,
        truncation_scale=truncation_scale,
    )


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
    sq_rescales = cfg.square_rescales

    def block3_approx_exp(x: Tensor) -> Tensor:
        truncation_scale = _configured_binary_truncation_fused_cuda_scale(
            x,
            cfg,
        )
        fused = _try_block3_fused_cuda(
            x,
            cfg,
            truncation_scale=truncation_scale,
        )
        if fused is not None:
            return _apply_configured_truncation(
                fused,
                cfg,
                binary_already_applied=truncation_scale is not None,
            )

        x = _sample_and_add_gaussian_for_point(x, cfg.x_fresh)

        noisy_inv_2n = _sample_gaussian_for_point(x, cfg.inv_2n_encode)
        noisy_inv_2n.add_(inv_2n_value)

        x_scaled = x * noisy_inv_2n
        if cfg.x_inv_2n_result_rescale is not None:
            x_scaled = _sample_and_add_gaussian_for_point(
                x_scaled, cfg.x_inv_2n_result_rescale,
            )

        y = 1.0 + x_scaled

        for k in range(degree):
            y = y * y
            rs = sq_rescales[k] if k < len(sq_rescales) else None
            if rs is not None:
                y = _sample_and_add_gaussian_for_point(y, rs)

        y = _apply_configured_truncation(y, cfg)
        return y

    return block3_approx_exp


def _make_block2_bsgs_mask_hook(
        mask1_encode: NoisePoint,
        mask1_rescale: Optional[NoisePoint],
        mask2_encode: NoisePoint,
        mask2_rescale: Optional[NoisePoint],
        rotation_after_mask1_rescale: int = 0,
        rotation_after_mask2_rescale: int = 0,
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

        noisy_mask1 = _sample_gaussian_for_point(tensor, mask1_encode)
        noisy_mask1.add_(1.0)
        out = tensor * noisy_mask1
        if mask1_rescale is not None:
            out = _sample_and_add_gaussian_for_point(out, mask1_rescale)
            out = _apply_rotation_noise(
                out, mask1_rescale, rotation_after_mask1_rescale,
            )

        noisy_mask2 = _sample_gaussian_for_point(out, mask2_encode)
        noisy_mask2.add_(1.0)
        out = out * noisy_mask2
        if mask2_rescale is not None:
            out = _sample_and_add_gaussian_for_point(out, mask2_rescale)
            out = _apply_rotation_noise(
                out, mask2_rescale, rotation_after_mask2_rescale,
            )
        return out
    return hook


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

    softmax_out_fresh: NoisePoint
    softmax_out_mask_encode: NoisePoint

    v_fresh: NoisePoint
    v_mask_encode: NoisePoint

    softmax_v_mask_encode: NoisePoint

    wo_encode: NoisePoint

    ln_mean_inv_d_encode: NoisePoint
    ln_var_inv_d_encode: NoisePoint


    softmax_out_mask_rescale: Optional[NoisePoint] = None
    v_mask_rescale: Optional[NoisePoint] = None
    softmax_v_matmul_rescale: Optional[NoisePoint] = None
    softmax_v_mask_rescale: Optional[NoisePoint] = None
    wo_result_rescale: Optional[NoisePoint] = None
    ln_mean_result_rescale: Optional[NoisePoint] = None
    ln_square_result_rescale: Optional[NoisePoint] = None
    ln_var_result_rescale: Optional[NoisePoint] = None

    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    output_truncation_ring_bits: int = 43
    output_truncation_source_fractional_bits: int = 24


    rotation_after_softmax_out_mask_rescale: bool = False
    rotation_after_v_mask_rescale: bool = False
    rotation_after_softmax_v_matmul_rescale: bool = False
    rotation_after_softmax_v_mask_rescale: bool = False
    rotation_after_wo_rescale: bool = False
    rotation_after_ln_square_rescale: bool = False
    rotation_repeat_counts: dict = field(default_factory=dict)


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
        rotation_after_mask_rescale: int = 0,
        ):
    """softmax 输出 / V 共用：fresh on tensor → ⊙ ones-mask (encode) → optional rescale。

    可选 rotation 紧跟 mask_rescale 之后；SF 继承自 mask_rescale_point。
    """
    def hook(tensor: Tensor) -> Tensor:

        out = _sample_and_add_gaussian_for_point(tensor, fresh_point)

        noisy_mask = _sample_gaussian_for_point(out, mask_encode_point)
        noisy_mask.add_(1.0)
        out = out * noisy_mask

        if mask_rescale_point is not None:
            out = _sample_and_add_gaussian_for_point(
                out, mask_rescale_point,
            )
            out = _apply_rotation_noise(
                out, mask_rescale_point, rotation_after_mask_rescale,
            )
        return out
    return hook


def _make_block4_softmax_v_hook(
        matmul_rescale: Optional[NoisePoint],
        mask_encode: NoisePoint,
        mask_rescale: Optional[NoisePoint],
        rotation_after_matmul_rescale: int = 0,
        rotation_after_mask_rescale: int = 0,
        ):
    """softmax×V matmul 之后：optional rescale on matmul → ⊙ ones-mask (encode) → optional rescale。

    matmul rescale 与 mask rescale 各支持一个独立 rotation 选项，SF 继承自各自的 rescale。
    """
    def hook(tensor: Tensor) -> Tensor:

        if matmul_rescale is not None:
            tensor = _sample_and_add_gaussian_for_point(
                tensor, matmul_rescale,
            )
            tensor = _apply_rotation_noise(
                tensor, matmul_rescale, rotation_after_matmul_rescale,
            )

        noisy_mask = _sample_gaussian_for_point(tensor, mask_encode)
        noisy_mask.add_(1.0)
        out = tensor * noisy_mask

        if mask_rescale is not None:
            out = _sample_and_add_gaussian_for_point(out, mask_rescale)
            out = _apply_rotation_noise(
                out, mask_rescale, rotation_after_mask_rescale,
            )
        return out
    return hook


def _make_block4_wo_forward(
        linear_module: nn.Linear,
        encode_point: NoisePoint,
        rescale_point: Optional[NoisePoint],
        rotation_after_rescale: int = 0,
        ):
    """Wo 投影包装：encode on W_o + 可选 rescale on Att = X·W_o 结果。

    可选 rotation 紧跟 rescale 之后；SF 继承自 rescale_point。
    """
    def block4_wo_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        weight = linear_module.weight
        noisy_weight = _sample_and_add_gaussian_for_point(
            weight, encode_point,
        )
        noisy_weight = noisy_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = linear_module.bias
        if bias is not None:
            bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        out = nn.functional.linear(hidden_states, noisy_weight, bias)
        if rescale_point is not None:
            out = _sample_and_add_gaussian_for_point(out, rescale_point)
            out = _apply_rotation_noise(
                out, rescale_point, rotation_after_rescale,
            )
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
        """Install the Block 5 LayerNorm-tail configuration."""
        self.cfg5 = cfg5

    def forward(self, x: Tensor) -> Tensor:
        D = int(x.shape[-1])
        cfg4 = self.cfg4


        sum_x = x.sum(dim=-1, keepdim=True)
        if cfg4 is not None:
            noisy_inv_d = _sample_gaussian_for_point(x, cfg4.ln_mean_inv_d_encode)
            noisy_inv_d.add_(1.0 / D)
            mean = sum_x * noisy_inv_d
            if cfg4.ln_mean_result_rescale is not None:
                mean = _sample_and_add_gaussian_for_point(
                    mean, cfg4.ln_mean_result_rescale,
                )
        else:
            mean = sum_x / float(D)

        x_centered = x - mean

        sq = x_centered * x_centered
        if cfg4 is not None and cfg4.ln_square_result_rescale is not None:
            sq = _sample_and_add_gaussian_for_point(
                sq, cfg4.ln_square_result_rescale,
            )

            sq = _apply_rotation_noise(
                sq,
                cfg4.ln_square_result_rescale,
                _rotation_repeat_count(cfg4, "rotation_after_ln_square_rescale"),
            )

        sum_sq = sq.sum(dim=-1, keepdim=True)
        if cfg4 is not None:
            noisy_inv_d_var = _sample_gaussian_for_point(sq, cfg4.ln_var_inv_d_encode)
            noisy_inv_d_var.add_(1.0 / D)
            var = sum_sq * noisy_inv_d_var
            if cfg4.ln_var_result_rescale is not None:
                var = _sample_and_add_gaussian_for_point(
                    var, cfg4.ln_var_result_rescale,
                )
        else:
            var = sum_sq / float(D)


        if cfg4 is not None:
            var = _apply_configured_truncation(var, cfg4)


        inv_std = torch.rsqrt(var + self.eps)


        cfg5 = self.cfg5
        if cfg5 is not None:
            if inv_std.shape != x.shape:
                inv_std = inv_std.expand_as(x).contiguous()
            noisy_inv_std = _sample_and_add_gaussian_for_point(
                inv_std, cfg5.inv_std_fresh,
            )
            noisy_x_centered = _sample_and_add_gaussian_for_point(
                x_centered, cfg5.x_centered_fresh,
            )
            normalized = noisy_x_centered * noisy_inv_std
            if cfg5.normalize_result_rescale is not None:
                normalized = _sample_and_add_gaussian_for_point(
                    normalized, cfg5.normalize_result_rescale,
                )
            gamma_broadcast = self.weight.expand_as(normalized)
            noisy_gamma = _sample_and_add_gaussian_for_point(
                gamma_broadcast, cfg5.gamma_encode,
            )
            gamma_mul = normalized * noisy_gamma
            if cfg5.gamma_result_rescale is not None:
                gamma_mul = _sample_and_add_gaussian_for_point(
                    gamma_mul, cfg5.gamma_result_rescale,
                )

                gamma_mul = _apply_rotation_noise(
                    gamma_mul,
                    cfg5.gamma_result_rescale,
                    _rotation_repeat_count(cfg5, "rotation_after_gamma_rescale"),
                )
            out = gamma_mul + self.bias
        else:
            normalized = x_centered * inv_std
            out = normalized * self.weight + self.bias
        return out


@dataclass
class Block5NoiseConfig:
    """BLB Block 5 噪声配置。

    Block 5 范围：post-attn LN tail (rsqrt 之后) → Wffn1 → GELU 多项式近似。
    GELU supports degree in {1, 2, 4}.

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

    gelu_power_rescales: Tuple[Optional[NoisePoint], ...] = field(default_factory=tuple)
    gelu_coeff_mul_rescales: Tuple[Optional[NoisePoint], ...] = field(default_factory=tuple)

    output_truncation_k: Optional[int] = None
    output_truncation_mode: str = "binary"
    output_truncation_ring_bits: int = 43
    output_truncation_source_fractional_bits: int = 24


    rotation_after_gamma_rescale: bool = False
    rotation_after_wffn1_rescale: bool = False
    rotation_repeat_counts: dict = field(default_factory=dict)


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
        gelu_degree:  GELU degree ∈ {0, 1, 2, 4}（0=ReLU，无多项式 GELU 噪声）
        N:            CKKS 多项式阶。None = 按 degree 自动选
                      （degree∈{0,1} → 8192，degree∈{2,4} → 16384）
        gelu_power_rescale_sfs:    长度 == degree-1；
                                   degree=1: ()；degree=2: (x²,)；degree=4: (x²,x³,x⁴)
        gelu_coeff_mul_rescale_sfs: 长度 == degree；按 c_1·x, c_2·x², ... 顺序
        其它 *_sf 含义同 Block 1/2 同名参数。

    每个 ``*_sf`` 都是 ``NOISE_VARIANCE_TABLE_BY_N`` 的 key（即 scale_bits）；
    σ² 严禁写死。
    """
    deg = int(gelu_degree)
    if deg not in (0, 1, 2, 4):
        raise ValueError(f"Block 5 GELU degree 必须 ∈ {{0, 1, 2, 4}}, got {deg}")
    if N is None:

        N = 8192 if deg <= 1 else 16384

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


    expected_power_len = max(0, deg - 1)
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


def _make_block5_wffn1_forward(
        linear_module: nn.Linear,
        encode_point: NoisePoint,
        rescale_point: Optional[NoisePoint],
        rotation_after_rescale: int = 0,
        ):
    """Wffn1 投影包装：encode on W_ffn1 + 可选 rescale on result（GELU 输入 x）
    + 可选 rotation 紧跟 rescale 之后（SF 继承自 rescale_point）。
    """
    def block5_wffn1_forward(hidden_states):
        if hidden_states is None:
            return hidden_states
        weight = linear_module.weight
        noisy_weight = _sample_and_add_gaussian_for_point(
            weight, encode_point,
        )
        noisy_weight = noisy_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = linear_module.bias
        if bias is not None:
            bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        out = nn.functional.linear(hidden_states, noisy_weight, bias)
        if rescale_point is not None:
            out = _sample_and_add_gaussian_for_point(out, rescale_point)

            out = _apply_rotation_noise(
                out, rescale_point, rotation_after_rescale,
            )
        return out
    return block5_wffn1_forward


def _select_piecewise_gelu_output(x: Tensor, y_neg: Tensor, y_pos: Tensor) -> Tensor:
    """Select GELU approximation pieces with a scalar-zero low/NaN branch."""
    out = torch.where(x < 0, y_neg, y_pos)
    out = torch.where(x >= -2.7, out, 0.0)
    return torch.where(x > 2.7, x, out)


_BLOCK5_FUSED_CUDA_ENABLED = str(
    _os.environ.get("BLB_STAGE2_BLOCK5_FUSED_CUDA", "1")
).strip().lower() not in {"0", "false", "no", "off"}
_BLOCK5_FUSED_CUDA_IMPL = None
_BLOCK5_FUSED_CUDA_RESOLVED = False
_BLOCK5_FUSED_CUDA_WORKSPACES = {}


def _get_block5_fused_cuda_noise_workspace(
        x: Tensor,
        point_count: int,
        ) -> Tensor:
    """Reuse noise storage for sequential Block5 calls on one thread/stream."""
    count = int(point_count)
    if count <= 0:
        raise ValueError(f"point_count must be positive, got {count}")
    stream = torch.cuda.current_stream(x.device)
    key = (
        int(_threading.get_ident()),
        str(x.device),
        int(stream.cuda_stream),
        x.dtype,
    )
    required = count * int(x.numel())
    workspace = _BLOCK5_FUSED_CUDA_WORKSPACES.get(key)
    if workspace is None or int(workspace.numel()) < required:
        workspace = torch.empty(
            required,
            device=x.device,
            dtype=x.dtype,
        )
        _BLOCK5_FUSED_CUDA_WORKSPACES[key] = workspace
    return workspace[:required].view(count, *x.shape)


def _resolve_block5_fused_cuda_impl():
    global _BLOCK5_FUSED_CUDA_IMPL, _BLOCK5_FUSED_CUDA_RESOLVED
    if not _BLOCK5_FUSED_CUDA_RESOLVED:
        _BLOCK5_FUSED_CUDA_RESOLVED = True
        try:
            from blb_stage2_rl.block5_fused_cuda import (
                block5_degree4_cuda,
                is_available,
            )

            if is_available():
                _BLOCK5_FUSED_CUDA_IMPL = block5_degree4_cuda
        except (ImportError, ModuleNotFoundError):
            _BLOCK5_FUSED_CUDA_IMPL = None
    return _BLOCK5_FUSED_CUDA_IMPL


def _try_block5_fused_cuda(
        x: Tensor,
        cfg: Block5NoiseConfig,
        coeff_dict,
        *,
        truncation_scale: Optional[float] = None,
        ) -> Optional[Tensor]:
    """Run the exact degree-4 CUDA specialization, or return ``None``."""
    power_rescales = tuple(getattr(cfg, "gelu_power_rescales", ()) or ())
    coefficient_rescales = tuple(
        getattr(cfg, "gelu_coeff_mul_rescales", ()) or ()
    )
    negative_coefficients = tuple(coeff_dict[1])
    positive_coefficients = tuple(coeff_dict[0])
    if (
            not _BLOCK5_FUSED_CUDA_ENABLED
            or int(getattr(cfg, "gelu_degree", 0)) != 4
            or len(power_rescales) != 3
            or len(coefficient_rescales) != 4
            or len(negative_coefficients) != 5
            or len(positive_coefficients) != 5
            or not x.is_cuda
            or x.dtype != torch.float32
            or x.requires_grad
            or not x.is_contiguous()
    ):
        return None

    implementation = _resolve_block5_fused_cuda_impl()
    if implementation is None:
        return None

    points = []
    indices = [-1] * 21

    def append_point(slot: int, point: Optional[NoisePoint]) -> None:
        if point is None:
            return
        indices[slot] = len(points)
        points.append(point)

    for slot, point in enumerate(power_rescales):
        append_point(slot, point)

    def append_piece(
            coefficient_slot: int,
            rescale_slot: int,
            ) -> None:
        for degree_index in range(5):
            append_point(coefficient_slot + degree_index, cfg.gelu_coeff_encode)
            if degree_index > 0:
                append_point(
                    rescale_slot + degree_index - 1,
                    coefficient_rescales[degree_index - 1],
                )

    append_piece(3, 8)
    append_piece(12, 17)

    stds = []
    for point in points:
        std = _noise_std_for_values(
            str(point.distribution).lower(),
            int(point.scaling_factor),
            int(point.N),
        )
        if std <= 0.0:
            return None
        stds.append(std)

    try:
        workspace = _get_block5_fused_cuda_noise_workspace(
            x,
            21,
        )
    except torch.cuda.OutOfMemoryError:
        return None
    generator = _get_noise_generator(x.device)
    return implementation(
        x,
        workspace,
        indices,
        stds,
        negative_coefficients,
        positive_coefficients,
        generator,
        truncation_scale,
    )


def _make_block5_gelu_forward(original_gelu, cfg5: Block5NoiseConfig):
    """构造 BLB Block 5 噪声版的 ``PolynomialGELU.forward``。

    替换 ``layer.intermediate.intermediate_act_fn.forward``。

    工作流（与原 PolynomialGELU 等价但带噪）：
      1. 计算 x 的幂 x², x³, x⁴（按 degree 决定哪些）：每个 power 之后加可选 rescale。
         共享 power 用于 piecewise 两段多项式，避免重复加噪。
      2. 对负段 (x ∈ [-2.7, 0)) 和正段 (x ∈ [0, 2.7]) 分别用各自 ``coeff[sign]``：
         a) 以 x 同形 [B, S, H] 采样每个系数的 encode 噪声，再加标量 c_k（per-slot 独立）；
         b) 常数项 c_0 直接累加（无乘法 → 无 rescale）；
         c) 非常数项 c_k * x^k：乘法后加可选 rescale（按 cfg5.gelu_coeff_mul_rescales[k-1]）。
      3. 用 mask 选段，与原 PolynomialGELU.forward 一致。
    """
    coeff_dict = original_gelu.coeff
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
        """返回 [None, x^1, ..., x^degree]，按 degree 决定中间 rescale。"""
        powers = [None] * (degree + 1)
        powers[1] = x
        if degree >= 2:
            x2 = x * x
            rs = pwr_rs[0] if len(pwr_rs) > 0 else None
            if rs is not None:
                x2 = _sample_and_add_gaussian_for_point(x2, rs)
            powers[2] = x2
        if degree >= 4:
            x3 = powers[2] * x
            rs = pwr_rs[1] if len(pwr_rs) > 1 else None
            if rs is not None:
                x3 = _sample_and_add_gaussian_for_point(x3, rs)
            powers[3] = x3
            x4 = powers[2] * powers[2]
            rs = pwr_rs[2] if len(pwr_rs) > 2 else None
            if rs is not None:
                x4 = _sample_and_add_gaussian_for_point(x4, rs)
            powers[4] = x4
        return powers

    def _compute_polynomial(powers, coeffs_for_piece, x_ref: Tensor) -> Tensor:
        """c_0 + c_1·x + c_2·x² + ... + c_n·x^n, 每个系数 encode + 每个乘法 rescale。"""
        if len(coeffs_for_piece) != degree + 1:

            raise RuntimeError(
                f"coeff piece 长度 {len(coeffs_for_piece)} != degree+1 = {degree+1}"
            )
        result = None
        for k in range(degree + 1):
            coeff_value = float(coeffs_for_piece[k])
            noisy_coeff = _sample_gaussian_for_point(x_ref, cfg5.gelu_coeff_encode)
            noisy_coeff.add_(coeff_value)
            if k == 0:

                term = noisy_coeff
            else:
                term = powers[k] * noisy_coeff
                rs = coeff_rs[k - 1] if (k - 1) < len(coeff_rs) else None
                if rs is not None:
                    term = _sample_and_add_gaussian_for_point(term, rs)
            result = term if result is None else result + term
        return result

    def block5_gelu_forward(x: Tensor) -> Tensor:
        truncation_scale = _configured_binary_truncation_fused_cuda_scale(
            x,
            cfg5,
        )
        fused = _try_block5_fused_cuda(
            x,
            cfg5,
            coeff_dict,
            truncation_scale=truncation_scale,
        )
        if fused is not None:
            return _apply_configured_truncation(
                fused,
                cfg5,
                binary_already_applied=truncation_scale is not None,
            )
        powers = _compute_powers(x)

        y1 = _compute_polynomial(powers, coeff_dict[1], x)
        y2 = _compute_polynomial(powers, coeff_dict[0], x)
        out = _select_piecewise_gelu_output(x, y1, y2)

        out = _apply_configured_truncation(out, cfg5)
        return out

    return block5_gelu_forward


_GELU_PAIRED_POLY_MIN_NUMEL = 12_000_000


def polynomial(x, coeff, sign):

    device = x.device
    dtype  = x.dtype


    powers = torch.stack([x.pow(i) for i in range(len(coeff[sign]))], dim=-1)


    coeff_tensor = torch.tensor(
        coeff[sign],
        device=device,
        dtype=dtype
    )


    return (powers * coeff_tensor).sum(dim=-1)

class PolynomialGELU(nn.Module):
    """可逆的三次多项式GELU近似"""
    def __init__(self, degree=4):
        super().__init__()
        self.coeff = GELU_COEEF[degree]
        self.degree = degree


        self._coeff_cache = {}
        self._paired_coeff_cache = {}

    def _coeff_tensor(self, sign: int, device, dtype) -> Tensor:
        key = (sign, device, dtype)
        t = self._coeff_cache.get(key)
        if t is None:
            t = torch.tensor(self.coeff[sign], device=device, dtype=dtype)
            self._coeff_cache[key] = t
        return t

    def _poly(self, x: Tensor, sign: int) -> Tensor:


        coeff_tensor = self._coeff_tensor(sign, x.device, x.dtype)
        n = coeff_tensor.shape[0]
        if n == 1:
            return coeff_tensor[0].expand_as(x).clone()
        out = coeff_tensor[0].expand_as(x).clone()
        power = x
        out = torch.addcmul(out, coeff_tensor[1], power)
        for i in range(2, n):
            power = power * x
            out = torch.addcmul(out, coeff_tensor[i], power)
        return out

    def _paired_coeff_tensor(self, device, dtype) -> Tensor:
        key = (device, dtype)
        paired = self._paired_coeff_cache.get(key)
        if paired is None:
            paired = torch.stack(
                (
                    self._coeff_tensor(1, device, dtype),
                    self._coeff_tensor(0, device, dtype),
                ),
                dim=0,
            )
            self._paired_coeff_cache[key] = paired
        return paired

    def _poly_pair(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        coeff_tensor = self._paired_coeff_tensor(x.device, x.dtype)
        view_shape = (2,) + (1,) * x.ndim
        out = coeff_tensor[:, 0].view(view_shape).expand((2,) + tuple(x.shape)).clone()
        power = x
        out = torch.addcmul(
            out,
            coeff_tensor[:, 1].view(view_shape),
            power.unsqueeze(0),
        )
        for i in range(2, coeff_tensor.shape[1]):
            power = power * x
            out = torch.addcmul(
                out,
                coeff_tensor[:, i].view(view_shape),
                power.unsqueeze(0),
            )
        return out[0], out[1]

    def forward(self, x: Tensor) -> Tensor:

        if self.degree == 0:

            return self._poly(x, 1)

        if (
            x.is_cuda
            and x.dtype == torch.float32
            and self.degree in (2, 4)
            and x.numel() >= _GELU_PAIRED_POLY_MIN_NUMEL
        ):
            y1, y2 = self._poly_pair(x)
        else:
            y1 = self._poly(x, 1)
            y2 = self._poly(x, 0)
        out = _select_piecewise_gelu_output(x, y1, y2)


        return out


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
        """近似计算指数函数"""


        t = 1 + x / (2 ** self.degree)
        for _ in range(self.degree):
            t = t * t
        return t


    def approximation_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """使用指数近似计算softmax"""


        x = x - x.max(dim=-1, keepdim=True)[0] + 1e-9


        exp_approx = self.approximation_exponential(x)
        exp_out = torch.where(x < self.lower_bound, 0.0, exp_approx)
        sum_exp = torch.sum(exp_out, dim=-1, keepdim=True) + 1e-9

        return exp_out / sum_exp


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

        tail_pos = 0

        if encoder_hidden_states is not None and tail_pos < len(tail):
            first = tail[tail_pos]
            if encoder_attention_mask is None and (first is None or self._looks_like_attention_mask(first)):
                encoder_attention_mask = first
                tail_pos += 1

        if past_key_value is None and past_key_values is None and tail_pos < len(tail):
            candidate = tail[tail_pos]
            tail_pos += 1
            if isinstance(candidate, bool):
                if output_attentions in (False, None):
                    output_attentions = candidate
                candidate = None
            elif (
                encoder_hidden_states is None
                and encoder_attention_mask is None
                and self._looks_like_attention_mask(candidate)
            ):


                encoder_attention_mask = candidate
                if tail_pos < len(tail):
                    candidate = tail[tail_pos]
                    tail_pos += 1
                else:
                    candidate = None
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


        kt = key_layer.transpose(-1, -2)
        block2_kt_hook = getattr(self, "_block2_kt_bsgs_hook", None)
        if block2_kt_hook is not None:
            kt = block2_kt_hook(kt)
        attention_scores = torch.matmul(query_layer, kt)


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
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:

            attention_scores = attention_scores + attention_mask


        attention_probs = self.approximation_softmax(attention_scores)


        attention_probs = self.dropout(attention_probs)


        if head_mask is not None:
            attention_probs = attention_probs * head_mask


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


        block4_softmax_v_hook = getattr(self, "_block4_softmax_v_hook", None)
        if block4_softmax_v_hook is not None:
            context_layer = block4_softmax_v_hook(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ReversibleLayerHandler:
    """Install and restore BERT approximation and BLB noise modules."""


    _BERT_PATHS = {
        "gelu_act": "intermediate.intermediate_act_fn",
        "wo_dense": "attention.output.dense",
        "wffn1_dense": "intermediate.dense",
        "wffn2_dense": "output.dense",
    }
    def __init__(self, model):
        self.model = model
        self._paths = self._BERT_PATHS
        self._resolved_layers_cache = {}
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


        self.reuse_approx_modules = True
        self._approx_softmax_cache = {}
        self._approx_gelu_cache = {}
        self._approx_softmax_rebuilds = 0
        self._approx_gelu_rebuilds = 0

        self.original_block1_ffn2 = {}
        self.original_block1_layernorm = {}
        self.block1_cfg_per_layer = {}

        self.original_block2_qproj = {}
        self.original_block2_kproj = {}
        self.original_block2_vproj = {}
        self.block2_cfg_per_layer = {}


        self.block3_installed_layers = set()
        self.block3_cfg_per_layer = {}

        self.original_block4_wo = {}
        self.original_block4_post_attn_ln = {}
        self.block4_cfg_per_layer = {}

        self.original_block5_wffn1 = {}
        self.original_block5_gelu = {}
        self.block5_cfg_per_layer = {}


        self.backup_model = copy.deepcopy(model)

    @staticmethod
    def _approx_attn_is_fresh_equivalent(module) -> bool:
        """True iff ``module`` is bit-identical to a freshly constructed
        ``BertSelfAttentionWithAproximation``: no block-3 instance override of
        ``approximation_exponential``, no softmax/value noise state, and no BLB
        per-instance hooks. Only then is "reuse the cached module + update the
        degree" equivalent to reconstructing it from scratch."""
        if "approximation_exponential" in vars(module):
            return False
        if getattr(module, "_softmax_value_noise_state", None) is not None:
            return False
        for hook in ("_block2_q_bsgs_hook", "_block2_kt_bsgs_hook",
                     "_block2_qkt_merge_hook", "_block4_softmax_out_hook",
                     "_block4_v_hook", "_block4_softmax_v_hook"):
            if getattr(module, hook, None) is not None:
                return False
        return True

    def _resolve_layers(self, layer_name):
        layers = self._resolved_layers_cache.get(layer_name)
        if layers is None:
            layers = tuple(eval("self." + layer_name))
            self._resolved_layers_cache[layer_name] = layers
        return layers

    def replace_layer_gelu(self, layer_indices=None, layer_name="model.model.layers", degree=1):
        """Replace GELU in selected BERT layers."""
        act_path = self._paths["gelu_act"]
        for i, layer in enumerate(self._resolve_layers(layer_name)):
            if i in layer_indices:
                if i not in self.original_gelu:
                    self.original_gelu[i] = {
                        "act_fn": _get_attr_path(layer, act_path),
                    }
                orig_act = _get_attr_path(layer, act_path)
                orig_training = getattr(orig_act, "training", layer.training)


                cached = (self._approx_gelu_cache.get((i, degree))
                          if self.reuse_approx_modules else None)
                if cached is not None and "forward" not in vars(cached):
                    new_act = cached
                else:

                    new_act = nn.ReLU() if int(degree) == 0 else PolynomialGELU(degree=degree)
                    self._approx_gelu_rebuilds += 1
                    if self.reuse_approx_modules:
                        self._approx_gelu_cache[(i, degree)] = new_act
                new_act.train(bool(orig_training))
                _set_attr_path(layer, act_path, new_act)

        print(f"已替换 {len(layer_indices)} 层的GELU函数（GELU function）")

    def replace_layer_softmax(self, layer_indices=None, layer_name="model.model.layers", attention_name = "attention", degree=1):
        """Replace softmax in selected BERT attention layers."""
        for i, layer in enumerate(self._resolve_layers(layer_name)):
            if i in layer_indices:

                if i not in self.original_attention:
                    self.original_attention[i] = {
                        'attention': eval("layer."+ attention_name)
                    }


                cached = self._approx_softmax_cache.get(i) if self.reuse_approx_modules else None
                if (cached is not None
                        and layer.attention.self is cached
                        and self._approx_attn_is_fresh_equivalent(cached)):
                    cached.degree = degree
                    cached.lower_bound = Exp_bound[degree]
                    continue


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

        layers = self._resolve_layers(layer_name)
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return


        self._check_noise_mode_conflict(selected, installing="single_table")

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

        layers = self._resolve_layers(layer_name)
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return


        self._check_noise_mode_conflict(selected, installing="single_table")

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

        layers = self._resolve_layers(layer_name)
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return


        self._check_noise_mode_conflict(selected, installing="single_table")


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

        layers = self._resolve_layers(layer_name)
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return


        self._check_noise_mode_conflict(selected, installing="single_table")

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

            linear_module.forward = _make_noisy_linear_forward(
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
        for i, layer in enumerate(self._resolve_layers(layer_name)):
            if i in layer_indices and i in self.original_gelu:
                _set_attr_path(layer, act_path, self.original_gelu[i]["act_fn"])

        print(f"已恢复 {len(layer_indices)} 层的原始GELU函数（original GELU function）")

    def restore_layer_softmax(self, layer_indices=None, layer_name="model.model.layers", attention_name = "attention"):
        """Restore softmax in selected BERT attention layers."""
        for i, layer in enumerate(self._resolve_layers(layer_name)):
            if i in layer_indices and i in self.original_attention:
                current_training = layer.attention.self.training
                restored_attention = self.original_attention[i]['attention']
                restored_attention.train(bool(current_training))
                layer.attention.self = restored_attention


    def restore_layer_input_noise(self, layer_indices=None, layer_name="model.model.layers"):
        """Restore original transformer-layer inputs for selected layers."""
        layers = self._resolve_layers(layer_name)
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
        layers = self._resolve_layers(layer_name)
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

        本方法会覆盖 ``replace_layer_ffn2_noise`` 对 Wffn2 forward 的包装，因为
        Block 1 是该包装的严格扩展；恢复 Block 1 后会还原原始 forward。
        """

        if cfg is None:
            cfg = make_block1_default_config()

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return


        self._check_noise_mode_conflict(selected, installing="blb")

        for i, layer in enumerate(layers):
            if i not in selected:
                continue


            ffn2_module = layer.output.dense
            stored_forward = self.original_block1_ffn2.get(i)

            if stored_forward is None or getattr(stored_forward, "__self__", None) is not ffn2_module:
                self.original_block1_ffn2[i] = ffn2_module.forward
            ffn2_module.forward = _make_block1_ffn2_forward(ffn2_module, cfg)


            current_ln = layer.output.LayerNorm
            if i not in self.original_block1_layernorm:

                self.original_block1_layernorm[i] = current_ln
            source_ln = self.original_block1_layernorm[i]
            new_ln = NoisyBlock1LayerNorm(source_ln, cfg)
            new_ln.train(source_ln.training)

            try:
                ref_param = source_ln.weight
                new_ln = new_ln.to(device=ref_param.device, dtype=ref_param.dtype)
            except Exception:
                pass
            layer.output.LayerNorm = new_ln


            self.block1_cfg_per_layer[i] = cfg

        rescale_summary = (
            f"wffn2_result={cfg.wffn2_result_rescale.scaling_factor if cfg.wffn2_result_rescale else 'off'}, "
            f"mean={cfg.mean_result_rescale.scaling_factor if cfg.mean_result_rescale else 'off'}, "
            f"square={cfg.square_result_rescale.scaling_factor if cfg.square_result_rescale else 'off'}, "
            f"var={cfg.var_result_rescale.scaling_factor if cfg.var_result_rescale else 'off'}"
        )
        mode_label = "噪声+截断" if bool(getattr(cfg, "noise_enabled", True)) else "仅截断"
        _print_blb_install(
            f"已为 {len(selected)} 层启用 BLB Block 1 {mode_label} "
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


    def replace_layer_block2_noise(
            self,
            layer_indices=None,
            layer_name="model.model.layers",
            cfg: Optional[Block2NoiseConfig] = None,
            ):
        """安装 BLB Block 2 噪声（22 个注入点）。

        Block 2 covers:
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

        与单表 ``replace_layer_input_noise`` 的关系：
          * Block 2 视角下，X 进 Wq/Wk/Wv 之前**不再加 fresh**（X 的 PPTI 噪声
            来自 LN tail γ 乘法的 rescale）。本方法不会触发单表
            input-X fresh 噪声；如果你之前装过 ``replace_layer_input_noise``，
            建议先 ``restore_*`` 再 install Block 2，避免双重加噪。
        """

        if cfg is None:
            cfg = make_block2_default_config()

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return


        self._check_noise_mode_conflict(selected, installing="blb")

        for i, layer in enumerate(layers):
            if i not in selected:
                continue


            current_ln = layer.output.LayerNorm
            if isinstance(current_ln, NoisyBlock1LayerNorm):

                current_ln.set_block2_cfg(cfg)
            else:

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


            attn_self = layer.attention.self


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
                rotation_after_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_wq_rescale",
                ),
            )

            k_module = attn_self.key
            if i not in self.original_block2_kproj:
                self.original_block2_kproj[i] = k_module.forward
            k_module.forward = _make_block2_qk_proj_forward(
                k_module, cfg.wk_encode, cfg.wk_result_rescale,
                rotation_after_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_wk_rescale",
                ),
            )

            v_module = attn_self.value
            if i not in self.original_block2_vproj:
                self.original_block2_vproj[i] = v_module.forward
            v_module.forward = _make_block2_qk_proj_forward(
                v_module, cfg.wv_encode, cfg.wv_result_rescale,
                rotation_after_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_wv_rescale",
                ),
            )


            attn_self._block2_q_bsgs_hook = _make_block2_bsgs_mask_hook(
                cfg.q_mask1_encode, cfg.q_mask1_result_rescale,
                cfg.q_mask2_encode, cfg.q_mask2_result_rescale,
                rotation_after_mask1_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_q_mask1_rescale",
                ),
                rotation_after_mask2_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_q_mask2_rescale",
                ),
            )
            attn_self._block2_kt_bsgs_hook = _make_block2_bsgs_mask_hook(
                cfg.kt_mask1_encode, cfg.kt_mask1_result_rescale,
                cfg.kt_mask2_encode, cfg.kt_mask2_result_rescale,
                rotation_after_mask1_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_kt_mask1_rescale",
                ),
                rotation_after_mask2_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_kt_mask2_rescale",
                ),
            )


            attn_self._block2_qkt_merge_hook = _make_block2_qkt_merge_hook(
                cfg.qkt_matmul_result_rescale,
                cfg.qkt_merge_mask_encode, cfg.qkt_merge_mask_result_rescale,
                truncation_cfg=cfg,
                rotation_after_qkt_matmul_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_qkt_matmul_rescale",
                ),
            )


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


            current_ln = layer.output.LayerNorm
            if isinstance(current_ln, NoisyBlock1LayerNorm):
                if i in self.block1_cfg_per_layer:

                    current_ln.set_block2_cfg(None)
                else:

                    if i in self.original_block1_layernorm:
                        layer.output.LayerNorm = self.original_block1_layernorm[i]
                        del self.original_block1_layernorm[i]
                    else:

                        current_ln.set_block2_cfg(None)

            self.block2_cfg_per_layer.pop(i, None)


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
        if cfg is not None and cfg_per_layer is not None:
            raise ValueError("cfg 与 cfg_per_layer 互斥，二选一。")

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return


        self._check_noise_mode_conflict(selected, installing="blb")

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

                if "approximation_exponential" in attn_self.__dict__:
                    del attn_self.__dict__["approximation_exponential"]
                self.block3_installed_layers.discard(i)
            self.block3_cfg_per_layer.pop(i, None)


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

        与单表 ``replace_layer_softmax_value_noise`` 的关系：
          * 装 Block 4 后，BertSelfAttentionWithAproximation.forward 会 short-circuit
            掉 ``_apply_softmax_value_noise``。恢复 Block 4 后回到单表路径。

        与单表 ``replace_layer_attention_output_noise`` 的关系：
          * Block 4 的 Wo wrap 是单表 Wo wrap 的严格扩展（多了 result rescale）。
            install Block 4 时会**覆盖**之前的 wo wrap；restore 时回到原始 forward。
        """
        if cfg is None:
            cfg = make_block4_default_config()

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return


        self._check_noise_mode_conflict(selected, installing="blb")

        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            attn_self = layer.attention.self
            if not isinstance(attn_self, BertSelfAttentionWithAproximation):
                raise RuntimeError(
                    f"layer {i} 的 attention.self 不是 BertSelfAttentionWithAproximation，"
                    f"无法安装 Block 4 hook。请先 replace_layer_softmax 安装 softmax 近似。"
                )


            attn_self._block4_softmax_out_hook = _make_block4_input_mask_hook(
                cfg.softmax_out_fresh, cfg.softmax_out_mask_encode, cfg.softmax_out_mask_rescale,
                rotation_after_mask_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_softmax_out_mask_rescale",
                ),
            )
            attn_self._block4_v_hook = _make_block4_input_mask_hook(
                cfg.v_fresh, cfg.v_mask_encode, cfg.v_mask_rescale,
                rotation_after_mask_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_v_mask_rescale",
                ),
            )
            attn_self._block4_softmax_v_hook = _make_block4_softmax_v_hook(
                cfg.softmax_v_matmul_rescale, cfg.softmax_v_mask_encode, cfg.softmax_v_mask_rescale,
                rotation_after_matmul_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_softmax_v_matmul_rescale",
                ),
                rotation_after_mask_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_softmax_v_mask_rescale",
                ),
            )


            wo_module = layer.attention.output.dense
            stored_forward = self.original_block4_wo.get(i)
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not wo_module:
                self.original_block4_wo[i] = wo_module.forward
            wo_module.forward = _make_block4_wo_forward(
                wo_module, cfg.wo_encode, cfg.wo_result_rescale,
                rotation_after_rescale=_rotation_repeat_count(
                    cfg, "rotation_after_wo_rescale",
                ),
            )


            current_ln = layer.attention.output.LayerNorm
            if isinstance(current_ln, NoisyBlock4LayerNorm):

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


            if i in self.original_block4_wo:
                wo_module = layer.attention.output.dense
                original_forward = self.original_block4_wo[i]
                if getattr(original_forward, "__self__", None) is wo_module:
                    wo_module.forward = original_forward
                del self.original_block4_wo[i]


            current_ln = layer.attention.output.LayerNorm
            if isinstance(current_ln, NoisyBlock4LayerNorm):
                if current_ln.cfg5 is not None:

                    current_ln.set_block4_cfg(None)
                else:

                    if i in self.original_block4_post_attn_ln:
                        layer.attention.output.LayerNorm = self.original_block4_post_attn_ln[i]
                        del self.original_block4_post_attn_ln[i]
                    else:
                        current_ln.set_block4_cfg(None)

            self.block4_cfg_per_layer.pop(i, None)


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
        if cfg is not None and cfg_per_layer is not None:
            raise ValueError("cfg 与 cfg_per_layer 互斥，二选一。")

        layers = list(eval("self." + layer_name))
        if layer_indices is None:
            selected = set(range(len(layers)))
        else:
            selected = set(layer_indices)
            if not selected:
                return


        self._check_noise_mode_conflict(selected, installing="blb")

        installed_summary = []
        for i, layer in enumerate(layers):
            if i not in selected:
                continue

            gelu_module = layer.intermediate.intermediate_act_fn


            if not isinstance(gelu_module, PolynomialGELU):
                raise RuntimeError(
                    f"layer {i} requires PolynomialGELU before Block 5 install"
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


            wffn1_module = layer.intermediate.dense
            stored_forward = self.original_block5_wffn1.get(i)
            if stored_forward is None or getattr(stored_forward, "__self__", None) is not wffn1_module:
                self.original_block5_wffn1[i] = wffn1_module.forward
            wffn1_module.forward = _make_block5_wffn1_forward(
                wffn1_module, this_cfg.wffn1_encode, this_cfg.wffn1_result_rescale,
                rotation_after_rescale=_rotation_repeat_count(
                    this_cfg, "rotation_after_wffn1_rescale",
                ),
            )


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


            if i in self.original_block5_gelu:
                gelu_module = layer.intermediate.intermediate_act_fn
                original_forward = self.original_block5_gelu[i]
                if getattr(original_forward, "__self__", None) is gelu_module:
                    gelu_module.forward = original_forward
                del self.original_block5_gelu[i]


            if i in self.original_block5_wffn1:
                wffn1_module = layer.intermediate.dense
                original_forward = self.original_block5_wffn1[i]
                if getattr(original_forward, "__self__", None) is wffn1_module:
                    wffn1_module.forward = original_forward
                del self.original_block5_wffn1[i]


            current_ln = layer.attention.output.LayerNorm
            if isinstance(current_ln, NoisyBlock4LayerNorm):
                if current_ln.cfg4 is not None:

                    current_ln.set_block5_cfg(None)
                else:

                    if i in self.original_block4_post_attn_ln:
                        layer.attention.output.LayerNorm = self.original_block4_post_attn_ln[i]
                        del self.original_block4_post_attn_ln[i]
                    else:
                        current_ln.set_block5_cfg(None)

            self.block5_cfg_per_layer.pop(i, None)


    def get_active_single_table_noise_layers(self) -> dict:
        """返回每种单表噪声类型当前安装到了哪些层。

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
        }

    def _check_noise_mode_conflict(self, target_layers, *, installing: str):
        """确认 BLB 与单表噪声不会安装在同一层。

        Args:
            target_layers: 即将操作的 layer 索引集合
            installing: "blb" 表示在装 BLB；
                        "single_table" 表示在装单表噪声。
                        二者互斥时抛 RuntimeError。
        """
        target = set(int(i) for i in target_layers)
        if not target:
            return

        if installing == "blb":
            single_table_active = self.get_active_single_table_noise_layers()
            conflicts = []
            for noise_type, layer_set in single_table_active.items():
                inter = layer_set & target
                if inter:
                    conflicts.append(f"single-table {noise_type}: layers {sorted(inter)}")
            if conflicts:
                raise RuntimeError(
                    "BLB 噪声与单表噪声互斥，检测到单表噪声残留：\n  - "
                    + "\n  - ".join(conflicts)
                    + "\n请先调用对应的 restore_layer_*_noise 还原单表噪声后再装 BLB。"
                )
        elif installing == "single_table":
            blb_active = self.get_active_blb_noise_layers()
            conflicts = []
            for block_name, layer_set in blb_active.items():
                inter = layer_set & target
                if inter:
                    conflicts.append(f"BLB {block_name}: layers {sorted(inter)}")
            if conflicts:
                raise RuntimeError(
                    "单表噪声与 BLB 噪声互斥，检测到 BLB 残留：\n  - "
                    + "\n  - ".join(conflicts)
                    + "\n请先调用对应的 restore_layer_block*_noise 还原 BLB 噪声后再装单表噪声。"
                )
        else:
            raise ValueError(
                f"installing 必须是 'blb' 或 'single_table'，不能是 {installing!r}"
            )

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
        self.original_block1_ffn2 = {}
        self.original_block1_layernorm = {}
        self.block1_cfg_per_layer = {}

        self.original_block2_qproj = {}
        self.original_block2_kproj = {}
        self.original_block2_vproj = {}
        self.block2_cfg_per_layer = {}

        self.block3_installed_layers = set()
        self.block3_cfg_per_layer = {}

        self.original_block4_wo = {}
        self.original_block4_post_attn_ln = {}
        self.block4_cfg_per_layer = {}

        self.original_block5_wffn1 = {}
        self.original_block5_gelu = {}
        self.block5_cfg_per_layer = {}

        print("已完全恢复原始模型状态")
