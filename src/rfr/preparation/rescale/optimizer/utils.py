"""
rescale_optimizer/utils.py

Utility functions: logging, pretty-printing, mathematical helpers.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union


logger = logging.getLogger("rescale_optimizer")


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    file_mode: str = "w",
) -> None:
    """
    Configure logging for the rescale optimizer.

    Parameters
    ----------
    level : int
        Logging level (``logging.INFO``, ``logging.DEBUG``, ...).
    log_file : str or Path, optional
        If given, also append/overwrite log records to this file (in
        addition to the stream handler).  Parent directories are
        auto-created.
    file_mode : str
        File-handler mode ("w" = overwrite, "a" = append).  Defaults to
        "w" so each run starts clean.  Ignored when ``log_file`` is
        ``None``.
    """
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    root = logging.getLogger("rescale_optimizer")
    root.handlers.clear()
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    root.setLevel(level)


def bits_for_value(value: float) -> float:
    """返回表示 |value| 所需的 bit 数 = ceil(log2(|value| + 1))。"""
    if value <= 0:
        return 0.0
    return math.ceil(math.log2(abs(value) + 1))


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_ntt_friendly_prime(bit_size: int, ring_degree: int = 1 << 15,
                            direction: str = "up") -> int:
    """
    寻找一个 NTT 友好的素数 q，满足:
      - q ≡ 1 (mod 2·ring_degree)
      - bit_length(q) == bit_size

    Parameters
    ----------
    bit_size : int
        目标素数的 bit 宽度。
    ring_degree : int
        多项式环度数 N（默认 2^15 = 32768）。
    direction : str
        "up" 从下界向上搜索, "down" 从上界向下搜索。

    Returns
    -------
    int
        满足条件的素数; 若找不到则返回 0。
    """
    m = 2 * ring_degree
    lower = 1 << (bit_size - 1)
    upper = (1 << bit_size) - 1

    if direction == "up":
        start = lower + (m - lower % m) % m + 1
        candidate = start
        while candidate <= upper:
            if is_prime(candidate):
                return candidate
            candidate += m
    else:
        start = upper - (upper % m) + 1
        if start > upper:
            start -= m
        candidate = start
        while candidate >= lower:
            if is_prime(candidate):
                return candidate
            candidate -= m
    return 0


def format_skeleton(skeleton: List[int],
                    drop_bits: Optional[List[float]] = None) -> str:
    """
    格式化打印一个 rescale skeleton。

    Parameters
    ----------
    skeleton : list of int
        cut point index 序列, e.g. [0, 3, 7, 12]。
    drop_bits : list of float, optional
        各 stage 的 drop bit 数, 长度 = len(skeleton)-1。
    """
    lines = [f"Skeleton: {skeleton}  (length = {len(skeleton)}, "
             f"rescales = {len(skeleton) - 1})"]
    if drop_bits:
        lines.append(f"  drop bits per stage: {drop_bits}")
        lines.append(f"  total drop bits:     {sum(drop_bits):.1f}")
    return "\n".join(lines)


def format_modulus_chain(primes_bits: List[float],
                         labels: Optional[List[str]] = None) -> str:
    """格式化打印模数链。"""
    if labels and len(labels) == len(primes_bits):
        parts = [f"{lbl}={b:.0f}" for lbl, b in zip(labels, primes_bits)]
    else:
        parts = [f"{b:.0f}" for b in primes_bits]
    return "ModulusChain: (" + ", ".join(parts) + f")  total={sum(primes_bits):.0f} bits"
