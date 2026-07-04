"""Small shared statistics helpers for reports and diagnostics.

These helpers intentionally cover only simple report math.  Keep model/reward
semantics in their owning modules.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def mean_or_none(values: Iterable[float]) -> float | None:
    count = 0

    def iter_floats():
        nonlocal count
        for value in values:
            count += 1
            yield float(value)

    total = math.fsum(iter_floats())
    if not count:
        return None
    return float(total) / float(count)


def mean_or_default(values: Iterable[float], *, default: float = 0.0) -> float:
    mean = mean_or_none(values)
    return float(default) if mean is None else float(mean)


def mean_from_total(total: float, count: int, *, default: float = 0.0) -> float:
    return float(total) / float(count) if int(count) else float(default)


def ratio_or_default(numer: float, denom: float, *, default: float = 0.0) -> float:
    return float(numer) / float(denom) if float(denom) else float(default)


def safe_div_or_none(numer: float, denom: float) -> float | None:
    if float(denom) <= 0.0:
        return None
    return float(numer) / float(denom)


def fraction_true(flags: Iterable[bool], *, default: float = 0.0) -> float:
    total = 0
    count = 0
    for flag in flags:
        total += 1
        if bool(flag):
            count += 1
    return float(count) / float(total) if total else float(default)


def median_sorted(values: Sequence[float], *, default: float = 0.0) -> float:
    count = len(values)
    if not count:
        return float(default)
    mid = count // 2
    if count % 2:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0
