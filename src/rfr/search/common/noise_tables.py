"""Torch-free access to the BLB noise-variance table.

The canonical source of truth is ``function_handler._NOISE_STD_RAW`` (the
``noise_std_table.csv`` import). ``function_handler`` imports torch at module
load, so importing it pulls torch in — unusable on a torch-free box (local dev)
or in the torch-free test lane. This module extracts ``_NOISE_STD_RAW`` from the
``function_handler.py`` *source* via ``ast`` (no execution, no torch) and rebuilds
the same ``NOISE_VARIANCE_TABLE_BY_N`` mapping.

The extraction mirrors ``scripts/blb_verify_noise_install.load_noise_variance_table``
exactly (``encoding=std_enc^2``, ``fresh=std_fresh^2``, ``rescale=std_rs^2``,
``rotation`` reuses the rescale column). Both read the single ``_NOISE_STD_RAW``
literal, so there is no data drift.

The table stays in the torch-free common package. It reads the model-handler
source without importing torch.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Dict, Tuple

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_FUNCTION_HANDLER = _THIS_DIR.parent / "runtime" / "model_handler.py"


def _extract_noise_std_raw() -> Dict[int, Dict[int, Tuple[float, float, float]]]:
    """Pull the ``_NOISE_STD_RAW`` literal out of function_handler.py via ast."""
    src = _FUNCTION_HANDLER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_NOISE_STD_RAW":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"could not extract _NOISE_STD_RAW from {_FUNCTION_HANDLER}")


def _build_table() -> Dict[int, Dict[int, Dict[str, float]]]:
    raw = _extract_noise_std_raw()
    table: Dict[int, Dict[int, Dict[str, float]]] = {}
    for N, by_sf in raw.items():
        table[int(N)] = {}
        for sf, stds in by_sf.items():
            table[int(N)][int(sf)] = {
                "encoding": float(stds[0]) ** 2,
                "fresh": float(stds[1]) ** 2,
                "rescale": float(stds[2]) ** 2,


                "rotation": float(stds[2]) ** 2,
            }
    return table


NOISE_VARIANCE_TABLE_BY_N: Dict[int, Dict[int, Dict[str, float]]] = _build_table()
ALLOWED_N: Tuple[int, ...] = tuple(sorted(NOISE_VARIANCE_TABLE_BY_N))
ALLOWED_SCALING_FACTORS_BY_N: Dict[int, Tuple[int, ...]] = {
    _N: tuple(sorted(_t)) for _N, _t in NOISE_VARIANCE_TABLE_BY_N.items()
}


def variance(N: int, scaling_factor: int, distribution: str) -> float:
    """Variance for ``N(0, var)`` at the given (N, scale_bits, distribution).

    A scaling factor ABOVE the table max installs no measurable noise: var(46) is
    ~2.8e-25 and each +1 SF is ~x0.25, so var(>46) is far below fp precision
    (var(49)~4e-27). Such a point is treated as 0 (no noise) — matching
    ``function_handler.get_input_noise_variance_by_N`` — so the precision boost can
    push an installed point past the table max instead of being blocked by it. SF
    BELOW the table min stays a KeyError (that regime is snapped to the table min
    upstream and never looked up here).
    """
    n, sf, dist = int(N), int(scaling_factor), str(distribution).lower()
    tbl = NOISE_VARIANCE_TABLE_BY_N.get(n)
    if tbl is not None and tbl and sf > max(tbl):
        return 0.0
    try:
        return float(NOISE_VARIANCE_TABLE_BY_N[n][sf][dist])
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"no noise variance for N={N}, sf={scaling_factor}, dist={distribution!r}") from exc
