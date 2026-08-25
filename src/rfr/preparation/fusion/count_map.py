"""Fusion-count action map: pluggable NoiseOrder + runtime map data/lookup.

torch-free. The fusion-count map (built offline by
``rfr.preparation.fusion.build_map``) replaces per-slot SF decisions with a
per-block ``(fusion_option, K)`` choice: each ``fusion_option`` is the
minimum-noise SF combination achieving a given ``fusion_count`` for a block-type.
This module is (1) the pluggable noise partial order used to pick "minimum noise"
and (2) the runtime data/lookup layer.

Imports only the repository noise tables, so map inspection remains torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import pathlib
from typing import Dict, List, Protocol, Sequence, runtime_checkable

import numpy as np

from rfr.search.common import noise_tables
from rfr.common.json_utils import read_json_file


FUSION_CONFIG_ROOT = (
    pathlib.Path(__file__).resolve().parents[4]
    / "configs"
    / "preparation"
    / "fusion"
)


@dataclass(frozen=True)
class InstalledNoisePoint:
    """One Gaussian noise point actually installed after replan + override.

    Fused-away rescale points are *absent* from the plan (0 contribution); K
    (truncation) is not Gaussian and never appears here.
    """

    scaling_factor: int
    distribution: str
    N: int


@runtime_checkable
class NoiseOrder(Protocol):
    """Pluggable partial order over installed noise plans. Lower == quieter.

    Swap the implementation to redefine "minimum noise" without touching the
    builder or the runtime.
    """

    name: str

    def total_variance(self, installed_points: Sequence[InstalledNoisePoint]) -> float: ...


class SummedInstalledVariance:
    """Default NoiseOrder: Σ σ² over the actually-installed Gaussian points."""

    name = "summed_installed_variance"

    def total_variance(self, installed_points: Sequence[InstalledNoisePoint]) -> float:
        total = 0.0
        for p in installed_points:
            total += noise_tables.variance(int(p.N), int(p.scaling_factor), str(p.distribution))
        return total


@dataclass(frozen=True)
class FusionOption:
    """One RL-selectable option for a block-type.

    ``action_indices`` is the FULL per-block slot index vector (length =
    ``block_num_slots``), with the K slot left at a baseline placeholder — the
    env overwrites it with the separately-decided K (see :meth:`FusionCountMap.expand`).
    ``option_id == 0`` is always the baseline (all-max SF) by construction.
    """

    option_id: int
    fusion_count: int
    tie_index: int
    total_variance: float
    total_bits: int
    slots: Dict[str, int]
    action_indices: List[int]


    boosted: bool = False
    explicit_field_values: Dict[str, int] = None  # type: ignore[assignment]


@dataclass
class BlockTypeFusionMap:
    graph_key: str
    k_slot_index: int
    block_num_slots: int
    options: List[FusionOption]


@dataclass
class FusionCountMap:
    profile: str
    graphs: Dict[str, BlockTypeFusionMap]
    _max_num_options: int

    @staticmethod
    def _option_from_dict(o: dict) -> FusionOption:
        return FusionOption(
            option_id=int(o["option_id"]),
            fusion_count=int(o["fusion_count"]),
            tie_index=int(o["tie_index"]),
            total_variance=float(o["total_variance"]),
            total_bits=int(o["total_bits"]),
            slots={str(k): int(v) for k, v in dict(o.get("slots", {})).items()},
            action_indices=[int(x) for x in o["action_indices"]],
            boosted=bool(o.get("boosted", False)),
            explicit_field_values=(
                {str(k): int(v) for k, v in dict(o["explicit_field_values"]).items()}
                if o.get("explicit_field_values") else None
            ),
        )

    @classmethod
    def from_payload(cls, payload: dict) -> FusionCountMap:
        graphs: Dict[str, BlockTypeFusionMap] = {}
        for gk, g in payload["graphs"].items():
            opts = [cls._option_from_dict(o) for o in g["options"]]
            assert opts and opts[0].option_id == 0, (
                f"{gk}: option 0 must be the baseline (option_id==0); got {opts[0].option_id if opts else 'none'}"
            )
            graphs[str(gk)] = BlockTypeFusionMap(
                graph_key=str(g["graph_key"]),
                k_slot_index=int(g["k_slot_index"]),
                block_num_slots=int(g["block_num_slots"]),
                options=opts,
            )
        return cls(
            profile=str(payload["profile"]),
            graphs=graphs,
            _max_num_options=int(payload.get("max_num_options", 0))
            or max((len(g.options) for g in graphs.values()), default=0),
        )

    @classmethod
    def load(cls, profile: str, root: str | None = None) -> FusionCountMap:
        map_dir = (
            pathlib.Path(root) / "fusion_maps" / profile
            if root
            else FUSION_CONFIG_ROOT / "maps" / profile
        )
        merged: dict = {"profile": profile, "graphs": {}, "max_num_options": 0}
        mx = 0
        for path in _iter_map_paths(map_dir):
            g = read_json_file(path)
            merged["graphs"][str(g["graph_key"])] = g
            mx = max(mx, len(g["options"]))
        if not merged["graphs"]:
            raise FileNotFoundError(f"no fusion-count map JSON under {map_dir}")
        merged["max_num_options"] = mx
        return cls.from_payload(merged)

    def options(self, graph_key: str) -> List[FusionOption]:
        return self.graphs[graph_key].options

    def num_options(self, graph_key: str) -> int:
        return len(self.graphs[graph_key].options)

    def baseline_option_id(self, graph_key: str) -> int:
        return 0

    def max_num_options(self) -> int:
        return self._max_num_options

    def k_slot_index(self, graph_key: str) -> int:
        return self.graphs[graph_key].k_slot_index

    def expand(self, graph_key: str, option_id: int, k_index: int) -> np.ndarray:
        """Full per-block slot index vector for ``option_id`` with the K slot
        overwritten by the separately-decided ``k_index``."""
        g = self.graphs[graph_key]
        opt = g.options[int(option_id)]
        vec = np.asarray(opt.action_indices, dtype=int).copy()
        vec[int(g.k_slot_index)] = int(k_index)
        return vec


def _iter_map_paths(map_dir: pathlib.Path) -> List[pathlib.Path]:
    try:
        names = sorted(
            entry.name
            for entry in os.scandir(map_dir)
            if entry.is_file()
            and entry.name.endswith(".json")
            and entry.name.startswith("block")
        )
    except OSError:
        return []
    return [map_dir / name for name in names]
