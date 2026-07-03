"""Sub-stage Stage-2 RL env.

Thin filter over :class:`BLBStage2SequentialEnv` so a single training run only
exposes decisions for ONE block (e.g. all layer's block 2). Other blocks'
actions come from a fixed ``frozen_base_action_vec`` supplied at construction
time -- those slots are pre-spliced into ``_pending_full_vec`` at every
``reset()`` so the terminal full-vec evaluation and the per-step
``ReplanSession`` call see the correct cfgs for non-active blocks.

Used by the new 4-sub-stage path (block 1 → 2 → 4 → 5; block 3 always
frozen to ``static_skeletons`` baseline). The full sequential path (47 steps
for L=12 after block 3 was removed from the schedule, C 2026-05-30) is untouched.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

import numpy as np

from .action_space import BlockStepSpec, step_schedule
from .env import BLBStage2Env
from .sequential_env import BLBStage2SequentialEnv, SequentialEnvConfig


class BLBStage2SubstageEnv(BLBStage2SequentialEnv):
    """Sub-stage sequential env: train one block, freeze every other slot.

    ``active_block_idx`` selects the block under RL control (1, 2, 4, or 5;
    block 3 is always part of the frozen base). ``frozen_base_action_vec``
    contains the values for every slot in the full 577-dim action; the active
    block's slots in that vec are still spliced into ``_pending_full_vec``
    at reset, but RL writes over them every step.
    """

    def __init__(
            self,
            *,
            base_env: BLBStage2Env,
            active_block_idx: int,
            frozen_base_action_vec: Sequence[int],
            env_cfg: Optional[SequentialEnvConfig] = None,
            ):
        if int(active_block_idx) not in (1, 2, 4, 5):
            raise ValueError(
                f"active_block_idx must be one of (1,2,4,5); got {active_block_idx}. "
                "Block 3 is frozen by design (see CLAUDE.md sub-stage path)."
            )
        self._active_block_idx = int(active_block_idx)
        self._frozen_base_action_vec = np.asarray(
            frozen_base_action_vec, dtype=np.int64
        ).reshape(-1).copy()
        # Parent __init__ calls _rebuild_schedule, which we override below.
        super().__init__(base_env=base_env, env_cfg=env_cfg)

    @property
    def active_block_idx(self) -> int:
        return self._active_block_idx

    @property
    def frozen_base_action_vec(self) -> np.ndarray:
        return self._frozen_base_action_vec.copy()

    def update_frozen_base(self, new_base: Sequence[int]) -> None:
        """Replace the frozen base used at next reset. Useful between sub-stages
        when the orchestrator wants to keep the same env instance alive."""
        new_arr = np.asarray(new_base, dtype=np.int64).reshape(-1)
        if new_arr.size != self._frozen_base_action_vec.size:
            raise ValueError(
                f"new_base size {new_arr.size} != current frozen_base size "
                f"{self._frozen_base_action_vec.size}"
            )
        self._frozen_base_action_vec = new_arr.copy()

    # ------------------------------------------------------------------
    # overrides
    # ------------------------------------------------------------------
    def _rebuild_schedule(self) -> None:
        """Build the substage schedule: filter to active block + renumber.

        Each emitted spec keeps its original ``full_vec_offsets`` (those index
        into the legacy 577-dim vec and are stable), but ``step_idx`` and
        ``terminal`` are renumbered for the new shorter horizon.
        """
        # Mirror the parent's degree-broadcast so block 3/5 graph keys pick up
        # the per-layer Stage-1 degrees.
        def _broadcast(value, default: int) -> List[int]:
            if isinstance(value, int):
                return [int(value)] * self.num_layers
            try:
                arr = list(np.asarray(value, dtype=int).reshape(-1).tolist())
            except Exception:
                return [int(default)] * self.num_layers
            if not arr:
                return [int(default)] * self.num_layers
            if len(arr) == 1:
                return [int(arr[0])] * self.num_layers
            if len(arr) != self.num_layers:
                if len(arr) > self.num_layers:
                    arr = arr[: self.num_layers]
                else:
                    arr = arr + [arr[-1]] * (self.num_layers - len(arr))
            return [int(x) for x in arr]

        attn = _broadcast(self.base.attn_degree, 4)
        gelu = _broadcast(self.base.gelu_degree, 4)
        full = step_schedule(
            self.num_layers,
            profile=self.profile,
            attn_degree_per_layer=attn,
            gelu_degree_per_layer=gelu,
        )
        # first_input fresh is deprecated and no longer appears in any substage
        # action. It stays frozen in the legacy full action vector and is ignored
        # by model installation/final eval.
        active_specs = [s for s in full if s.block_idx == self._active_block_idx]
        if not active_specs:
            raise RuntimeError(
                f"step_schedule produced no specs for active_block_idx="
                f"{self._active_block_idx}; check num_layers={self.num_layers}"
            )
        renumbered: List[BlockStepSpec] = []
        n = len(active_specs)
        for new_idx, spec in enumerate(active_specs):
            renumbered.append(dataclasses.replace(
                spec,
                step_idx=new_idx,
                terminal=(new_idx == n - 1),
            ))
        self._schedule = renumbered
        # Parent's horizon attribute is set in __init__ from the full schedule;
        # override here so _build_obs sees the substage horizon.
        self.horizon = n
        # _max_step_dim controls policy actor-head width and prev_actions
        # padding. Substage's max is over its own specs only.
        self._max_step_dim = max(len(s.slot_dims) for s in renumbered)

    def reset(self, *, seed: Optional[int] = None) -> np.ndarray:
        """Same as parent but pre-fill the accumulator with the frozen base."""
        obs = super().reset(seed=seed)
        # Parent reset seeds _pending_full_vec with the all-max baseline; replace
        # with frozen base so the per-step ReplanSession sees every other block's
        # chosen action and the terminal full-eval forward installs every
        # non-active block too. (frozen_base also carries block 3 at its baseline.)
        if self._frozen_base_action_vec.size != self._pending_full_vec.size:
            raise RuntimeError(
                f"frozen_base size {self._frozen_base_action_vec.size} != "
                f"full action vec size {self._pending_full_vec.size}"
            )
        self._pending_full_vec = self._frozen_base_action_vec.astype(
            self._pending_full_vec.dtype
        ).copy()
        # _build_obs uses the (now-substage) schedule; rebuild the obs.
        return self._build_obs()
