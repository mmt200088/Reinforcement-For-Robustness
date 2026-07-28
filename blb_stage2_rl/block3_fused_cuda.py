"""Optional exact-FP32 CUDA fusion for Block3 degree-4/6 hot paths."""
from __future__ import annotations

from typing import Sequence

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):  # pragma: no cover - CPU/local lane.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _add_rn_f32(left, right):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _mul_rn_f32(left, right):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _block3_degree4_kernel(
            x_ptr,
            fresh_ptr,
            inv_encode_ptr,
            square0_ptr,
            square1_ptr,
            square2_ptr,
            square3_ptr,
            out_ptr,
            numel,
            BLOCK_SIZE: tl.constexpr,
            ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask)
        fresh = tl.load(fresh_ptr + offsets, mask=mask)
        inv_encode = tl.load(inv_encode_ptr + offsets, mask=mask)
        square0 = tl.load(square0_ptr + offsets, mask=mask)
        square1 = tl.load(square1_ptr + offsets, mask=mask)
        square2 = tl.load(square2_ptr + offsets, mask=mask)
        square3 = tl.load(square3_ptr + offsets, mask=mask)

        noisy_x = _add_rn_f32(x, fresh)
        noisy_inv = _add_rn_f32(inv_encode, 0.0625)
        y = _add_rn_f32(_mul_rn_f32(noisy_x, noisy_inv), 1.0)
        y = _add_rn_f32(_mul_rn_f32(y, y), square0)
        y = _add_rn_f32(_mul_rn_f32(y, y), square1)
        y = _add_rn_f32(_mul_rn_f32(y, y), square2)
        y = _add_rn_f32(_mul_rn_f32(y, y), square3)
        tl.store(out_ptr + offsets, y, mask=mask)


    @triton.jit
    def _block3_degree6_kernel(
            x_ptr,
            fresh_ptr,
            inv_encode_ptr,
            square0_ptr,
            square1_ptr,
            square2_ptr,
            square3_ptr,
            square4_ptr,
            square5_ptr,
            out_ptr,
            numel,
            BLOCK_SIZE: tl.constexpr,
            ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask)
        fresh = tl.load(fresh_ptr + offsets, mask=mask)
        inv_encode = tl.load(inv_encode_ptr + offsets, mask=mask)
        square0 = tl.load(square0_ptr + offsets, mask=mask)
        square1 = tl.load(square1_ptr + offsets, mask=mask)
        square2 = tl.load(square2_ptr + offsets, mask=mask)
        square3 = tl.load(square3_ptr + offsets, mask=mask)
        square4 = tl.load(square4_ptr + offsets, mask=mask)
        square5 = tl.load(square5_ptr + offsets, mask=mask)

        noisy_x = _add_rn_f32(x, fresh)
        noisy_inv = _add_rn_f32(inv_encode, 0.015625)
        y = _add_rn_f32(_mul_rn_f32(noisy_x, noisy_inv), 1.0)
        y = _add_rn_f32(_mul_rn_f32(y, y), square0)
        y = _add_rn_f32(_mul_rn_f32(y, y), square1)
        y = _add_rn_f32(_mul_rn_f32(y, y), square2)
        y = _add_rn_f32(_mul_rn_f32(y, y), square3)
        y = _add_rn_f32(_mul_rn_f32(y, y), square4)
        y = _add_rn_f32(_mul_rn_f32(y, y), square5)
        tl.store(out_ptr + offsets, y, mask=mask)


def is_available() -> bool:
    return triton is not None


def block3_degree4_cuda(x: torch.Tensor, noises: Sequence[torch.Tensor]) -> torch.Tensor:
    """Apply degree-4 Block3 arithmetic to six pre-sampled noise tensors."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if len(noises) != 6:
        raise ValueError(f"expected six Block3 noise tensors, got {len(noises)}")
    out = torch.empty_like(x)
    numel = int(x.numel())
    _block3_degree4_kernel[(triton.cdiv(numel, 256),)](
        x,
        *noises,
        out,
        numel,
        BLOCK_SIZE=256,
    )
    return out


def block3_degree6_cuda(
        x: torch.Tensor,
        noises: Sequence[torch.Tensor],
        ) -> torch.Tensor:
    """Apply degree-6 Block3 arithmetic to eight pre-sampled noise tensors."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if len(noises) != 8:
        raise ValueError(
            f"expected eight Block3 noise tensors, got {len(noises)}"
        )
    out = torch.empty_like(x)
    numel = int(x.numel())
    _block3_degree6_kernel[(triton.cdiv(numel, 256),)](
        x,
        *noises,
        out,
        numel,
        BLOCK_SIZE=256,
    )
    return out
