"""Optional exact-FP32 CUDA fusion for binary output truncation."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except (ImportError, ModuleNotFoundError):  # pragma: no cover - CPU/local lane.
    triton = None
    tl = None
    libdevice = None


if triton is not None:
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
    def _div_rn_f32(left, right):
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def binary_truncation_rn_f32(value, scale):
        scaled = _mul_rn_f32(value, scale)
        truncated = libdevice.trunc(scaled)
        return _div_rn_f32(truncated, scale)


    @triton.jit
    def _binary_truncation_kernel(
            x_ptr,
            out_ptr,
            numel,
            scale,
            BLOCK_SIZE: tl.constexpr,
            ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        value = tl.load(x_ptr + offsets, mask=mask)
        out = binary_truncation_rn_f32(value, scale)
        tl.store(out_ptr + offsets, out, mask=mask)


def is_available() -> bool:
    return triton is not None


def binary_truncation_cuda(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return ``trunc(x * 2**k) / 2**k`` in one exact-FP32 CUDA kernel."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    out = torch.empty_like(x)
    numel = int(x.numel())
    scale = float(2 ** int(k))
    _binary_truncation_kernel[(triton.cdiv(numel, 256),)](
        x,
        out,
        numel,
        scale,
        BLOCK_SIZE=256,
    )
    return out
