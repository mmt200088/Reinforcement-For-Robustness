"""Optional exact-FP32 CUDA fusion for the Block5 degree-4 GELU hot path."""
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
    from .truncation_fused_cuda import binary_truncation_rn_f32


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
    def _power_kernel(
            left_ptr,
            right_ptr,
            noise_ptr,
            out_ptr,
            numel,
            HAS_NOISE: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
            ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        value = _mul_rn_f32(
            tl.load(left_ptr + offsets, mask=mask),
            tl.load(right_ptr + offsets, mask=mask),
        )
        if HAS_NOISE:
            value = _add_rn_f32(
                value,
                tl.load(noise_ptr + offsets, mask=mask),
            )
        tl.store(out_ptr + offsets, value, mask=mask)


    @triton.jit
    def _accumulate_piece_kernel(
            power_ptr,
            coefficient_noise_ptr,
            rescale_noise_ptr,
            accumulator_ptr,
            numel,
            coefficient,
            base_coefficient,
            HAS_RESCALE_NOISE: tl.constexpr,
            INITIALIZE_ACCUMULATOR: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
            ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        noisy_coefficient = _add_rn_f32(
            tl.load(coefficient_noise_ptr + offsets, mask=mask),
            coefficient,
        )
        term = _mul_rn_f32(
            tl.load(power_ptr + offsets, mask=mask),
            noisy_coefficient,
        )
        if HAS_RESCALE_NOISE:
            term = _add_rn_f32(
                term,
                tl.load(rescale_noise_ptr + offsets, mask=mask),
            )
        accumulator = tl.load(accumulator_ptr + offsets, mask=mask)
        if INITIALIZE_ACCUMULATOR:
            accumulator = _add_rn_f32(accumulator, base_coefficient)
        result = _add_rn_f32(accumulator, term)
        tl.store(accumulator_ptr + offsets, result, mask=mask)


    @triton.jit
    def _accumulate_and_select_piece_kernel(
            power_ptr,
            coefficient_noise_ptr,
            rescale_noise_ptr,
            accumulator_ptr,
            x_ptr,
            negative_ptr,
            out_ptr,
            numel,
            coefficient,
            truncation_scale,
            HAS_RESCALE_NOISE: tl.constexpr,
            APPLY_TRUNCATION: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
            ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        noisy_coefficient = _add_rn_f32(
            tl.load(coefficient_noise_ptr + offsets, mask=mask),
            coefficient,
        )
        term = _mul_rn_f32(
            tl.load(power_ptr + offsets, mask=mask),
            noisy_coefficient,
        )
        if HAS_RESCALE_NOISE:
            term = _add_rn_f32(
                term,
                tl.load(rescale_noise_ptr + offsets, mask=mask),
            )
        positive = _add_rn_f32(
            tl.load(accumulator_ptr + offsets, mask=mask),
            term,
        )
        x = tl.load(x_ptr + offsets, mask=mask)
        negative = tl.load(negative_ptr + offsets, mask=mask)
        out = tl.where(x < 0.0, negative, positive)
        out = tl.where(x >= -2.7, out, 0.0)
        out = tl.where(x > 2.7, x, out)
        if APPLY_TRUNCATION:
            out = binary_truncation_rn_f32(out, truncation_scale)
        tl.store(out_ptr + offsets, out, mask=mask)


def is_available() -> bool:
    return triton is not None


def block5_degree4_cuda(
        x: torch.Tensor,
        workspace: torch.Tensor,
        noise_indices: Sequence[int],
        noise_stds: Sequence[float],
        negative_coefficients: Sequence[float],
        positive_coefficients: Sequence[float],
        generator: torch.Generator,
        truncation_scale: float | None = None,
        ) -> torch.Tensor:
    """Run eager-order noise sampling with staged exact-FP32 arithmetic."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if len(noise_indices) != 21:
        raise ValueError(f"expected 21 noise indices, got {len(noise_indices)}")
    if len(negative_coefficients) != 5 or len(positive_coefficients) != 5:
        raise ValueError("degree-4 GELU requires five coefficients per piece")
    if workspace.dim() != x.dim() + 1 or tuple(workspace.shape[1:]) != tuple(x.shape):
        raise ValueError("workspace trailing dimensions must match x")
    if int(workspace.shape[0]) < 7:
        raise ValueError("degree-4 GELU requires seven workspace rows")

    indices = tuple(int(value) for value in noise_indices)
    stds = tuple(float(value) for value in noise_stds)
    negative = tuple(float(value) for value in negative_coefficients)
    positive = tuple(float(value) for value in positive_coefficients)
    for index in indices:
        if index >= len(stds):
            raise ValueError(f"noise index {index} exceeds {len(stds)} stds")

    noise0, noise1 = workspace[0], workspace[1]
    powers = (None, x, workspace[2], workspace[3], workspace[4])
    negative_out, positive_out = workspace[5], workspace[6]
    numel = int(x.numel())
    grid = (triton.cdiv(numel, 256),)

    def sample(slot: int, target: torch.Tensor) -> bool:
        index = indices[slot]
        if index < 0:
            return False
        target.normal_(0.0, stds[index], generator=generator)
        return True

    for slot, left, right, out in (
            (0, x, x, powers[2]),
            (1, powers[2], x, powers[3]),
            (2, powers[2], powers[2], powers[4]),
            ):
        has_noise = sample(slot, noise0)
        _power_kernel[grid](
            left,
            right,
            noise0,
            out,
            numel,
            HAS_NOISE=has_noise,
            BLOCK_SIZE=256,
        )

    def compute_piece(
            coefficients: Sequence[float],
            coefficient_slot: int,
            rescale_slot: int,
            accumulator: torch.Tensor,
            final_output: torch.Tensor | None = None,
            negative_output: torch.Tensor | None = None,
            ) -> None:
        if not sample(coefficient_slot, accumulator):
            raise RuntimeError("coefficient encode noise is required")
        for degree_index in range(1, 5):
            if not sample(coefficient_slot + degree_index, noise0):
                raise RuntimeError("coefficient encode noise is required")
            has_rescale = sample(rescale_slot + degree_index - 1, noise1)
            if final_output is not None and degree_index == 4:
                if negative_output is None:
                    raise RuntimeError("negative piece output is required")
                _accumulate_and_select_piece_kernel[grid](
                    powers[degree_index],
                    noise0,
                    noise1,
                    accumulator,
                    x,
                    negative_output,
                    final_output,
                    numel,
                    coefficients[degree_index],
                    float(truncation_scale or 1.0),
                    HAS_RESCALE_NOISE=has_rescale,
                    APPLY_TRUNCATION=truncation_scale is not None,
                    BLOCK_SIZE=256,
                )
            else:
                _accumulate_piece_kernel[grid](
                    powers[degree_index],
                    noise0,
                    noise1,
                    accumulator,
                    numel,
                    coefficients[degree_index],
                    coefficients[0],
                    HAS_RESCALE_NOISE=has_rescale,
                    INITIALIZE_ACCUMULATOR=degree_index == 1,
                    BLOCK_SIZE=256,
                )

    compute_piece(negative, 3, 8, negative_out)
    out = torch.empty_like(x)
    compute_piece(
        positive,
        12,
        17,
        positive_out,
        final_output=out,
        negative_output=negative_out,
    )
    return out
