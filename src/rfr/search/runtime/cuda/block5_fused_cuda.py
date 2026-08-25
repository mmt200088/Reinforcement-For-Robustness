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
    def _evaluate_polynomial_from_powers(
            x,
            x2,
            x3,
            x4,
            coefficient0_noise,
            coefficient1_noise,
            rescale1_noise,
            coefficient2_noise,
            rescale2_noise,
            coefficient3_noise,
            rescale3_noise,
            coefficient4_noise,
            rescale4_noise,
            coefficient0,
            coefficient1,
            coefficient2,
            coefficient3,
            coefficient4,
            HAS_RESCALE1_NOISE: tl.constexpr,
            HAS_RESCALE2_NOISE: tl.constexpr,
            HAS_RESCALE3_NOISE: tl.constexpr,
            HAS_RESCALE4_NOISE: tl.constexpr,
            ):
        result = _add_rn_f32(coefficient0_noise, coefficient0)
        noisy_coefficient1 = _add_rn_f32(
            coefficient1_noise,
            coefficient1,
        )
        term1 = _mul_rn_f32(x, noisy_coefficient1)
        if HAS_RESCALE1_NOISE:
            term1 = _add_rn_f32(term1, rescale1_noise)
        result = _add_rn_f32(result, term1)

        noisy_coefficient2 = _add_rn_f32(
            coefficient2_noise,
            coefficient2,
        )
        term2 = _mul_rn_f32(x2, noisy_coefficient2)
        if HAS_RESCALE2_NOISE:
            term2 = _add_rn_f32(term2, rescale2_noise)
        result = _add_rn_f32(result, term2)

        noisy_coefficient3 = _add_rn_f32(
            coefficient3_noise,
            coefficient3,
        )
        term3 = _mul_rn_f32(x3, noisy_coefficient3)
        if HAS_RESCALE3_NOISE:
            term3 = _add_rn_f32(term3, rescale3_noise)
        result = _add_rn_f32(result, term3)

        noisy_coefficient4 = _add_rn_f32(
            coefficient4_noise,
            coefficient4,
        )
        term4 = _mul_rn_f32(x4, noisy_coefficient4)
        if HAS_RESCALE4_NOISE:
            term4 = _add_rn_f32(term4, rescale4_noise)
        return _add_rn_f32(result, term4)


    @triton.jit
    def _load_workspace(workspace_ptr, offsets, mask, numel, row: tl.constexpr):
        return tl.load(workspace_ptr + row * numel + offsets, mask=mask)


    @triton.jit
    def _piecewise_polynomial_kernel(
            x_ptr,
            workspace_ptr,
            out_ptr,
            numel,
            negative_coefficient0,
            negative_coefficient1,
            negative_coefficient2,
            negative_coefficient3,
            negative_coefficient4,
            positive_coefficient0,
            positive_coefficient1,
            positive_coefficient2,
            positive_coefficient3,
            positive_coefficient4,
            truncation_scale,
            HAS_POWER2_NOISE: tl.constexpr,
            HAS_POWER3_NOISE: tl.constexpr,
            HAS_POWER4_NOISE: tl.constexpr,
            NEGATIVE_HAS_RESCALE1_NOISE: tl.constexpr,
            NEGATIVE_HAS_RESCALE2_NOISE: tl.constexpr,
            NEGATIVE_HAS_RESCALE3_NOISE: tl.constexpr,
            NEGATIVE_HAS_RESCALE4_NOISE: tl.constexpr,
            POSITIVE_HAS_RESCALE1_NOISE: tl.constexpr,
            POSITIVE_HAS_RESCALE2_NOISE: tl.constexpr,
            POSITIVE_HAS_RESCALE3_NOISE: tl.constexpr,
            POSITIVE_HAS_RESCALE4_NOISE: tl.constexpr,
            APPLY_TRUNCATION: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
            ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask)

        x2 = _mul_rn_f32(x, x)
        if HAS_POWER2_NOISE:
            x2 = _add_rn_f32(
                x2,
                _load_workspace(workspace_ptr, offsets, mask, numel, 0),
            )
        x3 = _mul_rn_f32(x2, x)
        if HAS_POWER3_NOISE:
            x3 = _add_rn_f32(
                x3,
                _load_workspace(workspace_ptr, offsets, mask, numel, 1),
            )
        x4 = _mul_rn_f32(x2, x2)
        if HAS_POWER4_NOISE:
            x4 = _add_rn_f32(
                x4,
                _load_workspace(workspace_ptr, offsets, mask, numel, 2),
            )

        negative = _evaluate_polynomial_from_powers(
            x,
            x2,
            x3,
            x4,
            _load_workspace(workspace_ptr, offsets, mask, numel, 3),
            _load_workspace(workspace_ptr, offsets, mask, numel, 4),
            _load_workspace(workspace_ptr, offsets, mask, numel, 5),
            _load_workspace(workspace_ptr, offsets, mask, numel, 6),
            _load_workspace(workspace_ptr, offsets, mask, numel, 7),
            _load_workspace(workspace_ptr, offsets, mask, numel, 8),
            _load_workspace(workspace_ptr, offsets, mask, numel, 9),
            _load_workspace(workspace_ptr, offsets, mask, numel, 10),
            _load_workspace(workspace_ptr, offsets, mask, numel, 11),
            negative_coefficient0,
            negative_coefficient1,
            negative_coefficient2,
            negative_coefficient3,
            negative_coefficient4,
            HAS_RESCALE1_NOISE=NEGATIVE_HAS_RESCALE1_NOISE,
            HAS_RESCALE2_NOISE=NEGATIVE_HAS_RESCALE2_NOISE,
            HAS_RESCALE3_NOISE=NEGATIVE_HAS_RESCALE3_NOISE,
            HAS_RESCALE4_NOISE=NEGATIVE_HAS_RESCALE4_NOISE,
        )
        positive = _evaluate_polynomial_from_powers(
            x,
            x2,
            x3,
            x4,
            _load_workspace(workspace_ptr, offsets, mask, numel, 12),
            _load_workspace(workspace_ptr, offsets, mask, numel, 13),
            _load_workspace(workspace_ptr, offsets, mask, numel, 14),
            _load_workspace(workspace_ptr, offsets, mask, numel, 15),
            _load_workspace(workspace_ptr, offsets, mask, numel, 16),
            _load_workspace(workspace_ptr, offsets, mask, numel, 17),
            _load_workspace(workspace_ptr, offsets, mask, numel, 18),
            _load_workspace(workspace_ptr, offsets, mask, numel, 19),
            _load_workspace(workspace_ptr, offsets, mask, numel, 20),
            positive_coefficient0,
            positive_coefficient1,
            positive_coefficient2,
            positive_coefficient3,
            positive_coefficient4,
            HAS_RESCALE1_NOISE=POSITIVE_HAS_RESCALE1_NOISE,
            HAS_RESCALE2_NOISE=POSITIVE_HAS_RESCALE2_NOISE,
            HAS_RESCALE3_NOISE=POSITIVE_HAS_RESCALE3_NOISE,
            HAS_RESCALE4_NOISE=POSITIVE_HAS_RESCALE4_NOISE,
        )
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
    if int(workspace.shape[0]) < 21:
        raise ValueError("degree-4 GELU requires twenty-one workspace rows")

    indices = tuple(int(value) for value in noise_indices)
    stds = tuple(float(value) for value in noise_stds)
    negative = tuple(float(value) for value in negative_coefficients)
    positive = tuple(float(value) for value in positive_coefficients)
    for index in indices:
        if index >= len(stds):
            raise ValueError(f"noise index {index} exceeds {len(stds)} stds")

    numel = int(x.numel())
    grid = (triton.cdiv(numel, 256),)

    def sample(slot: int, target: torch.Tensor) -> bool:
        index = indices[slot]
        if index < 0:
            return False
        target.normal_(0.0, stds[index], generator=generator)
        return True

    power_noise_flags = tuple(
        sample(slot, workspace[slot])
        for slot in range(3)
    )

    def sample_piece(
            coefficient_slot: int,
            rescale_slot: int,
            coefficient0_row: int,
            ) -> tuple[bool, bool, bool, bool]:
        if not sample(coefficient_slot, workspace[coefficient0_row]):
            raise RuntimeError("coefficient encode noise is required")
        rescale_noise_flags = []
        for degree_index in range(1, 5):
            coefficient_noise_row = coefficient0_row + degree_index * 2 - 1
            rescale_noise_row = coefficient_noise_row + 1
            if not sample(
                    coefficient_slot + degree_index,
                    workspace[coefficient_noise_row],
            ):
                raise RuntimeError("coefficient encode noise is required")
            rescale_noise_flags.append(sample(
                rescale_slot + degree_index - 1,
                workspace[rescale_noise_row],
            ))
        return tuple(rescale_noise_flags)

    negative_rescale_flags = sample_piece(3, 8, 3)
    positive_rescale_flags = sample_piece(12, 17, 12)
    out = torch.empty_like(x)
    _piecewise_polynomial_kernel[grid](
        x,
        workspace,
        out,
        numel,
        *negative,
        *positive,
        float(truncation_scale or 1.0),
        HAS_POWER2_NOISE=power_noise_flags[0],
        HAS_POWER3_NOISE=power_noise_flags[1],
        HAS_POWER4_NOISE=power_noise_flags[2],
        NEGATIVE_HAS_RESCALE1_NOISE=negative_rescale_flags[0],
        NEGATIVE_HAS_RESCALE2_NOISE=negative_rescale_flags[1],
        NEGATIVE_HAS_RESCALE3_NOISE=negative_rescale_flags[2],
        NEGATIVE_HAS_RESCALE4_NOISE=negative_rescale_flags[3],
        POSITIVE_HAS_RESCALE1_NOISE=positive_rescale_flags[0],
        POSITIVE_HAS_RESCALE2_NOISE=positive_rescale_flags[1],
        POSITIVE_HAS_RESCALE3_NOISE=positive_rescale_flags[2],
        POSITIVE_HAS_RESCALE4_NOISE=positive_rescale_flags[3],
        APPLY_TRUNCATION=truncation_scale is not None,
        BLOCK_SIZE=256,
    )
    return out
