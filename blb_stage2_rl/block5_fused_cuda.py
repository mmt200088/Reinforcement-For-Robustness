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


_NOISE_STD_TENSOR_CACHE = {}
_NOISE_STD_TENSOR_CACHE_MAXSIZE = 256


def _noise_std_tensor(
        workspace: torch.Tensor,
        stds: Sequence[float],
        ) -> torch.Tensor:
    stream = torch.cuda.current_stream(workspace.device)
    normalized_stds = tuple(float(value) for value in stds)
    key = (
        str(workspace.device),
        int(stream.cuda_stream),
        workspace.dtype,
        int(workspace.dim()),
        normalized_stds,
    )
    tensor = _NOISE_STD_TENSOR_CACHE.pop(key, None)
    if tensor is None:
        tensor = torch.tensor(
            normalized_stds,
            device=workspace.device,
            dtype=workspace.dtype,
        ).view(len(normalized_stds), *([1] * (workspace.dim() - 1)))
    _NOISE_STD_TENSOR_CACHE[key] = tensor
    if len(_NOISE_STD_TENSOR_CACHE) > _NOISE_STD_TENSOR_CACHE_MAXSIZE:
        del _NOISE_STD_TENSOR_CACHE[next(iter(_NOISE_STD_TENSOR_CACHE))]
    return tensor


def _sample_gaussian_rows_cuda(
        workspace: torch.Tensor,
        start_row: int,
        stds: Sequence[float],
        generator: torch.Generator,
        ) -> torch.Tensor:
    """Sample consecutive same-shape rows without changing CUDA RNG results."""
    start = int(start_row)
    count = len(stds)
    if start < 0 or start + count > int(workspace.shape[0]):
        raise ValueError("grouped noise rows exceed the workspace")
    target = workspace[start:start + count]
    if count == 0:
        return target

    properties = torch.cuda.get_device_properties(workspace.device)
    block_size = 256
    blocks_per_sm = properties.max_threads_per_multi_processor // block_size
    philox_grid_period = (
        block_size * blocks_per_sm * properties.multi_processor_count * 4
    )
    row_numel = int(target[0].numel())
    if (
            workspace.dtype != torch.float32
            or not target.is_contiguous()
            or row_numel % philox_grid_period != 0
    ):
        for row, std in zip(target, stds):
            row.normal_(0.0, float(std), generator=generator)
        return target

    scales = _noise_std_tensor(workspace, stds).expand_as(target)
    torch.normal(0.0, scales, generator=generator, out=target)
    return target


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
    def _evaluate_polynomial_piece(
            x,
            power2_noise,
            power3_noise,
            power4_noise,
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
            HAS_POWER2_NOISE: tl.constexpr,
            HAS_POWER3_NOISE: tl.constexpr,
            HAS_POWER4_NOISE: tl.constexpr,
            HAS_RESCALE1_NOISE: tl.constexpr,
            HAS_RESCALE2_NOISE: tl.constexpr,
            HAS_RESCALE3_NOISE: tl.constexpr,
            HAS_RESCALE4_NOISE: tl.constexpr,
            ):
        x2 = _mul_rn_f32(x, x)
        if HAS_POWER2_NOISE:
            x2 = _add_rn_f32(x2, power2_noise)
        x3 = _mul_rn_f32(x2, x)
        if HAS_POWER3_NOISE:
            x3 = _add_rn_f32(x3, power3_noise)
        x4 = _mul_rn_f32(x2, x2)
        if HAS_POWER4_NOISE:
            x4 = _add_rn_f32(x4, power4_noise)

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
    def _load_optional_workspace(
            workspace_ptr,
            offsets,
            mask,
            numel,
            row: tl.constexpr,
            enabled: tl.constexpr,
            ):
        if enabled:
            return _load_workspace(workspace_ptr, offsets, mask, numel, row)
        return tl.zeros(offsets.shape, tl.float32)


    @triton.jit
    def _polynomial_piece_kernel(
            x_ptr,
            workspace_ptr,
            out_ptr,
            numel,
            coefficient0,
            coefficient1,
            coefficient2,
            coefficient3,
            coefficient4,
            HAS_POWER2_NOISE: tl.constexpr,
            HAS_POWER3_NOISE: tl.constexpr,
            HAS_POWER4_NOISE: tl.constexpr,
            HAS_RESCALE1_NOISE: tl.constexpr,
            HAS_RESCALE2_NOISE: tl.constexpr,
            HAS_RESCALE3_NOISE: tl.constexpr,
            HAS_RESCALE4_NOISE: tl.constexpr,
            POWER2_ROW: tl.constexpr,
            POWER3_ROW: tl.constexpr,
            POWER4_ROW: tl.constexpr,
            COEFFICIENT0_ROW: tl.constexpr,
            COEFFICIENT1_ROW: tl.constexpr,
            RESCALE1_ROW: tl.constexpr,
            COEFFICIENT2_ROW: tl.constexpr,
            RESCALE2_ROW: tl.constexpr,
            COEFFICIENT3_ROW: tl.constexpr,
            RESCALE3_ROW: tl.constexpr,
            COEFFICIENT4_ROW: tl.constexpr,
            RESCALE4_ROW: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
            ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask)
        out = _evaluate_polynomial_piece(
            x,
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                POWER2_ROW, HAS_POWER2_NOISE,
            ),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                POWER3_ROW, HAS_POWER3_NOISE,
            ),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                POWER4_ROW, HAS_POWER4_NOISE,
            ),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT0_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT1_ROW),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                RESCALE1_ROW, HAS_RESCALE1_NOISE,
            ),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT2_ROW),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                RESCALE2_ROW, HAS_RESCALE2_NOISE,
            ),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT3_ROW),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                RESCALE3_ROW, HAS_RESCALE3_NOISE,
            ),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT4_ROW),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                RESCALE4_ROW, HAS_RESCALE4_NOISE,
            ),
            coefficient0,
            coefficient1,
            coefficient2,
            coefficient3,
            coefficient4,
            HAS_POWER2_NOISE=HAS_POWER2_NOISE,
            HAS_POWER3_NOISE=HAS_POWER3_NOISE,
            HAS_POWER4_NOISE=HAS_POWER4_NOISE,
            HAS_RESCALE1_NOISE=HAS_RESCALE1_NOISE,
            HAS_RESCALE2_NOISE=HAS_RESCALE2_NOISE,
            HAS_RESCALE3_NOISE=HAS_RESCALE3_NOISE,
            HAS_RESCALE4_NOISE=HAS_RESCALE4_NOISE,
        )
        tl.store(out_ptr + offsets, out, mask=mask)


    @triton.jit
    def _polynomial_piece_and_select_kernel(
            x_ptr,
            workspace_ptr,
            negative_ptr,
            out_ptr,
            numel,
            coefficient0,
            coefficient1,
            coefficient2,
            coefficient3,
            coefficient4,
            truncation_scale,
            HAS_POWER2_NOISE: tl.constexpr,
            HAS_POWER3_NOISE: tl.constexpr,
            HAS_POWER4_NOISE: tl.constexpr,
            HAS_RESCALE1_NOISE: tl.constexpr,
            HAS_RESCALE2_NOISE: tl.constexpr,
            HAS_RESCALE3_NOISE: tl.constexpr,
            HAS_RESCALE4_NOISE: tl.constexpr,
            POWER2_ROW: tl.constexpr,
            POWER3_ROW: tl.constexpr,
            POWER4_ROW: tl.constexpr,
            COEFFICIENT0_ROW: tl.constexpr,
            COEFFICIENT1_ROW: tl.constexpr,
            RESCALE1_ROW: tl.constexpr,
            COEFFICIENT2_ROW: tl.constexpr,
            RESCALE2_ROW: tl.constexpr,
            COEFFICIENT3_ROW: tl.constexpr,
            RESCALE3_ROW: tl.constexpr,
            COEFFICIENT4_ROW: tl.constexpr,
            RESCALE4_ROW: tl.constexpr,
            APPLY_TRUNCATION: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
            ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask)
        positive = _evaluate_polynomial_piece(
            x,
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                POWER2_ROW, HAS_POWER2_NOISE,
            ),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                POWER3_ROW, HAS_POWER3_NOISE,
            ),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                POWER4_ROW, HAS_POWER4_NOISE,
            ),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT0_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT1_ROW),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                RESCALE1_ROW, HAS_RESCALE1_NOISE,
            ),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT2_ROW),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                RESCALE2_ROW, HAS_RESCALE2_NOISE,
            ),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT3_ROW),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                RESCALE3_ROW, HAS_RESCALE3_NOISE,
            ),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT4_ROW),
            _load_optional_workspace(
                workspace_ptr, offsets, mask, numel,
                RESCALE4_ROW, HAS_RESCALE4_NOISE,
            ),
            coefficient0,
            coefficient1,
            coefficient2,
            coefficient3,
            coefficient4,
            HAS_POWER2_NOISE=HAS_POWER2_NOISE,
            HAS_POWER3_NOISE=HAS_POWER3_NOISE,
            HAS_POWER4_NOISE=HAS_POWER4_NOISE,
            HAS_RESCALE1_NOISE=HAS_RESCALE1_NOISE,
            HAS_RESCALE2_NOISE=HAS_RESCALE2_NOISE,
            HAS_RESCALE3_NOISE=HAS_RESCALE3_NOISE,
            HAS_RESCALE4_NOISE=HAS_RESCALE4_NOISE,
        )
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
    if int(workspace.shape[0]) < 13:
        raise ValueError("degree-4 GELU requires thirteen workspace rows")

    indices = tuple(int(value) for value in noise_indices)
    stds = tuple(float(value) for value in noise_stds)
    negative = tuple(float(value) for value in negative_coefficients)
    positive = tuple(float(value) for value in positive_coefficients)
    for index in indices:
        if index >= len(stds):
            raise ValueError(f"noise index {index} exceeds {len(stds)} stds")

    negative_out = workspace[12]
    numel = int(x.numel())
    grid = (triton.cdiv(numel, 256),)

    def sample_slots(slots: Sequence[int], start_row: int):
        active_slots = tuple(slot for slot in slots if indices[slot] >= 0)
        _sample_gaussian_rows_cuda(
            workspace,
            start_row,
            tuple(stds[indices[slot]] for slot in active_slots),
            generator,
        )
        return {
            slot: start_row + offset
            for offset, slot in enumerate(active_slots)
        }

    power_rows = sample_slots((0, 1, 2), 0)

    def compute_piece(
            coefficients: Sequence[float],
            coefficient_slot: int,
            rescale_slot: int,
            rows,
            final_output: torch.Tensor | None = None,
            negative_output: torch.Tensor | None = None,
            ) -> None:
        coefficient_rows = tuple(
            rows.get(coefficient_slot + degree_index, -1)
            for degree_index in range(5)
        )
        if any(row < 0 for row in coefficient_rows):
            raise RuntimeError("coefficient encode noise is required")
        rescale_rows = tuple(
            rows.get(rescale_slot + degree_index, -1)
            for degree_index in range(4)
        )

        common_kwargs = {
            "HAS_POWER2_NOISE": 0 in power_rows,
            "HAS_POWER3_NOISE": 1 in power_rows,
            "HAS_POWER4_NOISE": 2 in power_rows,
            "HAS_RESCALE1_NOISE": rescale_rows[0] >= 0,
            "HAS_RESCALE2_NOISE": rescale_rows[1] >= 0,
            "HAS_RESCALE3_NOISE": rescale_rows[2] >= 0,
            "HAS_RESCALE4_NOISE": rescale_rows[3] >= 0,
            "POWER2_ROW": power_rows.get(0, -1),
            "POWER3_ROW": power_rows.get(1, -1),
            "POWER4_ROW": power_rows.get(2, -1),
            "COEFFICIENT0_ROW": coefficient_rows[0],
            "COEFFICIENT1_ROW": coefficient_rows[1],
            "RESCALE1_ROW": rescale_rows[0],
            "COEFFICIENT2_ROW": coefficient_rows[2],
            "RESCALE2_ROW": rescale_rows[1],
            "COEFFICIENT3_ROW": coefficient_rows[3],
            "RESCALE3_ROW": rescale_rows[2],
            "COEFFICIENT4_ROW": coefficient_rows[4],
            "RESCALE4_ROW": rescale_rows[3],
            "BLOCK_SIZE": 256,
        }
        if final_output is None:
            _polynomial_piece_kernel[grid](
                x,
                workspace,
                negative_out,
                numel,
                *coefficients,
                **common_kwargs,
            )
            return
        if negative_output is None:
            raise RuntimeError("negative piece output is required")
        _polynomial_piece_and_select_kernel[grid](
            x,
            workspace,
            negative_output,
            final_output,
            numel,
            *coefficients,
            float(truncation_scale or 1.0),
            APPLY_TRUNCATION=truncation_scale is not None,
            **common_kwargs,
        )

    negative_slots = (3, 4, 8, 5, 9, 6, 10, 7, 11)
    negative_rows = sample_slots(negative_slots, 3)
    compute_piece(negative, 3, 8, negative_rows)
    out = torch.empty_like(x)
    positive_slots = (12, 13, 17, 14, 18, 15, 19, 16, 20)
    positive_rows = sample_slots(positive_slots, 3)
    compute_piece(
        positive,
        12,
        17,
        positive_rows,
        final_output=out,
        negative_output=negative_out,
    )
    return out
