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


_PHILOX_GRID_PERIODS = {}


def _cuda_philox_grid_period(device: torch.device) -> int:
    owned = torch.device(device)
    index = torch.cuda.current_device() if owned.index is None else int(owned.index)
    cached = _PHILOX_GRID_PERIODS.get(index)
    if cached is not None:
        return cached
    properties = torch.cuda.get_device_properties(index)
    block_size = 256
    blocks_per_sm = properties.max_threads_per_multi_processor // block_size
    period = block_size * blocks_per_sm * properties.multi_processor_count * 4
    _PHILOX_GRID_PERIODS[index] = int(period)
    return int(period)


def _can_group_block5_noise(workspace: torch.Tensor) -> bool:
    return bool(
        workspace.is_cuda
        and workspace.dtype == torch.float32
        and workspace.is_contiguous()
        and int(workspace[0].numel())
        % _cuda_philox_grid_period(workspace.device) == 0
    )


def _sample_standard_normal_rows_cuda(
        workspace: torch.Tensor,
        start_row: int,
        count: int,
        generator: torch.Generator,
        ) -> torch.Tensor:
    start = int(start_row)
    owned_count = int(count)
    if start < 0 or owned_count < 0 or start + owned_count > workspace.shape[0]:
        raise ValueError("standard-normal rows exceed the workspace")
    target = workspace[start:start + owned_count]
    if owned_count:
        target.normal_(0.0, 1.0, generator=generator)
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
    def _normal_transform_rn_f32(value, std):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, 0f00000000;",
            "=f,f,f",
            [value, std],
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
            power2_std,
            power3_std,
            power4_std,
            coefficient0_std,
            coefficient1_std,
            rescale1_std,
            coefficient2_std,
            rescale2_std,
            coefficient3_std,
            rescale3_std,
            coefficient4_std,
            rescale4_std,
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
            NOISE_IS_STANDARD_NORMAL: tl.constexpr,
            ):
        if NOISE_IS_STANDARD_NORMAL:
            power2_noise = _normal_transform_rn_f32(power2_noise, power2_std)
            power3_noise = _normal_transform_rn_f32(power3_noise, power3_std)
            power4_noise = _normal_transform_rn_f32(power4_noise, power4_std)
            coefficient0_noise = _normal_transform_rn_f32(
                coefficient0_noise, coefficient0_std,
            )
            coefficient1_noise = _normal_transform_rn_f32(
                coefficient1_noise, coefficient1_std,
            )
            rescale1_noise = _normal_transform_rn_f32(
                rescale1_noise, rescale1_std,
            )
            coefficient2_noise = _normal_transform_rn_f32(
                coefficient2_noise, coefficient2_std,
            )
            rescale2_noise = _normal_transform_rn_f32(
                rescale2_noise, rescale2_std,
            )
            coefficient3_noise = _normal_transform_rn_f32(
                coefficient3_noise, coefficient3_std,
            )
            rescale3_noise = _normal_transform_rn_f32(
                rescale3_noise, rescale3_std,
            )
            coefficient4_noise = _normal_transform_rn_f32(
                coefficient4_noise, coefficient4_std,
            )
            rescale4_noise = _normal_transform_rn_f32(
                rescale4_noise, rescale4_std,
            )
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
            power2_std,
            power3_std,
            power4_std,
            coefficient0_std,
            coefficient1_std,
            rescale1_std,
            coefficient2_std,
            rescale2_std,
            coefficient3_std,
            rescale3_std,
            coefficient4_std,
            rescale4_std,
            HAS_POWER2_NOISE: tl.constexpr,
            HAS_POWER3_NOISE: tl.constexpr,
            HAS_POWER4_NOISE: tl.constexpr,
            HAS_RESCALE1_NOISE: tl.constexpr,
            HAS_RESCALE2_NOISE: tl.constexpr,
            HAS_RESCALE3_NOISE: tl.constexpr,
            HAS_RESCALE4_NOISE: tl.constexpr,
            NOISE_IS_STANDARD_NORMAL: tl.constexpr,
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
            _load_workspace(workspace_ptr, offsets, mask, numel, POWER2_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, POWER3_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, POWER4_ROW),
            _load_workspace(
                workspace_ptr, offsets, mask, numel, COEFFICIENT0_ROW,
            ),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT1_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, RESCALE1_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT2_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, RESCALE2_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT3_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, RESCALE3_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT4_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, RESCALE4_ROW),
            power2_std,
            power3_std,
            power4_std,
            coefficient0_std,
            coefficient1_std,
            rescale1_std,
            coefficient2_std,
            rescale2_std,
            coefficient3_std,
            rescale3_std,
            coefficient4_std,
            rescale4_std,
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
            NOISE_IS_STANDARD_NORMAL=NOISE_IS_STANDARD_NORMAL,
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
            power2_std,
            power3_std,
            power4_std,
            coefficient0_std,
            coefficient1_std,
            rescale1_std,
            coefficient2_std,
            rescale2_std,
            coefficient3_std,
            rescale3_std,
            coefficient4_std,
            rescale4_std,
            truncation_scale,
            HAS_POWER2_NOISE: tl.constexpr,
            HAS_POWER3_NOISE: tl.constexpr,
            HAS_POWER4_NOISE: tl.constexpr,
            HAS_RESCALE1_NOISE: tl.constexpr,
            HAS_RESCALE2_NOISE: tl.constexpr,
            HAS_RESCALE3_NOISE: tl.constexpr,
            HAS_RESCALE4_NOISE: tl.constexpr,
            NOISE_IS_STANDARD_NORMAL: tl.constexpr,
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
            _load_workspace(workspace_ptr, offsets, mask, numel, POWER2_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, POWER3_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, POWER4_ROW),
            _load_workspace(
                workspace_ptr, offsets, mask, numel, COEFFICIENT0_ROW,
            ),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT1_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, RESCALE1_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT2_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, RESCALE2_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT3_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, RESCALE3_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, COEFFICIENT4_ROW),
            _load_workspace(workspace_ptr, offsets, mask, numel, RESCALE4_ROW),
            power2_std,
            power3_std,
            power4_std,
            coefficient0_std,
            coefficient1_std,
            rescale1_std,
            coefficient2_std,
            rescale2_std,
            coefficient3_std,
            rescale3_std,
            coefficient4_std,
            rescale4_std,
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
            NOISE_IS_STANDARD_NORMAL=NOISE_IS_STANDARD_NORMAL,
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
    """Run exact eager-order sampling, batching only Philox-aligned rows."""
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

    numel = int(x.numel())
    grid = (triton.cdiv(numel, 256),)
    group_standard_normals = _can_group_block5_noise(workspace)

    def sample_slots(slots: Sequence[int], start_row: int):
        active_slots = tuple(slot for slot in slots if indices[slot] >= 0)
        _sample_standard_normal_rows_cuda(
            workspace,
            start_row,
            len(active_slots),
            generator,
        )
        return {
            slot: start_row + offset
            for offset, slot in enumerate(active_slots)
        }

    def sample(slot: int, target: torch.Tensor) -> bool:
        index = indices[slot]
        if index < 0:
            return False
        target.normal_(0.0, stds[index], generator=generator)
        return True

    if group_standard_normals:
        power_rows = sample_slots((0, 1, 2), 0)
        power_noise_flags = tuple(indices[slot] >= 0 for slot in range(3))
        negative_out = workspace[12]
    else:
        power_noise_flags = tuple(
            sample(slot, workspace[slot])
            for slot in range(3)
        )
        power_rows = {slot: slot for slot in range(3)}
        negative_out = workspace[11]

    def noise_std(slot: int) -> float:
        index = indices[slot]
        if group_standard_normals and index >= 0:
            return stds[index]
        return 1.0

    def compute_piece(
            coefficients: Sequence[float],
            coefficient_slot: int,
            rescale_slot: int,
            coefficient0_row: int,
            final_output: torch.Tensor | None = None,
            negative_output: torch.Tensor | None = None,
            ) -> None:
        if group_standard_normals:
            piece_slots = [coefficient_slot]
            for degree_index in range(1, 5):
                piece_slots.extend((
                    coefficient_slot + degree_index,
                    rescale_slot + degree_index - 1,
                ))
            piece_rows = sample_slots(piece_slots, 3)
            coefficient_rows = tuple(
                piece_rows.get(coefficient_slot + degree_index, -1)
                for degree_index in range(5)
            )
            rescale_rows = tuple(
                piece_rows.get(rescale_slot + degree_index, 0)
                for degree_index in range(4)
            )
            rescale_noise_flags = tuple(
                indices[rescale_slot + degree_index] >= 0
                for degree_index in range(4)
            )
        else:
            if not sample(coefficient_slot, workspace[coefficient0_row]):
                raise RuntimeError("coefficient encode noise is required")
            coefficient_rows_list = [coefficient0_row]
            rescale_rows_list = []
            rescale_noise_flags_list = []
            for degree_index in range(1, 5):
                coefficient_noise_row = 3 + (degree_index - 1) * 2
                rescale_noise_row = coefficient_noise_row + 1
                if not sample(
                        coefficient_slot + degree_index,
                        workspace[coefficient_noise_row],
                ):
                    raise RuntimeError("coefficient encode noise is required")
                coefficient_rows_list.append(coefficient_noise_row)
                rescale_rows_list.append(rescale_noise_row)
                rescale_noise_flags_list.append(sample(
                    rescale_slot + degree_index - 1,
                    workspace[rescale_noise_row],
                ))
            coefficient_rows = tuple(coefficient_rows_list)
            rescale_rows = tuple(rescale_rows_list)
            rescale_noise_flags = tuple(rescale_noise_flags_list)

        if any(row < 0 for row in coefficient_rows):
            raise RuntimeError("coefficient encode noise is required")

        noise_scales = (
            noise_std(0),
            noise_std(1),
            noise_std(2),
            noise_std(coefficient_slot),
            noise_std(coefficient_slot + 1),
            noise_std(rescale_slot),
            noise_std(coefficient_slot + 2),
            noise_std(rescale_slot + 1),
            noise_std(coefficient_slot + 3),
            noise_std(rescale_slot + 2),
            noise_std(coefficient_slot + 4),
            noise_std(rescale_slot + 3),
        )

        common_kwargs = {
            "HAS_POWER2_NOISE": power_noise_flags[0],
            "HAS_POWER3_NOISE": power_noise_flags[1],
            "HAS_POWER4_NOISE": power_noise_flags[2],
            "HAS_RESCALE1_NOISE": rescale_noise_flags[0],
            "HAS_RESCALE2_NOISE": rescale_noise_flags[1],
            "HAS_RESCALE3_NOISE": rescale_noise_flags[2],
            "HAS_RESCALE4_NOISE": rescale_noise_flags[3],
            "NOISE_IS_STANDARD_NORMAL": group_standard_normals,
            "POWER2_ROW": power_rows.get(0, 0),
            "POWER3_ROW": power_rows.get(1, 0),
            "POWER4_ROW": power_rows.get(2, 0),
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
                *noise_scales,
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
            *noise_scales,
            float(truncation_scale or 1.0),
            APPLY_TRUNCATION=truncation_scale is not None,
            **common_kwargs,
        )

    compute_piece(negative, 3, 8, 11)
    out = torch.empty_like(x)
    compute_piece(
        positive,
        12,
        17,
        12,
        final_output=out,
        negative_output=negative_out,
    )
    return out
