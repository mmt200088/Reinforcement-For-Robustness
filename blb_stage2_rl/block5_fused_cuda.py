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
    def _load_noise(noise_ptr, offsets, mask, numel, index: tl.constexpr):
        return tl.load(noise_ptr + index * numel + offsets, mask=mask)


    @triton.jit
    def _block5_degree4_kernel(
            x_ptr,
            noise_ptr,
            out_ptr,
            numel,
            neg_c0,
            neg_c1,
            neg_c2,
            neg_c3,
            neg_c4,
            pos_c0,
            pos_c1,
            pos_c2,
            pos_c3,
            pos_c4,
            POWER2_IDX: tl.constexpr,
            POWER3_IDX: tl.constexpr,
            POWER4_IDX: tl.constexpr,
            NEG_C0_IDX: tl.constexpr,
            NEG_C1_IDX: tl.constexpr,
            NEG_C2_IDX: tl.constexpr,
            NEG_C3_IDX: tl.constexpr,
            NEG_C4_IDX: tl.constexpr,
            NEG_R1_IDX: tl.constexpr,
            NEG_R2_IDX: tl.constexpr,
            NEG_R3_IDX: tl.constexpr,
            NEG_R4_IDX: tl.constexpr,
            POS_C0_IDX: tl.constexpr,
            POS_C1_IDX: tl.constexpr,
            POS_C2_IDX: tl.constexpr,
            POS_C3_IDX: tl.constexpr,
            POS_C4_IDX: tl.constexpr,
            POS_R1_IDX: tl.constexpr,
            POS_R2_IDX: tl.constexpr,
            POS_R3_IDX: tl.constexpr,
            POS_R4_IDX: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
            ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask)

        x2 = _mul_rn_f32(x, x)
        if POWER2_IDX >= 0:
            x2 = _add_rn_f32(
                x2,
                _load_noise(noise_ptr, offsets, mask, numel, POWER2_IDX),
            )
        x3 = _mul_rn_f32(x2, x)
        if POWER3_IDX >= 0:
            x3 = _add_rn_f32(
                x3,
                _load_noise(noise_ptr, offsets, mask, numel, POWER3_IDX),
            )
        x4 = _mul_rn_f32(x2, x2)
        if POWER4_IDX >= 0:
            x4 = _add_rn_f32(
                x4,
                _load_noise(noise_ptr, offsets, mask, numel, POWER4_IDX),
            )

        y_neg = _add_rn_f32(
            _load_noise(noise_ptr, offsets, mask, numel, NEG_C0_IDX),
            neg_c0,
        )
        neg_term1 = _mul_rn_f32(
            x,
            _add_rn_f32(
                _load_noise(noise_ptr, offsets, mask, numel, NEG_C1_IDX),
                neg_c1,
            ),
        )
        if NEG_R1_IDX >= 0:
            neg_term1 = _add_rn_f32(
                neg_term1,
                _load_noise(noise_ptr, offsets, mask, numel, NEG_R1_IDX),
            )
        y_neg = _add_rn_f32(y_neg, neg_term1)

        neg_term2 = _mul_rn_f32(
            x2,
            _add_rn_f32(
                _load_noise(noise_ptr, offsets, mask, numel, NEG_C2_IDX),
                neg_c2,
            ),
        )
        if NEG_R2_IDX >= 0:
            neg_term2 = _add_rn_f32(
                neg_term2,
                _load_noise(noise_ptr, offsets, mask, numel, NEG_R2_IDX),
            )
        y_neg = _add_rn_f32(y_neg, neg_term2)

        neg_term3 = _mul_rn_f32(
            x3,
            _add_rn_f32(
                _load_noise(noise_ptr, offsets, mask, numel, NEG_C3_IDX),
                neg_c3,
            ),
        )
        if NEG_R3_IDX >= 0:
            neg_term3 = _add_rn_f32(
                neg_term3,
                _load_noise(noise_ptr, offsets, mask, numel, NEG_R3_IDX),
            )
        y_neg = _add_rn_f32(y_neg, neg_term3)

        neg_term4 = _mul_rn_f32(
            x4,
            _add_rn_f32(
                _load_noise(noise_ptr, offsets, mask, numel, NEG_C4_IDX),
                neg_c4,
            ),
        )
        if NEG_R4_IDX >= 0:
            neg_term4 = _add_rn_f32(
                neg_term4,
                _load_noise(noise_ptr, offsets, mask, numel, NEG_R4_IDX),
            )
        y_neg = _add_rn_f32(y_neg, neg_term4)

        y_pos = _add_rn_f32(
            _load_noise(noise_ptr, offsets, mask, numel, POS_C0_IDX),
            pos_c0,
        )
        pos_term1 = _mul_rn_f32(
            x,
            _add_rn_f32(
                _load_noise(noise_ptr, offsets, mask, numel, POS_C1_IDX),
                pos_c1,
            ),
        )
        if POS_R1_IDX >= 0:
            pos_term1 = _add_rn_f32(
                pos_term1,
                _load_noise(noise_ptr, offsets, mask, numel, POS_R1_IDX),
            )
        y_pos = _add_rn_f32(y_pos, pos_term1)

        pos_term2 = _mul_rn_f32(
            x2,
            _add_rn_f32(
                _load_noise(noise_ptr, offsets, mask, numel, POS_C2_IDX),
                pos_c2,
            ),
        )
        if POS_R2_IDX >= 0:
            pos_term2 = _add_rn_f32(
                pos_term2,
                _load_noise(noise_ptr, offsets, mask, numel, POS_R2_IDX),
            )
        y_pos = _add_rn_f32(y_pos, pos_term2)

        pos_term3 = _mul_rn_f32(
            x3,
            _add_rn_f32(
                _load_noise(noise_ptr, offsets, mask, numel, POS_C3_IDX),
                pos_c3,
            ),
        )
        if POS_R3_IDX >= 0:
            pos_term3 = _add_rn_f32(
                pos_term3,
                _load_noise(noise_ptr, offsets, mask, numel, POS_R3_IDX),
            )
        y_pos = _add_rn_f32(y_pos, pos_term3)

        pos_term4 = _mul_rn_f32(
            x4,
            _add_rn_f32(
                _load_noise(noise_ptr, offsets, mask, numel, POS_C4_IDX),
                pos_c4,
            ),
        )
        if POS_R4_IDX >= 0:
            pos_term4 = _add_rn_f32(
                pos_term4,
                _load_noise(noise_ptr, offsets, mask, numel, POS_R4_IDX),
            )
        y_pos = _add_rn_f32(y_pos, pos_term4)

        out = tl.where(x < 0.0, y_neg, y_pos)
        out = tl.where(x >= -2.7, out, 0.0)
        out = tl.where(x > 2.7, x, out)
        tl.store(out_ptr + offsets, out, mask=mask)


def is_available() -> bool:
    return triton is not None


def block5_degree4_cuda(
        x: torch.Tensor,
        noise_slab: torch.Tensor,
        noise_indices: Sequence[int],
        negative_coefficients: Sequence[float],
        positive_coefficients: Sequence[float],
        ) -> torch.Tensor:
    """Apply exact eager-order degree-4 GELU arithmetic to sampled noises."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if len(noise_indices) != 21:
        raise ValueError(f"expected 21 noise indices, got {len(noise_indices)}")
    if len(negative_coefficients) != 5 or len(positive_coefficients) != 5:
        raise ValueError("degree-4 GELU requires five coefficients per piece")

    out = torch.empty_like(x)
    numel = int(x.numel())
    indices = tuple(int(value) for value in noise_indices)
    negative = tuple(float(value) for value in negative_coefficients)
    positive = tuple(float(value) for value in positive_coefficients)
    _block5_degree4_kernel[(triton.cdiv(numel, 256),)](
        x,
        noise_slab,
        out,
        numel,
        *negative,
        *positive,
        POWER2_IDX=indices[0],
        POWER3_IDX=indices[1],
        POWER4_IDX=indices[2],
        NEG_C0_IDX=indices[3],
        NEG_C1_IDX=indices[4],
        NEG_C2_IDX=indices[5],
        NEG_C3_IDX=indices[6],
        NEG_C4_IDX=indices[7],
        NEG_R1_IDX=indices[8],
        NEG_R2_IDX=indices[9],
        NEG_R3_IDX=indices[10],
        NEG_R4_IDX=indices[11],
        POS_C0_IDX=indices[12],
        POS_C1_IDX=indices[13],
        POS_C2_IDX=indices[14],
        POS_C3_IDX=indices[15],
        POS_C4_IDX=indices[16],
        POS_R1_IDX=indices[17],
        POS_R2_IDX=indices[18],
        POS_R3_IDX=indices[19],
        POS_R4_IDX=indices[20],
        BLOCK_SIZE=256,
    )
    return out
