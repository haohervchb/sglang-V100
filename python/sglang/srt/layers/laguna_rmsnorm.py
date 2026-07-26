"""Fused RMSNorm for Laguna's BF16 residual emulation on SM70.

Laguna was trained with a BF16 residual stream.  V100 cannot execute the
model's GEMMs in BF16, so the model keeps projection outputs in FP16 while
transporting the residual in FP32 and explicitly rounding each residual update
to BF16.  Expressing that sequence with eager PyTorch launches many tiny
kernels per norm, which dominates batch-one decode latency.  This module keeps
the same rounding points in one Triton kernel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _round_fp32_to_bf16_rne(value):
    """Round FP32 to BF16 and return the rounded value represented as FP32."""
    bits = value.to(tl.uint32, bitcast=True)
    # IEEE round-to-nearest-even before clearing the low 16 mantissa bits.
    rounded_bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _laguna_rmsnorm_kernel(
    output_ptr,
    residual_output_ptr,
    x_ptr,
    residual_ptr,
    post_residual_ptr,
    weight_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    residual_scale: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_POST_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    value = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(
            residual_ptr + row_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        value = value * residual_scale + residual
        if HAS_POST_RESIDUAL:
            value += tl.load(
                post_residual_ptr + row_offsets, mask=mask, other=0.0
            ).to(tl.float32)

        value = _round_fp32_to_bf16_rne(value)
        tl.store(residual_output_ptr + row_offsets, value, mask=mask)

    variance = tl.sum(value * value, axis=0) / n_cols
    normalized = value * tl.rsqrt(variance + eps)
    normalized = _round_fp32_to_bf16_rne(normalized)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = _round_fp32_to_bf16_rne(weight)
    output = _round_fp32_to_bf16_rne(normalized * weight)
    tl.store(output_ptr + row_offsets, output, mask=mask)


def laguna_rmsnorm_sm70(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual_scale: float,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Apply Laguna RMSNorm with its BF16 rounding points in one kernel."""
    if x.ndim != 2:
        raise ValueError(f"Laguna SM70 RMSNorm expects a 2D input, got {x.shape}.")
    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    if post_residual_addition is not None and not post_residual_addition.is_contiguous():
        post_residual_addition = post_residual_addition.contiguous()

    n_rows, n_cols = x.shape
    if weight.ndim != 1 or weight.shape[0] != n_cols:
        raise ValueError(
            f"Laguna SM70 RMSNorm weight must have shape [{n_cols}], "
            f"got {tuple(weight.shape)}."
        )
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError(
            f"Laguna SM70 RMSNorm only supports hidden sizes <= 65536, got {n_cols}."
        )

    output = torch.empty_like(x)
    has_residual = residual is not None
    residual_output = torch.empty_like(residual) if has_residual else output
    residual_ptr = residual if has_residual else x
    post_residual_ptr = (
        post_residual_addition if post_residual_addition is not None else x
    )

    num_warps = max(min(triton.next_power_of_2(triton.cdiv(n_cols, 256)), 16), 4)
    _laguna_rmsnorm_kernel[(n_rows,)](
        output,
        residual_output,
        x,
        residual_ptr,
        post_residual_ptr,
        weight,
        n_cols=n_cols,
        eps=eps,
        residual_scale=residual_scale,
        HAS_RESIDUAL=has_residual,
        HAS_POST_RESIDUAL=post_residual_addition is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    if has_residual:
        return output, residual_output
    return output
