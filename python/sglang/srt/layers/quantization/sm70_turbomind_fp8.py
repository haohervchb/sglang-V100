"""TurboMind W8A16 block-FP8 linear kernels for NVIDIA SM70."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.parameter import Parameter

from sglang.srt.layers.attention.triton_ops.fp8_sm70 import (
    fp8_e4m3fn_to_fp32,
)

logger = logging.getLogger(__name__)

_DEFAULT_OPS_PATH = (
    Path(__file__).resolve().parents[3] / "jit_kernel" / "_sm70_turbomind_v100.so"
)
_OPS_LOAD_ATTEMPTED = False
_OPS_AVAILABLE = False

_PREFILL_BACKENDS = ("auto", "turbomind", "fp16")


def _get_sm70_fp8_prefill_backend() -> str:
    backend = os.environ.get("SGLANG_SM70_FP8_PREFILL_BACKEND", "auto").lower()
    if backend not in _PREFILL_BACKENDS:
        raise ValueError(
            "SGLANG_SM70_FP8_PREFILL_BACKEND must be auto, turbomind, or "
            f"fp16; got {backend!r}."
        )
    return backend


def _get_sm70_fp8_prefill_min_tokens() -> int:
    value = int(os.environ.get("SGLANG_SM70_FP8_PREFILL_MIN_TOKENS", "2048"))
    if value <= 0:
        raise ValueError(
            "SGLANG_SM70_FP8_PREFILL_MIN_TOKENS must be positive; " f"got {value}."
        )
    return value


@triton.jit
def _dequantize_block_fp8_weight_kernel(
    weight,
    scales,
    output,
    num_elements,
    scale_row_stride,
    scale_col_stride,
    num_columns: tl.constexpr,
    weight_block_rows: tl.constexpr,
    weight_block_columns: tl.constexpr,
    elements_per_program: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * elements_per_program + tl.arange(
        0, elements_per_program
    )
    mask = offsets < num_elements
    rows = offsets // num_columns
    columns = offsets - rows * num_columns
    scale_offsets = (rows // weight_block_rows) * scale_row_stride + (
        columns // weight_block_columns
    ) * scale_col_stride
    raw = tl.load(weight + offsets, mask=mask, other=0)
    scale = tl.load(scales + scale_offsets, mask=mask, other=0.0)
    tl.store(output + offsets, fp8_e4m3fn_to_fp32(raw) * scale, mask=mask)


def _dequantize_sm70_fp8_prefill_weight(layer: torch.nn.Module) -> torch.Tensor:
    weight = layer.sm70_fp8_prefill_weight
    scales = layer.sm70_fp8_prefill_scales
    output = torch.empty(weight.shape, dtype=torch.float16, device=weight.device)
    elements_per_program = 256
    _dequantize_block_fp8_weight_kernel[
        (triton.cdiv(weight.numel(), elements_per_program),)
    ](
        weight.view(torch.uint8),
        scales,
        output,
        weight.numel(),
        scales.stride(0),
        scales.stride(1),
        num_columns=weight.shape[1],
        weight_block_rows=128,
        weight_block_columns=128,
        elements_per_program=elements_per_program,
        num_warps=4,
    )
    return output


def _use_sm70_fp8_prefill_bridge(layer: torch.nn.Module, num_tokens: int) -> bool:
    return (
        getattr(layer, "sm70_fp8_prefill_bridge", False)
        and num_tokens >= layer.sm70_fp8_prefill_min_tokens
    )


def _load_sm70_turbomind_fp8_ops() -> bool:
    global _OPS_AVAILABLE, _OPS_LOAD_ATTEMPTED
    if _OPS_LOAD_ATTEMPTED:
        return _OPS_AVAILABLE
    _OPS_LOAD_ATTEMPTED = True

    ops_path = Path(os.environ.get("SGLANG_SM70_TURBOMIND_OPS_PATH", _DEFAULT_OPS_PATH))
    try:
        torch.ops.load_library(str(ops_path))
        _OPS_AVAILABLE = hasattr(
            torch.ops.sglang_sm70_turbomind, "fp8_prepare"
        ) and hasattr(torch.ops.sglang_sm70_turbomind, "fp8_gemm")
        if not _OPS_AVAILABLE:
            raise RuntimeError("the required fp8_prepare/fp8_gemm operators are absent")
    except Exception as exc:
        logger.warning("SM70 TurboMind FP8 operators unavailable: %s", exc)
        _OPS_AVAILABLE = False
    return _OPS_AVAILABLE


def can_use_sm70_turbomind_fp8(
    weight_block_size: Optional[list[int]],
) -> bool:
    """Return whether the local SM70 W8A16 kernel covers this quantization."""
    backend = os.environ.get("SGLANG_SM70_FP8_BACKEND", "auto").lower()
    if backend not in ("auto", "turbomind", "marlin"):
        raise ValueError(
            "SGLANG_SM70_FP8_BACKEND must be auto, turbomind, or marlin; "
            f"got {backend!r}."
        )
    if backend == "marlin":
        return False
    if (
        weight_block_size != [128, 128]
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (7, 0)
    ):
        if backend == "turbomind":
            raise RuntimeError(
                "The SM70 TurboMind FP8 backend requires an NVIDIA SM70 GPU "
                "and block-wise FP8 weights with block_size=[128, 128]."
            )
        return False

    available = _load_sm70_turbomind_fp8_ops()
    if backend == "turbomind" and not available:
        raise RuntimeError(
            "SGLANG_SM70_FP8_BACKEND=turbomind was requested, but the "
            "TurboMind FP8 extension could not be loaded."
        )
    return available


def prepare_sm70_turbomind_fp8_linear(layer: torch.nn.Module) -> None:
    if layer.orig_dtype != torch.float16:
        raise RuntimeError(
            "SM70 TurboMind FP8 requires FP16 activations, "
            f"but the layer dtype is {layer.orig_dtype}."
        )
    if getattr(layer, "weight_block_size", None) != [128, 128]:
        raise RuntimeError("SM70 TurboMind FP8 requires weight_block_size=[128, 128].")

    weight = layer.weight.data
    scales = layer.weight_scale_inv.data.to(torch.float32).contiguous()
    prefix = getattr(layer, "prefix", "")
    prefill_backend = _get_sm70_fp8_prefill_backend()
    use_prefill_bridge = prefill_backend != "turbomind" and ".layers." in prefix
    if use_prefill_bridge:
        # TurboMind is substantially faster for decode on V100, but its
        # on-the-fly W8A16 conversion loses to cuBLAS FP16 once the GEMM's M
        # dimension is large. Keep the checkpoint layout as a compact second
        # FP8 copy and materialize one temporary FP16 projection weight at a
        # time for prefill. Do not mirror lm_head: it is large and only sees a
        # small M in serving.
        layer.register_parameter(
            "sm70_fp8_prefill_weight",
            Parameter(weight, requires_grad=False),
        )
        layer.register_parameter(
            "sm70_fp8_prefill_scales",
            Parameter(scales, requires_grad=False),
        )
        layer.sm70_fp8_prefill_min_tokens = _get_sm70_fp8_prefill_min_tokens()
        layer.sm70_fp8_prefill_bridge = True
    logical_widths = getattr(layer, "logical_widths", None)
    is_gated_silu = (
        getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "gate_up_proj"
        and isinstance(logical_widths, list)
        and len(logical_widths) == 2
        and logical_widths[0] == logical_widths[1]
    )
    tm_weight, tm_scales, meta = torch.ops.sglang_sm70_turbomind.fp8_prepare(
        weight,
        scales,
        128,
        is_gated_silu,
    )
    layer.weight = Parameter(tm_weight, requires_grad=False)
    layer.weight_scale_inv = Parameter(tm_scales, requires_grad=False)
    layer.sm70_fp8_k_ld = int(meta[0].item())
    layer.sm70_fp8_q_ld = int(meta[1].item())
    layer.sm70_fp8_turbomind = True
    layer.sm70_fp8_gated_silu = is_gated_silu
    logger.info_once("SM70 (V100): using TurboMind W8A16 block-FP8 dense GEMM.")
    if is_gated_silu:
        logger.info_once(
            "SM70 (V100): using the TurboMind fused gate/up SiLU epilogue."
        )
    if use_prefill_bridge:
        logger.info_once(
            "SM70 (V100): using temporary FP16 weights for block-FP8 "
            "prefill projections with at least %d tokens.",
            layer.sm70_fp8_prefill_min_tokens,
        )


def apply_sm70_turbomind_fp8_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    if _use_sm70_fp8_prefill_bridge(layer, x_2d.shape[0]):
        weight = _dequantize_sm70_fp8_prefill_weight(layer)
        out_2d = F.linear(x_2d, weight, bias)
        return out_2d.reshape(*x.shape[:-1], layer.output_size_per_partition)
    out_2d = torch.empty(
        (x_2d.shape[0], layer.output_size_per_partition),
        dtype=x.dtype,
        device=x.device,
    )
    torch.ops.sglang_sm70_turbomind.fp8_gemm(
        out_2d,
        x_2d,
        layer.weight,
        layer.weight_scale_inv,
        128,
        layer.sm70_fp8_k_ld,
        layer.sm70_fp8_q_ld,
        False,
    )
    if getattr(layer, "sm70_fp8_gated_silu", False):
        # The packed weight is interleaved for the fused epilogue. Preserve the
        # ordinary linear contract when a caller does not request fusion.
        out_features = layer.output_size_per_partition // 2
        out_2d = (
            out_2d.reshape(x_2d.shape[0], out_features, 2)
            .transpose(1, 2)
            .reshape(x_2d.shape[0], layer.output_size_per_partition)
        )
    if bias is not None:
        out_2d.add_(bias)
    return out_2d.reshape(*x.shape[:-1], layer.output_size_per_partition)


def apply_sm70_turbomind_fp8_fused_silu_and_mul(
    layer: torch.nn.Module,
    x: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Apply a packed gate/up projection and SiLU multiplication in one GEMM."""
    if not getattr(layer, "sm70_fp8_gated_silu", False):
        return None

    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    out_features = layer.output_size_per_partition // 2
    if _use_sm70_fp8_prefill_bridge(layer, x_2d.shape[0]):
        weight = _dequantize_sm70_fp8_prefill_weight(layer)
        projected = F.linear(x_2d, weight)
        gate, up = projected.chunk(2, dim=-1)
        out_2d = F.silu(gate) * up
        return out_2d.reshape(*x.shape[:-1], out_features)
    out_2d = torch.empty(
        (x_2d.shape[0], out_features),
        dtype=x.dtype,
        device=x.device,
    )
    torch.ops.sglang_sm70_turbomind.fp8_gemm(
        out_2d,
        x_2d,
        layer.weight,
        layer.weight_scale_inv,
        128,
        layer.sm70_fp8_k_ld,
        layer.sm70_fp8_q_ld,
        True,
    )
    return out_2d.reshape(*x.shape[:-1], out_features)
