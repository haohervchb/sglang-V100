"""TurboMind W8A16 block-FP8 linear kernels for NVIDIA SM70."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

_DEFAULT_OPS_PATH = (
    Path(__file__).resolve().parents[3] / "jit_kernel" / "_sm70_turbomind_v100.so"
)
_OPS_LOAD_ATTEMPTED = False
_OPS_AVAILABLE = False

_PREFILL_BACKENDS = ("auto", "turbomind", "fp16")

# Qwen3.8-27B-FP8 TP4 shapes admitted by 1Cat's real-weight, bitwise gates.
# The TurboMind tensors have already been converted to logical KxN here.
_SM70_FP8_PREFILL_DENSE_MIN_M = 3920
_SM70_FP8_PREFILL_DENSE_SHAPES = {
    "gate_up_proj": (5120, 8704),
    "down_proj": (4352, 5120),
    "out_proj": (1536, 5120),
    "o_proj": (1536, 5120),
}
_SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS = max(
    k * n for k, n in _SM70_FP8_PREFILL_DENSE_SHAPES.values()
)
_SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES = (
    _SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS * torch.float16.itemsize
)
# Layers deliberately retain only a CUDA address. This strong cache owns the
# allocation and keeps the 85 MiB buffer out of torch.compile/CUDA-graph inputs.
_SM70_FP8_PREFILL_DENSE_WORKSPACES: dict[
    tuple[int, torch.dtype], torch.Tensor
] = {}
_SM70_FP8_PREFILL_DENSE_OOM_WARNED = False


def _get_sm70_fp8_prefill_backend() -> str:
    backend = os.environ.get("SGLANG_SM70_FP8_PREFILL_BACKEND", "auto").lower()
    if backend not in _PREFILL_BACKENDS:
        raise ValueError(
            "SGLANG_SM70_FP8_PREFILL_BACKEND must be auto, turbomind, or "
            f"fp16; got {backend!r}."
        )
    return backend


def _get_sm70_fp8_prefill_min_tokens() -> int:
    value = int(
        os.environ.get(
            "SGLANG_SM70_FP8_PREFILL_MIN_TOKENS",
            str(_SM70_FP8_PREFILL_DENSE_MIN_M),
        )
    )
    if value <= 0:
        raise ValueError(
            "SGLANG_SM70_FP8_PREFILL_MIN_TOKENS must be positive; " f"got {value}."
        )
    return value


def _env_flag(name: str, default: str) -> bool:
    value = os.environ.get(name, default).strip().lower()
    if value not in ("0", "false", "off", "no", "1", "true", "on", "yes"):
        raise ValueError(f"{name} must be a boolean value, got {value!r}.")
    return value in ("1", "true", "on", "yes")


def _is_sm70_fp8_prefill_exact_dense_layer(layer: torch.nn.Module) -> bool:
    if getattr(layer, "tp_size", 1) != 4:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected = _SM70_FP8_PREFILL_DENSE_SHAPES.get(suffix)
    return expected is not None and tuple(layer.weight.shape) == expected


def _get_sm70_fp8_prefill_exact_dense_workspace(
    weight: torch.Tensor,
) -> Optional[torch.Tensor]:
    global _SM70_FP8_PREFILL_DENSE_OOM_WARNED
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, torch.float16)
    workspace = _SM70_FP8_PREFILL_DENSE_WORKSPACES.get(key)
    if workspace is not None:
        return workspace
    try:
        workspace = torch.empty(
            _SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS,
            dtype=torch.float16,
            device=weight.device,
        )
    except torch.OutOfMemoryError:
        if not _SM70_FP8_PREFILL_DENSE_OOM_WARNED:
            logger.warning(
                "Insufficient memory for the bounded SM70 FP8 prefill "
                "workspace; falling back to TurboMind W8A16."
            )
            _SM70_FP8_PREFILL_DENSE_OOM_WARNED = True
        return None
    _SM70_FP8_PREFILL_DENSE_WORKSPACES[key] = workspace
    return workspace


def _use_sm70_fp8_prefill_dispatch(layer: torch.nn.Module) -> bool:
    return (
        getattr(layer, "sm70_fp8_prefill_exact_dense_workspace_ptr", 0) != 0
        and hasattr(torch.ops.sglang_sm70_turbomind, "fp8_prefill_dispatch")
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
    prefill_backend = _get_sm70_fp8_prefill_backend()
    exact_dense_enabled = _env_flag("SGLANG_SM70_FP8_PREFILL_EXACT_DENSE", "1")
    if (
        prefill_backend != "turbomind"
        and exact_dense_enabled
        and hasattr(torch.ops.sglang_sm70_turbomind, "fp8_prefill_dispatch")
        and _is_sm70_fp8_prefill_exact_dense_layer(layer)
    ):
        workspace = _get_sm70_fp8_prefill_exact_dense_workspace(layer.weight)
        if workspace is not None:
            layer.sm70_fp8_prefill_exact_dense_workspace_ptr = workspace.data_ptr()
            layer.sm70_fp8_prefill_min_tokens = (
                _get_sm70_fp8_prefill_min_tokens()
            )
    logger.info_once("SM70 (V100): using TurboMind W8A16 block-FP8 dense GEMM.")
    if is_gated_silu:
        logger.info_once(
            "SM70 (V100): using the TurboMind fused gate/up SiLU epilogue."
        )
    if _use_sm70_fp8_prefill_dispatch(layer):
        logger.info_once(
            "SM70 (V100): using the graph-safe shared 85 MiB exact-dense "
            "FP8 prefill workspace for admitted TP4 projections at M >= %d.",
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
    out_2d = torch.empty(
        (x_2d.shape[0], layer.output_size_per_partition),
        dtype=x.dtype,
        device=x.device,
    )
    if _use_sm70_fp8_prefill_dispatch(layer) and x_2d.dtype == torch.float16:
        torch.ops.sglang_sm70_turbomind.fp8_prefill_dispatch(
            out_2d,
            layer.sm70_fp8_prefill_exact_dense_workspace_ptr,
            x_2d,
            layer.weight,
            layer.weight_scale_inv,
            128,
            layer.sm70_fp8_k_ld,
            layer.sm70_fp8_q_ld,
            False,
            layer.sm70_fp8_prefill_min_tokens,
        )
    else:
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
    out_2d = torch.empty(
        (x_2d.shape[0], out_features),
        dtype=x.dtype,
        device=x.device,
    )
    if _use_sm70_fp8_prefill_dispatch(layer) and x_2d.dtype == torch.float16:
        torch.ops.sglang_sm70_turbomind.fp8_prefill_dispatch(
            out_2d,
            layer.sm70_fp8_prefill_exact_dense_workspace_ptr,
            x_2d,
            layer.weight,
            layer.weight_scale_inv,
            128,
            layer.sm70_fp8_k_ld,
            layer.sm70_fp8_q_ld,
            True,
            layer.sm70_fp8_prefill_min_tokens,
        )
    else:
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
