"""Measured batch-one FP16 fusions for Qwen3.8 on Volta."""

import os

import torch

from sglang.jit_kernel.sm70_dense_gemv import supported as gemv_supported
from sglang.jit_kernel.utils import cache_once, load_jit


def enabled() -> bool:
    return os.environ.get("SGLANG_SM70_QWEN_FUSIONS", "0") == "1"


def gate_supported(x, weight, value, bias=None) -> bool:
    return (
        enabled()
        and tuple(weight.shape) == (1, 2560)
        and value.shape == (1, 2560)
        and value.dtype == torch.float16
        and value.device == x.device
        and value.is_contiguous()
        and gemv_supported(x, weight, bias)
    )


def gate_up_supported(x, weight) -> bool:
    return (
        enabled() and tuple(weight.shape) == (320, 2560) and gemv_supported(x, weight)
    )


def qkv_ba_supported(x, weight, tail) -> bool:
    return (
        enabled()
        and isinstance(weight, torch.Tensor)
        and isinstance(tail, torch.Tensor)
        and tuple(weight.shape) == (4096, 2560)
        and tuple(tail.shape) == (24, 2560)
        and tail.dtype == torch.float16
        and tail.device == x.device
        and tail.is_contiguous()
        and tail.data_ptr() % 16 == 0
        and gemv_supported(x, weight)
    )


def reuse_qkv_prefix(projected, query, key, value):
    # This ratio-three GDN layout is already [Q512, K512, V1536, Z1536].
    # For one row its Q/K/V prefix is contiguous despite the parent row stride.
    if (
        enabled()
        and projected.is_cuda
        and projected.dtype == torch.float16
        and projected.shape == (1, 4096)
        and projected.is_contiguous()
        and query.shape == (1, 512)
        and key.shape == (1, 512)
        and value.shape == (1, 1536)
        and query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and query.data_ptr() == projected.data_ptr()
        and key.data_ptr() == projected.data_ptr() + 1024
        and value.data_ptr() == projected.data_ptr() + 2048
        and torch.cuda.get_device_capability(projected.device) == (7, 0)
    ):
        return projected[:, :2560]
    return None


@cache_once
def _module():
    return load_jit(
        "sm70_qwen_fusions",
        cuda_files=["elementwise/sm70_qwen_fusions.cuh"],
        cuda_wrappers=[
            ("gate", "sglang::sm70_qwen_fusions::run<512>"),
            ("gate_up", "sglang::sm70_qwen_fusions::gate_up"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


@cache_once
def _projection_module():
    return load_jit(
        "sm70_qwen_qkv_ba",
        cuda_files=["elementwise/sm70_dense_gemv.cuh"],
        cuda_wrappers=[("qkv_ba", "sglang::sm70_dense_gemv::qkv_ba")],
    )


def gate(x, weight, value):
    output = torch.empty_like(value)
    _module().gate(x, weight, value, output)
    return output


def gate_up(x, weight):
    output = torch.empty((1, 160), dtype=x.dtype, device=x.device)
    _module().gate_up(x, weight, output)
    return output


def qkv_ba(x, weight, tail):
    output = torch.empty((1, 4120), dtype=x.dtype, device=x.device)
    _projection_module().qkv_ba(x, weight, tail, output)
    return output[:, :4096], output[:, 4096:]
