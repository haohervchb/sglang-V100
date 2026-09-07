"""Native batch-one FP16 GEMVs for the measured Qwen3.8 TP4 shapes."""

import os

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

# (output features, input features) -> (threads, lanes per row, vector width).
_CONFIGS = {
    (4096, 2560): (64, 16, 8),
    (3584, 2560): (64, 32, 8),
    (2560, 1536): (128, 32, 8),
    (320, 2560): (256, 32, 8),
    (2560, 160): (128, 8, 4),
    (512, 2560): (256, 32, 8),
    (640, 2560): (128, 32, 8),
    (24, 2560): (64, 32, 8),
    (1, 2560): (256, 32, 8),
    (10240, 2560): (64, 32, 8),
    (2560, 2560): (128, 32, 8),
}


def supported(x: torch.Tensor, weight: torch.Tensor, bias=None) -> bool:
    return (
        os.environ.get("SGLANG_SM70_DENSE_GEMV", "0") == "1"
        and x.is_cuda
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and weight.device == x.device
        and x.ndim == 2
        and x.shape[0] == 1
        and weight.ndim == 2
        and x.shape[1] == weight.shape[1]
        and bias is None
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and (
            tuple(weight.shape) in _CONFIGS
            or (weight.shape[1] == 2560 and weight.shape[0] >= 32768)
        )
        and torch.cuda.get_device_capability(x.device) == (7, 0)
    )


@cache_once
def _module(threads: int, lanes: int, vector: int):
    return load_jit(
        "sm70_dense_gemv",
        threads,
        lanes,
        vector,
        cuda_files=["elementwise/sm70_dense_gemv.cuh"],
        cuda_wrappers=[
            ("gemv", f"sglang::sm70_dense_gemv::gemv<{threads},{lanes},{vector}>")
        ],
    )


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    config = _CONFIGS.get(tuple(weight.shape), (64, 32, 8))
    out = torch.empty((1, weight.shape[0]), dtype=x.dtype, device=x.device)
    _module(*config).gemv(x, weight, out)
    return out
