"""Native FP16 kernels for Qwen3.8's batch-one hyperconnection mix on SM70."""

import torch

from sglang.jit_kernel.utils import cache_once, load_jit


@cache_once
def _module():
    return load_jit(
        "sm70_hc_mix",
        cuda_files=["elementwise/sm70_hc_mix.cuh"],
        cuda_wrappers=[
            ("down", "sglang::sm70_hc::down"),
            ("up", "sglang::sm70_hc::up"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


def hc_down(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    output = torch.empty((1, 320), device=x.device, dtype=x.dtype)
    _module().down(x, weight, output)
    return output


def hc_up(
    activated: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    output = torch.empty((1, 2560), device=x.device, dtype=x.dtype)
    _module().up(activated, x, weight, output)
    return output
