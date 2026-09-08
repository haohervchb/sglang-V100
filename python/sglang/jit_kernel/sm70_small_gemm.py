"""FP16 projections for measured two/four-token Qwen verification shapes."""

import os

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

# (rows, output features, input features) -> (threads, lanes per output).
# Shapes where cuBLAS is faster deliberately stay on the existing path.
_CONFIGS = {
    (2, 24, 2560): (64, 32),
    (2, 640, 2560): (256, 32),
    (2, 1, 2560): (64, 32),
    (2, 2560, 2560): (128, 32),
    (4, 24, 2560): (64, 32),
    (4, 640, 2560): (128, 32),
    (4, 2560, 2560): (64, 32),
    (2, 4096, 2560): (256, 32),
    (4, 4096, 2560): (64, 32),
    (2, 3584, 2560): (64, 32),
    (4, 3584, 2560): (64, 32),
    (2, 2560, 1536): (64, 32),
    (4, 2560, 1536): (128, 16),
    (2, 320, 2560): (256, 32),
    (4, 320, 2560): (256, 32),
    (2, 2560, 160): (128, 8),
    (4, 2560, 160): (128, 8),
    (2, 512, 2560): (256, 32),
    (4, 512, 2560): (256, 32),
    (2, 62080, 2560): (128, 32),
    (4, 62080, 2560): (128, 32),
}


def shape_supported(x, weight):
    return (
        os.environ.get("SGLANG_SM70_MTP_SMALL_GEMM", "1") == "1"
        and (x.shape[0], *weight.shape) in _CONFIGS
    )


@cache_once
def _module(rows, threads, lanes):
    return load_jit(
        "sm70_small_gemm",
        rows,
        threads,
        lanes,
        cuda_files=["elementwise/sm70_small_gemm.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::sm70_small_gemm::run<{rows},{threads},{lanes}>")
        ],
    )


def linear(x, weight):
    threads, lanes = _CONFIGS[(x.shape[0], *weight.shape)]
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _module(x.shape[0], threads, lanes).run(x, weight, out)
    return out
