"""Opt-in hand-written CUDA long-context grouped-decode partial (SM70).

Replaces the TileLang split-KV decode partial with a native SM70 kernel for the
exact Qwen3.8-27B TP4 shape (H6 / Hkv1 / D256, E5M2 byte KV, page size 16).
The kernel streams each split's K/V from the paged cache exactly once, which
cuts DRAM traffic versus the TileLang codegen on the same layout. The partial
output ABI matches ``_decode_partial_kernel`` exactly so downstream code can
reuse the unchanged TileLang combine kernel.

Gated by ``SGLANG_V100_DECODE_CUDA=1``; falls back to TileLang otherwise.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_SRC_PATH = (
    Path(__file__).resolve().parents[4]
    / "jit_kernel"
    / "csrc"
    / "sm70_longctx_decode.cu"
)
_EXT = None
_OPS_LOAD_ATTEMPTED = False

PAGE_SIZE = 16  # fixed page granularity supported by the CUDA kernel


def sm70_cuda_decode_enabled() -> bool:
    """Whether the CUDA decode partial is requested.

    Defaults to on for SM70 (the kernel is bit-exact and faster than the
    TileLang codegen on the same layout); set ``SGLANG_V100_DECODE_CUDA=0`` to
    fall back to the TileLang partial.
    """
    return os.environ.get("SGLANG_V100_DECODE_CUDA", "1") == "1"


def sm70_cuda_decode_available() -> bool:
    """Whether the CUDA partial can be used on this GPU."""
    if not sm70_cuda_decode_enabled():
        return False
    if not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability()
    except Exception:
        return False
    return capability == (7, 0)


def _load_sm70_cuda_decode_ops():
    """Lazy-load the standalone SM70 long-context decode extension (JIT-built)."""
    global _EXT, _OPS_LOAD_ATTEMPTED
    if _EXT is not None:
        return _EXT
    if _OPS_LOAD_ATTEMPTED:
        return None
    _OPS_LOAD_ATTEMPTED = True
    if not _SRC_PATH.is_file():
        logger.warning(
            "SM70 CUDA decode partial source not found: %s", _SRC_PATH
        )
        return None
    from torch.utils.cpp_extension import load_inline

    build_directory = os.environ.get(
        "SGLANG_V100_DECODE_CUDA_BUILD_DIR", "/tmp/sglang_sm70_longctx_decode"
    )
    os.makedirs(build_directory, exist_ok=True)
    try:
        _EXT = load_inline(
            name="sglang_sm70_longctx_decode_v100",
            cpp_sources="",
            cuda_sources=_SRC_PATH.read_text(),
            functions=None,
            is_python_module=True,
            verbose=False,
            build_directory=build_directory,
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        )
    except Exception:  # pragma: no cover - build/environment failures
        logger.exception("SM70 CUDA decode partial failed to build")
        return None
    logger.info_once("SM70 (V100): CUDA long-context decode partial loaded.")
    return _EXT


def sm70_cuda_decode_partial(
    q,
    k_cache,
    v_cache,
    page_table,
    seq_lens,
    max_splits,
    min_tokens_per_split,
    softmax_scale,
    k_scale,
    v_scale,
):
    """Run the CUDA split-KV partial, returning (partial_o, partial_lse)."""
    ext = _load_sm70_cuda_decode_ops()
    if ext is None:
        raise RuntimeError(
            "SM70 CUDA decode partial requested but extension is unavailable."
        )
    batch, heads, dim = q.shape
    page_size = k_cache.shape[1]
    partial_o = torch.empty(
        (batch, max_splits, heads, dim), dtype=torch.float16, device=q.device
    )
    partial_lse = torch.empty(
        (batch, max_splits, heads), dtype=torch.float32, device=q.device
    )
    ext.sm70_longctx_decode(
        q.contiguous(),
        k_cache.view(torch.uint8).contiguous(),
        v_cache.view(torch.uint8).contiguous(),
        page_table.to(dtype=torch.int32).contiguous(),
        seq_lens.to(dtype=torch.int32).contiguous(),
        int(max_splits),
        int(min_tokens_per_split),
        float(softmax_scale),
        float(k_scale),
        float(v_scale),
        partial_o,
        partial_lse,
    )
    return partial_o, partial_lse
