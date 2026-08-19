"""1Cat's exact SM70 Split-D D256 prefill operators.

The extension is built separately from SGLang so the small, Volta-only FA2
translation unit does not enlarge the normal sgl-kernel build.  Loading stays
lazy: installations without the optional binary continue to use TileLang.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_OPS_CHECKED = False
_DENSE_OP = None
_SPLITKV3_OP = None
_SPLITKV3_WORKSPACES = {}


def _get_ops():
    global _OPS_CHECKED, _DENSE_OP, _SPLITKV3_OP
    if _OPS_CHECKED:
        return _DENSE_OP, _SPLITKV3_OP
    _OPS_CHECKED = True

    namespace = getattr(torch.ops, "_vllm_fa2_C", None)
    if namespace is None or not hasattr(
        namespace, "sm70_d256_splitd_n32_dense_fwd"
    ):
        library = (
            Path(__file__).resolve().parents[4]
            / "jit_kernel"
            / "_sm70_fa2_d256.so"
        )
        if not library.is_file():
            return None, None
        try:
            torch.ops.load_library(str(library))
        except (OSError, RuntimeError) as exc:
            logger.warning(
                "Could not load the optional SM70 D256 FA2 extension (%s); "
                "using TileLang.",
                exc,
            )
            return None, None
        namespace = torch.ops._vllm_fa2_C

    _DENSE_OP = getattr(
        namespace, "sm70_d256_splitd_n32_dense_fwd", None
    )
    _SPLITKV3_OP = getattr(
        namespace, "sm70_d256_splitd_n32_dense_splitkv3_fwd", None
    )
    return _DENSE_OP, _SPLITKV3_OP


def native_dense_d256_available() -> bool:
    dense, _ = _get_ops()
    return dense is not None


def _get_splitkv3_workspace(q4: torch.Tensor):
    stream = torch.cuda.current_stream(q4.device)
    key = (
        q4.device.index,
        stream.cuda_stream,
        q4.dtype,
        tuple(q4.shape),
    )
    workspace = _SPLITKV3_WORKSPACES.get(key)
    if workspace is None:
        partial_out = torch.empty(
            (3, *q4.shape), dtype=torch.float32, device=q4.device
        )
        partial_max = torch.empty(
            (3, *q4.shape[:-1]), dtype=torch.float32, device=q4.device
        )
        workspace = (partial_out, partial_max, torch.empty_like(partial_max))
        _SPLITKV3_WORKSPACES[key] = workspace
    return workspace


def run_native_dense_d256(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    *,
    splitkv3: bool = False,
) -> torch.Tensor | None:
    """Run dense causal attention for contiguous ``[M,H,D]`` tensors."""
    dense_op, splitkv3_op = _get_ops()
    if dense_op is None:
        return None
    if (
        q.device.type != "cuda"
        or q.dtype != torch.float16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or q.ndim != 3
        or k.ndim != 3
        or v.ndim != 3
        or q.shape[-1] != 256
        or k.shape[-1] != 256
        or v.shape != k.shape
        or q.shape[0] % 64 != 0
        or k.shape[0] % 32 != 0
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or v.stride(-1) != 1
    ):
        return None

    q4 = q.unsqueeze(0)
    k4 = k.unsqueeze(0)
    v4 = v.unsqueeze(0)
    out = torch.empty_like(q4)
    if splitkv3 and splitkv3_op is not None:
        try:
            partial_out, partial_max, partial_sum = _get_splitkv3_workspace(q4)
        except torch.OutOfMemoryError:
            splitkv3 = False
        else:
            splitkv3_op(
                q4,
                k4,
                v4,
                partial_out,
                partial_max,
                partial_sum,
                out,
                softmax_scale,
                True,
            )
            return out[0]

    dense_op(q4, k4, v4, out, softmax_scale, True)
    return out[0]
