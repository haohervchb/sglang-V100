from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# Constants matching device::marlin_moe:: in marlin.cuh
_MAX_THREAD_N = 256


# --- SM70 (V100) marlin_v100 integration ---------------------------------------
# The stock JIT Marlin MoE kernel (csrc/gemm/marlin_moe/marlin_template.h) is an
# empty-body stub for __CUDA_ARCH__ < 800: it launches but writes nothing, so on
# V100 the routed experts silently contribute zero (the unquantized shared
# expert + residual mask the corruption). marlin_v100 (a vLLM-derived SM70 fork
# with real WMMA kernels) replaces it. Built+installed by
# scripts/setup_v100_marlin.sh; auto-detected here on SM70 with no env var.

_IS_SM70: bool = False
try:
    if torch.cuda.is_available():
        _IS_SM70 = torch.cuda.get_device_capability()[0] == 7
except Exception:
    pass

# Cache for the loaded op. States: None = not attempted yet; callable = loaded;
# False = attempted but unavailable (so we only warn once).
_marlin_v100_op = None


def _load_marlin_v100_op():
    """Lazily auto-detect and load the marlin_v100 MoE op on SM70.

    Returns the registered ``torch.ops._moe_C.moe_wna16_marlin_gemm`` callable,
    or ``None`` if it could not be found. Subsequent calls return the cached
    result. No-op (returns None) on non-SM70 devices.
    """
    global _marlin_v100_op
    if _marlin_v100_op is not False and _marlin_v100_op is not None:
        return _marlin_v100_op
    if _marlin_v100_op is False:
        return None
    _marlin_v100_op = False  # mark attempted

    if not _IS_SM70:
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser("~")
    candidates = []
    # 1. installed next to sglang's jit_kernel package (setup_v100_marlin.sh)
    candidates += sorted(glob.glob(os.path.join(here, "_sm70_marlin_v100_moe*.so")))
    # 2. dev build in a ~/marlin_v100 checkout
    candidates += sorted(glob.glob(os.path.join(home, "marlin_v100", "vllm", "_moe_C*.so")))

    for path in candidates:
        try:
            torch.ops.load_library(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("SM70 (V100): failed to load marlin_v100 .so at %s: %s", path, e)
            continue
        op = getattr(torch.ops._moe_C, "moe_wna16_marlin_gemm", None)
        if op is not None:
            _marlin_v100_op = op
            logger.info("SM70 (V100): using marlin_v100 MoE kernel from %s", path)
            return op

    logger.warning(
        "SM70 (V100) detected but the marlin_v100 MoE kernel was not found. "
        "The stock JIT Marlin kernel is an empty stub on SM70, so routed-expert "
        "output will be ZERO (incorrect). Build it with: "
        "`bash scripts/setup_v100_marlin.sh`. Searched: %s",
        candidates,
    )
    return None


@cache_once
def _jit_moe_wna16_marlin_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "moe_wna16_marlin",
        *args,
        cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],
        cuda_wrappers=[
            (
                "moe_wna16_marlin_gemm",
                f"moe_wna16_marlin_gemm<{args}>",
            )
        ],
    )


def _or_empty(
    t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)


@debug_kernel_api
def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    perm_or_none: Optional[torch.Tensor],
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    device = a.device

    # Allocate output if not provided
    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.empty((size_m * top_k, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    # SM70 (V100): dispatch to the marlin_v100 kernel when available. Its op
    # signature differs from the JIT module: it takes `a_scales` (None for
    # W4A16) and tuning ints (thread_k/n, blocks_per_sm; -1 = auto-select), and
    # it computes has_act_order/has_bias/has_zp/num_groups/group_size/is_ep
    # internally from the tensor shapes, so those are not forwarded. The raw
    # Optional tensors are passed through unchanged (None => std::nullopt) so
    # the kernel's has_value()-based presence checks fire correctly; do NOT
    # convert None to an empty tensor (the kernel treats a present-but-empty
    # global_scale as an nvfp4-only input and rejects it for GPTQ/AWQ).
    if _IS_SM70:
        op = _load_marlin_v100_op()
        if op is not None:
            op(
                a,
                c,
                b_q_weight,
                b_bias_or_none,
                b_scales,
                None,  # a_scales (W4A16 has no activation quantization)
                global_scale_or_none,
                b_zeros_or_none,
                g_idx_or_none,
                perm_or_none,
                workspace,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights,
                moe_block_size,
                top_k,
                mul_topk_weights,
                b_q_type.id,
                size_m,
                size_n,
                size_k,
                is_k_full,
                use_atomic_add,
                use_fp32_reduce,
                is_zp_float,
                -1,  # thread_k  (-1 => C++ model-specific auto-select)
                -1,  # thread_n
                -1,  # blocks_per_sm
            )
            return c
        # fall through to the stock JIT path (empty stub on SM70) with the
        # warning already emitted by _load_marlin_v100_op.

    # Determine activation ordering
    has_act_order = (
        g_idx_or_none is not None
        and perm_or_none is not None
        and g_idx_or_none.numel() > 0
        and perm_or_none.numel() > 0
        and g_idx_or_none.size(-1) > 0
        and perm_or_none.size(-1) > 0
    )

    # Determine has_zp
    has_zp = b_zeros_or_none is not None and b_zeros_or_none.numel() > 0

    # Determine has_bias
    has_bias = b_bias_or_none is not None

    # Derive num_groups and group_size from b_scales
    num_groups = b_scales.size(1)
    if has_act_order:
        if is_k_full:
            group_size = size_k // num_groups
        else:
            group_size = 0
    else:
        if num_groups > 1:
            group_size = size_k // num_groups
        else:
            group_size = -1

    # Allocate a_tmp for act_order column permutation
    if has_act_order:
        a_tmp = torch.empty((size_m * top_k, size_k), dtype=a.dtype, device=device)
    else:
        a_tmp = torch.empty(0, dtype=a.dtype, device=device)

    # Allocate c_tmp for fp32 reduce
    if use_fp32_reduce and not use_atomic_add:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        # max num of threadblocks is sms * 4
        max_c_tmp_size = min(
            size_n * sorted_token_ids.size(0),
            sms * 4 * moe_block_size * _MAX_THREAD_N,
        )
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = torch.empty(max_c_tmp_size, dtype=torch.float32, device=device)
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    # Convert Optional tensors to empty tensors
    g_idx_t = _or_empty(g_idx_or_none, device, torch.int32)
    perm_t = _or_empty(perm_or_none, device, torch.int32)
    b_zeros_t = _or_empty(b_zeros_or_none, device, a.dtype)
    b_bias_t = _or_empty(b_bias_or_none, device, a.dtype)
    global_scale_t = _or_empty(global_scale_or_none, device, a.dtype)

    module = _jit_moe_wna16_marlin_module(a.dtype)
    module.moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias_t,
        b_scales,
        global_scale_t,
        b_zeros_t,
        g_idx_t,
        perm_t,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        a_tmp,
        c_tmp,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        has_act_order,
        has_bias,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )

    return c
