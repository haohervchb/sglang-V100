"""Adapter: calls TileLang paged kernel on sglang's 4D paged K/V cache.
Page-by-page loading handles scattered physical blocks correctly.
Kernel uses T.Parallel for per-page element-wise load (correct for non-consecutive pages).
"""

import math
import os
import warnings

import torch

from ._kernels_paged import get_paged_kernel
from ._kernels_paged_verify import VERIFY_Q_BLOCK, get_paged_verify_kernels

warnings.filterwarnings("ignore", message="Field.*duplicates an ancestor field")

_FP8_E4M3FN_LUT = {}

# One reusable dense page buffer per CUDA stream.  Stream-local ownership makes
# reuse ordered without a host synchronization and avoids races when a process
# drives more than one model stream.
_D256_DENSE_WORKSPACES = {}
_D256_TAIL_WORKSPACES = {}
_D256_GATHER_OOM_WARNED = False

_D256_GATHER_MIN_QUERY_TOKENS = 3920
_D256_GATHER_MIN_CONTEXT = 8192


def _env_flag(name, default):
    value = os.environ.get(name, default).strip().lower()
    if value not in (
        "0",
        "false",
        "off",
        "no",
        "1",
        "true",
        "on",
        "yes",
    ):
        raise ValueError(f"{name} must be a boolean value, got {value!r}.")
    return value in ("1", "true", "on", "yes")


def _should_use_d256_gather(
    *,
    batch,
    heads,
    heads_kv,
    dim,
    num_tokens,
    max_seq_len,
    causal,
    sliding_window_size,
    fp8_kv,
    fp16,
    logical_dense_kv=False,
):
    """Evidence-bounded policy for the exact dense D256 prefill route."""
    # A bridge workspace has already paid the page-resolution cost and is
    # physically logical/dense. Use Split-D from the first full 4096-token
    # chunk; the 8K threshold only amortizes page-16 index_select gathers.
    # 1Cat's production path admits arbitrary q lengths and tail-pads them to
    # 64. SGLang's preferred 15680-token chunk is already aligned; 3920 is the
    # exact-FP8 projection cutoff and also covers a measured 4000-token prompt.
    min_context = (
        num_tokens
        if logical_dense_kv
        else _D256_GATHER_MIN_CONTEXT
    )
    return (
        _env_flag("SGLANG_V100_PREFILL_D256_GATHER", "1")
        and batch == 1
        and heads == 6
        and heads_kv == 1
        and dim == 256
        and num_tokens >= _D256_GATHER_MIN_QUERY_TOKENS
        and max_seq_len >= min_context
        and causal
        and sliding_window_size < 0
        and not fp8_kv
        and fp16
    )


def _get_d256_dense_workspace(k_cache, v_cache, required_pages):
    """Return geometrically-grown, stream-local K/V page workspaces."""
    stream = torch.cuda.current_stream(k_cache.device)
    key = (
        k_cache.device.index,
        stream.cuda_stream,
        k_cache.shape[1:],
        k_cache.dtype,
    )
    workspace = _D256_DENSE_WORKSPACES.get(key)
    if workspace is None or workspace[0].shape[0] < required_pages:
        capacity = 1 << (required_pages - 1).bit_length()
        shape = (capacity, *k_cache.shape[1:])
        workspace = (
            torch.empty(shape, dtype=k_cache.dtype, device=k_cache.device),
            torch.empty(shape, dtype=v_cache.dtype, device=v_cache.device),
        )
        _D256_DENSE_WORKSPACES[key] = workspace
    return workspace


def _get_d256_tail_workspace(q, k, v, required_tokens):
    """Return stream-ordered padding buffers for an unaligned Q tail."""
    stream = torch.cuda.current_stream(q.device)
    key = (
        q.device.index,
        stream.cuda_stream,
        q.shape[1:],
        k.shape[1:],
        q.dtype,
    )
    workspace = _D256_TAIL_WORKSPACES.get(key)
    if workspace is None or workspace[0].shape[0] < required_tokens:
        capacity = 1 << (required_tokens - 1).bit_length()
        workspace = (
            torch.empty((capacity, *q.shape[1:]), dtype=q.dtype, device=q.device),
            torch.empty((capacity, *k.shape[1:]), dtype=k.dtype, device=k.device),
            torch.empty((capacity, *v.shape[1:]), dtype=v.dtype, device=v.device),
        )
        _D256_TAIL_WORKSPACES[key] = workspace
    return workspace


def _run_d256_gathered_dense(
    q,
    k_cache,
    v_cache,
    block_table,
    max_seq_len,
    num_tokens,
    heads_kv,
    softmax_scale,
    logical_dense_kv=False,
):
    """Gather logical pages once, then run exact N32 dense attention."""
    from ._kernels_dense_d256 import get_dense_prefix_d256_kernel
    from ._native_d256 import run_native_dense_d256

    active_pages = (max_seq_len + k_cache.shape[1] - 1) // k_cache.shape[1]
    if logical_dense_kv:
        dense_k_pages = k_cache[:active_pages]
        dense_v_pages = v_cache[:active_pages]
    else:
        dense_k_pages, dense_v_pages = _get_d256_dense_workspace(
            k_cache, v_cache, active_pages
        )
        logical_pages = block_table[0, :active_pages]
        torch.index_select(
            k_cache, 0, logical_pages, out=dense_k_pages[:active_pages]
        )
        torch.index_select(
            v_cache, 0, logical_pages, out=dense_v_pages[:active_pages]
        )

    dense_k = dense_k_pages[:active_pages].flatten(0, 1)[:max_seq_len]
    dense_v = dense_v_pages[:active_pages].flatten(0, 1)[:max_seq_len]
    use_native = _env_flag("SGLANG_V100_PREFILL_D256_NATIVE", "1")
    use_splitkv3 = (
        _env_flag("SGLANG_V100_PREFILL_D256_SPLITKV3", "0")
        and max_seq_len >= 32768
    )
    if use_native:
        native_q = q
        native_k = dense_k
        native_v = dense_v
        result_start = 0
        if q.shape[0] % 64 != 0:
            padded_q_tokens = ((q.shape[0] + 63) // 64) * 64
            padded_q, padded_k, padded_v = _get_d256_tail_workspace(
                q, dense_k, dense_v, padded_q_tokens
            )
            if padded_q_tokens <= dense_k.shape[0] and dense_k.shape[0] % 32 == 0:
                # FlashAttention's causal mask is bottom-right aligned. Prefix
                # padding Q preserves the original Q/K alignment when K is
                # already long enough, exactly as in 1Cat's bridge path.
                result_start = padded_q_tokens - q.shape[0]
                padded_q[:result_start].zero_()
                padded_q[result_start:padded_q_tokens].copy_(q)
                native_q = padded_q[:padded_q_tokens]
            elif q.shape[0] == dense_k.shape[0]:
                # A full 4000-token prompt has no prefix space for left Q
                # padding. Append inert Q/K/V rows instead; causal outputs for
                # every original row are unchanged because the padding is
                # strictly after them.
                padded_q[: q.shape[0]].copy_(q)
                padded_k[: dense_k.shape[0]].copy_(dense_k)
                padded_v[: dense_v.shape[0]].copy_(dense_v)
                padded_q[q.shape[0] : padded_q_tokens].zero_()
                padded_k[dense_k.shape[0] : padded_q_tokens].zero_()
                padded_v[dense_v.shape[0] : padded_q_tokens].zero_()
                native_q = padded_q[:padded_q_tokens]
                native_k = padded_k[:padded_q_tokens]
                native_v = padded_v[:padded_q_tokens]
        native_result = run_native_dense_d256(
            native_q,
            native_k,
            native_v,
            softmax_scale,
            splitkv3=use_splitkv3,
        )
        if native_result is not None:
            return native_result[result_start : result_start + q.shape[0]]

    if use_splitkv3:
        from ._kernels_dense_d256_splitkv import (
            get_dense_prefix_d256_splitkv3_kernels,
        )

        partial, merge = get_dense_prefix_d256_splitkv3_kernels(
            q.shape[1], heads_kv
        )
        partial_o, partial_max, partial_sum = partial(
            q,
            dense_k,
            dense_v,
            max_seq_len - num_tokens,
            softmax_scale,
        )
        return merge(
            partial_o,
            partial_max,
            partial_sum,
            softmax_scale,
        )

    kernel = get_dense_prefix_d256_kernel(q.shape[1], heads_kv)
    return kernel(
        q,
        dense_k,
        dense_v,
        max_seq_len - num_tokens,
        softmax_scale,
    )


def _get_fp8_e4m3fn_lut(device):
    key = str(device)
    lut = _FP8_E4M3FN_LUT.get(key)
    if lut is None:
        raw = torch.arange(256, dtype=torch.uint8, device=device)
        lut = raw.view(torch.float8_e4m3fn).to(torch.float16)
        _FP8_E4M3FN_LUT[key] = lut
    return lut


def paged_forward(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    query_start_loc,
    prefix_kv_lens,
    out=None,
    block_size=16,
    num_kv_heads=None,
    softmax_scale=None,
    causal=True,
    sliding_window_size=-1,
    linear_verify=False,
    k_scale=1.0,
    v_scale=1.0,
    max_seq_len_hint=None,
    logical_dense_kv=False,
):
    num_tokens, num_heads, D = q.shape
    heads_kv = num_kv_heads or k_cache.shape[2]
    B = block_table.shape[0]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)

    num_blocks = k_cache.shape[0]
    max_blocks = block_table.shape[1]
    fp8_kv = (
        k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == torch.float8_e4m3fn
    )
    if max_seq_len_hint is None:
        max_seq_len_hint = int(seq_lens.max().item()) if B > 0 else 0

    query_start_loc = query_start_loc.contiguous().to(
        device=q.device, dtype=torch.int32
    )
    prefix_kv_lens = prefix_kv_lens.contiguous().to(device=q.device, dtype=torch.int32)

    # Fixed linear verify blocks are small in Q and potentially huge in KV.
    # Keep bounded sliding-window layers on the ordinary paged kernel.
    if (
        linear_verify
        and sliding_window_size < 0
        and D <= 256
        and num_tokens <= B * VERIFY_Q_BLOCK
    ):
        partial, combine, _ = get_paged_verify_kernels(
            batch=B,
            heads=num_heads,
            heads_kv=heads_kv,
            dim=D,
            block_size=block_size,
            num_pages=num_blocks,
            max_blocks=max_blocks,
            causal=causal,
            fp8_kv=fp8_kv,
        )
        if fp8_kv:
            # TileLang cannot lower a native FP8 operand for SM70. The verify
            # kernel accepts the same storage as bytes and decodes each value
            # into FP16 shared memory before WMMA.
            k_cache = k_cache.view(torch.uint8)
            v_cache = v_cache.view(torch.uint8)
        partial_o, partial_lse = partial(
            q,
            k_cache,
            v_cache,
            _get_fp8_e4m3fn_lut(q.device),
            block_table,
            seq_lens,
            query_start_loc,
            prefix_kv_lens,
            softmax_scale * k_scale,
        )
        result = combine(partial_o, partial_lse, seq_lens, query_start_loc)
    elif _should_use_d256_gather(
        batch=B,
        heads=num_heads,
        heads_kv=heads_kv,
        dim=D,
        num_tokens=num_tokens,
        max_seq_len=max_seq_len_hint,
        causal=causal,
        sliding_window_size=sliding_window_size,
        fp8_kv=fp8_kv,
        fp16=(
            q.dtype == torch.float16
            and k_cache.dtype == torch.float16
            and v_cache.dtype == torch.float16
        ),
        logical_dense_kv=logical_dense_kv,
    ) and not torch.cuda.is_current_stream_capturing():
        global _D256_GATHER_OOM_WARNED
        try:
            result = _run_d256_gathered_dense(
                q,
                k_cache,
                v_cache,
                block_table,
                max_seq_len_hint,
                num_tokens,
                heads_kv,
                softmax_scale * k_scale,
                logical_dense_kv=logical_dense_kv,
            )
        except torch.OutOfMemoryError:
            if not _D256_GATHER_OOM_WARNED:
                warnings.warn(
                    "Could not allocate the D256 dense-gather workspace; "
                    "falling back to direct paged attention.",
                    RuntimeWarning,
                )
                _D256_GATHER_OOM_WARNED = True
            kernel_compiled = get_paged_kernel(
                batch=B,
                heads=num_heads,
                heads_kv=heads_kv,
                dim=D,
                block_size=block_size,
                num_pages=num_blocks,
                max_blocks=max_blocks,
                causal=causal,
                sliding_window_size=sliding_window_size,
            )
            result = kernel_compiled(
                q,
                k_cache,
                v_cache,
                block_table,
                seq_lens,
                query_start_loc,
                prefix_kv_lens,
                num_tokens,
                softmax_scale * k_scale,
            )
    else:
        kernel_compiled = get_paged_kernel(
            batch=B,
            heads=num_heads,
            heads_kv=heads_kv,
            dim=D,
            block_size=block_size,
            num_pages=num_blocks,
            max_blocks=max_blocks,
            causal=causal,
            sliding_window_size=sliding_window_size,
        )

        # Pass 4D cache directly + block_table as page indices. block_table[b, L]
        # maps each logical page to its physical cache page.
        result = kernel_compiled(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            query_start_loc,
            prefix_kv_lens,
            num_tokens,
            softmax_scale * k_scale,
        )

    # Copy kernel output into the caller-provided 'out' tensor
    out.copy_(result[:num_tokens])
    if v_scale != 1.0:
        out.mul_(v_scale)

    softmax_lse = torch.empty(
        num_heads, num_tokens, dtype=torch.float32, device=q.device
    )
    return out, softmax_lse
