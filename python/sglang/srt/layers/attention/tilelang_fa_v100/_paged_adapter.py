"""Adapter: calls TileLang paged kernel on sglang's 4D paged K/V cache.
Page-by-page loading handles scattered physical blocks correctly.
Kernel uses T.Parallel for per-page element-wise load (correct for non-consecutive pages).
"""
import math
import warnings
import torch

from ._kernels_paged import get_paged_kernel

warnings.filterwarnings("ignore", message="Field.*duplicates an ancestor field")


def paged_forward(q, k_cache, v_cache, block_table, seq_lens,
                  query_start_loc, prefix_kv_lens, out=None,
                  block_size=16, num_kv_heads=None, softmax_scale=None,
                  causal=True, sliding_window_size=-1):
    num_tokens, num_heads, D = q.shape
    heads_kv = num_kv_heads or k_cache.shape[2]
    B = block_table.shape[0]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)

    num_blocks = k_cache.shape[0]
    max_blocks = block_table.shape[1]

    kernel_compiled = get_paged_kernel(
        batch=B, heads=num_heads, heads_kv=heads_kv, dim=D,
        block_size=block_size, num_pages=num_blocks,
        max_blocks=max_blocks, causal=causal,
        sliding_window_size=sliding_window_size,
    )

    # Pass 4D cache directly + block_table as page indices
    # Kernel loads each tile page-by-page: block_table[b, L] gives physical page
    result = kernel_compiled(
        q, k_cache, v_cache, block_table, seq_lens,
        query_start_loc.contiguous().to(device=q.device, dtype=torch.int32),
        prefix_kv_lens.contiguous().to(device=q.device, dtype=torch.int32),
        num_tokens,
        softmax_scale,
    )

    # Copy kernel output into the caller-provided 'out' tensor
    out.copy_(result)

    softmax_lse = torch.empty(num_heads, num_tokens, dtype=torch.float32, device=q.device)
    return out, softmax_lse
