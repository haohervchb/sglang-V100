"""Exact dense-prefix D256 prefill kernel for SM70.

Long paged prefixes are gathered into logical order before this kernel is
called.  The attention body keeps the existing TileLang N32 reduction order
while removing page-table lookup, integer divide, and scattered-page address
resolution from every K/V element load.
"""

import tilelang
import tilelang.language as T

from ._kernels_paged import pass_configs

_BLOCK_M = 64
_BLOCK_N = 32
_THREADS = 256


@tilelang.jit(out_idx=[5], pass_configs=pass_configs)
def _dense_prefix_d256_kernel(heads, heads_kv):
    dim = 256
    nt = T.dynamic("nt")
    nk = T.dynamic("nk")

    @T.prim_func
    def main(
        Q: T.Tensor([nt, heads, dim], T.float16),
        K: T.Tensor([nk, heads_kv, dim], T.float16),
        V: T.Tensor([nk, heads_kv, dim], T.float16),
        prefix_kv_len: T.int32,
        sm_scale: T.float32,
        Output: T.Tensor([nt, heads, dim], T.float16),
    ):
        with T.Kernel(
            T.ceildiv(nt, _BLOCK_M), heads, threads=_THREADS
        ) as (q_tile, q_head):
            Q_shared = T.alloc_shared([_BLOCK_M, dim], T.float16)
            K_shared = T.alloc_shared([_BLOCK_N, dim], T.float16)
            V_shared = T.alloc_shared([_BLOCK_N, dim], T.float16)
            P_shared = T.alloc_shared([_BLOCK_M, _BLOCK_N], T.float16)

            scores = T.alloc_fragment([_BLOCK_M, _BLOCK_N], T.float32)
            probabilities = T.alloc_fragment([_BLOCK_M, _BLOCK_N], T.float16)
            output = T.alloc_fragment([_BLOCK_M, dim], T.float32)
            row_max = T.alloc_fragment([_BLOCK_M], T.float32)
            previous_max = T.alloc_fragment([_BLOCK_M], T.float32)
            row_sum = T.alloc_fragment([_BLOCK_M], T.float32)
            tile_sum = T.alloc_fragment([_BLOCK_M], T.float32)
            rescale = T.alloc_fragment([_BLOCK_M], T.float32)

            kv_head = q_head // (heads // heads_kv)
            query_start = q_tile * _BLOCK_M

            T.clear(Q_shared)
            for row, d in T.Parallel(_BLOCK_M, dim):
                if query_start + row < nt:
                    Q_shared[row, d] = Q[query_start + row, q_head, d]

            T.fill(output, 0)
            T.fill(row_max, -T.infinity(T.float32))
            T.fill(row_sum, 0)

            loop_end = T.min(
                T.ceildiv(nk, _BLOCK_N),
                T.ceildiv(
                    prefix_kv_len + query_start + _BLOCK_M,
                    _BLOCK_N,
                ),
            )
            for kv_tile in T.Pipelined(loop_end, num_stages=0):
                tile_start = kv_tile * _BLOCK_N
                T.clear(K_shared)
                for n, d in T.Parallel(_BLOCK_N, dim):
                    kv_index = tile_start + n
                    if kv_index < nk:
                        K_shared[n, d] = K[kv_index, kv_head, d]

                for row, n in T.Parallel(_BLOCK_M, _BLOCK_N):
                    kv_index = tile_start + n
                    scores[row, n] = T.if_then_else(
                        (query_start + row < nt)
                        & (kv_index < nk)
                        & (
                            kv_index
                            <= prefix_kv_len + query_start + row
                        ),
                        0,
                        -T.infinity(T.float32),
                    )

                T.gemm(
                    Q_shared,
                    K_shared,
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(row_max, previous_max)
                T.reduce_max(scores, row_max, dim=1, clear=False)
                for row in T.Parallel(_BLOCK_M):
                    row_max[row] = T.if_then_else(
                        row_max[row] == -T.infinity(T.float32),
                        0,
                        row_max[row],
                    )
                    row_max[row] = T.max(
                        row_max[row], previous_max[row]
                    )
                    rescale[row] = T.exp(
                        (previous_max[row] - row_max[row]) * sm_scale
                    )
                    row_sum[row] *= rescale[row]
                for row, d in T.Parallel(_BLOCK_M, dim):
                    output[row, d] *= rescale[row]
                for row, n in T.Parallel(_BLOCK_M, _BLOCK_N):
                    scores[row, n] = T.exp(
                        (scores[row, n] - row_max[row]) * sm_scale
                    )
                T.reduce_sum(scores, tile_sum, dim=1)
                for row in T.Parallel(_BLOCK_M):
                    row_sum[row] += tile_sum[row]

                T.clear(V_shared)
                for n, d in T.Parallel(_BLOCK_N, dim):
                    kv_index = tile_start + n
                    if kv_index < nk:
                        V_shared[n, d] = V[kv_index, kv_head, d]

                for row, n in T.Parallel(_BLOCK_M, _BLOCK_N):
                    P_shared[row, n] = T.cast(
                        scores[row, n], T.float16
                    )
                T.copy(P_shared, probabilities)
                T.gemm(
                    probabilities,
                    V_shared,
                    output,
                    policy=T.GemmWarpPolicy.Square,
                )

            for row, d in T.Parallel(_BLOCK_M, dim):
                if query_start + row < nt:
                    Output[query_start + row, q_head, d] = T.cast(
                        output[row, d] / row_sum[row], T.float16
                    )

    return main


_KERNEL_CACHE = {}


def get_dense_prefix_d256_kernel(heads, heads_kv):
    key = (heads, heads_kv)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _dense_prefix_d256_kernel(
            heads=heads, heads_kv=heads_kv
        )
    return _KERNEL_CACHE[key]
