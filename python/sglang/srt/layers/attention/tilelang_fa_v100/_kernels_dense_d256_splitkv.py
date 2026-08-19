"""Experimental exact three-way split-KV form of the SM70 D256 kernel.

This mirrors 1Cat's split-state contract: each CTA writes an unnormalised FP32
numerator plus local max/sum, and a second kernel performs the associative
softmax merge.  It remains an opt-in fallback experiment; the native Split-D
operator is faster on the measured V100 stack.
"""

import tilelang
import tilelang.language as T

from ._kernels_paged import pass_configs

_BLOCK_M = 64
_BLOCK_N = 32
_THREADS = 256
_SPLITS = 3


@tilelang.jit(out_idx=[5, 6, 7], pass_configs=pass_configs)
def _dense_prefix_d256_splitkv3_partial(heads, heads_kv):
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
        Partial_O: T.Tensor([_SPLITS, nt, heads, dim], T.float32),
        Partial_Max: T.Tensor([_SPLITS, nt, heads], T.float32),
        Partial_Sum: T.Tensor([_SPLITS, nt, heads], T.float32),
    ):
        with T.Kernel(
            T.ceildiv(nt, _BLOCK_M), heads, _SPLITS, threads=_THREADS
        ) as (q_tile, q_head, split_id):
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
            visible_blocks = T.min(
                T.ceildiv(nk, _BLOCK_N),
                T.ceildiv(prefix_kv_len + query_start + _BLOCK_M, _BLOCK_N),
            )
            split_begin = visible_blocks * split_id // _SPLITS
            split_end = visible_blocks * (split_id + 1) // _SPLITS
            for local_tile in T.Pipelined(split_end - split_begin, num_stages=0):
                kv_tile = split_begin + local_tile
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
                        & (kv_index <= prefix_kv_len + query_start + row),
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
                    row_max[row] = T.max(row_max[row], previous_max[row])
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
                    P_shared[row, n] = T.cast(scores[row, n], T.float16)
                T.copy(P_shared, probabilities)
                T.gemm(
                    probabilities,
                    V_shared,
                    output,
                    policy=T.GemmWarpPolicy.Square,
                )

            for row in T.Parallel(_BLOCK_M):
                if query_start + row < nt:
                    Partial_Max[split_id, query_start + row, q_head] = row_max[row]
                    Partial_Sum[split_id, query_start + row, q_head] = row_sum[row]
            for row, d in T.Parallel(_BLOCK_M, dim):
                if query_start + row < nt:
                    Partial_O[split_id, query_start + row, q_head, d] = output[
                        row, d
                    ]

    return main


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def _dense_prefix_d256_splitkv3_merge(heads):
    dim = 256
    nt = T.dynamic("nt")

    @T.prim_func
    def main(
        Partial_O: T.Tensor([_SPLITS, nt, heads, dim], T.float32),
        Partial_Max: T.Tensor([_SPLITS, nt, heads], T.float32),
        Partial_Sum: T.Tensor([_SPLITS, nt, heads], T.float32),
        sm_scale: T.float32,
        Output: T.Tensor([nt, heads, dim], T.float16),
    ):
        with T.Kernel(T.ceildiv(nt, _BLOCK_M), heads, threads=_THREADS) as (
            q_tile,
            q_head,
        ):
            weights = T.alloc_shared([_BLOCK_M, _SPLITS], T.float32)
            denominator = T.alloc_shared([_BLOCK_M], T.float32)
            query_start = q_tile * _BLOCK_M
            for row in T.Parallel(_BLOCK_M):
                query_row = query_start + row
                if query_row < nt:
                    global_max = T.max(
                        Partial_Max[0, query_row, q_head],
                        T.max(
                            Partial_Max[1, query_row, q_head],
                            Partial_Max[2, query_row, q_head],
                        ),
                    )
                    weights[row, 0] = T.exp(
                        (Partial_Max[0, query_row, q_head] - global_max)
                        * sm_scale
                    )
                    weights[row, 1] = T.exp(
                        (Partial_Max[1, query_row, q_head] - global_max)
                        * sm_scale
                    )
                    weights[row, 2] = T.exp(
                        (Partial_Max[2, query_row, q_head] - global_max)
                        * sm_scale
                    )
                    denominator[row] = (
                        Partial_Sum[0, query_row, q_head] * weights[row, 0]
                        + Partial_Sum[1, query_row, q_head] * weights[row, 1]
                        + Partial_Sum[2, query_row, q_head] * weights[row, 2]
                    )
            T.sync_threads()
            for row, d in T.Parallel(_BLOCK_M, dim):
                query_row = query_start + row
                if query_row < nt:
                    numerator = (
                        Partial_O[0, query_row, q_head, d] * weights[row, 0]
                        + Partial_O[1, query_row, q_head, d] * weights[row, 1]
                        + Partial_O[2, query_row, q_head, d] * weights[row, 2]
                    )
                    Output[query_row, q_head, d] = T.cast(
                        numerator / denominator[row], T.float16
                    )

    return main


_CACHE = {}


def get_dense_prefix_d256_splitkv3_kernels(heads, heads_kv):
    key = (heads, heads_kv)
    if key not in _CACHE:
        _CACHE[key] = (
            _dense_prefix_d256_splitkv3_partial(heads, heads_kv),
            _dense_prefix_d256_splitkv3_merge(heads),
        )
    return _CACHE[key]
