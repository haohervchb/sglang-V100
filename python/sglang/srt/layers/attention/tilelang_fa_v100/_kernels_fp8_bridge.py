"""TileLang FP8 paged-KV gather/dequantization for SM70.

Volta cannot consume FP8 in tensor cores. Long prefill therefore resolves each
logical page once into a reusable FP16 workspace, amortizing the linear copy
over thousands of query rows. Both E4M3 and E5M2 are decoded by a 256-entry
FP16 lookup supplied by the adapter.
"""

from functools import lru_cache

import tilelang
import tilelang.language as T

from ._kernels_paged import pass_configs


@tilelang.jit(pass_configs=pass_configs)
def _fp8_paged_gather_kernel(
    batch: int,
    heads_kv: int,
    dim: int,
    page_size: int,
    num_pages: int,
    max_blocks: int,
):
    @T.prim_func
    def main(
        KCache: T.Tensor([num_pages, page_size, heads_kv, dim], T.uint8),
        VCache: T.Tensor([num_pages, page_size, heads_kv, dim], T.uint8),
        Lut: T.Tensor([256], T.float16),
        PageTable: T.Tensor([batch, max_blocks], T.int32),
        SeqLens: T.Tensor([batch], T.int32),
        KOutput: T.Tensor([batch * max_blocks, page_size, heads_kv, dim], T.float16),
        VOutput: T.Tensor([batch * max_blocks, page_size, heads_kv, dim], T.float16),
    ):
        with T.Kernel(max_blocks, heads_kv, batch, threads=128) as (
            logical_page,
            head,
            sequence,
        ):
            active_pages = T.ceildiv(SeqLens[sequence], page_size)
            if logical_page < active_pages:
                physical_page = PageTable[sequence, logical_page]
                output_page = sequence * max_blocks + logical_page
                for offset, d in T.Parallel(page_size, dim):
                    KOutput[output_page, offset, head, d] = Lut[
                        T.cast(
                            KCache[physical_page, offset, head, d],
                            T.int32,
                        )
                    ]
                    VOutput[output_page, offset, head, d] = Lut[
                        T.cast(
                            VCache[physical_page, offset, head, d],
                            T.int32,
                        )
                    ]

    return main


@lru_cache(maxsize=None)
def get_fp8_paged_gather_kernel(
    batch: int,
    heads_kv: int,
    dim: int,
    page_size: int,
    num_pages: int,
    max_blocks: int,
):
    return _fp8_paged_gather_kernel(
        batch,
        heads_kv,
        dim,
        page_size,
        num_pages,
        max_blocks,
    )
