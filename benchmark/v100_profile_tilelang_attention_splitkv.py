#!/usr/bin/env python3
# ruff: noqa: B023, BLE001
"""Profile exact SM70 D=256 split-KV under SGLang's FP8-model shape."""

from __future__ import annotations

import argparse
import statistics

import torch
from sglang.srt.layers.attention.tilelang_fa_v100._kernels_dense_d256 import (
    get_dense_prefix_d256_kernel,
)
from sglang.srt.layers.attention.tilelang_fa_v100._kernels_dense_d256_splitkv import (
    get_dense_prefix_d256_splitkv3_kernels,
)
from sglang.srt.layers.attention.tilelang_fa_v100._kernels_paged import (
    get_paged_kernel,
)


def timed(call, *, warmup: int, iterations: int):
    output = None
    for _ in range(warmup):
        output = call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=int, default=8192)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[32768, 65536, 131072]
    )
    parser.add_argument("--splits", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--paged-control", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("This profile requires an NVIDIA V100 (SM70).")

    torch.manual_seed(43)
    q = torch.randn(args.query, 6, 256, device="cuda", dtype=torch.float16) * 0.1
    dense = get_dense_prefix_d256_kernel(6, 1)
    for context in args.contexts:
        if context < args.query:
            raise SystemExit("Every context must be at least the query length")
        k = torch.randn(context, 1, 256, device="cuda", dtype=torch.float16) * 0.1
        v = torch.randn_like(k) * 0.1
        prefix = context - args.query
        baseline_ms, baseline = timed(
            lambda: dense(q, k, v, prefix, 256**-0.5),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(f"shape=q{args.query}/k{context}/h6/hkv1/d256")
        print(f"  dense: {baseline_ms:.4f} ms")
        if args.paged_control:
            page_size = 16
            max_blocks = (context + page_size - 1) // page_size
            paged = get_paged_kernel(
                1,
                6,
                1,
                256,
                page_size,
                max_blocks,
                max_blocks,
                True,
            )
            page_table = torch.arange(
                max_blocks, device="cuda", dtype=torch.int32
            ).view(1, -1)
            seq_lens = torch.tensor([context], device="cuda", dtype=torch.int32)
            query_start_loc = torch.tensor(
                [0, args.query], device="cuda", dtype=torch.int32
            )
            prefix_lens = torch.tensor([prefix], device="cuda", dtype=torch.int32)
            paged_ms, paged_output = timed(
                lambda: paged(
                    q,
                    k.view(max_blocks, page_size, 1, 256),
                    v.view(max_blocks, page_size, 1, 256),
                    page_table,
                    seq_lens,
                    query_start_loc,
                    prefix_lens,
                    args.query,
                    256**-0.5,
                ),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            max_abs = (paged_output.float() - baseline.float()).abs().max().item()
            print(
                f"  direct-paged: {paged_ms:.4f} ms "
                f"dense-speedup={paged_ms / baseline_ms:.3f}x "
                f"max_abs={max_abs:.3e}"
            )
        for splits in args.splits:
            try:
                partial, merge = get_dense_prefix_d256_splitkv3_kernels(
                    6, 1, splits=splits
                )

                def call():
                    partial_o, partial_max, partial_sum = partial(
                        q, k, v, prefix, 256**-0.5
                    )
                    return merge(partial_o, partial_max, partial_sum, 256**-0.5)

                latency, output = timed(
                    call,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                max_abs = (output.float() - baseline.float()).abs().max().item()
                print(
                    f"  split={splits}: {latency:.4f} ms "
                    f"speedup={baseline_ms / latency:.3f}x "
                    f"max_abs={max_abs:.3e}"
                )
            except Exception as error:
                detail = str(error).splitlines()[0]
                print(f"  split={splits}: ERROR {type(error).__name__}: {detail}")


if __name__ == "__main__":
    main()
