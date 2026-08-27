#!/usr/bin/env python3
# ruff: noqa: B023, BLE001
"""Profile the native TileLang E5M2 paged-KV prefill bridge on V100."""

from __future__ import annotations

import argparse
import statistics

import torch
from sglang.srt.layers.attention.tilelang_fa_v100._kernels_fp8_bridge import (
    get_fp8_paged_gather_kernel,
)


def timed(call, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[8192, 65536, 131072]
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=31)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("This profile requires an NVIDIA V100 (SM70).")

    page_size, heads_kv, dim = 16, 1, 256
    raw = torch.arange(256, device="cuda", dtype=torch.uint8)
    lut = raw.view(torch.float8_e5m2).to(torch.float16)
    torch.manual_seed(47)
    for context in args.contexts:
        max_blocks = (context + page_size - 1) // page_size
        num_pages = max_blocks + 17
        k_cache = torch.randn(
            num_pages,
            page_size,
            heads_kv,
            dim,
            device="cuda",
            dtype=torch.float16,
        ).to(torch.float8_e5m2)
        v_cache = torch.randn_like(k_cache, dtype=torch.float16).to(torch.float8_e5m2)
        page_table = (
            torch.randperm(num_pages, device="cuda", dtype=torch.int64)[:max_blocks]
            .to(torch.int32)
            .view(1, -1)
        )
        seq_lens = torch.tensor([context], device="cuda", dtype=torch.int32)
        k_output = torch.empty(
            max_blocks, page_size, heads_kv, dim, device="cuda", dtype=torch.float16
        )
        v_output = torch.empty_like(k_output)

        lut_kernel = get_fp8_paged_gather_kernel(
            1, heads_kv, dim, page_size, num_pages, max_blocks, False, 128
        )
        lut_call = lambda: lut_kernel(
            k_cache.view(torch.uint8),
            v_cache.view(torch.uint8),
            lut,
            page_table,
            seq_lens,
            k_output,
            v_output,
        )
        lut_ms = timed(lut_call, warmup=args.warmup, iterations=args.iterations)
        lut_call()
        torch.cuda.synchronize()
        baseline_k = k_output.clone()
        baseline_v = v_output.clone()
        print(f"context={context} pages={max_blocks} lut_t128={lut_ms:.4f} ms")

        for threads in args.threads:
            try:
                kernel = get_fp8_paged_gather_kernel(
                    1,
                    heads_kv,
                    dim,
                    page_size,
                    num_pages,
                    max_blocks,
                    True,
                    threads,
                )
                call = lambda: kernel(
                    k_cache.view(torch.uint8),
                    v_cache.view(torch.uint8),
                    lut,
                    page_table,
                    seq_lens,
                    k_output,
                    v_output,
                )
                latency = timed(call, warmup=args.warmup, iterations=args.iterations)
                call()
                torch.cuda.synchronize()
                k_equal = torch.equal(k_output, baseline_k)
                v_equal = torch.equal(v_output, baseline_v)
                print(
                    f"  bits_t{threads}: {latency:.4f} ms "
                    f"speedup={lut_ms / latency:.3f}x exact={k_equal and v_equal}"
                )
            except Exception as error:
                detail = str(error).splitlines()[0]
                print(f"  bits_t{threads}: ERROR {type(error).__name__}: {detail}")


if __name__ == "__main__":
    main()
