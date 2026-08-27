#!/usr/bin/env python3
# ruff: noqa: B023
"""Profile exact SM70 grouped decode partition policies.

The harness uses Qwen3.8-27B TP4 geometry (B1/H6/Hkv1/D256), E5M2 paged KV,
and the same CUDA partial plus TileLang reducer used by production SGLang.  It
keeps the largest contexts in the sweep because partition policies that look
neutral at 64K can change CTA-wave efficiency substantially at 150K-250K.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch
from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import (
    sm70_cuda_decode_partial,
)
from sglang.srt.layers.attention.tilelang_fa_v100._kernels_paged_decode import (
    _decode_combine_kernel,
)


@dataclass(frozen=True)
class Config:
    name: str
    max_splits: int
    min_tokens_per_split: int


CONFIGS = (
    Config("current_s160_t32", 160, 32),
    Config("p1024_s256", 256, 1024),
    Config("p512_s512", 512, 512),
    Config("p256_s1024", 1024, 256),
)


def run_once(config, q, k, v, page_table, seq_lens, scale):
    partial_o, partial_lse = sm70_cuda_decode_partial(
        q,
        k,
        v,
        page_table,
        seq_lens,
        config.max_splits,
        config.min_tokens_per_split,
        scale,
        1.0,
        1.0,
    )
    combine = _decode_combine_kernel(
        1,
        6,
        256,
        config.max_splits,
        256,
        config.min_tokens_per_split,
    )
    return combine(partial_o, partial_lse, seq_lens)


def bench(fn, *, warmup, iterations):
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[4096, 32768, 65536, 131072, 180224, 245760],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--configs", nargs="*", default=[])
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("This profile requires an NVIDIA V100 (SM70).")
    selected = [
        config for config in CONFIGS if not args.configs or config.name in args.configs
    ]
    missing = set(args.configs) - {config.name for config in selected}
    if missing:
        raise SystemExit(f"Unknown configs: {sorted(missing)}")

    max_context = max(args.contexts)
    torch.manual_seed(73)
    q = torch.randn(1, 6, 256, device="cuda", dtype=torch.float16) * 0.1
    k = (torch.randn(max_context, 1, 256, device="cuda") * 0.1).to(torch.float8_e5m2)
    v = (torch.randn(max_context, 1, 256, device="cuda") * 0.1).to(torch.float8_e5m2)
    k = k.view(-1, 16, 1, 256)
    v = v.view(-1, 16, 1, 256)
    page_table = torch.arange(k.shape[0], dtype=torch.int32, device="cuda").view(1, -1)
    scale = 256**-0.5

    baseline_config = CONFIGS[0]
    for context in args.contexts:
        seq_lens = torch.tensor([context], dtype=torch.int32, device="cuda")
        baseline_ms, baseline = bench(
            lambda: run_once(baseline_config, q, k, v, page_table, seq_lens, scale),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(f"context={context} baseline={baseline_ms:.5f} ms")
        for config in selected:
            latency, output = bench(
                lambda config=config: run_once(
                    config, q, k, v, page_table, seq_lens, scale
                ),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            diff = (output.float() - baseline.float()).abs()
            active = min(
                config.max_splits,
                max(
                    1,
                    (context + config.min_tokens_per_split - 1)
                    // config.min_tokens_per_split,
                ),
            )
            print(
                f"  {config.name:18s} active={active:4d} "
                f"{latency:9.5f} ms speedup={baseline_ms / latency:6.3f}x "
                f"max_abs={diff.max().item():.3e}"
            )


if __name__ == "__main__":
    main()
