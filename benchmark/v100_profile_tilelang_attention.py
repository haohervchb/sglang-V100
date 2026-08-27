#!/usr/bin/env python3
# ruff: noqa: BLE001
"""Profile native SM70 TileLang D=256 attention schedules.

This is deliberately an operator-level harness: it uses the Qwen3.8 TP4
geometry that the production adapter dispatches (Hq=6, Hkv=1, D=256), checks
each generated schedule against the retained baseline, and prints median CUDA
event latency.  It does not import or execute any external attention package.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch
from sglang.srt.layers.attention.tilelang_fa_v100._kernels_dense_d256 import (
    get_dense_prefix_d256_kernel,
)


@dataclass(frozen=True)
class Config:
    name: str
    block_m: int
    threads: int
    block_n: int = 32
    qk_policy: int = 2
    pv_policy: int = 1
    kv_union: bool = False
    gemm_version: int = 2
    num_stages: int = 0


CONFIGS = (
    Config("bm64_t256", 64, 256),
    Config("bm64_bn16_t256", 64, 256, block_n=16),
    Config("bm64_bn16_t256_union", 64, 256, block_n=16, kv_union=True),
    Config(
        "bm64_bn16_t256_union_qk1",
        64,
        256,
        block_n=16,
        qk_policy=1,
        kv_union=True,
    ),
    Config(
        "bm64_bn16_t256_union_qk3",
        64,
        256,
        block_n=16,
        qk_policy=3,
        kv_union=True,
    ),
    Config(
        "bm64_bn16_t256_union_pv2",
        64,
        256,
        block_n=16,
        pv_policy=2,
        kv_union=True,
    ),
    Config(
        "bm64_bn16_t256_union_pv3",
        64,
        256,
        block_n=16,
        pv_policy=3,
        kv_union=True,
    ),
    Config(
        "bm64_bn16_t256_union_v1",
        64,
        256,
        block_n=16,
        kv_union=True,
        gemm_version=1,
    ),
    Config("bm64_bn64_t256", 64, 256, block_n=64),
    Config("bm64_bn48_t256", 64, 256, block_n=48),
    Config("bm64_t128", 64, 128),
    Config("bm64_t512", 64, 512),
    Config("bm64_t256_s1", 64, 256, num_stages=1),
    Config("bm64_t256_s2", 64, 256, num_stages=2),
    Config("bm64_t256_qk1", 64, 256, qk_policy=1),
    Config("bm64_t256_qk3", 64, 256, qk_policy=3),
    Config("bm64_t256_pv2", 64, 256, pv_policy=2),
    Config("bm64_t256_pv3", 64, 256, pv_policy=3),
    Config("bm64_t256_v1", 64, 256, gemm_version=1),
    Config("bm64_t256_union", 64, 256, kv_union=True),
    Config("bm64_t128_union", 64, 128, kv_union=True),
    Config("bm64_t512_union", 64, 512, kv_union=True),
    Config("bm32_t128", 32, 128),
    Config("bm32_bn16_t128", 32, 128, block_n=16),
    Config("bm32_bn16_t128_union", 32, 128, block_n=16, kv_union=True),
    Config("bm32_bn64_t128", 32, 128, block_n=64),
    Config("bm32_bn64_t256", 32, 256, block_n=64),
    Config("bm32_bn128_t512", 32, 512, block_n=128),
    Config("bm32_t128_union", 32, 128, kv_union=True),
    Config("bm32_t256", 32, 256),
    Config("bm32_t256_union", 32, 256, kv_union=True),
    Config("bm32_t256_qk1", 32, 256, qk_policy=1),
    Config("bm32_t256_qk3", 32, 256, qk_policy=3),
    Config("bm32_t256_pv2", 32, 256, pv_policy=2),
    Config("bm32_t256_pv3", 32, 256, pv_policy=3),
    Config("bm32_t512_union", 32, 512, kv_union=True),
    Config("bm32_t512_qk1", 32, 512, qk_policy=1),
    Config("bm32_t512_qk3", 32, 512, qk_policy=3),
    Config("bm32_t512_pv2", 32, 512, pv_policy=2),
    Config("bm32_t512_pv3", 32, 512, pv_policy=3),
    Config("bm16_t128_union", 16, 128, kv_union=True),
    Config("bm16_t256_union", 16, 256, kv_union=True),
    Config("bm64_t256_union_qk1", 64, 256, qk_policy=1, kv_union=True),
    Config("bm64_t256_union_qk3", 64, 256, qk_policy=3, kv_union=True),
    Config("bm64_t256_union_pv2", 64, 256, pv_policy=2, kv_union=True),
    Config("bm64_t256_union_pv3", 64, 256, pv_policy=3, kv_union=True),
    Config("bm64_t256_union_v1", 64, 256, kv_union=True, gemm_version=1),
    Config("bm32_t256_union_qk1", 32, 256, qk_policy=1, kv_union=True),
    Config("bm32_t256_union_qk3", 32, 256, qk_policy=3, kv_union=True),
    Config("bm32_t256_union_pv2", 32, 256, pv_policy=2, kv_union=True),
    Config("bm32_t256_union_pv3", 32, 256, pv_policy=3, kv_union=True),
    Config("bm32_t256_union_v1", 32, 256, kv_union=True, gemm_version=1),
)


def bench(fn, args, *, warmup: int, iterations: int) -> tuple[float, torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = fn(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn(*args)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    assert output is not None
    return statistics.median(samples), output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=int, default=4096)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--configs", nargs="*", default=[])
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("This profile requires an NVIDIA V100 (SM70).")
    if args.context < args.query:
        raise SystemExit("--context must be at least --query")

    torch.manual_seed(37)
    q = torch.randn(args.query, 6, 256, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(args.context, 1, 256, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn_like(k) * 0.1
    kernel_args = (q, k, v, args.context - args.query, 256**-0.5)
    selected = [
        config for config in CONFIGS if not args.configs or config.name in args.configs
    ]
    missing = set(args.configs) - {config.name for config in selected}
    if missing:
        raise SystemExit(f"Unknown configs: {sorted(missing)}")

    baseline_config = CONFIGS[0]
    baseline_fn = get_dense_prefix_d256_kernel(
        6,
        1,
        qk_policy=baseline_config.qk_policy,
        pv_policy=baseline_config.pv_policy,
        block_m=baseline_config.block_m,
        block_n=baseline_config.block_n,
        threads=baseline_config.threads,
        kv_union=baseline_config.kv_union,
        gemm_version=baseline_config.gemm_version,
        num_stages=baseline_config.num_stages,
    )
    baseline_ms, baseline = bench(
        baseline_fn, kernel_args, warmup=args.warmup, iterations=args.iterations
    )
    print(
        f"shape=q{args.query}/k{args.context}/h6/hkv1/d256 "
        f"baseline={baseline_ms:.4f} ms"
    )

    rows = []
    for config in selected:
        try:
            fn = get_dense_prefix_d256_kernel(
                6,
                1,
                qk_policy=config.qk_policy,
                pv_policy=config.pv_policy,
                block_m=config.block_m,
                block_n=config.block_n,
                threads=config.threads,
                kv_union=config.kv_union,
                gemm_version=config.gemm_version,
                num_stages=config.num_stages,
            )
            latency, output = bench(
                fn, kernel_args, warmup=args.warmup, iterations=args.iterations
            )
            max_abs = (output.float() - baseline.float()).abs().max().item()
            rows.append((latency, config.name, max_abs))
            print(
                f"{config.name:28s} {latency:9.4f} ms "
                f"speedup={baseline_ms / latency:6.3f}x max_abs={max_abs:.3e}"
            )
        except Exception as error:  # profile invalid code-generation choices too
            detail = str(error).splitlines()[0]
            print(f"{config.name:28s} ERROR {type(error).__name__}: {detail}")

    print("ranking")
    for latency, name, max_abs in sorted(rows):
        print(
            f"{name:28s} {latency:9.4f} ms "
            f"speedup={baseline_ms / latency:6.3f}x max_abs={max_abs:.3e}"
        )


if __name__ == "__main__":
    main()
