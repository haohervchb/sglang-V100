"""Benchmark the graph-safe SM70 FP8 exact-dense prefill dispatch.

This isolates the Qwen3.8-27B-FP8 TP4 projection shapes admitted by the
production policy.  It compares compact TurboMind W8A16 with the runtime
dispatch that reconstructs one FP16 KxN weight in a shared workspace and
calls cuBLAS.  Run on one V100; no model checkpoint is required.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.srt.layers.quantization.sm70_turbomind_fp8 import (
    _load_sm70_turbomind_fp8_ops,
)


SHAPES = {
    "gate_up_proj": (5120, 8704, True),
    "down_proj": (4352, 5120, False),
    "out_proj": (1536, 5120, False),
}


def timed_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples)


def run_shape(name: str, m: int, warmup: int, repeats: int) -> None:
    k, n, gated = SHAPES[name]
    weight = (
        torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.05
    ).to(torch.float8_e4m3fn)
    scales = torch.ones(
        ((n + 127) // 128, (k + 127) // 128),
        device="cuda",
        dtype=torch.float32,
    )
    packed, packed_scales, meta = (
        torch.ops.sglang_sm70_turbomind.fp8_prepare(weight, scales, 128, gated)
    )
    dense = torch.empty((k, n), device="cuda", dtype=torch.float16)
    torch.ops.sglang_sm70_turbomind.fp8_dequantize_out(
        dense, packed, packed_scales, 128
    )
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    out_n = n // 2 if gated else n
    compact_out = torch.empty((m, out_n), device="cuda", dtype=torch.float16)
    exact_out = torch.empty_like(compact_out)

    def compact():
        torch.ops.sglang_sm70_turbomind.fp8_gemm(
            compact_out,
            x,
            packed,
            packed_scales,
            128,
            int(meta[0]),
            int(meta[1]),
            gated,
        )

    def exact():
        torch.ops.sglang_sm70_turbomind.fp8_prefill_dispatch(
            exact_out,
            dense.data_ptr(),
            x,
            packed,
            packed_scales,
            128,
            int(meta[0]),
            int(meta[1]),
            gated,
            1,
        )

    compact()
    exact()
    torch.cuda.synchronize()
    max_abs = float((compact_out - exact_out).abs().max())
    relative_l2 = float(
        torch.linalg.vector_norm((compact_out - exact_out).float())
        / torch.linalg.vector_norm(compact_out.float())
    )
    compact_ms = timed_ms(compact, warmup, repeats)
    exact_ms = timed_ms(exact, warmup, repeats)
    print(
        f"{name:12s} M={m} K={k} N={n}: "
        f"compact={compact_ms:.4f} ms exact={exact_ms:.4f} ms "
        f"speedup={compact_ms / exact_ms:.3f}x "
        f"max_abs={max_abs:.6g} rel_l2={relative_l2:.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--shape", choices=["all", *SHAPES], default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("This benchmark requires an NVIDIA V100 (SM70).")
    if not _load_sm70_turbomind_fp8_ops():
        raise SystemExit("The private SM70 TurboMind extension is unavailable.")
    torch.manual_seed(20260819)
    names = SHAPES if args.shape == "all" else (args.shape,)
    for name in names:
        run_shape(name, args.m, args.warmup, args.repeats)


if __name__ == "__main__":
    main()
