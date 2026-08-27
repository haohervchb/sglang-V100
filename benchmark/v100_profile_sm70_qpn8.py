#!/usr/bin/env python3
# ruff: noqa: B023
"""Tune the SM70 QPN8 FP8 decode kernel on Qwen3.8-27B TP4 shapes."""

from __future__ import annotations

import argparse
import statistics

import torch
from sglang.srt.layers.quantization.sm70_turbomind_fp8 import (
    _load_sm70_qpn8_ops,
)

SHAPES = {
    "gate_up_proj": (5120, 8704),
    "down_proj": (4352, 5120),
    "in_proj_qkvz": (5120, 4096),
    "out_proj": (1536, 5120),
}


def bench(fn, warmup, iterations, inner):
    for _ in range(warmup):
        for _ in range(inner):
            output = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            output = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    return statistics.median(samples), output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--shapes", nargs="*", default=list(SHAPES))
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=31)
    parser.add_argument("--inner", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.m <= 8:
        raise SystemExit("--m must be in [1, 8]")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("This profile requires an NVIDIA V100 (SM70).")
    unknown = set(args.shapes) - set(SHAPES)
    if unknown:
        raise SystemExit(f"Unknown shapes: {sorted(unknown)}")

    ext = _load_sm70_qpn8_ops()
    torch.manual_seed(89)
    for name in args.shapes:
        k, n = SHAPES[name]
        x = torch.randn(args.m, k, device="cuda", dtype=torch.float16) * 0.1
        codes = torch.randint(0, 256, (n * k,), dtype=torch.uint8, device="cuda")
        scales = torch.randn(
            (k // 128) * (n // 32), device="cuda", dtype=torch.float16
        ).abs_()

        reference_ms, reference = bench(
            lambda: ext.qpn8_linear(x, codes, scales, n, k, 16, 1),
            args.warmup,
            args.iterations,
            args.inner,
        )
        print(f"shape={name} M={args.m} K={k} N={n} baseline={reference_ms:.5f} ms")
        for splitk in (4, 8, 16, 32):
            if (k // 16) % splitk:
                continue
            for nacc_arg in (1, 2, 3, 4):
                if splitk == 4 and nacc_arg == 4:
                    continue
                latency, output = bench(
                    lambda splitk=splitk, nacc_arg=nacc_arg: ext.qpn8_linear(
                        x, codes, scales, n, k, splitk, nacc_arg
                    ),
                    args.warmup,
                    args.iterations,
                    args.inner,
                )
                diff = (output.float() - reference.float()).abs()
                decoder = "fast" if nacc_arg >= 3 else "scalar"
                nacc = nacc_arg - 2 if nacc_arg >= 3 else nacc_arg
                print(
                    f"  split={splitk:2d} nacc={nacc} {decoder:6s} "
                    f"{latency:8.5f} ms speedup={reference_ms / latency:6.3f}x "
                    f"max_abs={diff.max().item():.3e}"
                )
        if name == "gate_up_proj":
            out_features = n // 2

            def unfused_gate_up():
                gate_up = ext.qpn8_linear(x, codes, scales, n, k, 16, 3)
                return (
                    torch.nn.functional.silu(gate_up[:, :out_features])
                    * gate_up[:, out_features:]
                )

            unfused_ms, unfused = bench(
                unfused_gate_up,
                args.warmup,
                args.iterations,
                args.inner,
            )
            fused_ms, fused = bench(
                lambda: ext.qpn8_gated_silu(x, codes, scales, out_features, k),
                args.warmup,
                args.iterations,
                args.inner,
            )
            diff = (fused.float() - unfused.float()).abs()
            exact = torch.equal(fused, unfused)
            print(
                f"  gate_silu unfused={unfused_ms:.5f} ms "
                f"fused={fused_ms:.5f} ms speedup={unfused_ms / fused_ms:.3f}x "
                f"bitwise={exact} max_abs={diff.max().item():.3e}"
            )


if __name__ == "__main__":
    main()
