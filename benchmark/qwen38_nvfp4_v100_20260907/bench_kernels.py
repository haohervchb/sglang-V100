"""Graph-replay timings for the exact TP4 NVFP4 decode shapes; run with idle GPUs."""

import argparse
import json
import os
import statistics

import torch
import triton

from sglang.jit_kernel.sm70_nvfp4_moe_decode import (
    sm70_nvfp4_moe_decode,
    sm70_topk10_softmax,
)
from sglang.srt.layers.hc_mix_triton import (
    sm70_hc_down_gemv_silu,
    sm70_hc_up_gemv_reduce,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        parser.error("This benchmark requires SM70")
    torch.manual_seed(31)

    def measure(name, fn, **shape):
        fn()
        torch.cuda.synchronize()
        times = [
            triton.testing.do_bench_cudagraph(fn, rep=150) * 1000
            for _ in range(args.repeats)
        ]
        print(
            json.dumps(
                dict(
                    label=args.label,
                    kernel=name,
                    **shape,
                    us=times,
                    median_us=statistics.median(times)
                )
            ),
            flush=True,
        )

    x = torch.randn(1, 10240, device="cuda", dtype=torch.float16)
    down_weight = torch.randn(320, 10240, device="cuda", dtype=torch.float16) * 0.01
    up_weight = torch.randn(10240, 320, device="cuda", dtype=torch.float16) * 0.01
    activated = sm70_hc_down_gemv_silu(x, down_weight, 4)
    for native in ("0", "1"):
        os.environ["SGLANG_SM70_HC_NATIVE"] = native
        measure(
            "hc_down", lambda: sm70_hc_down_gemv_silu(x, down_weight, 4), native=native
        )
        measure(
            "hc_up",
            lambda: sm70_hc_up_gemv_reduce(activated, x, up_weight, 4, 2560),
            native=native,
        )

    # Use all 512 experts so the routed weight set does not fit in Volta's L2.
    w13 = torch.empty((512, 160, 640), dtype=torch.int32, device="cuda").random_()
    w2 = torch.empty((512, 10, 5120), dtype=torch.int32, device="cuda").random_()
    s13 = torch.full((512, 160, 320), 0x70, dtype=torch.uint8, device="cuda")
    s2 = torch.full((512, 10, 2560), 0x70, dtype=torch.uint8, device="cuda")
    g13 = torch.full((512,), 100.0, device="cuda")
    g2 = torch.full((512,), 100.0, device="cuda")
    for batch in (1, 2, 4):
        logits = torch.randn(batch, 512, dtype=torch.float16, device="cuda")
        weights, ids = sm70_topk10_softmax(logits)
        hidden = torch.randn(batch, 2560, dtype=torch.float16, device="cuda")
        measure("topk10", lambda: sm70_topk10_softmax(logits), batch=batch)
        measure(
            "nvfp4_moe",
            lambda: sm70_nvfp4_moe_decode(
                hidden, w13, w2, s13, s2, g13, g2, ids, weights
            ),
            batch=batch,
        )


if __name__ == "__main__":
    main()
