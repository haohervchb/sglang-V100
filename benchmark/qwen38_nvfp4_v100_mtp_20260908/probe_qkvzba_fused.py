import json
from pathlib import Path

import torch
import triton

from sglang.jit_kernel.sm70_small_gemm import linear
from sglang.jit_kernel.utils import load_jit

p = Path("/tmp/qwen38_mtp_20260908")
mod = load_jit(
    "mtp_qkvzba_probe",
    cuda_files=[str(p / "qkvzba_fused.cuh")],
    cuda_wrappers=[
        (f"m{m}t{t}", f"mtp_qkvzba_probe::run<{m},{t}>")
        for m in [2, 4]
        for t in [64, 128, 256]
    ],
)
for m in [2, 4]:
    torch.manual_seed(31)
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, 2560, device="cuda", dtype=torch.float16) * 0.01
    tail = torch.randn(24, 2560, device="cuda", dtype=torch.float16) * 0.01
    qkv = torch.empty(m, 2560, device="cuda", dtype=torch.float16)
    z = torch.empty(m, 1536, device="cuda", dtype=torch.float16)
    b = torch.empty(m, 12, device="cuda", dtype=torch.float16)
    a = torch.empty_like(b)

    def baseline():
        p = linear(x, w)
        ba = linear(x, tail)
        return (
            p[:, :2560].contiguous(),
            p[:, 2560:].contiguous(),
            ba[:, :12].contiguous(),
            ba[:, 12:].contiguous(),
        )

    ref = baseline()
    base = triton.testing.do_bench_cudagraph(baseline, rep=150) * 1000
    for t in [64, 128, 256]:
        fn = lambda: getattr(mod, f"m{m}t{t}")(x, w, tail, qkv, z, b, a)
        fn()
        for u, v in zip(ref, [qkv, z, b, a]):
            torch.testing.assert_close(u, v, rtol=0, atol=0)
        print(
            json.dumps(
                dict(
                    m=m,
                    threads=t,
                    baseline_us=base,
                    candidate_us=triton.testing.do_bench_cudagraph(fn, rep=150) * 1000,
                    bitwise_equal=True,
                )
            ),
            flush=True,
        )
(p / "qkvzba_fused_probe.done").write_text("done")
