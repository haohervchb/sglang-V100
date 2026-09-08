import json
from pathlib import Path

import torch
import triton

from sglang.jit_kernel.sm70_small_gemm import linear
from sglang.jit_kernel.utils import load_jit

p = Path("/tmp/qwen38_mtp_20260908")
mod = load_jit(
    "mtp_batch_fusion_probe",
    cuda_files=[str(p / "batch_fusions.cuh")],
    cuda_wrappers=[(f"gate{t}", f"mtp_fusion_probe::run<{t}>") for t in [128, 256, 512]]
    + [(f"up{m}", f"mtp_fusion_probe::gate_up<{m}>") for m in [2, 4]],
    extra_cuda_cflags=["--fmad=false"],
)
for m in [2, 4]:
    torch.manual_seed(31)
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    w = torch.randn(1, 2560, device="cuda", dtype=torch.float16) * 0.01
    v = torch.randn_like(x)
    o = torch.empty_like(x)

    def baseline():
        return torch.sigmoid(torch.nn.functional.linear(x, w)) * v

    ref = baseline()
    base = triton.testing.do_bench_cudagraph(baseline, rep=150) * 1000
    for t in [128, 256, 512]:
        fn = lambda: getattr(mod, f"gate{t}")(x, w, v, o)
        fn()
        torch.testing.assert_close(o, ref, rtol=0.002, atol=0.001)
        print(
            json.dumps(
                dict(
                    op="gate",
                    m=m,
                    threads=t,
                    baseline_us=base,
                    candidate_us=triton.testing.do_bench_cudagraph(fn, rep=150) * 1000,
                    max_error=(o - ref).abs().max().item(),
                )
            ),
            flush=True,
        )
    w = torch.randn(320, 2560, device="cuda", dtype=torch.float16) * 0.01
    o = torch.empty(m, 160, device="cuda", dtype=torch.float16)

    def baseline_up():
        z = linear(x, w)
        return (
            torch.nn.functional.silu(z[:, :160].float()) * z[:, 160:].float()
        ).half()

    from sgl_kernel import silu_and_mul

    act = silu_and_mul
    basefn = lambda: act(linear(x, w))
    ref = baseline_up()
    fn = lambda: getattr(mod, f"up{m}")(x, w, o)
    fn()
    torch.testing.assert_close(o, ref, rtol=0.002, atol=0.001)
    print(
        json.dumps(
            dict(
                op="gate_up",
                m=m,
                baseline_us=triton.testing.do_bench_cudagraph(basefn, rep=150) * 1000,
                candidate_us=triton.testing.do_bench_cudagraph(fn, rep=150) * 1000,
                max_error=(o - ref).abs().max().item(),
            )
        ),
        flush=True,
    )
(p / "batch_fusions_probe.done").write_text("done")
