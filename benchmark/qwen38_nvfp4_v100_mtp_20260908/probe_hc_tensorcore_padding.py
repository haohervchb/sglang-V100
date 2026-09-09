import json
from pathlib import Path

import torch
import triton

p = Path("/tmp/qwen38_mtp_20260908")
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
for n, k in [(320, 10240), (10240, 320), (4096, 2560)]:
    torch.manual_seed(n + k)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.01
    x = torch.randn(4, k, device="cuda", dtype=torch.float16)
    reference = x.float() @ w.float().T
    for m in [4, 8, 16, 32]:
        padded = torch.zeros(m, k, device="cuda", dtype=torch.float16)
        padded[:4].copy_(x)
        o = torch.empty(m, n, device="cuda", dtype=torch.float16)

        def run():
            torch.mm(padded, w.T, out=o)

        run()
        torch.testing.assert_close(o[:4].float(), reference, rtol=0.003, atol=0.001)
        times = [
            triton.testing.do_bench_cudagraph(run, rep=150) * 1000 for _ in range(3)
        ]
        row = dict(
            shape=[n, k],
            padded_rows=m,
            times_us=times,
            max_abs_error=(o[:4].float() - reference).abs().max().item(),
        )
        print(json.dumps(row), flush=True)
        with (p / "hc_tensorcore_padding_timing.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
(p / "hc_tensorcore_padding_probe.done").write_text("done")
