import ctypes as C
import json
from pathlib import Path

import torch
import triton

p = Path("/tmp/qwen38_mtp_20260908")
cu = C.CDLL("libcuda.so.1")
torch.cuda.init()
torch.empty(1, device="cuda")


def ok(code):
    if code:
        raise RuntimeError(f"CUDA driver error {code}")


syms = [
    "_ZN14sm70_nvfp4_moe22gate_up_partial_kernelEPK6__halfPKjPKhPKiiPf",
    "_ZN14sm70_nvfp4_moe26gate_up_reduce_silu_kernelEPKfS1_PKiiP6__half",
    "_ZN14sm70_nvfp4_moe19down_partial_kernelEPK6__halfPKjPKhPKfPKiS8_iPf",
    "_ZN14sm70_nvfp4_moe18down_reduce_kernelEPKfiP6__half",
]
cu.cuLaunchKernel.argtypes = [
    C.c_void_p,
    C.c_uint,
    C.c_uint,
    C.c_uint,
    C.c_uint,
    C.c_uint,
    C.c_uint,
    C.c_uint,
    C.c_void_p,
    C.POINTER(C.c_void_p),
    C.c_void_p,
]


def load(v):
    mod = C.c_void_p()
    ok(cu.cuModuleLoad(C.byref(mod), str(p / f"{v}.cubin").encode()))
    fs = []
    for name in syms:
        f = C.c_void_p()
        ok(cu.cuModuleGetFunction(C.byref(f), mod, name.encode()))
        fs.append(f)
    return mod, fs


def runner(fs, nt, args, work):
    holders = [
        [
            C.c_uint64(a.data_ptr()) if isinstance(a, torch.Tensor) else C.c_int(a)
            for a in aa
        ]
        for aa in args
    ]
    params = [
        (C.c_void_p * len(aa))(*[C.cast(C.pointer(x), C.c_void_p) for x in aa])
        for aa in holders
    ]

    def fn():
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        for f, pa, n in zip(fs, params, work):
            ok(
                cu.cuLaunchKernel(
                    f, (n + nt - 1) // nt, 1, 1, nt, 1, 1, 0, stream, pa, None
                )
            )

    fn.holders = holders
    return fn


variants = [
    ("baseline", 64),
    ("constant", 64),
    ("constant", 128),
    ("vector", 64),
    ("vector", 128),
    ("vector_input", 64),
    ("vector_input", 128),
    ("vector_input_gate", 64),
    ("vector_input_gate", 128),
]
mods = {
    (mode, nt): load(
        "moe_g4_r16_t64" if mode == "baseline" else f"moe_scale_{mode}_t{nt}"
    )
    for mode, nt in variants
}
for m in [4, 2, 1]:
    torch.manual_seed(123)
    w13 = torch.empty(512, 160, 640, device="cuda", dtype=torch.int32).random_()
    w2 = torch.empty(512, 10, 5120, device="cuda", dtype=torch.int32).random_()
    s13 = torch.randint(64, 112, (512, 160, 320), device="cuda", dtype=torch.uint8)
    s2 = torch.randint(64, 112, (512, 10, 2560), device="cuda", dtype=torch.uint8)
    g13 = torch.full((512,), 100.0, device="cuda")
    g2 = g13.clone()
    ids = torch.rand(m, 512, device="cuda").topk(10, dim=-1).indices.int().flatten()
    weights = torch.softmax(torch.randn(m, 10, device="cuda"), -1).flatten()
    x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
    for (mode, nt), (mod, fs) in mods.items():
        split, down = 40, 5
        part = torch.empty(split, m * 10, 320, device="cuda")
        a = torch.empty(m * 10, 160, device="cuda", dtype=torch.float16)
        dp = torch.empty(m * 10, down, 2560, device="cuda")
        o = torch.empty_like(x)
        args = [
            [x, w13, s13, ids, m * 10, part],
            [part, g13, ids, m * 10, a],
            [a, w2, s2, g2, ids, weights, m * 10, dp],
            [dp, m, o],
        ]
        work = [m * 10 * split * 40, m * 10 * 160, m * 10 * down * 320, m * 2560]
        fn = runner(fs, nt, args, work)
        fn()
        torch.cuda.synchronize()
        if mode == "baseline":
            ref = o.clone()
        torch.testing.assert_close(o, ref, rtol=0, atol=0)
        err = (o.float() - ref.float()).abs().max().item()
        rel = (o.float() - ref.float()).norm().item() / ref.float().norm().item()
        times = [
            triton.testing.do_bench_cudagraph(fn, rep=100) * 1000 for _ in range(2)
        ]
        stages = [
            triton.testing.do_bench_cudagraph(runner([f], nt, [aa], [ww]), rep=80)
            * 1000
            for f, aa, ww in zip(fs, args, work)
        ]
        row = dict(
            batch=m,
            scale_mode=mode,
            threads=nt,
            times_us=times,
            stages_us=stages,
            max_error=err,
            rel_l2=rel,
        )
        print(json.dumps(row), flush=True)
        with (p / "moe_scale_timing.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    del w13, w2, s13, s2, ref
(p / "moe_scale_probe.done").write_text("done")
