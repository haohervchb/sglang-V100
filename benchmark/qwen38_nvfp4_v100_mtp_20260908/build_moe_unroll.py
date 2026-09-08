import json
import subprocess
from pathlib import Path

p = Path("/tmp/qwen38_mtp_20260908")
src = Path("python/sglang/jit_kernel/csrc/sm70_nvfp4_moe_decode.cu").read_text()
src = (
    src[
        src.index("namespace sm70_nvfp4_moe {") : src.index("void decode(torch::Tensor")
    ]
    + "\n}\n"
)
src = (
    "#include <cuda_fp16.h>\n#include <cfloat>\n#include <climits>\n#include <cstdint>\n#include <cub/block/block_radix_sort.cuh>\n"
    + src
)
variants = [
    (g, r, t)
    for g, r in [
        (4, 16),
        (1, 16),
        (1, 8),
        (1, 4),
        (1, 2),
        (1, 1),
        (2, 4),
        (2, 8),
        (4, 4),
    ]
    for t in [64, 128]
]
for g, r, t in variants:
    name = f"moe_g{g}_r{r}_t{t}"
    s = (
        src.replace("kThreads = 64", f"kThreads = {t}")
        .replace(
            "#pragma unroll\n  for (int group_it",
            f"#pragma unroll {g}\n  for (int group_it",
        )
        .replace(
            "#pragma unroll\n    for (int r = 0;",
            f"#pragma unroll {r}\n    for (int r = 0;",
        )
    )
    (p / f"{name}.cu").write_text(s)
    with (p / f"{name}.build.log").open("w") as log:
        subprocess.run(
            [
                "/usr/local/cuda-12.8/bin/nvcc",
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-arch=sm_70",
                "-Xptxas=-v",
                "-cubin",
                str(p / f"{name}.cu"),
                "-o",
                str(p / f"{name}.cubin"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(name, flush=True)
(p / "moe_unroll_variants.json").write_text(json.dumps(variants))
