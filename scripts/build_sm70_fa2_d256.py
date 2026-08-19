#!/usr/bin/env python3
"""Build 1Cat's exact SM70 D256 FA2 operators against the active Torch ABI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

repo = Path(__file__).resolve().parents[1]
source = Path(
    os.environ.get(
        "SGLANG_SM70_FA2_ROOT",
        "~/.cache/sglang-v100-sources/flash-attention-v100-fa2",
    )
).expanduser()
required_source = source / "csrc/flash_attn/src/flash_fwd_d256_splitd_sm70.cu"
if not required_source.is_file():
    raise SystemExit(
        f"Patched SM70 FA2 source not found at {required_source}. Apply 1Cat's "
        "sm70_flash_attn_d256_pipeline.patch and splitkv3.patch first."
    )

cuda_tag = str(torch.version.cuda or "cpu").replace(".", "")
torch_tag = torch.__version__.split("+")[0].replace(".", "_")
build_root = source / "build" / f"sglang-torch{torch_tag}-cu{cuda_tag}"
build_temp = build_root / "temp"
build_lib = build_root / "lib"
build_temp.mkdir(parents=True, exist_ok=True)
build_lib.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
env.setdefault("CUDACXX", str(Path(env.get("CUDA_HOME", "/usr/local/cuda-12.8")) / "bin/nvcc"))
env.setdefault("CUDAHOSTCXX", "/usr/bin/g++-12")
env.setdefault("CC", "/usr/bin/gcc-12")
env.setdefault("CXX", "/usr/bin/g++-12")
env.setdefault("CMAKE_BUILD_TYPE", "Release")

subprocess.check_call(
    [
        sys.executable,
        "setup.py",
        "build_ext",
        "--build-temp",
        str(build_temp),
        "--build-lib",
        str(build_lib),
    ],
    cwd=source,
    env=env,
)
artifacts = list(build_lib.glob("vllm_flash_attn/_vllm_fa2_C*.so"))
if len(artifacts) != 1:
    raise SystemExit(f"Expected one SM70 FA2 artifact under {build_lib}, got {artifacts}")
destination = repo / "python/sglang/jit_kernel/_sm70_fa2_d256.so"
shutil.copy2(artifacts[0], destination)
print(f"Installed {destination}")
