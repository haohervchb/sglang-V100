# syntax=docker/dockerfile:1.7

# Reproducible SGLang image for NVIDIA V100 (Volta, SM70). Expensive native
# builds are deliberately separate layers; a later validation failure never
# invalidates completed FlashInfer, sglang-kernel, or Marlin compilation.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
    CUDAHOSTCXX=/usr/bin/g++-12 \
    TORCH_CUDA_ARCH_LIST=7.0 \
    NVCC_THREADS=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.cargo/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git g++-12 libnuma-dev \
      ninja-build patch pkg-config protobuf-compiler python3.12 python3.12-dev \
      python3-pip python3-venv \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && python -m venv "$VIRTUAL_ENV"

FROM base AS builder

ARG MAX_JOBS
ENV MAX_JOBS=${MAX_JOBS}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal

WORKDIR /opt/sglang
COPY scripts/v100_safe_jobs.sh /usr/local/bin/v100-safe-jobs
RUN chmod +x /usr/local/bin/v100-safe-jobs

# Install SGLang's current runtime dependency list while deliberately excluding
# the four packages replaced below by CUDA 12.8 / SM70 builds.
COPY python/pyproject.toml /tmp/sglang-pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade \
      pip setuptools wheel setuptools-scm setuptools-rust scikit-build-core \
      ninja psutil packaging \
    && python -m pip install \
      torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
      --index-url https://download.pytorch.org/whl/cu128 \
    && python - <<'PY'
import subprocess
import sys
import tomllib
from packaging.requirements import Requirement

with open("/tmp/sglang-pyproject.toml", "rb") as file:
    project = tomllib.load(file)["project"]

dependencies = [
    *project["dependencies"],
    *project["optional-dependencies"]["diffusion-v100"],
]

replaced = {
    "flashinfer-python",
    "sglang-kernel",
    "torch",
    "torchaudio",
    "torchvision",
}
dependencies = [
    dependency
    for dependency in dependencies
    if Requirement(dependency).name.lower().replace("_", "-") not in replaced
]
subprocess.check_call([sys.executable, "-m", "pip", "install", *dependencies])
PY
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install \
      grpcio==1.81.1 grpcio-health-checking==1.81.1 \
      grpcio-reflection==1.81.1 protobuf==6.33.6 tilelang==0.1.8 \
    && python -m pip uninstall -y nvidia-nccl-cu13 || true \
    && python -m pip install --force-reinstall --no-deps \
      nvidia-nccl-cu12==2.27.5

# Patched FlashInfer SM70 source. This is editable because its JIT headers and
# Python sources are both needed at runtime.
COPY patches/flashinfer-sm70.patch \
      /opt/sglang/patches/flashinfer-sm70.patch
RUN git clone https://github.com/haohervchb/flashinfer.git \
      /opt/deps/flashinfer-sm70 \
    && git -C /opt/deps/flashinfer-sm70 checkout --detach \
      c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833 \
    && git -C /opt/deps/flashinfer-sm70 apply \
      /opt/sglang/patches/flashinfer-sm70.patch
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip uninstall -y flashinfer-python flashinfer-cubin || true \
    && python -m pip install --no-deps --no-build-isolation \
      -e /opt/deps/flashinfer-sm70

# TurboMind is the one temporarily retained external SM70 component. Fetch
# only its attributed source directories; never install the 1Cat vLLM or
# attention packages and never place a full 1Cat checkout in the image.
RUN git clone --filter=blob:none --sparse --no-checkout \
      https://github.com/1CatAI/1Cat-vLLM.git \
      /opt/deps/turbomind-sm70-source \
    && git -C /opt/deps/turbomind-sm70-source sparse-checkout set \
      LICENSE csrc/core csrc/sm70_turbomind csrc/moe \
    && git -C /opt/deps/turbomind-sm70-source checkout --detach \
      6ada86ed64af6d1a7b3cb0f34df237fd86f06d48
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git \
      /opt/deps/cutlass-turbomind \
    && git -C /opt/deps/cutlass-turbomind checkout --detach \
      da5e086dab31d63815acafdac9a9c5893b1c69e2

# Build SGLang's adapter for the attributed TurboMind W8A16 block-FP8 and
# FP16 MoE source against the pinned CUTLASS revision used for host validation.
COPY scripts/build_sm70_turbomind.py /opt/sglang/scripts/build_sm70_turbomind.py
COPY python/sglang/jit_kernel/csrc/sm70_turbomind_bindings.cpp \
     /opt/sglang/python/sglang/jit_kernel/csrc/sm70_turbomind_bindings.cpp
COPY python/sglang/jit_kernel/csrc/sm70_fp16_moe_gemm.cu \
     /opt/sglang/python/sglang/jit_kernel/csrc/sm70_fp16_moe_gemm.cu
COPY python/sglang/jit_kernel/csrc/sm70_fp8_e5m2_cache.cu \
     /opt/sglang/python/sglang/jit_kernel/csrc/sm70_fp8_e5m2_cache.cu
RUN --mount=type=cache,target=/root/.cache/torch_extensions,sharing=locked \
    export MAX_JOBS="$(v100-safe-jobs)" \
    && export SGLANG_TURBOMIND_SM70_ROOT=/opt/deps/turbomind-sm70-source \
    && export SGLANG_TURBOMIND_CUTLASS_ROOT=/opt/deps/cutlass-turbomind \
    && python /opt/sglang/scripts/build_sm70_turbomind.py

# Only this source tree invalidates the sglang-kernel layer. The context ignores
# all local .so/build outputs, preventing the stale-binary bug from the host.
COPY sgl-kernel /opt/sglang/sgl-kernel
RUN python -m pip uninstall -y sglang-kernel || true
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    find /opt/sglang/sgl-kernel/python/sgl_kernel \
      -type f -name 'common_ops*.so' -delete \
    && JOBS="$(v100-safe-jobs)" \
    && export MAX_JOBS="${JOBS}" \
    && export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
    && export CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=1" \
    && python -m pip install --no-deps --no-build-isolation \
      /opt/sglang/sgl-kernel

# Marlin uses the dedicated marlin_v100 repository and the exact proven local
# compatibility/tuning patches. Its smoke test is deferred to the next layer.
RUN git clone --depth 1 --branch v4.2.1 \
      https://github.com/NVIDIA/cutlass.git /opt/cutlass
COPY scripts/setup_v100_marlin.sh /opt/sglang/scripts/setup_v100_marlin.sh
COPY patches/marlin-v100-qwen-sm70-tuning.patch \
      /opt/sglang/patches/marlin-v100-qwen-sm70-tuning.patch
RUN --mount=type=cache,target=/opt/deps/marlin-v100,sharing=locked \
    export CUTLASS_DIR=/opt/cutlass \
    && export MARLIN_V100_REPO=/opt/deps/marlin-v100 \
    && export MARLIN_V100_REF=6d72a49939701d26b15b617a4cd2423174adb2d1 \
    && export MARLIN_V100_INSTALL_DIR=/opt/v100-artifacts \
    && export MARLIN_V100_SKIP_SMOKE=1 \
    && export MARLIN_V100_SKIP_BF16_COMPAT=1 \
    && export MAX_JOBS="$(v100-safe-jobs)" \
    && bash /opt/sglang/scripts/setup_v100_marlin.sh

# Application changes do not invalidate any native compilation layer.
COPY python /opt/sglang/python
RUN --mount=type=bind,source=rust/sglang-grpc,target=/mnt/sglang-grpc,ro \
    --mount=type=bind,source=proto,target=/mnt/proto,ro \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    mkdir -p /opt/sglang/rust \
    && cp -a /mnt/sglang-grpc /opt/sglang/rust/sglang-grpc \
    && cp -a /mnt/proto /opt/sglang/proto \
    && python -m pip install --no-deps --no-build-isolation -e /opt/sglang/python \
    && python -m pip install cuda-tile==1.5.0 \
    && cp /opt/v100-artifacts/_sm70_marlin_v100_*.abi3.so \
      /opt/sglang/python/sglang/jit_kernel/

# GPU-independent artifact validation is intentionally after every expensive
# compilation RUN, so BuildKit retains those layers if this check ever changes.
RUN python - <<'PY'
import importlib.util
from pathlib import Path
import subprocess

site = Path("/opt/venv/lib/python3.12/site-packages")
common_ops = list((site / "sgl_kernel" / "sm70").glob("common_ops*.so"))
assert len(common_ops) == 1, common_ops
assert common_ops[0].name == "common_ops.abi3.so", common_ops[0]
marlin_dir = Path("/opt/sglang/python/sglang/jit_kernel")
marlin = [
    marlin_dir / "_sm70_marlin_v100_dense.abi3.so",
    marlin_dir / "_sm70_marlin_v100_moe.abi3.so",
]
turbomind = marlin_dir / "_sm70_turbomind_v100.so"
grpc_core = list(Path("/opt/sglang/python/sglang/srt/grpc").glob("_core*.so"))
assert len(grpc_core) == 1, grpc_core
assert Path("/opt/deps/flashinfer-sm70/flashinfer/sampling.py").is_file()
assert Path(
    "/opt/sglang/python/sglang/srt/layers/attention/"
    "tilelang_fa_v100/_kernels_paged_decode.py"
).is_file()
for module in ("flash_attn_v100", "flash_attn_v100_cuda", "flash_qla_v100"):
    assert importlib.util.find_spec(module) is None, module

def validate_sm70(binary, required_strings):
    binary = Path(binary)
    assert binary.stat().st_size > 100_000, binary
    cubins = subprocess.check_output(
        ["cuobjdump", "--list-elf", str(binary)], text=True
    )
    assert ".sm_70.cubin" in cubins, (binary, cubins)
    assert all(f".sm_{arch}." not in cubins for arch in (75, 80, 86, 89, 90, 100))
    strings = subprocess.check_output(["strings", str(binary)], text=True)
    for value in required_strings:
        assert value in strings, (binary, value)

validate_sm70(common_ops[0], ["all_reduce", "gptq_gemm", "causal_conv1d_fwd"])
validate_sm70(marlin[0], ["marlin_gemm"])
validate_sm70(marlin[1], ["moe_wna16_marlin_gemm"])
validate_sm70(turbomind, ["fp8_gemm", "f16_moe_gemm", "fp8_e5m2_cache_write"])
print("SM70 build artifacts validated")
PY

FROM base AS runtime

ENV NCCL_P2P_LEVEL=NVL \
    SGLANG_MAMBA_CONV_DTYPE=float16 \
    SGLANG_MAMBA_SSM_DTYPE=float16 \
    HF_HOME=/root/.cache/huggingface \
    FLASHINFER_WORKSPACE_BASE=/root/sglang-v100-jit \
    TILELANG_CACHE_DIR=/root/sglang-v100-jit/tilelang \
    TRITON_CACHE_DIR=/root/sglang-v100-jit/triton \
    TORCHINDUCTOR_CACHE_DIR=/root/sglang-v100-jit/torchinductor \
    SGLANG_V100_PYTHON=/opt/venv/bin/python

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/deps/flashinfer-sm70 /opt/deps/flashinfer-sm70
COPY --from=builder /opt/sglang/python /opt/sglang/python
COPY scripts/smoke_v100.sh /opt/sglang/scripts/smoke_v100.sh
COPY docker/v100-entrypoint.sh /usr/local/bin/v100-entrypoint
RUN chmod +x /opt/sglang/scripts/smoke_v100.sh /usr/local/bin/v100-entrypoint

WORKDIR /opt/sglang
EXPOSE 8082

ENTRYPOINT ["/usr/local/bin/v100-entrypoint"]
CMD ["--help"]
