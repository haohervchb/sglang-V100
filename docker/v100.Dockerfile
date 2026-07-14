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
      ninja-build pkg-config protobuf-compiler python3.12 python3.12-dev \
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
    dependencies = tomllib.load(file)["project"]["dependencies"]

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

COPY patches /opt/sglang/patches

# Patched FlashInfer SM70 source. This is editable because its JIT headers and
# Python sources are both needed at runtime.
RUN git clone https://github.com/haohervchb/flashinfer.git \
      /opt/deps/flashinfer-sm70 \
    && git -C /opt/deps/flashinfer-sm70 checkout --detach \
      c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833 \
    && git -C /opt/deps/flashinfer-sm70 apply \
      /opt/sglang/patches/flashinfer-sm70.patch
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --no-deps --no-build-isolation \
      -e /opt/deps/flashinfer-sm70

# Native SM70 attention fallback. Docker has nvcc but no GPU during build, so
# the final small patch permits the explicit sm_70 cross-compilation.
RUN git clone https://github.com/ai-bond/flash-attention-v100.git \
      /opt/deps/flash-attention-v100 \
    && git -C /opt/deps/flash-attention-v100 checkout --detach \
      d89800edf608d85744f3ab6188be5fd0736acf39 \
    && git -C /opt/deps/flash-attention-v100 apply \
      /opt/sglang/patches/flash-attention-v100-sglang.patch \
    && git -C /opt/deps/flash-attention-v100 apply \
      /opt/sglang/patches/flash-attention-v100-torch291.patch \
    && git -C /opt/deps/flash-attention-v100 apply \
      /opt/sglang/patches/flash-attention-v100-cross-compile.patch
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export MAX_JOBS="$(v100-safe-jobs)" SGLANG_V100_CROSS_COMPILE=1 \
    && python -m pip install --no-deps --no-build-isolation \
      /opt/deps/flash-attention-v100

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
RUN export CUTLASS_DIR=/opt/cutlass \
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
    && cp /opt/v100-artifacts/_sm70_marlin_v100_*.abi3.so \
      /opt/sglang/python/sglang/jit_kernel/

# GPU-independent artifact validation is intentionally after every expensive
# compilation RUN, so BuildKit retains those layers if this check ever changes.
RUN python - <<'PY'
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
native_attention = list(site.glob("flash_attn_v100_cuda*.so"))
grpc_core = list(Path("/opt/sglang/python/sglang/srt/grpc").glob("_core*.so"))
assert len(native_attention) == 1, native_attention
assert len(grpc_core) == 1, grpc_core
assert Path("/opt/deps/flashinfer-sm70/flashinfer/sampling.py").is_file()

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
print("SM70 build artifacts validated")
PY

FROM base AS runtime

ENV FLASHINFER_DISABLE_VERSION_CHECK=1 \
    NCCL_P2P_LEVEL=NVL \
    SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
    SGLANG_MAMBA_CONV_DTYPE=float16 \
    SGLANG_MAMBA_SSM_DTYPE=float16 \
    HF_HOME=/root/.cache/huggingface \
    SGLANG_V100_PYTHON=/opt/venv/bin/python

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/deps/flashinfer-sm70 /opt/deps/flashinfer-sm70
COPY --from=builder /opt/sglang/python /opt/sglang/python
COPY scripts/smoke_v100.sh /opt/sglang/scripts/smoke_v100.sh
COPY docker/v100-entrypoint.sh /usr/local/bin/v100-entrypoint
RUN chmod +x /opt/sglang/scripts/smoke_v100.sh /usr/local/bin/v100-entrypoint

WORKDIR /opt/sglang
EXPOSE 8082
VOLUME ["/root/.cache/huggingface", "/root/.cache/flashinfer", "/root/.tilelang", "/root/.triton", "/tmp/torchinductor_root"]

ENTRYPOINT ["/usr/local/bin/v100-entrypoint"]
CMD ["--help"]
