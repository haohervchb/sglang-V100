# SGLang for 4× NVIDIA V100 32 GB NVLink

This fork adapts SGLang for four SM70 V100 GPUs. It includes a TileLang
FlashAttention backend, V100-aware paged KV-cache and decode tuning, custom
all-reduce support, and Marlin GPTQ/AWQ MoE kernels.

The commands below target x86-64 Ubuntu, four 32 GB V100s connected by NVLink,
and a sufficiently recent NVIDIA driver. They intentionally create a dedicated
Conda environment named `sglang-v100`.

## Copy, paste, and build

Paste this entire block into a terminal. It is safe to rerun: existing clones
and the Conda environment are reused.

```bash
set -euo pipefail

# OS compiler tools and the CUDA 12.8 toolkit used by the V100 extensions.
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates cmake curl git g++-12 ninja-build pkg-config wget

if [[ ! -x /usr/local/cuda-12.8/bin/nvcc ]]; then
  . /etc/os-release
  CUDA_REPO="ubuntu${VERSION_ID//./}"
  wget -q \
    "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb" \
    -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-8
fi

# Install Miniconda only when conda is not already available.
if ! command -v conda >/dev/null 2>&1; then
  curl -fsSL \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
eval "$(conda shell.bash hook)"

# Clone or update this fork.
if [[ -d "$HOME/sglang-V100/.git" ]]; then
  git -C "$HOME/sglang-V100" pull --ff-only
else
  git clone https://github.com/haohervchb/sglang-V100.git "$HOME/sglang-V100"
fi
cd "$HOME/sglang-V100"

# Create the runtime environment.
if ! conda env list | awk '{print $1}' | grep -qx sglang-v100; then
  conda create -y -n sglang-v100 python=3.12 pip
fi
conda activate sglang-v100
python -m pip install --upgrade pip setuptools wheel scikit-build-core

# Install the CUDA 12.8 PyTorch build first and keep incompatible CUDA 13/
# FlashAttention-4 wheels out of this SM70 environment.
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

# Pin the transitive gRPC stack used by the tested environment, then install
# this fork and its V100-compatible dependency set.
python -m pip install \
  grpcio==1.81.1 grpcio-health-checking==1.81.1 \
  grpcio-reflection==1.81.1 protobuf==6.33.6 tilelang==0.1.8
python -m pip install -e ./python

# Marlin needs CUTLASS headers. The tested version is used here.
if [[ ! -d "$HOME/cutlass/.git" ]]; then
  git clone --depth 1 --branch v4.2.1 \
    https://github.com/NVIDIA/cutlass.git "$HOME/cutlass"
fi

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/g++-12
export CUTLASS_DIR="$HOME/cutlass"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"

# The PyPI sglang-kernel wheel contains newer-GPU binaries. Replace it with
# this fork's lean SM70-only build; this skips all SM80/89/90/100 targets.
export CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=1"
python -m pip install --force-reinstall --no-deps --no-build-isolation ./sgl-kernel

bash scripts/setup_v100_marlin.sh

# Final smoke checks, including the compiled kernel that serving imports.
FLASHINFER_DISABLE_VERSION_CHECK=1 python - <<'PY'
import torch
import tilelang
import sglang
import sgl_kernel

assert torch.__version__.startswith("2.9.1")
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 0)
assert "/sm70/" in sgl_kernel.common_ops.__file__.replace("\\", "/")
print("SGLang V100 environment is ready:", torch.__version__)
print("SM70 kernel:", sgl_kernel.common_ops.__file__)
PY
```

The FlashInfer package and cubin versions used by the working SM70 stack differ
slightly, so the serving commands set `FLASHINFER_DISABLE_VERSION_CHECK=1`.
This is expected for this fork.

## Serve models

Run `conda activate sglang-v100` first. Each example listens on port 8082 and
uses all four visible GPUs. Stop one server before starting another.

### Qwen3.5-122B-A10B GPTQ Int4

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
python -m sglang.launch_server \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --dtype float16 \
  --quantization gptq_marlin \
  --attention-backend flash_attn_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.78 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 16384 \
  --enable-profile-cuda-graph
```

### Qwen3.5-122B-A10B AWQ

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
python -m sglang.launch_server \
  --model QuantTrio/Qwen3.5-122B-A10B-AWQ \
  --dtype float16 \
  --quantization awq_marlin \
  --attention-backend flash_attn_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.78 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 16384 \
  --enable-profile-cuda-graph
```

### Qwen3.6-35B-A3B AWQ

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
python -m sglang.launch_server \
  --model QuantTrio/Qwen3.6-35B-A3B-AWQ \
  --dtype float16 \
  --quantization awq_marlin \
  --attention-backend flash_attn_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.78 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 16384 \
  --enable-profile-cuda-graph
```

### Qwen3.6-27B dense FP16

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
python -m sglang.launch_server \
  --model Qwen/Qwen3.6-27B \
  --dtype float16 \
  --attention-backend flash_attn_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.80 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 16384 \
  --enable-profile-cuda-graph
```

`NCCL_P2P_LEVEL=NVL` selects direct NVLink peer-to-peer transport on the V100s.
NCCL NVLS (NVLink SHARP) is a distinct newer feature, so the V100 commands do not
use `--enable-nccl-nvls`.

The first launch downloads the model from Hugging Face. For gated models,
export `HF_TOKEN` before launching. Reduce `--context-length` or
`--mem-fraction-static` if other processes are using GPU memory.

## Scope

This is a hardware-specific fork, not a replacement for upstream SGLang. The
tested target is four V100 32 GB GPUs with NVLink, CUDA 12.8, FP16 activations,
the `flash_attn_v100` attention backend, and the Marlin GPTQ/AWQ path. The
upstream project remains the source for general SGLang documentation.

SGLang is licensed under the Apache License 2.0; see [LICENSE](LICENSE).
