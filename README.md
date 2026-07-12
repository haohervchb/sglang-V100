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
python -m pip install --upgrade pip setuptools wheel packaging

# Install the CUDA 12.8 PyTorch build first and keep incompatible CUDA 13/
# FlashAttention-4 wheels out of this SM70 environment.
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

python - <<'PY'
import subprocess
import sys
import tomllib
from pathlib import Path
from packaging.requirements import Requirement

deps = tomllib.loads(Path("python/pyproject.toml").read_text())["project"]["dependencies"]
skip = {
    "flash-attn-4",
    "flashinfer-cubin",
    "flashinfer-python",
    "sglang-kernel",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
}
deps = [dep for dep in deps if Requirement(dep).name.lower() not in skip]
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
PY

python -m pip install \
  flashinfer-python==0.6.12 flashinfer-cubin==0.6.11.post1 \
  sglang-kernel==0.4.3 tilelang==0.1.8 transformers==5.8.1
python -m pip install --no-deps -e ./python

# Marlin needs CUTLASS headers. The tested version is used here.
if [[ ! -d "$HOME/cutlass/.git" ]]; then
  git clone --depth 1 --branch v4.2.1 \
    https://github.com/NVIDIA/cutlass.git "$HOME/cutlass"
fi

export CUDA_HOME=/usr/local/cuda-12.8
export CUDAHOSTCXX=/usr/bin/g++-12
export CUTLASS_DIR="$HOME/cutlass"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=8
bash scripts/setup_v100_marlin.sh

# Final smoke checks.
FLASHINFER_DISABLE_VERSION_CHECK=1 python - <<'PY'
import torch
import tilelang
import sglang

assert torch.__version__.startswith("2.9.1")
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 0)
print("SGLang V100 environment is ready:", torch.__version__)
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
  --enable-profile-cuda-graph \
  --enable-nccl-nvls
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
  --enable-profile-cuda-graph \
  --enable-nccl-nvls
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
  --enable-profile-cuda-graph \
  --enable-nccl-nvls
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
  --enable-profile-cuda-graph \
  --enable-nccl-nvls
```

The first launch downloads the model from Hugging Face. For gated models,
export `HF_TOKEN` before launching. Reduce `--context-length` or
`--mem-fraction-static` if other processes are using GPU memory.

## Scope

This is a hardware-specific fork, not a replacement for upstream SGLang. The
tested target is four V100 32 GB GPUs with NVLink, CUDA 12.8, FP16 activations,
the `flash_attn_v100` attention backend, and the Marlin GPTQ/AWQ path. The
upstream project remains the source for general SGLang documentation.

SGLang is licensed under the Apache License 2.0; see [LICENSE](LICENSE).
