#!/usr/bin/env bash
# Reproducible host installer for SGLang on NVIDIA V100 (SM70).
# Run this script as a child process; do not source it from an SSH login shell.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  printf '[install_v100] Do not source this file; run: bash %q\n' \
    "${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail

on_error() {
  local rc=$?
  printf '\n[install_v100] FAILED (exit %d) at line %d: %s\n' \
    "$rc" "${BASH_LINENO[0]}" "$BASH_COMMAND" >&2
  printf '[install_v100] Your login shell is still active; fix the error and rerun this script.\n' >&2
  exit "$rc"
}
trap on_error ERR

log() { printf '\n\033[1;34m[install_v100]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[install_v100] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_ROOT="${SGLANG_V100_DEPS_DIR:-$HOME/.cache/sglang-v100-sources}"
FLASHINFER_REV="c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833"
FLASH_ATTN_V100_REV="d89800edf608d85744f3ab6188be5fd0736acf39"

[[ -d "$REPO_ROOT/.git" ]] || die "$REPO_ROOT is not an SGLang-V100 checkout."

if [[ ${EUID} -eq 0 ]]; then
  SUDO=()
else
  command -v sudo >/dev/null || die "sudo is required to install system packages."
  SUDO=(sudo)
fi

log "Installing host compiler and CUDA 12.8 prerequisites"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  build-essential ca-certificates cmake curl git g++-12 ninja-build \
  pkg-config wget

if [[ ! -x /usr/local/cuda-12.8/bin/nvcc ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  CUDA_REPO="ubuntu${VERSION_ID//./}"
  wget -q \
    "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb" \
    -O /tmp/cuda-keyring.deb
  "${SUDO[@]}" dpkg -i /tmp/cuda-keyring.deb
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y cuda-toolkit-12-8
fi

if ! command -v conda >/dev/null 2>&1; then
  log "Installing Miniconda"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi

CONDA_EXE="$(command -v conda || true)"
[[ -n "$CONDA_EXE" ]] || CONDA_EXE="$HOME/miniconda3/bin/conda"
[[ -x "$CONDA_EXE" ]] || die "conda was not found after installation."
# shellcheck disable=SC1090
. "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx sglang-v100; then
  conda create -y -n sglang-v100 python=3.12 pip
fi
conda activate sglang-v100

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=7.0

# Use every CPU only when RAM can sustain that many compiler processes.  The
# previous unconditional nproc (88 jobs on the reference host, with no swap)
# could invoke the global OOM killer.  This is computed, never hard-coded to 8.
CPU_JOBS="$(nproc)"
MEM_AVAILABLE_KIB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
MEM_JOBS=$(( (MEM_AVAILABLE_KIB - 16 * 1024 * 1024) / (4 * 1024 * 1024) ))
(( MEM_JOBS < 1 )) && MEM_JOBS=1
SAFE_JOBS="$CPU_JOBS"
(( MEM_JOBS < SAFE_JOBS )) && SAFE_JOBS="$MEM_JOBS"
export MAX_JOBS="${MAX_JOBS:-$SAFE_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
log "Build parallelism: MAX_JOBS=$MAX_JOBS (CPU=$CPU_JOBS, RAM-safe=$SAFE_JOBS), NVCC_THREADS=$NVCC_THREADS"

python -m pip install --upgrade pip setuptools wheel scikit-build-core ninja psutil
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  grpcio==1.81.1 grpcio-health-checking==1.81.1 \
  grpcio-reflection==1.81.1 protobuf==6.33.6 tilelang==0.1.8
python -m pip install -e "$REPO_ROOT/python"

prepare_patched_repo() {
  local name=$1 url=$2 rev=$3 destination=$4
  shift 4
  local patch_file patch_fingerprint stamp_file

  if [[ ! -d "$destination/.git" ]]; then
    log "Cloning $name at $rev"
    mkdir -p "$(dirname "$destination")"
    git clone "$url" "$destination"
    git -C "$destination" checkout --detach "$rev"
  elif [[ "$(git -C "$destination" rev-parse HEAD)" != "$rev" ]]; then
    git -C "$destination" diff --quiet && \
      git -C "$destination" diff --cached --quiet || \
      die "$destination has local changes; move it aside or set SGLANG_V100_DEPS_DIR."
    git -C "$destination" fetch origin "$rev"
    git -C "$destination" checkout --detach "$rev"
  fi

  patch_fingerprint="$({ sha256sum "$@"; } | sha256sum | awk '{print $1}')"
  stamp_file="$destination/.sglang-v100-patches"
  if [[ -f "$stamp_file" ]] && [[ "$(<"$stamp_file")" == "$patch_fingerprint" ]]; then
    return
  fi
  git -C "$destination" diff --quiet && \
    git -C "$destination" diff --cached --quiet || \
    die "$destination has untracked installer patch state; remove it and rerun."

  for patch_file in "$@"; do
    git -C "$destination" apply --check "$patch_file" || \
      die "$name patch does not apply cleanly: $patch_file"
    git -C "$destination" apply "$patch_file"
  done
  printf '%s\n' "$patch_fingerprint" >"$stamp_file"
}

FLASHINFER_DIR="$DEPS_ROOT/flashinfer-sm70"
prepare_patched_repo \
  FlashInfer https://github.com/haohervchb/flashinfer.git \
  "$FLASHINFER_REV" "$FLASHINFER_DIR" \
  "$REPO_ROOT/patches/flashinfer-sm70.patch"
log "Installing the proven FlashInfer SM70 source"
python -m pip install --no-deps --no-build-isolation -e "$FLASHINFER_DIR"

FLASH_ATTN_V100_DIR="$DEPS_ROOT/flash-attention-v100"
prepare_patched_repo \
  flash-attention-v100 https://github.com/ai-bond/flash-attention-v100.git \
  "$FLASH_ATTN_V100_REV" "$FLASH_ATTN_V100_DIR" \
  "$REPO_ROOT/patches/flash-attention-v100-sglang.patch" \
  "$REPO_ROOT/patches/flash-attention-v100-torch291.patch"
log "Building the native SM70 attention fallback"
python -m pip install --force-reinstall --no-deps --no-build-isolation \
  "$FLASH_ATTN_V100_DIR"

if [[ ! -d "$HOME/cutlass/.git" ]]; then
  git clone --depth 1 --branch v4.2.1 \
    https://github.com/NVIDIA/cutlass.git "$HOME/cutlass"
fi
export CUTLASS_DIR="$HOME/cutlass"

log "Building lean SM70-only sglang-kernel"
# Remove the newer-GPU wheel first so pip cannot leave an orphaned common_ops
# filename beside the locally built ABI3 module.
python -m pip uninstall -y sglang-kernel || true
export CMAKE_ARGS="-DSGL_KERNEL_V100_ONLY=ON -DSGL_KERNEL_COMPILE_THREADS=$NVCC_THREADS"
python -m pip install --no-deps --no-build-isolation "$REPO_ROOT/sgl-kernel"

log "Restoring the CUDA 12 NCCL required by torch 2.9.1"
python -m pip uninstall -y nvidia-nccl-cu13 || true
python -m pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.27.5

log "Building V100 Marlin GPTQ/AWQ kernels"
export MARLIN_V100_REPO="${MARLIN_V100_REPO:-$DEPS_ROOT/marlin-v100}"
export MARLIN_V100_REF="${MARLIN_V100_REF:-6d72a49939701d26b15b617a4cd2423174adb2d1}"
bash "$REPO_ROOT/scripts/setup_v100_marlin.sh"

log "Running SM70 smoke checks and precompiling first-chat sampling"
FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 \
  SGLANG_V100_FLASHINFER_DIR="$FLASHINFER_DIR" python - <<'PY'
import glob
import os
from pathlib import Path

import torch
import flashinfer
import flash_attn_v100_cuda
import sgl_kernel
import sglang
import tilelang
from flashinfer.sampling import top_k_top_p_sampling_from_probs
from sglang.srt.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

expected_flashinfer = Path(os.environ["SGLANG_V100_FLASHINFER_DIR"]).resolve()
assert Path(flashinfer.__file__).resolve().is_relative_to(expected_flashinfer)
assert torch.__version__.startswith("2.9.1")
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 0)
assert "/sm70/" in sgl_kernel.common_ops.__file__.replace("\\", "/")
assert NCCLLibrary().ncclGetRawVersion() == 22705

common_ops = glob.glob(
    str(Path(sgl_kernel.__file__).parent / "sm70" / "common_ops*.so")
)
assert len(common_ops) == 1, f"stale common_ops variants remain: {common_ops}"

# FlashInfer builds this native module lazily.  Paying the compilation cost now
# prevents the first non-greedy chat request from stalling for ~90 seconds.
probs = torch.full((1, 128), 1.0 / 128, device="cuda", dtype=torch.float32)
top_k = torch.tensor([20], device="cuda", dtype=torch.int32)
top_p = torch.tensor([0.8], device="cuda", dtype=torch.float32)
top_k_top_p_sampling_from_probs(
    probs, top_k, top_p, filter_apply_order="joint"
)
torch.cuda.synchronize()

print("SGLang V100 environment is ready:", torch.__version__)
print("FlashInfer SM70:", flashinfer.__file__)
print("Native attention:", flash_attn_v100_cuda.__file__)
print("SM70 kernel:", sgl_kernel.common_ops.__file__)
print("NCCL:", NCCLLibrary().ncclGetVersion())
PY

log "Complete. Run: conda activate sglang-v100"
