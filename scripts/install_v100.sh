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
ONECAT_VLLM_REV="3ec0c68c6596d6ab31fbdee9fa676254a52c2b7d"
ONECAT_CUTLASS_REV="da5e086dab31d63815acafdac9a9c5893b1c69e2"

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
  local backup_destination old_fingerprint patch_file patch_fingerprint stamp_file

  if (( $# > 0 )); then
    patch_fingerprint="$({ sha256sum "$@"; } | sha256sum | awk '{print $1}')"
  else
    patch_fingerprint="$(printf '%s' "$rev" | sha256sum | awk '{print $1}')"
  fi
  stamp_file="$destination/.sglang-v100-patches"

  # An installer-managed checkout is expected to be dirty because patches are
  # applied without creating commits. If those patches change, retain that
  # checkout verbatim and build a clean replacement rather than either
  # rejecting a normal upgrade or discarding possible local edits.
  if [[ -d "$destination/.git" ]] && [[ -f "$stamp_file" ]] && \
      [[ "$(<"$stamp_file")" != "$patch_fingerprint" ]] && \
      ! git -C "$destination" diff --quiet; then
    old_fingerprint="$(<"$stamp_file")"
    backup_destination="${destination}.sglang-v100-backup-${old_fingerprint:0:12}"
    [[ ! -e "$backup_destination" ]] || \
      die "managed dependency backup already exists: $backup_destination"
    log "Preserving previous patched $name checkout at $backup_destination"
    mv "$destination" "$backup_destination"
    stamp_file="$destination/.sglang-v100-patches"
  fi

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
python -m pip uninstall -y flashinfer-python flashinfer-cubin || true
python -m pip install --no-deps --no-build-isolation -e "$FLASHINFER_DIR"

ONECAT_VLLM_DIR="$DEPS_ROOT/1cat-vllm"
prepare_patched_repo \
  1Cat-vLLM https://github.com/1CatAI/1Cat-vLLM.git \
  "$ONECAT_VLLM_REV" "$ONECAT_VLLM_DIR" \
  "$REPO_ROOT/patches/1cat-vllm-sm70-sglang.patch"
FLASH_ATTN_V100_DIR="$ONECAT_VLLM_DIR/flash-attention-v100"
log "Building enhanced SM70 attention with direct E4M3 XQA"
python -m pip install --force-reinstall --no-deps --no-build-isolation \
  "$FLASH_ATTN_V100_DIR"

ONECAT_CUTLASS_DIR="$DEPS_ROOT/cutlass-1cat"
prepare_patched_repo \
  CUTLASS https://github.com/NVIDIA/cutlass.git \
  "$ONECAT_CUTLASS_REV" "$ONECAT_CUTLASS_DIR"

log "Building the TurboMind SM70 block-FP8 GEMM backend"
SGLANG_1CAT_VLLM_ROOT="$ONECAT_VLLM_DIR" \
SGLANG_1CAT_CUTLASS_ROOT="$ONECAT_CUTLASS_DIR" \
  python "$REPO_ROOT/scripts/build_sm70_turbomind.py"

if [[ ! -d "$HOME/cutlass/.git" ]]; then
  git clone --depth 1 --branch v4.2.1 \
    https://github.com/NVIDIA/cutlass.git "$HOME/cutlass"
fi
export CUTLASS_DIR="$HOME/cutlass"

log "Building lean SM70-only sglang-kernel"
# Remove the newer-GPU wheel first so pip cannot leave an orphaned common_ops
# filename beside the locally built ABI3 module.
python -m pip uninstall -y sglang-kernel || true
# A prior in-place build can leave an ignored CPython .so inside the source
# package. pip then silently bundles it beside the fresh ABI3 SM70 extension.
find "$REPO_ROOT/sgl-kernel/python/sgl_kernel" \
  -type f -name 'common_ops*.so' -delete
python - <<'PY'
import site
from pathlib import Path

for root in site.getsitepackages():
    for artifact in (Path(root) / "sgl_kernel").glob("*/common_ops*.so"):
        artifact.unlink()
PY
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
if ! SGLANG_V100_FLASHINFER_DIR="$FLASHINFER_DIR" \
  bash "$REPO_ROOT/scripts/smoke_v100.sh"; then
  printf '\n[install_v100] All expensive builds completed successfully.\n' >&2
  printf '[install_v100] Rerun only validation (no rebuild) with:\n' >&2
  printf '  bash %q\n' "$REPO_ROOT/scripts/smoke_v100.sh" >&2
  exit 1
fi

log "Complete. Run: conda activate sglang-v100"
