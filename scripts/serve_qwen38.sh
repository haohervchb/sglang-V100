#!/usr/bin/env bash
# Foreground, manual-only Flash-Next server for a prepared host or the image.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'EOF'
Usage: bash serve_qwen38.sh [SGLang arguments]

Serve Flash-Next on port 8082 in this terminal; Ctrl-C stops it.
Defaults: 4x V100, MTP-3/4, eight requests, FP16 KV, 262K context.
Set PORT to change the port, or SGLANG_PYTHON to select the serving Python.
No service, boot startup or automatic restart is installed.
EOF
    exit 0
fi

q38_script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
q38_script_dir=$(dirname -- "$q38_script_path")
q38_profile="$q38_script_dir/serve_qwen38_flash_next_nvfp4_v100_throughput.sh"
if [[ ! -r $q38_profile ]]; then
    echo "This command needs the prepared Flash-Next checkout or Docker image." >&2
    exit 1
fi

if [[ -z ${SGLANG_PYTHON:-} ]]; then
    q38_candidates=()
    if [[ -n ${CONDA_PREFIX:-} ]]; then q38_candidates+=("$CONDA_PREFIX/bin/python"); fi
    q38_candidates+=(
        "$q38_script_dir/../.venv/bin/python"
        "$HOME/miniconda3/envs/sglang-v100/bin/python"
        "$HOME/miniforge3/envs/sglang-v100/bin/python"
        "$HOME/anaconda3/envs/sglang-v100/bin/python"
    )
    q38_path_python=$(command -v python || command -v python3 || true)
    if [[ -n $q38_path_python ]]; then q38_candidates+=("$q38_path_python"); fi
    for q38_python in "${q38_candidates[@]}"; do
        if [[ -x $q38_python ]] && "$q38_python" -c \
            'import importlib.util as u; raise SystemExit(not all(u.find_spec(m) for m in ("torch", "transformers", "tvm_ffi", "sgl_kernel", "flashinfer")))' \
            >/dev/null 2>&1; then
            export SGLANG_PYTHON="$q38_python"
            break
        fi
    done
    if [[ -z ${SGLANG_PYTHON:-} ]]; then
        echo "No serving Python found. Set SGLANG_PYTHON to your prepared environment's Python, or use Docker." >&2
        exit 1
    fi
fi

export PORT=${PORT:-8082}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_SM70_MTP_LOCAL_TOP1=0 SGLANG_SM70_HC_COMBINE_NORM=0 SGLANG_SM70_AR_PUSH16=0
exec bash "$q38_profile" RadixArk/Qwen3.8-Flash-Next-NVFP4 "$@"
