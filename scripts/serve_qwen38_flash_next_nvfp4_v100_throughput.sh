#!/usr/bin/env bash
# Validated TP4 V100 profile for concurrent agents and shared documents.
set -euo pipefail

q38_profile_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
q38_model=${1:-RadixArk/Qwen3.8-Flash-Next-NVFP4}
if (( $# )); then shift; fi

export SGLANG_SM70_BATCHED_MOE=${SGLANG_SM70_BATCHED_MOE:-1}

exec bash "$q38_profile_dir/serve_qwen38_flash_next_nvfp4_v100_mtp.sh" "$q38_model" \
    --max-running-requests 4 \
    --cuda-graph-bs 1 2 3 4 \
    --mem-fraction-static 0.85 \
    --max-total-tokens 277280 \
    --max-mamba-cache-size 24 \
    --prefill-max-requests 1 \
    "$@"
