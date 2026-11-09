#!/usr/bin/env bash
# Validated TP4 V100 profile for concurrent agents and shared documents.
set -euo pipefail

q38_profile_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
q38_model=${1:-RadixArk/Qwen3.8-Flash-Next-NVFP4}
if (( $# )); then shift; fi

export SGLANG_SM70_BATCHED_MOE=${SGLANG_SM70_BATCHED_MOE:-1}
export SGLANG_SM70_BATCHED_MOE32=${SGLANG_SM70_BATCHED_MOE32:-1}
export SGLANG_SM70_BATCHED_FUSIONS=${SGLANG_SM70_BATCHED_FUSIONS:-1}

exec bash "$q38_profile_dir/serve_qwen38_flash_next_nvfp4_v100_mtp.sh" "$q38_model" \
    --max-running-requests 8 \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 \
    --mem-fraction-static 0.86 \
    --max-total-tokens 277280 \
    --max-mamba-cache-size 48 \
    --prefill-max-requests 1 \
    --mm-attention-backend fa2_v100 \
    "$@"
