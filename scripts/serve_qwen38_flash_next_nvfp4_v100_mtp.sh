#!/usr/bin/env bash
# Validated TP4 V100 single-request profile: FP16 KV, native MTP-3/4, 262K.
set -euo pipefail

q38_profile_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
q38_repo_dir=$(cd -- "$q38_profile_dir/.." && pwd)
q38_cache_dir=${XDG_CACHE_HOME:-$HOME/.cache}/sglang-v100/qwen38
q38_model=${1:-RadixArk/Qwen3.8-Flash-Next-NVFP4}
if (( $# )); then shift; fi
q38_revision=7b719225242aacd3dbd3f9407468c2ee9a9d2594

export PORT=${PORT:-8082}
export KV_CACHE_DTYPE=auto
export PYTHONPATH="$q38_repo_dir/python"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=${MAX_JOBS:-2}
# Pixel/JPEG preprocessing otherwise competes with TP0 outside the KV budget.
export SGLANG_MM_PREPROCESS_ON_CPU=${SGLANG_MM_PREPROCESS_ON_CPU:-1}
export SGLANG_SM70_TURBOMIND_OPS_PATH=${SGLANG_SM70_TURBOMIND_OPS_PATH:-$q38_repo_dir/python/sglang/jit_kernel/_sm70_turbomind_v100.so}
export SGLANG_V100_DECODE_CUDA_BUILD_DIR=${SGLANG_V100_DECODE_CUDA_BUILD_DIR:-$q38_cache_dir/attention}
export SGLANG_V100_NVFP4_MOE_BUILD_DIR=${SGLANG_V100_NVFP4_MOE_BUILD_DIR:-$q38_cache_dir/moe}
export SGLANG_SM70_QSA_FUSED_GRAPH_METADATA=0
export SGLANG_SM70_PROMPT_LOOKUP=0
export SGLANG_SM70_MTP_WIDE_MOE=0
export SGLANG_SM70_MTP_SHARED_GATE=${SGLANG_SM70_MTP_SHARED_GATE:-1}
export SGLANG_SM70_MOE_PREFILL_2K=${SGLANG_SM70_MOE_PREFILL_2K:-1}
export SGLANG_SM70_QSA_GROUPED_PREFILL=${SGLANG_SM70_QSA_GROUPED_PREFILL:-1}
# 4K prefill chunks; end each prompt's final chunk on a 2K boundary so recurrent
# state is still checkpointed at the same prefix-cache points as 2K chunks.
export SGLANG_CHUNKED_PREFILL_CACHE_ALIGN=${SGLANG_CHUNKED_PREFILL_CACHE_ALIGN:-2048}
unset SGLANG_SPEC_PROBE_DIR SGLANG_SPEC_PROBE_ON_PROFILE SGLANG_SPEC_PROBE_ROUNDS

if [[ ! -r "$SGLANG_SM70_TURBOMIND_OPS_PATH" ]]; then
    echo "Missing V100 FP16 MoE library: $SGLANG_SM70_TURBOMIND_OPS_PATH" >&2
    exit 1
fi
mkdir -p "$SGLANG_V100_DECODE_CUDA_BUILD_DIR" "$SGLANG_V100_NVFP4_MOE_BUILD_DIR"

exec bash "$q38_profile_dir/serve_qwen38_flash_next_nvfp4_v100.sh" "$q38_model" \
    --revision "$q38_revision" \
    --mem-fraction-static 0.83 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 4096 \
    --enable-multimodal \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path "$q38_model" \
    --speculative-draft-model-revision "$q38_revision" \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    "$@"
