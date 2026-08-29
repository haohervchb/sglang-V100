#!/usr/bin/env bash
set -euo pipefail

q38_model_path=${1:-RadixArk/Qwen3.8-Flash-Next-NVFP4}
q38_port=${PORT:-30000}
q38_kv_cache_dtype=${KV_CACHE_DTYPE:-fp8_e5m2}
q38_extra_args=("${@:2}")

export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export SGLANG_CUSTOM_ALLREDUCE_ALGO=${SGLANG_CUSTOM_ALLREDUCE_ALGO:-1stage}
export SGLANG_MAMBA_CONV_DTYPE=${SGLANG_MAMBA_CONV_DTYPE:-float16}
export SGLANG_MAMBA_SSM_DTYPE=${SGLANG_MAMBA_SSM_DTYPE:-float16}
export SGLANG_SM70_FORCE_FP16=${SGLANG_SM70_FORCE_FP16:-1}
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-0}
# Dense flash-attn-v100 is faster than QSA's sparse gather at these bounded
# prefill lengths. Set this to 0 to retain sparse QSA semantics everywhere.
export SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=${SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS:-8192}

exec python -m sglang.launch_server \
  --trust-remote-code \
  --model-path "$q38_model_path" \
  --served-model-name qwen \
  --dtype float16 \
  --quantization modelopt_fp4 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --kv-cache-dtype "$q38_kv_cache_dtype" \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port "$q38_port" \
  --mem-fraction-static 0.80 \
  --context-length 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 8192 \
  --cuda-graph-bs 1 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-full-memory-ratio 0.2 \
  "${q38_extra_args[@]}"
