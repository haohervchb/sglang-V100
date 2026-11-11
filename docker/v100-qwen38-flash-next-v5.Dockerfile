# syntax=docker/dockerfile:1.7
# Balanced Flash-Next profile, with the validated native SM70 stack from v4.
FROM geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v4@sha256:9aa44672321e21e38e6727a218cd67710e683c195fc2fc713f482e85e2d61735

ENV SGLANG_PYTHON=/opt/venv/bin/python \
    PORT=8082 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONUNBUFFERED=1 \
    MAX_JOBS=2 \
    NVCC_THREADS=2 \
    CXX=/usr/bin/g++-12 \
    SGLANG_SM70_MTP_LOCAL_TOP1=0 \
    SGLANG_SM70_HC_COMBINE_NORM=0 \
    SGLANG_SM70_AR_PUSH16=0 \
    SGLANG_MM_PREPROCESS_ON_CPU=1 \
    TVM_FFI_CACHE_DIR=/root/sglang-v100-jit/tvm-ffi \
    TORCH_EXTENSIONS_DIR=/root/sglang-v100-jit/torch_extensions \
    SGLANG_V100_NVFP4_MOE_BUILD_DIR=/root/sglang-v100-jit/nvfp4_moe \
    SGLANG_V100_DECODE_CUDA_BUILD_DIR=/root/sglang-v100-jit/longctx_decode

# Build the patched grouped-QSA dependency inside the image, against its Torch
# ABI. Host shared libraries are excluded from the Docker build context.
COPY scripts/setup_v100_grouped_qsa.py /opt/sglang/scripts/
COPY python/sglang/jit_kernel/third_party/flash_v100/grouped_masked_kv.patch \
     python/sglang/jit_kernel/third_party/flash_v100/vision_host_compat.patch \
     python/sglang/jit_kernel/third_party/flash_v100/vision_container_build.patch \
     /opt/sglang/python/sglang/jit_kernel/third_party/flash_v100/
RUN git clone --filter=blob:none --sparse --no-checkout \
      https://github.com/1CatAI/1Cat-vLLM.git /opt/deps/qwen38-grouped-qsa \
    && git -C /opt/deps/qwen38-grouped-qsa sparse-checkout set flash-attention-v100 \
    && git -C /opt/deps/qwen38-grouped-qsa checkout --detach d8f42b391af06410ab01bb71657ededf7dce7799 \
    && python /opt/sglang/scripts/setup_v100_grouped_qsa.py \
      --source /opt/deps/qwen38-grouped-qsa/flash-attention-v100 \
      --build-dir /opt/deps/qwen38-grouped-qsa-build

# The same bidirectional vision kernels as the validated host profile. The
# container-build patch removes only the build-time requirement for a live GPU.
RUN git clone https://github.com/ai-bond/flash-attention-v100.git /opt/deps/qwen38-vision \
    && git -C /opt/deps/qwen38-vision checkout --detach d89800edf608d85744f3ab6188be5fd0736acf39 \
    && git -C /opt/deps/qwen38-vision apply \
      /opt/sglang/python/sglang/jit_kernel/third_party/flash_v100/vision_host_compat.patch \
    && git -C /opt/deps/qwen38-vision apply \
      /opt/sglang/python/sglang/jit_kernel/third_party/flash_v100/vision_container_build.patch \
    && MAX_JOBS=4 python -m pip install --no-deps --no-build-isolation /opt/deps/qwen38-vision

# Preserve the native libraries inherited/built above, and remove stale Python
# files before installing the current source tree.
RUN find /opt/sglang/python/sglang -type f ! -name '*.so' -delete
COPY python/sglang /opt/sglang/python/sglang
COPY scripts/serve_qwen38_flash_next_nvfp4_v100.sh \
     scripts/serve_qwen38_flash_next_nvfp4_v100_mtp.sh \
     scripts/serve_qwen38_flash_next_nvfp4_v100_throughput.sh \
     scripts/serve_qwen38.sh scripts/qwen38.sh \
     scripts/smoke_v100.sh /opt/sglang/scripts/
COPY docker/v100-entrypoint.sh /usr/local/bin/v100-entrypoint

RUN python -m compileall -q /opt/sglang/python/sglang \
    && python -c 'import torch, flash_attn_v100; from sglang.srt.layers.attention.tilelang_fa_v100._grouped_qsa import load_grouped_qsa_ops; assert load_grouped_qsa_ops().sglang_masked_sparse_kv' \
    && for q38_script in /usr/local/bin/v100-entrypoint /opt/sglang/scripts/*qwen38*.sh; do bash -n "$q38_script" || exit; done \
    && chmod +x /usr/local/bin/v100-entrypoint /opt/sglang/scripts/serve_qwen38.sh /opt/sglang/scripts/qwen38.sh \
    && ln -s /opt/sglang/scripts/serve_qwen38.sh /usr/local/bin/serve-qwen38 \
    && ln -s /opt/sglang/scripts/qwen38.sh /usr/local/bin/qwen38.sh

# Revision metadata comes last so it does not invalidate native build layers.
ARG SGLANG_SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="SGLang Qwen3.8 Flash Next on V100" \
      org.opencontainers.image.version="v100-qwen38-flash-next-v5.2" \
      org.opencontainers.image.revision="${SGLANG_SOURCE_REVISION}"

HEALTHCHECK --interval=30s --timeout=5s --start-period=60m --retries=3 \
    CMD python -c 'import os,urllib.request; urllib.request.urlopen("http://127.0.0.1:"+os.environ.get("PORT","8082")+"/health",timeout=3)'

ENTRYPOINT ["/usr/local/bin/v100-entrypoint"]
CMD ["/usr/local/bin/serve-qwen38"]
