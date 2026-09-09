# Qwen3.8 Flash Next V100 v4: validated target-only and MTP optimizations.
# Keep v3's native SM70 dependencies; compile the updated JIT kernels in the
# container. The build context excludes host .so files via .dockerignore.
FROM geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v3@sha256:da2714e969a492464610a1bdf58fcbbc0d2cf587b0633117ca505d13f3e7b617

ARG SGLANG_SOURCE_REVISION=3b9bf5b0b0
LABEL org.opencontainers.image.title="SGLang Qwen3.8 Flash Next on V100" \
      org.opencontainers.image.version="v100-qwen38-flash-next-v4" \
      org.opencontainers.image.revision="${SGLANG_SOURCE_REVISION}"

ENV SGLANG_SM70_DENSE_GEMV=1 \
    SGLANG_SM70_QWEN_FUSIONS=1 \
    CXX=/usr/bin/g++-12 \
    MAX_JOBS=2

# TorchCodec's wheel requires the FFmpeg shared libraries for video input.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python -c 'from torchcodec.decoders import VideoDecoder'

COPY python/sglang /opt/sglang/python/sglang
COPY scripts/serve_qwen38_flash_next_nvfp4_v100.sh \
     scripts/smoke_v100.sh /opt/sglang/scripts/

RUN python -m compileall -q /opt/sglang/python/sglang \
    && test -f /opt/sglang/python/sglang/jit_kernel/_sm70_marlin_v100_moe.abi3.so \
    && test -f /opt/sglang/python/sglang/jit_kernel/_sm70_marlin_v100_dense.abi3.so \
    && bash -n /opt/sglang/scripts/serve_qwen38_flash_next_nvfp4_v100.sh
