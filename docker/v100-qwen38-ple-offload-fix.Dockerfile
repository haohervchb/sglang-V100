# Lightweight test overlay for the published Qwen3.8 Flash Next V100 v2 image.
# This does not rebuild FlashInfer, Marlin, sglang-kernel, or TurboMind.
FROM geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v2

COPY patches/qwen38-ple-offload-init.patch /tmp/qwen38-ple-offload-init.patch
RUN cd /opt/sglang \
    && patch --dry-run -p1 < /tmp/qwen38-ple-offload-init.patch \
    && patch -p1 < /tmp/qwen38-ple-offload-init.patch \
    && rm /tmp/qwen38-ple-offload-init.patch
