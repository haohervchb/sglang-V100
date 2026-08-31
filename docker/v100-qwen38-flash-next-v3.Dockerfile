# Qwen3.8 Flash Next V100 v3 release image.
# v3 keeps the validated v2 native SM70 stack and adds the PLE host-offload
# initialization fix from main, avoiding a needless ~11.9 GiB transient CUDA
# allocation per TP4 rank before the table is moved to pinned host memory.
FROM geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v2

COPY python/sglang/srt/models/qwen4_exp_v100.py \
     /opt/sglang/python/sglang/srt/models/qwen4_exp_v100.py

RUN python -m py_compile /opt/sglang/python/sglang/srt/models/qwen4_exp_v100.py
