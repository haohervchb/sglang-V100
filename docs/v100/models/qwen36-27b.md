# Qwen3.6-27B on V100

[Back to the main README](../../../README.md) · [Host setup](../../../README.md#install-on-the-host) · [Docker setup](../../../README.md#docker)

Activate `sglang-v100` before running host commands. Examples use four
V100 32 GB GPUs; each command specifies its listening port.

## Qwen3.6-27B FP16 target only

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
sglang serve \
  --model Qwen/Qwen3.6-27B \
  --dtype float16 \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.80 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls \
  --enable-multimodal \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

## Qwen3.6-27B FP16 with DFlash

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --model Qwen/Qwen3.6-27B \
  --dtype float16 \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.70 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls \
  --enable-multimodal \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-27B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

## Qwen3.6-27B FP16 with MTP

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_SPEC_V2=1 \
sglang serve \
  --model Qwen/Qwen3.6-27B \
  --dtype float16 \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.70 \
  --context-length 8192 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```
