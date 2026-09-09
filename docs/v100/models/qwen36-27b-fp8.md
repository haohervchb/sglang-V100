# Qwen3.6-27B-FP8 on V100

[Back to the main README](../../../README.md) · [Host setup](../../../README.md#install-on-the-host) · [Docker setup](../../../README.md#docker)

Activate `sglang-v100` before running host commands. Examples use four
V100 32 GB GPUs; each command specifies its listening port.

## Qwen3.6-27B-FP8 target only

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
sglang serve \
  --model Qwen/Qwen3.6-27B-FP8 \
  --dtype float16 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.80 \
  --context-length 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 4096 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 1 \
  --cuda-graph-bs 1 \
  --enable-nccl-nvls \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

## Qwen3.6-27B-FP8 with DFlash

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --model Qwen/Qwen3.6-27B-FP8 \
  --dtype float16 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.75 \
  --context-length 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 4096 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 1 \
  --cuda-graph-bs 1 \
  --enable-nccl-nvls \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-27B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

Use `--kv-cache-dtype auto` for FP16 KV cache.

## Docker: Qwen3.6-27B-FP8 with DFlash

```bash
mkdir -p "$HOME/.cache/huggingface"

docker run --rm --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v sglang-v100-jit:/root/sglang-v100-jit \
  -e SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
  -e SGLANG_ENABLE_SPEC_V2=1 \
  -e SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
  geesegeesegeese/sglang-v100:latest \
  --model Qwen/Qwen3.6-27B-FP8 \
  --dtype float16 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.75 \
  --context-length 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 4096 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 1 \
  --cuda-graph-bs 1 \
  --enable-nccl-nvls \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-27B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```
