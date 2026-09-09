# Laguna-S-2.1-INT4 on V100

[Back to the main README](../../../README.md) · [Host setup](../../../README.md#install-on-the-host) · [Docker setup](../../../README.md#docker)

Activate `sglang-v100` before running host commands. Examples use four
V100 32 GB GPUs; each command specifies its listening port.

## Poolside Laguna-S-2.1 INT4 target only

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
sglang serve \
  --model poolside/Laguna-S-2.1-INT4 \
  --trust-remote-code \
  --dtype float16 \
  --kv-cache-dtype auto \
  --attention-backend tilelang_fa_v100 \
  --moe-runner-backend marlin \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.76 \
  --swa-full-tokens-ratio 0.08 \
  --context-length 262144 \
  --page-size 16 \
  --max-running-requests 2 \
  --chunked-prefill-size 4096 \
  --triton-attention-num-kv-splits 128 \
  --cuda-graph-max-bs 2 \
  --cuda-graph-bs 1 2 \
  --enable-nccl-nvls \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1
```

## Poolside Laguna-S-2.1 INT4 with DFlash

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --model poolside/Laguna-S-2.1-INT4 \
  --trust-remote-code \
  --dtype float16 \
  --kv-cache-dtype auto \
  --attention-backend tilelang_fa_v100 \
  --moe-runner-backend marlin \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.76 \
  --swa-full-tokens-ratio 0.08 \
  --context-length 262144 \
  --page-size 16 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --triton-attention-num-kv-splits 128 \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-INT4 \
  --speculative-dflash-block-size 8 \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1
```

For target-only serving, remove the three `--speculative-*` arguments.
`SGLANG_ENABLE_SPEC_V2` can also be omitted.
