# Qwen3.8 Flash Next NVFP4 on V100

[Back to the main README](../../../README.md) · [Host setup](../../../README.md#install-on-the-host) · [Docker setup](../../../README.md#docker)

Activate `sglang-v100` before running host commands. Examples use four
V100 32 GB GPUs; each command specifies its listening port.

Use the [current quick start](../../../README.md#qwen38-flash-next-nvfp4)
for the validated v4 deployment. The explicit host commands and older v3
four-request configuration below are retained for reproduction.

## Qwen3.8 Flash Next NVFP4 target-only

This is the non-speculative command used for the current V100 result. It uses
E5M2 KV cache and is sized for one request at the model's full 262,144-token
context. Run it from a clone at `$HOME/sglang-V100`, or change the `cd` path.
The checkpoint's Qwen3-Coder-style XML tool format is detected automatically;
the startup log should report `tool_call_parser=qwen3_coder`.

```bash
cd "$HOME/sglang-V100"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_SM70_FORCE_FP16=1 \
SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192 \
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
PYTHONPATH="$PWD/python" \
conda run --no-capture-output -n sglang-v100 \
python -m sglang.launch_server \
  --trust-remote-code \
  --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen \
  --dtype float16 \
  --quantization modelopt_fp4 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --kv-cache-dtype fp8_e5m2 \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.80 \
  --context-length 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 8192 \
  --cuda-graph-bs 1 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-full-memory-ratio 0.2
```

## Qwen3.8 Flash Next NVFP4 with MTP

This loads the checkpoint's built-in MTP module from the same RadixArk repo as
the target. The setting is three speculative steps with four draft tokens. It
keeps the same E5M2 KV cache and full-context, single-request memory
configuration as target-only.

The September 8 host optimization work enables QSA draft-extend graph capture,
two/four-token projections and HC mixing, narrower recurrent-attention tiles,
and CUDA attention/expert kernel improvements on V100. Two ordinary prose/code
requests at each of 1K, 8K, 25K and 70K context measure
**108.4–119.1 decode tok/s** over 1,024 output
tokens; this completed optimization pass does not achieve consistent 120 tok/s. See the
[MTP measurements and validation](../../../benchmark/qwen38_nvfp4_v100_mtp_20260908/README.md)
for per-context results, acceptance counts and the asynchronous result-copy fix.

```bash
cd "$HOME/sglang-V100"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_SM70_FORCE_FP16=1 \
SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192 \
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
PYTHONPATH="$PWD/python" \
conda run --no-capture-output -n sglang-v100 \
python -m sglang.launch_server \
  --trust-remote-code \
  --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen \
  --dtype float16 \
  --quantization modelopt_fp4 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --kv-cache-dtype fp8_e5m2 \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.80 \
  --context-length 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 8192 \
  --cuda-graph-bs 1 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-full-memory-ratio 0.2 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

## Build and serve the optimized Qwen3.8 Docker v4 image

The v4 overlay packages the September 8 target-only and MTP optimizations on
the pinned v3 native SM70 stack. It includes FFmpeg shared libraries for video
decoding. See the [image/video validation and baseline comparison](../../../benchmark/qwen38_nvfp4_v100_multimodal_20260909/README.md).
Pull the validated image:

```bash
docker pull geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v4
```

Or build it from this checkout:

```bash
docker build --network=host \
  --build-arg SGLANG_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -f docker/v100-qwen38-flash-next-v4.Dockerfile \
  -t geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v4 .
```

With the RadixArk model already present in the shared Hugging Face cache,
start MTP with the same single-request configuration as the host benchmark:

```bash
docker run -d --name qwen38-flash-next-mtp-v4 \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v sglang-v100-jit-v4:/root/sglang-v100-jit \
  -e HF_HUB_OFFLINE=1 -e PORT=8082 \
  -e TVM_FFI_CACHE_DIR=/root/sglang-v100-jit/tvm-ffi \
  -e TORCH_EXTENSIONS_DIR=/root/sglang-v100-jit/torch_extensions \
  -e SGLANG_V100_NVFP4_MOE_BUILD_DIR=/root/sglang-v100-jit/nvfp4_moe \
  -e SGLANG_V100_DECODE_CUDA_BUILD_DIR=/root/sglang-v100-jit/longctx_decode \
  geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v4 \
  bash /opt/sglang/scripts/serve_qwen38_flash_next_nvfp4_v100.sh \
  RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

This uses TP4, E5M2 KV, 262,144-token context capacity, 8,192-token prefill
chunks, memory fraction 0.80 and one running request. For target-only mode,
choose another container name and omit the five speculative arguments. The
API is available at `http://127.0.0.1:8082/v1`. Docker compiles the updated
kernels into its own JIT cache on first use.

See the [Docker/host comparison and reproduction commands](../../../benchmark/qwen38_nvfp4_v100_docker_v4_20260908/README.md).
The measured v4 target-only decode matches the host within 0.03%, and
prose/code MTP per-context mean decode matches within 2.9% across 1K–70K.
The 1K natural prefill mean is 10.7% faster in Docker; random-token output
rates vary with acceptance, as recorded in the report.
The published `latest` tag points to the same validated v4 image. The v3
commands below retain the earlier four-request deployment configuration.

## Serve Qwen3.8 Flash Next NVFP4 from Docker

Target-only on four V100 32 GB GPUs, with up to four live requests and the
full 262,144-token per-request context limit:

```bash
docker run --rm --name qwen38-flash-next \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v sglang-v100-jit:/root/sglang-v100-jit \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_P2P_LEVEL=NVL \
  -e SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
  -e SGLANG_MAMBA_CONV_DTYPE=float16 \
  -e SGLANG_MAMBA_SSM_DTYPE=float16 \
  -e SGLANG_SM70_FORCE_FP16=1 \
  -e SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192 \
  -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v3 \
  --trust-remote-code \
  --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen \
  --dtype float16 \
  --quantization modelopt_fp4 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --kv-cache-dtype fp8_e5m2 \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8082 \
  --mem-fraction-static 0.85 \
  --context-length 262144 \
  --max-running-requests 4 \
  --max-mamba-cache-size 20 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-full-memory-ratio 0.2
```

Built-in MTP-3/4 uses the same RadixArk checkpoint as both target and draft and
keeps the same four-request, full-context sizing. MTP deliberately uses
`--mem-fraction-static 0.80` rather than the target-only `0.85`; the extra
headroom is required for the first real prompt's transient speculative/prefill
allocations on 32 GB V100s:

```bash
docker run --rm --name qwen38-flash-next-mtp \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v sglang-v100-jit:/root/sglang-v100-jit \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_P2P_LEVEL=NVL \
  -e SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
  -e SGLANG_MAMBA_CONV_DTYPE=float16 \
  -e SGLANG_MAMBA_SSM_DTYPE=float16 \
  -e SGLANG_SM70_FORCE_FP16=1 \
  -e SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192 \
  -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v3 \
  --trust-remote-code \
  --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen \
  --dtype float16 \
  --quantization modelopt_fp4 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --kv-cache-dtype fp8_e5m2 \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8082 \
  --mem-fraction-static 0.80 \
  --context-length 262144 \
  --max-running-requests 4 \
  --max-mamba-cache-size 20 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-full-memory-ratio 0.2 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

Both commands expose the OpenAI-compatible API at `http://127.0.0.1:8082/v1`.
Startup should report `tool_call_parser=qwen3_coder`. Replace the pinned image
tag with `latest` only if tracking the newest published build is desired.
`--context-length 262144` preserves the maximum context of an individual
request; four simultaneous maximum-length requests cannot fit in the aggregate
KV cache. With the `extra_buffer` scheduler, each live request consumes five
Mamba slots, so four requests require `--max-mamba-cache-size 20`. CUDA graphs
are captured only for the supported live batch sizes 1, 2, and 4.

## Performance history

The [September 7–8 host source optimization](../../../benchmark/qwen38_nvfp4_v100_70tps_20260907/README.md)
reaches **4,876 prefill tok/s** and **71.87–72.00 decode tok/s**
with 25K inputs and three 2K-output runs. An 8K-output run averages
**71.83 decode tok/s** (slowest interval window: **71.71**).
These target-only source results are separate from the Docker rows in the
[main performance table](../../../README.md#current-v100-performance).
The [first-pass main comparison](../../../benchmark/qwen38_nvfp4_v100_20260907/README.md)
records the earlier 1K/8K/25K prefill and decode results. The later target-only
and MTP changes are packaged in the published v4 image, with
[September 8–9 Docker/host validation](../../../benchmark/qwen38_nvfp4_v100_docker_v4_20260908/README.md).
The registry's published v3 image retains the earlier source.

## Historical Qwen3.8 Flash Next Docker v2 concurrency benchmark

The fixed v2 image completed every requested load point with no request errors.
These rows are retained as benchmark provenance; use the v4 serving command
above for the optimized deployment. Output throughput is aggregate across each exact
8,192-input/1,024-output workload:

| Concurrency | Target only | MTP-3/4 | MTP delta |
| ---: | ---: | ---: | ---: |
| 1 | 55.12 tok/s | 74.82 tok/s | +35.7% |
| 4 | 137.26 tok/s | 149.25 tok/s | +8.7% |
| 8 | 172.98 tok/s | 194.13 tok/s | +12.2% |
| 16 | 239.36 tok/s | 183.87 tok/s | -23.2% |

For the cold 1,000-input/25,000-output request at concurrency 1, target-only
reached 60.09 output tok/s and MTP reached 88.07 output tok/s (+46.6%). See the
[full fixed-image report](../../../benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md)
for TTFT, TPOT, acceptance length, and memory sizing.
