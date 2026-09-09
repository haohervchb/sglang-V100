# SGLang for 4x NVIDIA V100 32 GB NVLink

SGLang serving commands and measured performance for SM70 V100 GPUs.

[Performance](#current-v100-performance) · [Install](#install-on-the-host) ·
[Docker](#docker) · [Latest models](#latest-model-serving-guides) ·
[Older models](#older-model-serving-guides) · [Attribution](#references-and-attribution)

**Docker image:** [geesegeesegeese/sglang-v100](https://hub.docker.com/r/geesegeesegeese/sglang-v100/tags)

## Current V100 performance

Models are grouped newest to oldest by when V100 serving support was added
here, with configurations kept together. All current rows are retained. Unless
a row says otherwise, LLM results use TP4, one cold request, and 256 greedy
output tokens.
Prefill is the exact input length divided by client time to first token; decode
excludes that first-token time. The measurements came from separate tuning
runs, so treat this as a practical reference rather than a perfectly controlled
cross-model leaderboard.
H3 reports wall-clock video generation time in the Results column rather than
LLM token throughput. `—` means the metric does not apply or the supported
configuration has no comparable retained end-to-end benchmark.

| Model checkpoint | Measured configuration | 1K prefill | 1K decode | 25K prefill | 25K decode | Results |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | Target only, E5M2 KV, Docker v4; 2K output | 2,704 tok/s | 74.265 tok/s | 4,872 tok/s | 73.548 tok/s | Fresh host decode matched within 0.03%; [Docker/host validation](benchmark/qwen38_nvfp4_v100_docker_v4_20260908/README.md) |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | MTP-3/4, E5M2 KV, Docker v4; prose/code, 1K output | 3,269 tok/s | 117.55 tok/s | 4,693 tok/s | 119.84 tok/s | 1K–70K mean decode within 2.9% of fresh host; individual requests 114–126 tok/s; [Docker/host validation](benchmark/qwen38_nvfp4_v100_docker_v4_20260908/README.md) |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | Target only, E5M2 KV, optimized host source | 2,444 tok/s | 73.46 tok/s | 4,865 tok/s | 71.96 tok/s | **25K→8K: 71.83 decode tok/s; slowest 256-token window: 71.71.** [Source benchmark and limits](benchmark/qwen38_nvfp4_v100_70tps_20260907/README.md); source changes are included in Docker v4 |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | Target only, E5M2 KV, Docker v2 | 2,299 tok/s¶ | 60.2 tok/s¶ | — | — | **1K→25K: 60.09 output tok/s; 8K→1K at c4: 137.26 aggregate output tok/s.** [Fixed-image benchmark](benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md); [Docker command](#serve-qwen38-flash-next-nvfp4-from-docker) |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | Built-in MTP-3/4, E5M2 KV, Docker v2 | 2,055 tok/s¶ | 88.2 tok/s¶ | — | — | **1K→25K: 88.07 output tok/s; 8K→1K at c4: 149.25 aggregate output tok/s.** Acceptance length: 3.493 and 3.089. [Fixed-image benchmark](benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md); [Docker command](#serve-qwen38-flash-next-nvfp4-from-docker) |
| `Qwen/Qwen3.8-27B-FP8` | Target only, E5M2 KV | 2,992 tok/s | 60.9 tok/s | 3,714 tok/s | 56.3 tok/s | **4K prefill/decode: 4,224/63.2; 70K decode: 59.1; 200K decode: 49.6 tok/s** with the SM70 CUDA split-KV decode partial and fused QPN8 gate/up path; [audited FP8 sweep](benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/README.md) |
| `Qwen/Qwen3.8-27B-FP8` | DFlash2-8, E5M2 KV | 1,803 tok/s | 136.6 tok/s | 2,701 tok/s | 102.3 tok/s | 118.1 tok/s warm short decode; **cold 150K: 134; cold 200K: 112 tok/s**; 79.2 tok/s at 70K (warm); [docker 1K/25K runs](benchmark/qwen38_27b_fp8_dflash2_e5m2_v100_20260821/README.md)‡ |
| `Qwen/Qwen3.8-27B` | DFlash2-8, FP16 KV | 2,094 tok/s | 86.7 tok/s | 2,992 tok/s | 68.6 tok/s | [docker 1K/25K runs](benchmark/qwen38_27b_fp16_dflash2_v100_20260821/README.md) |
| `Qwen/Qwen3.8-27B` | DSpark-7, FP16 KV | 2,020 tok/s | 73.6 tok/s | 3,001 tok/s | 74.5 tok/s | [docker 1K/25K runs](benchmark/qwen38_27b_fp16_dspark_v100_20260821/README.md) |
| `MiniMaxAI/MiniMax-H3` | TP4 W4A16, 960×544, 15 s clip, 10 steps | — | — | — | — | ~500 s/video |
| `Qwen/Qwen3.6-27B-FP8` | DFlash-16, FP16 KV | 2,774 tok/s | 154.0 tok/s | 3,128 tok/s | 126.2 tok/s | [13-point TP2/TP4 sweep](benchmark/qwen36_27b_fp8_tp_scaling_20260802/README.md) |
| `poolside/Laguna-S-2.1-INT4` | Marlin, DFlash-8 | 3,334 tok/s† | 77.3 tok/s | 4,327 tok/s† | 67.0 tok/s | [Context sweep](benchmark/dflash_v100_20260716/README.md) and [Laguna tuning](https://github.com/haohervchb/sglang-V100/commit/491bb6095a) |
| `Qwen/Qwen3.6-35B-A3B` | FP16, DFlash-16 | 4,240 tok/s | 150.1 tok/s | 12,258 tok/s | 136.4 tok/s | [35B optimization results](https://github.com/haohervchb/sglang-V100/commit/7b8615f26e) |
| `Qwen/Qwen3.6-27B` | FP16, DFlash-16 | 3,261 tok/s | 101.2 tok/s | 3,631 tok/s | 86.6 tok/s | [Audited context sweep](benchmark/dflash_v100_20260716/README.md) |
| `QuantTrio/Qwen3.6-35B-A3B-AWQ` | AWQ target/DFlash | — | — | — | — | Supported; comparable run not retained |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | GPTQ-Marlin, DFlash-16 | 3,426 tok/s | 109.2 tok/s | 4,718 tok/s | 81.9 tok/s | [Audited context sweep](benchmark/dflash_v100_20260716/README.md) |
| `QuantTrio/Qwen3.5-122B-A10B-AWQ` | AWQ-Marlin target only | — | — | — | — | Supported; comparable run not retained |

Target-only and MTP modes are also supported where commands are provided
below. †Laguna prefill comes from the retained DFlash sweep; its later Marlin
selector changed low-token-count decode and left effective prefill unchanged
within normal cold-run variation. The Laguna decode columns are the later tuned
block-8 results. ‡The DFlash2 1K/25K figures are docker single-request
measurements. The 150K and 200K figures are clean cold-cache host runs (single
request, freshly restarted server, zero cached prompt tokens); the older
70K/200K bring-up rows were warm periodic-synthetic measurements. They validate
the long-context path but are not directly comparable with the cold 1K/25K
sweep columns. ¶The Flash Next rows use the 2026-08-30 fixed-image Docker
validation, whose long request had 1,000 input and 25,000 output tokens rather
than the standard 256-token output. Its 1K prefill and decode columns are
derived from TTFT and mean TPOT. The 25K-input columns are empty because that
validation did not rerun a 25K-prompt point. The c4 figures are aggregate output
throughput for four exact 8,192-input/1,024-output requests.

## Install on the host

```bash
if [[ -d "$HOME/sglang-V100/.git" ]]; then
  git -C "$HOME/sglang-V100" pull --ff-only
else
  git clone https://github.com/haohervchb/sglang-V100.git "$HOME/sglang-V100"
fi

bash "$HOME/sglang-V100/scripts/install_v100.sh"
conda activate sglang-v100
```

Validate an existing installation:

```bash
conda activate sglang-v100
bash "$HOME/sglang-V100/scripts/smoke_v100.sh"
```

## Docker

Open the [Docker Hub repository and tag list](https://hub.docker.com/r/geesegeesegeese/sglang-v100/tags),
or pull the current image directly:

```bash
docker pull geesegeesegeese/sglang-v100:latest
```

For a reproducible deployment, pin the tested Qwen3.8 Flash Next release:

```bash
docker pull geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v4
```

Build the current checkout:

```bash
cd "$HOME/sglang-V100"
DOCKER_BUILDKIT=1 docker build --network=host \
  -f docker/v100.Dockerfile \
  -t sglang-v100:latest .
```

Create the shared model and JIT caches used by the Docker examples:

```bash
mkdir -p "$HOME/.cache/huggingface"
docker volume create sglang-v100-jit
```

## Latest model serving guides

The three most recently implemented models on `main` are Qwen3.8 Flash Next,
Qwen3.8-27B and MiniMax-H3. Their current launch examples are below. Run one
server at a time on the four GPUs and activate `sglang-v100` for host commands.

### Qwen3.8 Flash Next NVFP4

The packaged launcher uses TP4, FP16 activations, E5M2 KV, 262,144-token
context, 8,192-token prefill chunks and one running request. It selects the
measured V100 optimizations and detects the checkpoint's tool format.

Target-only on the host:

```bash
conda activate sglang-v100
cd "$HOME/sglang-V100"
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$PWD/python" PORT=30000 \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh
```

Built-in MTP uses the same checkpoint with three speculative steps and four
draft tokens:

```bash
conda activate sglang-v100
cd "$HOME/sglang-V100"
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$PWD/python" PORT=30000 \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh \
  RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

#### Serve Qwen3.8 Flash Next NVFP4 from Docker

Pull the pinned v4 image from the Docker section above. With the RadixArk
checkpoint already in the shared Hugging Face cache, start MTP:

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

For target-only mode, choose another container name and omit the five
speculative arguments. The API is at `http://127.0.0.1:8082/v1`; the host
commands use port 30000. First use compiles kernels into the Docker JIT cache.
The v4 image includes FFmpeg for video decoding.

See [Docker/host validation](benchmark/qwen38_nvfp4_v100_docker_v4_20260908/README.md),
[image/video validation](benchmark/qwen38_nvfp4_v100_multimodal_20260909/README.md),
and the [full model guide](docs/v100/models/qwen38-flash-next-nvfp4.md) for
explicit flags, overlay builds, older deployment configurations and performance
history.

### Qwen3.8-27B

These commands use `Qwen/Qwen3.8-27B-FP8`, TP4 and E5M2 target KV. Keep
`--max-total-tokens` equal to the 262,144-token context for this one-request
configuration so that KV allocation leaves enough activation headroom.

#### Target-only

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --dtype float16 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.75 \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 8192 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 1 \
  --cuda-graph-bs 1 \
  --enable-nccl-nvls \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

#### DFlash2

The checkpoint uses block size 8 and a 2,048-token draft window. The target
cache remains E5M2; the smaller draft cache uses FP16.

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --dtype float16 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.75 \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 8192 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 1 \
  --cuda-graph-bs 1 \
  --enable-nccl-nvls \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 \
  --speculative-dflash-block-size 8 \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-window-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

For DSpark, use the same speculative command with
`--speculative-algorithm DSPARK`, draft model
`RadixArk/Qwen3.8-27B-DSpark`, and `--speculative-dspark-block-size 7`.
Remove the two DFlash-specific arguments (`--speculative-dflash-block-size`
and `--speculative-draft-window-size`).

The [full Qwen3.8-27B guide](docs/v100/models/qwen38-27b.md) includes the
complete DSpark and Docker commands, KV choices, performance comparisons and
native-kernel tuning notes. FP16-checkpoint results remain in the performance
table and linked reports.

### MiniMax-H3 video and audio

#### Serve H3 on the host

Use the W4A16 configuration for text-to-video-and-audio or first/last-frame
generation:

```bash
conda activate sglang-v100

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_LEVEL=NVL \
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --tp-size 4 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --performance-mode speed \
  --quantization v100_w4a16_awq \
  --attention-backend tilelang_fa_v100 \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload \
  --vae-cpu-offload \
  --enable-torch-compile false \
  --warmup false \
  --server-warmup false \
  --host 0.0.0.0 \
  --port 30010
```

#### Serve H3 from Docker

```bash
mkdir -p "$HOME/.cache/huggingface"

docker run --rm --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v sglang-v100-jit:/root/sglang-v100-jit \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  geesegeesegeese/sglang-v100:latest \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --tp-size 4 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --performance-mode speed \
  --quantization v100_w4a16_awq \
  --attention-backend tilelang_fa_v100 \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload \
  --vae-cpu-offload \
  --enable-torch-compile false \
  --warmup false \
  --server-warmup false \
  --host 0.0.0.0 \
  --port 30010
```

#### Generate a text-to-video-and-audio clip

This produces 960x544 output:

```bash
video_id=$(
  curl -sS -X POST http://127.0.0.1:30010/v1/videos \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "MiniMaxAI/MiniMax-H3",
      "prompt": "A tiger walks slowly through morning fog while birds and leaves are heard around it.",
      "task": "t2va",
      "conditions": [],
      "target": {
        "short_edge": 544,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0
      },
      "num_inference_steps": 50,
      "flow_shift": 12.0,
      "audio_flow_shift": 3.0,
      "seed": 1101
    }' | jq -r '.id'
)

while true; do
  status=$(curl -sS "http://127.0.0.1:30010/v1/videos/$video_id" | jq -r '.status')
  [ "$status" = completed ] && break
  [ "$status" = failed ] && exit 1
  sleep 2
done

curl -sS -L "http://127.0.0.1:30010/v1/videos/$video_id/content" \
  -o minimax-h3-t2va.mp4
```

For W8A16 with DiT offload, first/last-frame requests and reference-media
generation, see the [full MiniMax-H3 guide](docs/v100/models/minimax-h3.md).

## Older model serving guides

Each guide keeps the model's target-only, speculative and Docker examples
where available. The performance table above retains all measured models.

| Model checkpoint | Serving guide |
| --- | --- |
| `Qwen/Qwen3.6-27B-FP8` | [Target-only, DFlash and Docker](docs/v100/models/qwen36-27b-fp8.md) |
| `poolside/Laguna-S-2.1-INT4` | [Target-only and DFlash](docs/v100/models/laguna-s-2-1-int4.md) |
| `Qwen/Qwen3.6-35B-A3B` | [FP16 target-only and DFlash](docs/v100/models/qwen36-35b-a3b.md) |
| `Qwen/Qwen3.6-27B` | [FP16 target-only, DFlash and MTP](docs/v100/models/qwen36-27b.md) |
| `QuantTrio/Qwen3.6-35B-A3B-AWQ` | [AWQ target-only and DFlash](docs/v100/models/qwen36-35b-a3b-awq.md) |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | [GPTQ target-only, DFlash and MTP](docs/v100/models/qwen35-122b-a10b-gptq-int4.md) |
| `QuantTrio/Qwen3.5-122B-A10B-AWQ` | [AWQ target-only](docs/v100/models/qwen35-122b-a10b-awq.md) |

## OpenAI-compatible chat request

This example targets the Flash Next host launcher above. Use the model name
and port configured for your server; its Docker example uses port 8082.

```bash
curl -sS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Write a short hello-world program."}],
    "temperature": 0,
    "max_tokens": 256
  }' | jq
```

## References and attribution

<!-- Keep the full references and attribution in this main README. -->

The following upstream projects were used as code dependencies, algorithmic
references, performance references, or model assets for the V100 work in this
repository. The relationship column states which kind of use applies.

| Work in this repository | Upstream project | Relationship | License / revision |
| --- | --- | --- | --- |
| SGLang serving runtime and model integration | [sgl-project/sglang](https://github.com/sgl-project/sglang) | Framework this V100 fork is based on. | Apache-2.0 |
| TileLang attention, FP8-KV bridge, GDN, and fused normalization kernels | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | Kernel language, compiler, and runtime; host and Docker currently install `tilelang==0.1.8`. | MIT / `0.1.8` |
| SM70 long-context attention and FP8 optimization campaign | [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM/tree/6ada86ed64af6d1a7b3cb0f34df237fd86f06d48) | Performance and design reference for D=256 paged attention, dense-KV gathering, K-axis splitting, FP8 E5M2 KV conversion, and long-context decode. | Apache-2.0 / `6ada86e` |
| Chunked GDN / gated-delta-rule algorithm | [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA/tree/v0.1.2) | Algorithm and API reference for chunk-64 KKT solving, gating, recurrent-state propagation, and variable-length GDN prefill. | MIT / `v0.1.2` |
| GDN utility and correctness-reference operators | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | Source of the adapted FLA utilities already carried under `python/sglang/srt/layers/attention/fla`; also used as the numerical reference for the SM70 GDN path. | MIT |
| TurboMind GEMM and MoE kernel lineage | [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) | Original TurboMind project and kernel architecture. | Apache-2.0 |
| SM70 TurboMind FP8, AWQ, and FP16-MoE build source | [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM/tree/6ada86ed64af6d1a7b3cb0f34df237fd86f06d48/csrc/sm70_turbomind) | Pinned sparse source snapshot used to build the current SGLang TurboMind adapter; its embedded TurboMind sources derive from LMDeploy. | Apache-2.0 / `6ada86e` |
| SM70 Marlin GPTQ/AWQ dense and MoE kernels | [zhinianqin/marlin_v100](https://github.com/zhinianqin/marlin_v100/tree/6d72a49939701d26b15b617a4cd2423174adb2d1) | Native extension built by `scripts/setup_v100_marlin.sh`, with the compatibility and Qwen tuning patches in this repository. | Apache-2.0 / `6d72a49` |
| QPN8 SM70 W8A16 decode kernel | [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) | Kernel architecture adapted for Qwen3.8 block-wise scales; this repository adds its own word-parallel FP8 decoder and paired gate/up SiLU epilogue. | MIT |
| FlashInfer sampling and remaining SM70-compatible runtime operations | [haohervchb/flashinfer](https://github.com/haohervchb/flashinfer/tree/c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833), derived from [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Pinned source dependency with this repository's reduced SM70 compatibility patch. | Apache-2.0 / `c3c40a7` |
| Tensor-core templates used by TurboMind and Marlin builds | [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) | Header/template build dependency. TurboMind uses `da5e086`; Marlin uses CUTLASS `v4.2.1`. | BSD-3-Clause |
| Qwen3.8 DFlash2 speculative decoding | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | Draft checkpoint, published block configuration, and model contract used by the DFlash2 integration and benchmarks. | Apache-2.0 / model revision `ac04198` |
| Qwen3.8 DSpark speculative decoding | [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) | Draft checkpoint and model configuration used by the DSpark serving path and benchmarks. | See model card |
