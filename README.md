# SGLang for 4x NVIDIA V100 32 GB NVLink

SGLang serving commands for four SM70 V100 GPUs.

**Docker image:** [Download `geesegeesegeese/sglang-v100` from Docker Hub](https://hub.docker.com/r/geesegeesegeese/sglang-v100/tags)

```bash
docker pull geesegeesegeese/sglang-v100:latest
```

## Current V100 performance

All currently documented model checkpoints are listed below. Unless a row says
otherwise, LLM results use TP4, one cold request, and 256 greedy output tokens.
Prefill is the exact input length divided by client time to first token; decode
excludes that first-token time. The measurements came from separate tuning
runs, so treat this as a practical reference rather than a perfectly controlled
cross-model leaderboard.
H3 reports wall-clock video generation time in the Results column rather than
LLM token throughput. `—` means the metric does not apply or the supported
configuration has no comparable retained end-to-end benchmark.

| Model checkpoint | Measured configuration | 1K prefill | 1K decode | 25K prefill | 25K decode | Results |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `MiniMaxAI/MiniMax-H3` | TP4 W4A16, 960×544, 15 s clip, 10 steps | — | — | — | — | ~500 s/video |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | Target only, E5M2 KV, Docker v2 | 2,299 tok/s¶ | 60.2 tok/s¶ | — | — | **1K→25K: 60.09 output tok/s; 8K→1K at c4: 137.26 aggregate output tok/s.** [Fixed-image benchmark](benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md); [Docker command](#serve-qwen38-flash-next-nvfp4-from-docker) |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | Built-in MTP-3/4, E5M2 KV, Docker v2 | 2,055 tok/s¶ | 88.2 tok/s¶ | — | — | **1K→25K: 88.07 output tok/s; 8K→1K at c4: 149.25 aggregate output tok/s.** Acceptance length: 3.493 and 3.089. [Fixed-image benchmark](benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md); [Docker command](#serve-qwen38-flash-next-nvfp4-from-docker) |
| `Qwen/Qwen3.8-27B-FP8` | Target only, E5M2 KV | 2,992 tok/s | 60.9 tok/s | 3,714 tok/s | 56.3 tok/s | **4K prefill/decode: 4,224/63.2; 70K decode: 59.1; 200K decode: 49.6 tok/s** with the SM70 CUDA split-KV decode partial and fused QPN8 gate/up path; [audited FP8 sweep](benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/README.md) |
| `Qwen/Qwen3.8-27B-FP8` | DFlash2-8, E5M2 KV | 1,803 tok/s | 136.6 tok/s | 2,701 tok/s | 102.3 tok/s | 118.1 tok/s warm short decode; **cold 150K: 134; cold 200K: 112 tok/s**; 79.2 tok/s at 70K (warm); [docker 1K/25K runs](benchmark/qwen38_27b_fp8_dflash2_e5m2_v100_20260821/README.md)‡ |
| `Qwen/Qwen3.8-27B` | DFlash2-8, FP16 KV | 2,094 tok/s | 86.7 tok/s | 2,992 tok/s | 68.6 tok/s | [docker 1K/25K runs](benchmark/qwen38_27b_fp16_dflash2_v100_20260821/README.md) |
| `Qwen/Qwen3.8-27B` | DSpark-7, FP16 KV | 2,020 tok/s | 73.6 tok/s | 3,001 tok/s | 74.5 tok/s | [docker 1K/25K runs](benchmark/qwen38_27b_fp16_dspark_v100_20260821/README.md) |
| `Qwen/Qwen3.6-27B-FP8` | DFlash-16, FP16 KV | 2,774 tok/s | 154.0 tok/s | 3,128 tok/s | 126.2 tok/s | [13-point TP2/TP4 sweep](benchmark/qwen36_27b_fp8_tp_scaling_20260802/README.md) |
| `Qwen/Qwen3.6-27B` | FP16, DFlash-16 | 3,261 tok/s | 101.2 tok/s | 3,631 tok/s | 86.6 tok/s | [Audited context sweep](benchmark/dflash_v100_20260716/README.md) |
| `Qwen/Qwen3.6-35B-A3B` | FP16, DFlash-16 | 4,240 tok/s | 150.1 tok/s | 12,258 tok/s | 136.4 tok/s | [35B optimization results](https://github.com/haohervchb/sglang-V100/commit/7b8615f26e) |
| `QuantTrio/Qwen3.6-35B-A3B-AWQ` | AWQ target/DFlash | — | — | — | — | Supported; comparable run not retained |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | GPTQ-Marlin, DFlash-16 | 3,426 tok/s | 109.2 tok/s | 4,718 tok/s | 81.9 tok/s | [Audited context sweep](benchmark/dflash_v100_20260716/README.md) |
| `QuantTrio/Qwen3.5-122B-A10B-AWQ` | AWQ-Marlin target only | — | — | — | — | Supported; comparable run not retained |
| `poolside/Laguna-S-2.1-INT4` | Marlin, DFlash-8 | 3,334 tok/s† | 77.3 tok/s | 4,327 tok/s† | 67.0 tok/s | [Context sweep](benchmark/dflash_v100_20260716/README.md) and [Laguna tuning](https://github.com/haohervchb/sglang-V100/commit/491bb6095a) |

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

### Historical Qwen3.8 Flash Next Docker v2 concurrency benchmark

The fixed v2 image completed every requested load point with no request errors.
These rows are retained as benchmark provenance; use the v3 image in the live
serve commands below. Output throughput is aggregate across each exact
8,192-input/1,024-output workload:

| Concurrency | Target only | MTP-3/4 | MTP delta |
| ---: | ---: | ---: | ---: |
| 1 | 55.12 tok/s | 74.82 tok/s | +35.7% |
| 4 | 137.26 tok/s | 149.25 tok/s | +8.7% |
| 8 | 172.98 tok/s | 194.13 tok/s | +12.2% |
| 16 | 239.36 tok/s | 183.87 tok/s | -23.2% |

For the cold 1,000-input/25,000-output request at concurrency 1, target-only
reached 60.09 output tok/s and MTP reached 88.07 output tok/s (+46.6%). See the
[full fixed-image report](benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md)
for TTFT, TPOT, acceptance length, and memory sizing.

### Native SM70 optimization status

The acceptance workload for these changes is the actual
`Qwen/Qwen3.8-27B-FP8` checkpoint with TP4, E5M2 KV, and speculative decoding
off. The post-port cold validation reaches 4,224 prefill tok/s and 63.2 decode
tok/s at 4K, 59.1 decode tok/s at 70K, and 49.6 decode tok/s at 200K. The SM70
CUDA read-once split-KV decode partial removed the old severe context decay;
the fused QPN8 gate/up path then reduced TPOT another 2.6% at 4K and 1.7% at
70K in controlled A/B runs. The full curve and profiler
breakdown are in the [FP8 target-only report](benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/README.md).
The operator measurements below explain individual paths; they are not being
used as a substitute for that FP8 end-to-end result.

| Path | Measured shape | Result |
| --- | --- | ---: |
| Chunked GDN prefill | Qwen3.8 TP4, 2,048 tokens | 1.455 ms vs 2.017 ms previous native schedule (27.9% lower) |
| Chunked GDN prefill | Qwen3.8 TP4, 4,096 tokens | 2.199 ms vs 3.236 ms previous native schedule (32.0% lower) |
| Chunked GDN prefill | Qwen3.8 TP4, 8,192 tokens | 3.725 ms vs 5.674 ms previous native schedule (34.3% lower) |
| Exact D256 tail split-KV | Q=64, K=245,760, Hq/Hkv=6/1 | 4.306 ms vs 40.675 ms unsplit (9.45x) |
| Fused QPN8 gate/up + SiLU | M=1, K=5,120, N=8,704 | 0.059 ms vs 0.062 ms materialized gate/up, bitwise exact (about 1.04x) |
| Mixed FP16/FP32 Gemma RMSNorm | 4,096 x 5,120 | 0.315 ms vs 1.397 ms PyTorch (4.44x) |
| Experimental BFLA sparse attention | Q=4,096, K=32,768, D=256, 10% keep | about 9.8 ms including selection vs 23.9 ms dense (about 2.4x) |

The GDN prefill dispatcher chooses the direct recurrent kernel through 448
tokens and the tensor-core 64-token chunk kernel above that boundary. It
supports packed variable-length batches, row-strided mixed QKV, indexed FP32
state, direct output, and a column-group CTA schedule. Keep decode on Triton in
the reference command: the native fused recurrent decoder is correct, but the
existing Triton decoder remains faster for Qwen3.8 TP4's one-token shape.

The mixed-dtype Gemma residual/RMSNorm route is automatic on SM70 for at least
256 rows with hidden size 5,120, but only for the exact FP16 activation plus
FP32 residual contract. Qwen3.8-27B-FP8 normally keeps this residual in FP16,
so the 4.44x operator result is not claimed as a gain for the primary model.
Set `SGLANG_V100_GEMMA_RMSNORM=0` for an A/B rollback. BFLA is intentionally
disabled by default. An exact all-keep control can be enabled with
`SGLANG_V100_BFLA_PREFILL=1`; actually dropping blocks additionally requires
`SGLANG_V100_BFLA_ALLOW_APPROXIMATE=1` and a keep ratio such as
`SGLANG_V100_BFLA_KEEP_RATIO=0.1`. Sparse mode changes model semantics and must
pass retrieval and long-context quality evaluation before production use.

For strictly greedy, non-speculative requests, the opt-in
`SGLANG_V100_GREEDY_TP_TOP1=1` route exchanges only each TP rank's top candidate
instead of gathering full-vocabulary logits. It fails closed to the ordinary
logits path for sampling, speculative decoding, logprobs, penalties, grammar,
custom logits processors, or logits bias. It therefore does not affect the
official sampling benchmark.

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
docker pull geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v3
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

### Serve Qwen3.8 Flash Next NVFP4 from Docker

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

## MiniMax-H3 video and audio

### Serve H3 on the host

Recommended W4A16 command:

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

W8A16 with DiT offload:

```bash
conda activate sglang-v100

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
  --quantization v100_w8a16 \
  --attention-backend tilelang_fa_v100 \
  --dit-cpu-offload \
  --text-encoder-cpu-offload \
  --vae-cpu-offload \
  --enable-torch-compile false \
  --host 0.0.0.0 \
  --port 30010
```

### Serve H3 from Docker

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

### Generate a text-to-video-and-audio clip

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

### Generate from a first frame

Replace `/absolute/path/first-frame.png` with a file visible to every server
rank:

```bash
video_id=$(
  curl -sS -X POST http://127.0.0.1:30010/v1/videos \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "MiniMaxAI/MiniMax-H3",
      "prompt": "Continue this scene with calm natural motion and synchronized ambient sound.",
      "task": "fl2va",
      "conditions": [{
        "type": "image",
        "uri": "file:///absolute/path/first-frame.png",
        "role": "keyframe",
        "frame_index": 0
      }],
      "target": {
        "short_edge": 544,
        "aspect_ratio": "auto",
        "duration_seconds": 5.0
      },
      "num_inference_steps": 50,
      "flow_shift": 12.0,
      "audio_flow_shift": 3.0,
      "seed": 2101
    }' | jq -r '.id'
)

while true; do
  status=$(curl -sS "http://127.0.0.1:30010/v1/videos/$video_id" | jq -r '.status')
  [ "$status" = completed ] && break
  [ "$status" = failed ] && exit 1
  sleep 2
done

curl -sS -L "http://127.0.0.1:30010/v1/videos/$video_id/content" \
  -o minimax-h3-fl2va.mp4
```

Use `"frame_index": -1` for a last frame. Launch with
`--model-variant ref2va` for reference-image, reference-audio, reference-video,
or video-to-video jobs.

## LLM serving examples

The Flash Next commands below invoke the repository environment directly and
need no wrapper script. The older examples assume `conda activate sglang-v100`
first. These commands use all four V100s; the listening port is shown in each
command.

### Qwen3.8 Flash Next NVFP4 target-only

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

### Qwen3.8 Flash Next NVFP4 with MTP

This loads the checkpoint's built-in MTP module from the same RadixArk repo as
the target. The setting is three speculative steps with four draft tokens. It
keeps the same E5M2 KV cache and full-context, single-request memory
configuration as target-only.

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

### Qwen3.8-27B-FP8 target-only

This is the reference command for ordinary, non-speculative decode. No
speculative environment switch or `--speculative-*` argument is present. The
SM70 CUDA read-once split-KV decode partial and the QPN8 W8A16 decode GEMMs are
enabled by default; set `SGLANG_V100_DECODE_CUDA=0` or
`SGLANG_SM70_FP8_DECODE_QPN8=0` to opt out. With the current CUDA partial and
QPN8 paths, target-only decode measures 59.1 tok/s at 70K and 49.6 tok/s at
200K, up from about 30 tok/s at 128K before. QPN8's word-parallel decoder and
fused gate/up SiLU path are also default-on; use
`SGLANG_SM70_FP8_QPN8_FASTDEC=0` and
`SGLANG_SM70_FP8_QPN8_FUSED_GATE=0` for a controlled rollback. The CUDA
partial covers TP1/TP2/TP4 (the GQA ratio is fixed at 6:1, so tensor-parallel
splits change only the per-rank KV-head count).

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

### Qwen3.8-27B-FP8 with DSpark

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
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
  --speculative-dspark-block-size 7 \
  --speculative-draft-model-quantization unquant \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

`fp8_e5m2` is the optimized compact-KV route for this model on V100: cache
writes, long D=256 prefill, long decode, and DSpark target verification all
have native SM70 paths. Use `--kv-cache-dtype auto` for the faster FP16 KV
cache when its lower capacity is acceptable. `fp8_e4m3` remains a compatibility
option, but is not the preferred compact-KV format for Qwen3.8-27B-FP8.
For this single-request 262K configuration, keep `--max-total-tokens` equal to
the context length. Leaving it automatic can allocate target and draft KV pools
for over one million tokens, wasting the activation headroom required by the
optimized prefill kernels.

The following historical sweep was measured on V100-SXM2-32GB with FP8
weights, **FP16 KV**, DSpark block 7, one cold-cache request, and 256 greedy
output tokens. It is not an E5M2 result:

| Input | TP2 prefill | TP4 prefill | TP2 decode | TP4 decode |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,761 tok/s | 2,749 tok/s | 76.5 tok/s | 107.8 tok/s |
| 9K | 1,888 tok/s | 3,355 tok/s | 60.7 tok/s | 84.7 tok/s |
| 17K | 1,778 tok/s | 3,278 tok/s | 56.9 tok/s | 86.7 tok/s |
| 25K | 1,686 tok/s | 3,140 tok/s | 51.9 tok/s | 78.3 tok/s |

Across the complete 1K-to-25K sweep, TP4 is 1.81x faster for prefill and
1.44x faster for decode by geometric mean. See the
[full 13-point benchmark](benchmark/qwen38_27b_fp8_dspark_tp_scaling_20260815/README.md).

### Qwen3.8-27B-FP8 with DFlash2

The published DFlash2 checkpoint uses block size 8: one anchor plus seven
proposed tokens. Its selector top-k of 16 is the number of candidates scored at
each proposal position, not the proposal length.

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

The V100 path keeps the target KV cache in E5M2 but uses FP16 for the much
smaller five-layer draft cache. Both the DFlash2 candidate selector and draft
forward are captured in the CUDA graph. Block 16 can be A/B tested by changing
only `--speculative-dflash-block-size 8` to `16`; this is outside the checkpoint's
published block-8 configuration and should be selected by measured output
throughput rather than acceptance length alone.

Bring-up results on four V100-SXM2-32GB GPUs, TP4, E5M2 target KV, CUDA graphs,
and 256 or 512 greedy output tokens:

| Input/workload | Context handling | Output throughput | Average commit length |
| --- | --- | ---: | ---: |
| Short prompt, warmed | Warm graph and kernels | 118.1 tok/s | 3.66 |
| 4K periodic synthetic prompt | Full prompt included in client time | 121.5 tok/s end-to-end | 7.11 |
| 70K periodic synthetic prompt | 69,952 prompt tokens reused | 79.2 tok/s end-to-end | 4.13 |
| 200K periodic synthetic prompt | Steady scheduler decode intervals | ~60 tok/s | 4.45 |
| 150K cold-cache prompt | 150,000 prompt tokens, fresh server, zero cached tokens | 134 tok/s | ~3.5 |
| 200K cold-cache prompt | 200,000 prompt tokens, fresh server, zero cached tokens | 112 tok/s | ~3.5 |

Acceptance is workload-dependent. The synthetic rows validate graph capture,
long-context KV operation, and selector stability; use the same prompt corpus
for block-8 versus block-16 comparisons. The cold-cache rows used
`--random-input-len 150000/200000`, one request, and 256 greedy output tokens
on a freshly restarted server.

#### Serve DFlash2 in Docker

Build (or pull) the container image as shown in the
[Docker section](#docker), then launch the same DFlash2 workload in a
container. The image bakes in the V100 defaults (`NCCL_P2P_LEVEL=NVL`,
`SGLANG_MAMBA_CONV_DTYPE=float16`, `SGLANG_MAMBA_SSM_DTYPE=float16`); the
remaining flags are passed explicitly.

```bash
docker rm -f v100-dflash2 2>/dev/null

docker run --rm --name v100-dflash2 \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
  -e SGLANG_ENABLE_SPEC_V2=1 \
  -e SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
  sglang-v100:latest \
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

`--network host` is required for TP4 NCCL on the tested single-node host.
Startup takes several minutes (weight load plus CUDA-graph capture), and the
first request after boot hits a one-time kernel/JIT warmup. The >128K prefill
path uses the native D256 split-D operator that is compiled into this image;
host and container were measured within 0.2% of each other at 131K and 200K.

### Qwen3.6-27B-FP8 with DFlash

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

### Qwen3.6-35B-A3B FP16 with DFlash

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --model Qwen/Qwen3.6-35B-A3B \
  --dtype float16 \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.70 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 8192 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls \
  --enable-multimodal \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

### Qwen3.5-122B-A10B GPTQ-Int4 with DFlash

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --dtype float16 \
  --quantization gptq_marlin \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.78 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls \
  --enable-multimodal \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-122B-A10B-DFlash \
  --speculative-dflash-block-size 16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

### Poolside Laguna-S-2.1 INT4 with DFlash

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

### Docker: Qwen3.6-27B-FP8 with DFlash

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

## OpenAI-compatible chat request

```bash
curl -sS http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Write a short hello-world program."}],
    "temperature": 0,
    "max_tokens": 256
  }' | jq
```

## Additional example serve commands

### Qwen3.5-122B-A10B GPTQ-Int4 target only

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
sglang serve \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --dtype float16 \
  --quantization gptq_marlin \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.78 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls
```

### Qwen3.5-122B-A10B AWQ target only

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
sglang serve \
  --model QuantTrio/Qwen3.5-122B-A10B-AWQ \
  --dtype float16 \
  --quantization awq_marlin \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.78 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls
```

### Qwen3.6-35B-A3B FP16 target only

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --model Qwen/Qwen3.6-35B-A3B \
  --dtype float16 \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.70 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 8192 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls \
  --enable-multimodal \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

### Qwen3.6-35B-A3B AWQ target only

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
sglang serve \
  --model QuantTrio/Qwen3.6-35B-A3B-AWQ \
  --dtype float16 \
  --quantization awq_marlin \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.78 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls
```

### Qwen3.6-27B FP16 target only

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

### Qwen3.6-27B-FP8 target only

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

### Poolside Laguna-S-2.1 INT4 target only

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

### Qwen3.6-27B FP16 with DFlash

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

### Qwen3.6-35B-A3B AWQ with DFlash

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --model QuantTrio/Qwen3.6-35B-A3B-AWQ \
  --dtype float16 \
  --quantization awq_marlin \
  --attention-backend tilelang_fa_v100 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.78 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-nccl-nvls \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-35B-A3B-DFlash \
  --speculative-dflash-block-size 8 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

### Qwen3.5-122B-A10B GPTQ-Int4 with MTP

```bash
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_SPEC_V2=1 \
sglang serve \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --dtype float16 \
  --quantization gptq_marlin \
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

### Qwen3.6-27B FP16 with MTP

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

## References and attribution

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
