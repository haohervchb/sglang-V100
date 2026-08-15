# SGLang for 4x NVIDIA V100 32 GB NVLink

SGLang serving commands for four SM70 V100 GPUs.

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

Pull the published image:

```bash
docker pull geesegeesegeese/sglang-v100:latest
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

Run `conda activate sglang-v100` first. These commands use all four V100s and
listen on port 8082.

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
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flash_attn_v100 \
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
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
  --speculative-dspark-block-size 7 \
  --speculative-draft-model-quantization unquant \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

Use `--kv-cache-dtype auto` for the faster FP16 KV cache when its lower
capacity is acceptable.

Measured on V100-SXM2-32GB with FP8 weights, FP16 KV, DSpark block 7, one
cold-cache request, and 256 greedy output tokens:

| Input | TP2 prefill | TP4 prefill | TP2 decode | TP4 decode |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,761 tok/s | 2,749 tok/s | 76.5 tok/s | 107.8 tok/s |
| 9K | 1,888 tok/s | 3,355 tok/s | 60.7 tok/s | 84.7 tok/s |
| 17K | 1,778 tok/s | 3,278 tok/s | 56.9 tok/s | 86.7 tok/s |
| 25K | 1,686 tok/s | 3,140 tok/s | 51.9 tok/s | 78.3 tok/s |

Across the complete 1K-to-25K sweep, TP4 is 1.81x faster for prefill and
1.44x faster for decode by geometric mean. See the
[full 13-point benchmark](benchmark/qwen38_27b_fp8_dspark_tp_scaling_20260815/README.md).

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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
  --attention-backend flash_attn_v100 \
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
