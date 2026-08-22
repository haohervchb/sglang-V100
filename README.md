# SGLang for 4x NVIDIA V100 32 GB NVLink

SGLang serving commands for four SM70 V100 GPUs.

## Current V100 performance

All currently documented model checkpoints are listed below. LLM results use
TP4, one cold request, and 256 greedy output tokens. Prefill is the exact input
length divided by client time to first token; decode excludes that first-token
time. The measurements came from separate tuning runs, so treat this as a
practical reference rather than a perfectly controlled cross-model leaderboard.
H3 reports wall-clock video generation time in the Results column rather than
LLM token throughput. `—` means the metric does not apply or the supported
configuration has no comparable retained end-to-end benchmark.

| Model checkpoint | Measured configuration | 1K prefill | 1K decode | 25K prefill | 25K decode | Results |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `MiniMaxAI/MiniMax-H3` | TP4 W4A16, 960×544, 15 s clip, 10 steps | — | — | — | — | ~500 s/video |
| `Qwen/Qwen3.8-27B-FP8` | Target only, E5M2 KV | 2,992 tok/s | 58.2 tok/s | 3,714 tok/s | 50.8 tok/s | **4K: 4,137/57.6; 70K: 2,980/38.8; 128K: 2,356/30.0 tok/s**; [audited FP8 sweep](benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/README.md) |
| `Qwen/Qwen3.8-27B-FP8` | DFlash2-8, E5M2 KV | 1,803 tok/s | 136.6 tok/s | 2,701 tok/s | 102.3 tok/s | 118.1 tok/s warm short decode; 79.2 tok/s at 70K; ~60 tok/s steady at 200K; [docker 1K/25K runs](benchmark/qwen38_27b_fp8_dflash2_e5m2_v100_20260821/README.md)‡ |
| `Qwen/Qwen3.8-27B-FP8` | DSpark-7, FP16 KV | 2,749 tok/s | 107.8 tok/s | 3,140 tok/s | 78.3 tok/s | [13-point TP2/TP4 sweep](benchmark/qwen38_27b_fp8_dspark_tp_scaling_20260815/README.md) |
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
block-8 results. ‡The DFlash2 figures are single-request bring-up measurements
on synthetic prompts. The 70K request reused 69,952 cached prompt tokens; the
200K figure is the scheduler's steady decode rate. They validate the long-context
path but are not directly comparable with the cold 1K/25K sweep columns.

### Native SM70 optimization status

The acceptance workload for these changes is the actual
`Qwen/Qwen3.8-27B-FP8` checkpoint with TP4, E5M2 KV, and speculative decoding
off. The retained cold sweep reaches 4,137 prefill tok/s at 4K and measures
2,356/30.0 prefill/decode tok/s at 128K. The full curve and profiler breakdown
are in the [FP8 target-only report](benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/README.md).
The operator measurements below explain individual paths; they are not being
used as a substitute for that FP8 end-to-end result.

| Path | Measured shape | Result |
| --- | --- | ---: |
| Chunked GDN prefill | Qwen3.8 TP4, 2,048 tokens | 1.291 ms vs 1.745 ms Triton (1.35x) |
| Chunked GDN prefill | Qwen3.8 TP4, 4,096 tokens | 2.547 ms vs 3.300 ms Triton (1.30x) |
| Chunked GDN prefill | Qwen3.8 TP4, 8,192 tokens | 5.043 ms vs 6.232 ms Triton (1.24x) |
| Mixed FP16/FP32 Gemma RMSNorm | 4,096 x 5,120 | 0.315 ms vs 1.397 ms PyTorch (4.44x) |
| Experimental BFLA sparse attention | Q=4,096, K=32,768, D=256, 10% keep | about 9.8 ms including selection vs 23.9 ms dense (about 2.4x) |

The GDN prefill dispatcher chooses the direct recurrent kernel through 1,280
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

### Source and dependency boundary

Attention, FP8 KV conversion, D256 dense/split-KV/sparse prefill, grouped
decode, GDN, and the mixed-dtype RMSNorm fusion are implemented in this
repository's `python/sglang/.../tilelang*` sources. Neither the host installer
nor Docker installs 1Cat-vLLM, FlashQLA, or zhinianqin's
FlashAttention-V100 package. The chunked GDN equations are informed by Qwen's
MIT-licensed public FlashQLA algorithm, but the SM70 kernels and SGLang
integration here are an independent TileLang implementation.

TurboMind is the temporary exception requested for quantized GEMM and MoE.
The installer makes a pinned sparse checkout containing only `LICENSE`,
`csrc/core`, `csrc/sm70_turbomind`, and `csrc/moe` from
[1CatAI/1Cat-vLLM at `6ada86e`](https://github.com/1CatAI/1Cat-vLLM/tree/6ada86ed64af6d1a7b3cb0f34df237fd86f06d48),
then builds SGLang's private adapter against pinned NVIDIA CUTLASS. It never
installs or imports that repository's vLLM or attention packages.

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

### Qwen3.8-27B-FP8 target-only

This is the reference command for ordinary, non-speculative decode. No
speculative environment switch or `--speculative-*` argument is present.

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

Acceptance is workload-dependent. The synthetic rows validate graph capture,
long-context KV operation, and selector stability; use the same prompt corpus
for block-8 versus block-16 comparisons.

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
