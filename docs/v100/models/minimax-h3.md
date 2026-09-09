# MiniMax-H3 on V100

[Back to the main README](../../../README.md) · [Host setup](../../../README.md#install-on-the-host) · [Docker setup](../../../README.md#docker)

Activate `sglang-v100` before running host commands. Examples use four
V100 32 GB GPUs; each command specifies its listening port.


## Serve H3 on the host

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

## Serve H3 from Docker

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

## Generate a text-to-video-and-audio clip

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

## Generate from a first frame

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
