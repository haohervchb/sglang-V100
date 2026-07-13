# SGLang for 4× NVIDIA V100 32 GB NVLink

This fork adapts SGLang for four SM70 V100 GPUs. It includes a TileLang
FlashAttention backend, V100-aware paged KV-cache and decode tuning, custom
all-reduce support, and Marlin GPTQ/AWQ MoE kernels.

The commands below target x86-64 Ubuntu, four 32 GB V100s connected by NVLink,
and a sufficiently recent NVIDIA driver. They intentionally create a dedicated
Conda environment named `sglang-v100`.

## Copy, paste, and build

Paste this block into a terminal. It is safe to rerun: the checkout, dependency
sources, build products, and Conda environment are reused.

```bash
if [[ -d "$HOME/sglang-V100/.git" ]]; then
  git -C "$HOME/sglang-V100" pull --ff-only
else
  git clone https://github.com/haohervchb/sglang-V100.git "$HOME/sglang-V100"
fi
bash "$HOME/sglang-V100/scripts/install_v100.sh"
```

Run the installer with `bash`; do not source it. Its strict error handling then
lives in a child process, so a failed command reports its line and returns to
the SSH prompt instead of exiting the login shell. Compilation uses all CPU
threads that available RAM can safely sustain, rather than a hard-coded job
count. Set `MAX_JOBS=16` before the command only if you want to impose a limit.

The installer builds only SM70 `sglang-kernel` targets, applies the exact
FlashInfer SM70, native V100 attention, and Marlin SM70 compatibility patches
kept in this repository, verifies NCCL, and precompiles FlashInfer's sampling
module. The last step moves its roughly minute-long cold JIT cost from the
first chat into installation/startup.

If only the final validation step fails, do not rerun the installer or rebuild
anything. After pulling the latest changes, rerun validation directly:

```bash
bash "$HOME/sglang-V100/scripts/smoke_v100.sh"
```

The FlashInfer package and cubin versions used by the working SM70 stack differ
slightly, so the serving commands set `FLASHINFER_DISABLE_VERSION_CHECK=1`.
This is expected for this fork.

## Docker: copy, paste, and run

The Docker image mirrors the host build: CUDA 12.8, Torch 2.9.1, the patched
FlashInfer SM70 source, native V100 attention fallback, SM70-only
`sglang-kernel`, V100 Marlin, and NCCL 2.27.5. BuildKit keeps every expensive
native component in its own layer, so rerunning this exact command resumes from
the last completed component:

```bash
cd "$HOME/sglang-V100"
DOCKER_BUILDKIT=1 docker build --network=host \
  -f docker/v100.Dockerfile \
  -t sglang-v100:latest .
```

Build parallelism is selected from the CPUs and available memory visible to
Docker. To impose a manual limit, add `--build-arg MAX_JOBS=16`.

Launch the default Qwen3.5 GPTQ model on four V100s:

```bash
docker volume create sglang-v100-flashinfer
docker volume create sglang-v100-tilelang

docker run --rm --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v sglang-v100-flashinfer:/root/.cache/flashinfer \
  -v sglang-v100-tilelang:/root/.cache/tilelang \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  sglang-v100:latest \
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
  --enable-nccl-nvls
```

The container validates the GPU stack and warms the FlashInfer sampler before
starting SGLang. Its FlashInfer and TileLang caches are persistent volumes, so
that cold JIT work survives container replacement. If the Docker Compose v2
plugin is installed, the equivalent shorter command is:

```bash
docker compose -f docker/v100-compose.yaml up --build
```

## Serve models

Run `conda activate sglang-v100` first. Each example listens on port 8082 and
uses all four visible GPUs. Stop one server before starting another.

### Qwen3.5-122B-A10B GPTQ Int4

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  --enable-nccl-nvls
```

### Qwen3.5-122B-A10B AWQ

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  --enable-nccl-nvls
```

### Qwen3.6-35B-A3B AWQ

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  --chunked-prefill-size 16384
```

### Qwen3.6-27B dense FP16

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  --enable-nccl-nvls
```

The first launch downloads the model from Hugging Face. For gated models,
export `HF_TOKEN` before launching. Reduce `--context-length` or
`--mem-fraction-static` if other processes are using GPU memory.

Startup now warms the batch-one and batch-two SM70 prefill paths and the
FlashInfer sampler before the API becomes ready. `--enable-profile-cuda-graph`
is deliberately absent from normal serving commands: it is a diagnostic flag
that writes profiling artifacts, not a performance switch.

## Scope

This is a hardware-specific fork, not a replacement for upstream SGLang. The
tested target is four V100 32 GB GPUs with NVLink, CUDA 12.8, FP16 activations,
the `flash_attn_v100` attention backend, and the Marlin GPTQ/AWQ path. The
upstream project remains the source for general SGLang documentation.

SGLang is licensed under the Apache License 2.0; see [LICENSE](LICENSE).
