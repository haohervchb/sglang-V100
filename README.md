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
docker volume create sglang-v100-triton
docker volume create sglang-v100-inductor

docker run --rm --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v sglang-v100-flashinfer:/root/.cache/flashinfer \
  -v sglang-v100-tilelang:/root/.tilelang \
  -v sglang-v100-triton:/root/.triton \
  -v sglang-v100-inductor:/tmp/torchinductor_root \
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

The container validates the GPU stack, verifies the real SM70 Marlin repack,
and warms the FlashInfer sampler before starting SGLang. Its FlashInfer,
TileLang, Triton, and TorchInductor caches are persistent volumes, so cold JIT
work survives container replacement. If the Docker Compose v2 plugin is
installed, the equivalent shorter command is:

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

## Decode acceleration with MTP or DFlash

MTP and DFlash are alternative speculative decoders; enable one at a time.
Both target checkpoints above include a built-in MTP layer. To enable it, add
these arguments to either the Qwen3.5-122B GPTQ or Qwen3.6-27B command:

```bash
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mamba-scheduler-strategy extra_buffer
```

Do not set `--speculative-draft-model-path` for MTP. SGLang selects the MTP
layer from the target checkpoint, including the unquantized MTP weights stored
inside the factory GPTQ-Int4 checkpoint.

To pair the factory Qwen3.5-122B-A10B GPTQ-Int4 target with its DFlash draft,
add:

```bash
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-122B-A10B-DFlash \
  --speculative-dflash-block-size 16 \
  --mamba-scheduler-strategy extra_buffer
```

To pair the dense Qwen3.6-27B target with its DFlash draft, add:

```bash
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-27B-DFlash \
  --speculative-dflash-block-size 16 \
  --mamba-scheduler-strategy extra_buffer
```

On V100, DFlash draft attention and linear target verification automatically
use the native SM70 paged-attention kernel. The kernel applies each draft
layer's causal sliding window or bidirectional full-attention semantics. MTP
draft extension uses Triton, while its linear target verification uses the
native SM70 kernel.
Top-k-1 MTP target verification uses the native TileLang paged-prefill kernel;
tree verification continues to use Triton's custom-mask path. Ordinary target
prefill remains on `flash_attn_v100`. Block size 16 is the low-concurrency
starting point. Try block size 8 for the 122B draft when serving more concurrent
requests and compare acceptance rate and inter-token latency on the actual
workload.

For full-attention 16-token verification blocks, the long-context SM70 path
partitions KV work across up to 80 V100 SMs and processes GQA heads in groups of
up to four. This replaces the ordinary extend kernel's serial full-prefix scan
and avoids reloading the same K/V data once per query head.

The long-context kernel improvement is best measured independently of draft
acceptance. The following is the raw time for one full-attention target verify
layer with a 16-token block on one V100 (TP4 per-rank head shapes):

| Target shape | Context | Old serial scan | Split-KV verify |
| --- | ---: | ---: | ---: |
| Qwen3.6-27B | 100,000 | 18.83 ms | 1.04 ms |
| Qwen3.6-27B | 150,000 | 27.58 ms | 1.54 ms |
| Qwen3.5-122B-A10B | 100,000 | 19.17 ms | 1.04 ms |
| Qwen3.5-122B-A10B | 150,000 | 27.34 ms | 1.53 ms |

Do not use repeated-token prompts to estimate application throughput. They can
produce unusually high DFlash acceptance and make end-to-end numbers too
optimistic. The following batch-one, greedy measurements use source files from
this repository as the prompt and report the steady decode bucket after
prefill. They were collected with the validated spec-v2 commands below:

| Target and workload | Prompt context | Accept length | Output tok/s |
| --- | ---: | ---: | ---: |
| Qwen3.6-27B, short quicksort | short | 5.07 | 119.0 |
| Qwen3.6-27B, code-context quicksort | 93,477 | 6.78 | 120.5 |
| Qwen3.6-27B, code-context quicksort | 161,058 | 6.72 | 100.6 |
| Qwen3.6-27B, hard code analysis | 93,489 | 2.10 | 37.6 |
| Qwen3.5-122B-A10B, short quicksort | short | 7.01 | 182.6 |
| Qwen3.5-122B-A10B, code-context quicksort | 93,477 | 6.48 | 134.9 |

The 27B model therefore holds about 100 output tok/s through 161K context when
the draft accepts roughly 6.7 tokens per verify. Acceptance is still the main
workload-dependent variable: the harder analysis prompt accepted only about
2.1 tokens and reached 37.6 tok/s on the same server. Benchmark actual prompts
and sampling settings, and always record `spec_accept_length` beside output
tok/s.

This is consistent with the checkpoint status: the
[Qwen3.6-27B DFlash model card](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)
says that checkpoint is still under training and publishes no benchmark
results. The mature
[Qwen3.5-122B-A10B DFlash model card](https://huggingface.co/z-lab/Qwen3.5-122B-A10B-DFlash)
reports mean block-16 accept lengths from about 5.4 to 8.7 depending on the
workload.

The serving implementation uses the cumulative mainline spec-v2 relay and
scheduling architecture by default while retaining the native SM70 draft and
verification kernels. Set `SGLANG_ENABLE_SPEC_V2=0` only to diagnose the legacy
v1 worker. The complete known-good v1 source is tagged
`dev-dflash-v1-sm70-baseline-20260716`.

### Validated 4x V100 serving commands

The following commands are the configurations used for end-to-end validation
on four V100-SXM2-32GB GPUs. Like the examples above, each command includes its
runtime environment variables inline and listens on port 8082. The conservative
`--mem-fraction-static 0.70` leaves room for the speculative draft weights and
CUDA graph capture.

Qwen3.5-122B-A10B GPTQ-Int4 with its built-in MTP layer:

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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

Qwen3.5-122B-A10B GPTQ-Int4 with DFlash:

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  --mem-fraction-static 0.70 \
  --context-length 262144 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 2 \
  --cuda-graph-bs 1 2 \
  --enable-nccl-nvls \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-122B-A10B-DFlash \
  --speculative-dflash-block-size 16
```

Qwen3.6-27B dense FP16 with its built-in MTP layer:

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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

Qwen3.6-27B dense FP16 with DFlash:

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  --cuda-graph-max-bs 2 \
  --cuda-graph-bs 1 2 \
  --enable-nccl-nvls \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-27B-DFlash \
  --speculative-dflash-block-size 16
```

For MTP, do not set `--speculative-draft-model-path`; SGLang loads the built-in
MTP layer from the target checkpoint. `--context-length` is an upper bound, not
a promise that the KV pool can hold that many live tokens. At
`--mem-fraction-static 0.70`, the 122B DFlash spec-v2 configuration has 116,064
target/draft KV slots on this machine and resolves the requested
`--max-running-requests 4` to three concurrent requests. Leave space for
generated tokens and concurrent requests. Increase `--mem-fraction-static`
only after checking the memory left for draft weights, JIT compilation, and
CUDA graphs.

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
