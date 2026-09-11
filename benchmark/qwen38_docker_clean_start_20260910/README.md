# Qwen3.8 Flash Next Docker v4: startup from empty caches

The published v4 image downloaded the full checkpoint, compiled its JIT kernels,
and served inference with both the Hugging Face and JIT caches initially empty.
No host checkpoint, compiled kernels, source checkout, launcher script, or
Hugging Face token was supplied to the container.

Tested on 2026-09-10 with four Tesla V100-SXM2 32 GB GPUs, NVIDIA driver
580.178.04, Docker 29.1.3, and a 256 GiB RAM host (251.6 GiB visible to Linux).
This checkpoint offloads embedding weights to host RAM; this run does not
establish a minimum RAM requirement for smaller hosts.

## Image and isolation

- Image: `geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v4`.
- Published manifest, checked against Docker Hub:
  `sha256:9aa44672321e21e38e6727a218cd67710e683c195fc2fc713f482e85e2d61735`.
- Local image ID:
  `sha256:4373c93d636ff83fba0e06eb0c88c10edee07621d7ba505eef8f51d9c84601af`.
- Model revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- Two new Docker volumes were created and asserted empty before launch:
  `sglang-v100-hf-clean-start-20260910` and
  `sglang-v100-jit-clean-start-20260910`.
- Preflight confirmed online Hugging Face access, no supplied token, the packaged
  launcher, and the image's `python`, `nvcc`, `g++-12`, and `ninja` executables.

The launch used the script-based README command current at test time, with
TP4, E5M2 KV, MTP-3/4, and the documented JIT environment settings.
Only the test container/volume names, port 8083, and an isolated named model
volume in place of the host cache bind mount differed. The exact argument list
and empty-cache assertions are retained in [validation.json](validation.json).
The container had no source or script bind mounts.
The [current README command](../../README.md#serve-qwen38-flash-next-nvfp4-from-docker)
expands the script's environment and server arguments directly, explicitly sets
`--enable-multimodal`, and uses foreground `--rm` container lifecycle settings.

## Results

| Check | Observed result |
| --- | --- |
| Model download | All 206 indexed shards present; 135,195,303,851 weight-file bytes, approximately 126 GiB |
| JIT compilation | 56 shared libraries and 988 cache files created during startup and inference |
| Readiness | `/health` returned HTTP 200 after 3,122.6 seconds (52 minutes 3 seconds) |
| Arithmetic chat request | `17 × 23` returned exactly `391` |
| Geography chat request | Capital of France returned exactly `Paris` |
| Generation probe | Exactly 512 output tokens, length stop, zero cached prompt tokens, no retractions |
| MTP exercised | 174 verification steps; 337 accepted draft tokens; mean acceptance length 2.943 |

The compiled artifacts include the NVFP4 MoE and long-context decode CUDA
extensions, FlashInfer sampling, SGLang TVM-FFI kernels, TileLang kernels, and
Triton launchers. They were built with the tools already in the published image.
No image patch or manual installation inside the container was needed.

Startup timing begins after `docker run`, so it excludes pulling the image.
It includes downloading weights, loading them, compilation, and application
warm-up. The 512-token request was submitted as the server began listening and
took 31.37 seconds, including first-use warm-up. This is a startup/inference
validation, not a new steady-state throughput measurement. Retained
[Docker/host performance comparisons](../qwen38_nvfp4_v100_docker_v4_20260908/README.md)
remain the performance reference.

The former `HF_HUB_OFFLINE=1` plus read-only cache example was unsuitable for
first startup. A separate empty-cache probe reproduced the offline missing-file
failure and then downloaded model/tokenizer configuration anonymously with
online access enabled. The public command now uses a writable cache and lets
the server download missing files. The launcher lives inside the image, and
Docker/runtime cache directories are created automatically.

The original `qwen38-flash-next-mtp-v4` service was restored and returned HTTP
200 from `/health` on port 8082. The test container and its two isolated volumes
were then removed, releasing approximately 126 GiB of temporary disk usage.
The original service's model and JIT caches were preserved.

## Evidence

- [Validation, launch arguments, and initial empty-cache checks](validation.json)
- [Hardware and published image manifest](environment.json)
- [Downloaded model inventory](model_cache.json)
- [Compiled JIT library inventory](jit_cache.json)
- [Server configuration](server_info.json)
- [Chat responses](responses.json)
- [512-token request and response with MTP counters](generation_smoke.json)
- [Selected startup log events](startup_events.txt)
- [Service restoration and temporary-cache cleanup](cleanup.json)
