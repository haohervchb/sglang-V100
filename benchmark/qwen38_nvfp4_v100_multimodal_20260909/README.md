# Qwen3.8 NVFP4 V100: image and video validation

**The corrected Docker image matches the optimized host and pre-optimization
main on all 12 response texts. No new multimodal regression was found.**
Each full sweep passes 11 of 12 ground-truth checks. The remaining OCR case
reads `V100 CHECK 4827` as `V10 CHECK 4827` in every configuration, including
main. This pre-existing error is retained as a failed check; the suite is
not reported as a complete ground-truth pass.

| Runtime | Ground-truth checks | Exact response texts matching main |
| --- | ---: | ---: |
| Main baseline, target-only | 11/12 | 12/12 |
| Optimized host, MTP | 11/12 | 12/12 |
| Corrected Docker, target-only | 11/12 | 12/12 |
| Corrected Docker, MTP | 11/12 | 12/12 |

The initial Docker run additionally failed both video requests. Installing
FFmpeg fixes those HTTP 500 failures. The rebuilt image passes the native
V100 smoke script, and three further image/video/text checks pass after
restarting the final MTP container. The local image is tagged
`sglang-v100:v100-qwen38-flash-next-v4` and `sglang-v100:latest`; container
`qwen38-flash-next-mtp-v4` serves `http://127.0.0.1:8082/v1`. It has not been
pushed to a registry. [Service record](service_final.json).

The dependency fix also preserves text decode performance:

| Natural-prompt input | Previous fresh host tok/s | Corrected Docker tok/s | Change |
| --- | ---: | ---: | ---: |
| 1,000 | 116.81 | 116.62 | -0.16% |
| 25,000 | 117.86 | 119.74 | +1.60% |

These are means of two 1,024-output requests per input length. The wider
[original text comparison](../qwen38_nvfp4_v100_docker_v4_20260908/README.md)
remains separate from this recheck. Consistent 120 tok/s is still not achieved.

[Comparison summary](summary.json), [main baseline responses](main_target_results.json),
[host MTP responses](host_mtp_results.json),
[Docker target-only responses](docker_mm_target_results.json),
[Docker MTP responses](docker_mm_mtp_results.json),
[checks after restart](docker_final_results.json),
[text-throughput measurements](docker_mm_mtp_perf_measurements.jsonl).

## Change and protocol

The initial Docker v4 image served images but returned HTTP 500 for video:
TorchCodec 0.9.1 could not load its FFmpeg shared libraries, then its fallback
failed because `decord` was absent. The v4 overlay and full V100 Dockerfile
now install FFmpeg. The overlay also imports `VideoDecoder` during the build
so missing decoder libraries fail the build. No model or inference-kernel
source was changed.

The rebuilt image is
`sha256:4373c93d636ff83fba0e06eb0c88c10edee07621d7ba505eef8f51d9c84601af`.
Its 3,083 tracked application files/symlinks match optimized source commit
`3b9bf5b0b043c8978b5a40cf3ffb113dc4724c2c`. It uses its own native dependencies
and JIT cache, with no host source or binary mount. The model remains
RadixArk/Qwen3.8-Flash-Next-NVFP4 snapshot
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

The suite uses four V100 SXM2 32 GB GPUs, TP4, FP16 activations, E5M2 KV,
262,144-token context capacity, 8,192-token prefill chunks and one running
request. MTP uses three draft steps and four verification tokens. Requests
use temperature zero, thinking disabled, and a 256-token output limit with
normal EOS termination. Every completed case must return HTTP 200 and finish
with `stop`. The streaming case also requires multiple SSE events and `[DONE]`.

The committed fixtures contain two images with swapped circle/square colors,
an OCR label, a bar chart and two six-second H.264 videos with reversed color
sequences. Expected answers are specified independently of model output in
[validate.py](validate.py). Swapped media use identical question text; neither
the question nor filenames reveal the answer. The fixtures are sent as data
URLs, avoiding remote media downloads or container file mounts.

| Case | Check |
| --- | --- |
| Image color | Red circle, blue square; identify the circle color |
| Image swap | Same question, blue circle/red square, without cache flush |
| Image repeat | Original image again without cache flush |
| Image follow-up | Ask about the square using the preceding image conversation |
| OCR | Transcribe `V100 CHECK 4827` exactly |
| Chart | Identify the tallest bar and value: `BETA 7` |
| Multiple images | Identify circle colors in image order: red, blue |
| Streaming | Describe both shapes, colors and left/right positions |
| Chunked prefill | Identify the circle after a 10,017-token multimodal prompt |
| Video order | Read red → green → blue from the decoded video |
| Reversed video | Read blue → green → red without flushing the cache |
| Text recovery | Answer `19 + 23` after the image/video requests |

The four comparison sweeps use the same fixture bytes and request payloads:
optimized host MTP, corrected Docker target-only, corrected Docker MTP and
an isolated checkout of pre-optimization `main` at
`72ef2f5f367a6c15eb21cf1b708ad3557b805305` in target-only mode. The baseline
uses the archived main Python source and launch script, with the host's
existing native SM70 binaries. The working checkout remains on the feature
branch during validation.

The host also lacked the FFmpeg libraries. For the comparison, Ubuntu
packages were downloaded and extracted under a private test directory, then
added to the host server's `LD_LIBRARY_PATH`. No system package installation
was needed. Exact versions and hashes are in
[host_ffmpeg_packages.json](host_ffmpeg_packages.json); Docker's FFmpeg
libraries are version `7:6.1.1-3ubuntu5`, listed in
[docker_ffmpeg_versions.txt](docker_ffmpeg_versions.txt).

## Evidence and reproduction

[Initial Docker failures](initial_docker/docker_mtp_results.json),
[build log](build.log), [image metadata](image_inspect.json),
[Docker source/dependencies](docker_environment.json),
[host source/dependencies](host_environment.json),
[native V100 smoke checks](smoke.log).

Build and serve using the [Docker v4 instructions](../../README.md#build-and-serve-the-optimized-qwen38-docker-v4-image).
Against an otherwise idle server on port 8082:

```bash
conda activate sglang-v100
python benchmark/qwen38_nvfp4_v100_multimodal_20260909/validate.py \
  --tag docker_mm_mtp --output-dir /tmp/qwen38_multimodal_recheck
```

The validator retains failed checks and exits nonzero if any ground-truth
answer fails. It refuses to overwrite a result. The committed fixtures are
the exact measured inputs. [make_fixtures.py](make_fixtures.py) documents how
they were generated; encoder versions can affect regenerated video bytes.

The text-throughput recheck uses the unchanged natural-prompt client from
the preceding performance report, with one discarded 256-output warmup and
two distinct 1,024-output requests at each of 1K and 25K input lengths:

```bash
export PYTHONPATH="$PWD/python"
python benchmark/qwen38_nvfp4_v100_mtp_20260908/bench_natural.py \
  --tag docker_mm_mtp_perf --lengths 1000 25000 --output-len 1024 \
  --output-dir /tmp/qwen38_multimodal_recheck
```

After collecting the four named multimodal sweeps and the throughput recheck,
recompute the baseline comparison with:

```bash
python benchmark/qwen38_nvfp4_v100_multimodal_20260909/summarize.py \
  --output-dir /tmp/qwen38_multimodal_recheck
```

This is a bounded image/video regression suite, not a broad model-quality
evaluation. The checkpoint has no audio configuration; audio was not tested.
Recorded request latency includes preprocessing and cold compilation where
applicable, and is not a steady-state multimodal throughput benchmark.
