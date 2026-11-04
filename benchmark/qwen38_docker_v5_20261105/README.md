# Balanced Flash-Next Docker v5

Docker v5 packages the balanced native MTP-3/4 profile used by the office host.
The standard start command is now `bash scripts/qwen38.sh`; the image itself
also defaults to this profile. Model/kernel arguments are in the serving
scripts. The launcher manages GPU access, cache volumes and container lifecycle.

The profile retains TP4 on four V100 SXM2 32 GB GPUs, FP16 KV, eight request
slots, 2K prefill, 277,280 shared KV tokens, 48 Mamba slots and 262K context.
CPU image preprocessing and the native vision backend are enabled. Wider MTP,
adaptive speculation and the three C1–C4 experimental flags remain off.

## Packaging

The image inherits the digest-pinned v4 CUDA 12.8 / PyTorch 2.9.1 SM70 runtime.
Its TurboMind binary matches the host. Patched grouped-QSA and native vision
libraries are built inside the image from pinned, attributed sources; host
shared libraries are excluded. The additional vision setup patch only permits
building without a visible GPU. It changes no CUDA kernel arithmetic.

The generic Docker entrypoint still accepts explicit commands and `sglang serve`
arguments. The default command selects the balanced profile. Model weights are
external to the image, and model/JIT caches persist across recreate operations.

Runtime source revision tested: `7e280c7ce3e99efd6d98882f4aba378c3ec7f052`.
Image tested: `sha256:d51f28aaf8aa424d8f688f942ef663dad2b7622dcb08367190f3de25488843c1`.
Pytest was added in a separate test-only image; its runtime layers are identical
to the production image. GPU tests mounted only the test files, with no host
runtime source or native binaries. The published release is checked to retain
these exact runtime filesystem layers when updating its source-revision label
to include the evidence commit.

## Validation

- 18 grouped-QSA/vision GPU tests passed, including masked
  unused KV pages, graph replay, ragged images and 72-wide head scaling.
- All text checks passed, including 262K retrieval and oversized-input recovery.
- Tool calls and retained shared-document prefix reuse passed, with at least 30,720 cached tokens per answer.
- Multimodal: 11/12, retaining the same known OCR miss.
- The 32-image request passed in 13.60 seconds.
- All 48/48 growing-document requests passed in
  266.69 seconds; GPU 0 retained at least 2720.7 MiB free.
- Host port 8082 was restored afterward. Parallel text, streaming and 32 images
  passed; final health was HTTP 200.

The container used an already populated model cache and the JIT-volume state
recorded in `jit_volume_before.json` (the named volume did not yet exist).
Logged readiness took about 19 minutes, including cached-weight loading and
first-use kernel compilation. Later starts reuse the compiled kernels.
This did not repeat the v4 test that
downloaded every model shard into an empty host cache.

## Throughput control

Cold 1K-input / 1,024-output-token prose, eight requests per batch. Host uses one
batch per concurrency; Docker uses two, reversing concurrency order in the
second pass. Rates include prefill and queueing. This sequential comparison is
a packaging regression check, not a claim of a new optimization.

| Clients | Host output tok/s | Docker output tok/s | Difference |
| --- | ---: | ---: | ---: |
| C1 | 115.4 | 112.9 | -2.1% |
| C4 | 255.6 | 250.4 | -2.0% |
| C8 | 360.9 | 356.5 | -1.2% |

The 20-case natural-EOS SPEED screen measured
306.8 aggregate output tok/s at C8.
Output lengths vary in that screen. See the raw request records for token
counts, acceptance, TTFT, output hashes and memory. Generated output prose is
omitted from the public archive.

[summary.json](summary.json) and [metadata.json](metadata.json) retain the
measurements, source/build hashes, configurations and validation logs. The
original run directory is `/home/rah/qwen38-runs/20261105-default-docker-075702`.
