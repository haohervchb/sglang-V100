# Qwen3.8 Flash Next NVFP4 on V100

[Back to the main README](../../../README.md) · [Foreground shell launcher](../../../scripts/serve_qwen38.sh)

The default is the balanced MTP-3/4 profile: four V100 32 GB GPUs, NVLink,
FP16 KV, eight request slots, 4K prefill chunks, 262K context, shared-prefix
checkpoints and image/video support. The three experimental kernel switches,
wider MTP and adaptive speculation are off. The serving arguments are kept in
[shell scripts](../../../scripts/serve_qwen38_flash_next_nvfp4_v100_throughput.sh).

## Start Docker

With Docker and the NVIDIA Container Toolkit installed, run these commands in
any directory. No repository checkout is required:

```bash
export QWEN38_IMAGE=geesegeesegeese/sglang-v100:latest
docker pull "$QWEN38_IMAGE"
docker run --rm --entrypoint cat "$QWEN38_IMAGE" /usr/local/bin/qwen38.sh > qwen38.sh
bash qwen38.sh
```

This starts `geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v5.2` in the
background, using container name `qwen38` and port 8082. Docker pulls the image
if needed. The checkpoint revision is pinned to
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`; weights download into a writable
host cache when missing. The image contains the runtime and CUDA build tools.

```bash
bash qwen38.sh logs
bash qwen38.sh status
curl --fail http://127.0.0.1:8082/health
```

Use `http://127.0.0.1:8082/v1` and model name `qwen` in your client. An active
container may still be loading; HTTP 200 from `/health` means it is ready.
Startup and restarts are manual. The container does not restart after failure,
a Docker-daemon restart or a reboot. Starting or restarting an existing managed
container also clears any older automatic restart policy.

```bash
bash qwen38.sh stop
bash qwen38.sh start
```

To use a different port or model-cache directory:

```bash
PORT=8083 HF_HOME=/srv/huggingface bash qwen38.sh
```

`HF_HOME` defaults to `~/.cache/huggingface`. The named volume
`qwen38-v5-jit` keeps compiled kernels separate from older images. Model files
and compiled kernels survive stop/recreate operations. First startup needs
space for approximately 126 GiB of model weights plus the image and JIT cache.
A populated model cache avoids downloading them again. `HF_TOKEN` and
`HF_HUB_OFFLINE`, when set, are passed through to the container.

The launcher refuses to start a new container when the selected port is
already occupied. Stop an existing foreground host server with Ctrl-C, or an
existing Docker server with `bash qwen38.sh stop`, before reusing the GPUs.
Use one serving instance per group of four GPUs. Startup remains manual.

## Update or build the image

```bash
bash qwen38.sh pull
bash qwen38.sh recreate
```

`recreate` applies the selected image and current settings to the managed
container, preserving both caches. `start` reuses an existing container and its
saved configuration. Set `QWEN38_IMAGE` to select another image tag, or
`QWEN38_CONTAINER` to select another managed container name.

Building requires the source checkout. Docker-only users can use `pull` and
`recreate` above instead:

```bash
bash scripts/qwen38.sh build
bash scripts/qwen38.sh recreate
```

The [v5 Dockerfile](../../../docker/v100-qwen38-flash-next-v5.Dockerfile)
reuses the digest-pinned v4 SM70 stack and adds the current serving source,
patched grouped-QSA and native vision libraries. Those libraries are compiled
inside the image against its PyTorch ABI; host `.so` files are excluded.
The vision build needs no GPU. Its runtime is checked on V100 after building.
The build retains dependency sources, licenses and metadata under `/opt/deps`.

The [v5 container validation](../../../benchmark/qwen38_docker_v5_20261105/README.md)
passed full-context, tool, image, shared-document and 48-request queue checks.
With weights already cached and a fresh JIT volume, first readiness took about
19 minutes on the test host. Later starts reuse the compiled kernels.
The [v5.1 launcher checks](../../../benchmark/qwen38_launcher_20261109/README.md)
verify image-only extraction and operation from an empty directory, with the
same runtime files and serving arguments as v5.
[v5.2](../../../benchmark/qwen38_v52_20261111/README.md) adds the decode kernel
bundle and 4K prefill: prose 114 -> 142 tok/s at one request and 359 -> 504 at
eight, and 11-19% shorter time to first token from 8K to 160K prompts.

For a direct Docker invocation, the image's default command already selects
the balanced profile. The host wrapper supplies GPU access, shared memory,
cache mounts and lifecycle options. Explicit commands such as
`docker run --rm IMAGE python -V` remain available for diagnostics, and the
existing generic `sglang serve` argument entrypoint is retained.

## Run on the host

After installing the V100 environment and optional grouped-QSA/vision libraries,
run from that prepared source checkout:

```bash
bash scripts/serve_qwen38.sh
```

The script finds a serving Python in the prepared environment, or uses
`SGLANG_PYTHON` when explicitly set. It runs in the foreground and Ctrl-C stops
it. The image also contains this command as `serve-qwen38`, which is its default
command. No systemd installation or boot startup is part of this workflow.
`bash scripts/qwen38.sh host` remains a compatible shortcut.

Dependency build instructions are in the
[third-party notice](../../../python/sglang/jit_kernel/third_party/flash_v100/README.md).
Docker already includes those dependencies.

Advanced SGLang arguments can follow the foreground script or Docker `start`/`recreate`:

```bash
bash scripts/serve_qwen38.sh --max-running-requests 4 --cuda-graph-bs 1 2 3 4
```

The 262K limit applies to each request; concurrent agents share a 277,280-token
KV pool. Prefill runs in 4K chunks; a prompt's last long chunk ends on a 2K
boundary (`SGLANG_CHUNKED_PREFILL_CACHE_ALIGN=2048`) so prefix checkpoints land
where 2K chunks put them, with about 2.6 GB left for temporary activations. Pixel preprocessing
stays on CPU to avoid competing with TP0 for GPU memory. Document extraction
still needs to produce input within the context limit.

For the older target-only single-request profile, use
`PORT=8082 bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh` (its own default
port is 30000). For the earlier
single-request MTP profile, use
`bash scripts/serve_qwen38_flash_next_nvfp4_v100_mtp.sh`.

## Measurements and earlier configurations

- [Balanced eight-agent profile and GPU core audit](../../../benchmark/qwen38_concurrent_core_audit_20261104/README.md)
- [Vision performance and multimodal checks](../../../benchmark/qwen38_vision_v100_20261104/README.md)
- [C1–C4 experiments and rejected wider/adaptive defaults](../../../benchmark/qwen38_c14_strata_20261104/README.md)
- [v4 Docker/host comparison](../../../benchmark/qwen38_nvfp4_v100_docker_v4_20260908/README.md)
- [v4 two-full-context deployment](../../../benchmark/qwen38_nvfp4_v100_full_context_mtp_concurrency_20260925/README.md)
- [v4 empty-cache startup](../../../benchmark/qwen38_docker_clean_start_20260910/README.md), measured on a 256 GiB RAM host
- [Community 8× V100 agent deployment](../../../benchmark/qwen38_nvfp4_v100_opencode_field_report_20260927/README.md)

The v4 measurements describe that historical image and configuration. They
are not a new v5 throughput claim.

## Hosts without NVLink (PCIe P2P)

With a source build containing this change, on four V100 SXM2 32 GB behind one PCIe switch (P2P works, no NVLink), replace
`NCCL_P2P_LEVEL=NVL` with `NCCL_P2P_LEVEL=PXB`, otherwise NCCL goes through
host memory. SGLang disables its custom all-reduce on more than two PCIe-only
GPUs and uses NCCL, which costs about 20 us per decode all-reduce. Setting
`SGLANG_CUSTOM_AR_ALLOW_PCIE=1` opts the v2 communicator back in: only the one-shot push kernel is
used (the contributor observed pull kernels hanging on that PCIe topology) and only for messages up to
`SGLANG_CUSTOM_AR_PCIE_MAX_BYTES` (default 128 KiB, the crossover with NCCL);
larger all-reduces still use NCCL. Actual NVLink hosts retain their existing
algorithm and size limits even when this opt-in is set. The legacy communicator
does not opt in; P2P and single-node checks remain required. The byte cap must
be a positive multiple of 16. The v4 image predates this change; v5 includes the source support. The default profile targets NVLink.

Contributor-reported measurements using a patched v4 runtime and the same command
(MTP-3/4, client-side, 4x V100 SXM2 on one PLX switch):

| Configuration | 1K decode | 6K decode | 25K decode | 4 requests, 1K in / 512 out |
| --- | ---: | ---: | ---: | ---: |
| NCCL over PCIe P2P | 92-97 tok/s | 98 tok/s | 94 tok/s | 145 tok/s |
| `SGLANG_CUSTOM_AR_ALLOW_PCIE=1` | 106 tok/s | 119 tok/s | 109 tok/s | 161 tok/s |

The contributor reports eight concurrent requests unchanged (234 vs 236 tok/s): their all-reduces
exceed the byte cap. The push kernel output is bit-identical to NCCL on integer
inputs from 4 KB to 512 KB, eager and inside CUDA graphs. These results come
from [PR #2](https://github.com/haohervchb/sglang-V100/pull/2), not a new PCIe
benchmark on the maintainer's NVLink host.

## Performance history

The [September 7–8 host source optimization](../../../benchmark/qwen38_nvfp4_v100_70tps_20260907/README.md)
reaches **4,876 prefill tok/s** and **71.87–72.00 decode tok/s**
with 25K inputs and three 2K-output runs. An 8K-output run averages
**71.83 decode tok/s** (slowest interval window: **71.71**).
These target-only source results are separate from the Docker rows in the
[main performance table](../../../README.md#current-v100-performance).
The [first-pass main comparison](../../../benchmark/qwen38_nvfp4_v100_20260907/README.md)
records the earlier 1K/8K/25K prefill and decode results. The later target-only
and MTP changes are packaged in the published v4 image, with
[September 8–9 Docker/host validation](../../../benchmark/qwen38_nvfp4_v100_docker_v4_20260908/README.md).
The registry's published v3 image retains the earlier source.

## Historical Qwen3.8 Flash Next Docker v2 concurrency benchmark

The fixed v2 image completed every requested load point with no request errors.
These rows are retained as benchmark provenance; use the current launcher
above for the balanced deployment. Output throughput is aggregate across each exact
8,192-input/1,024-output workload:

| Concurrency | Target only | MTP-3/4 | MTP delta |
| ---: | ---: | ---: | ---: |
| 1 | 55.12 tok/s | 74.82 tok/s | +35.7% |
| 4 | 137.26 tok/s | 149.25 tok/s | +8.7% |
| 8 | 172.98 tok/s | 194.13 tok/s | +12.2% |
| 16 | 239.36 tok/s | 183.87 tok/s | -23.2% |

For the cold 1,000-input/25,000-output request at concurrency 1, target-only
reached 60.09 output tok/s and MTP reached 88.07 output tok/s (+46.6%). See the
[full fixed-image report](../../../benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md)
for TTFT, TPOT, acceptance length, and memory sizing.
