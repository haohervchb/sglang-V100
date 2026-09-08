# Qwen3.8 NVFP4: Docker v4 versus the optimized host

The Docker and fresh host benchmarks are complete, measured September 8–9,
2026. **Docker target-only decode matches the host within 0.03%; prose/code
MTP per-context mean decode rates match within 2.9%.** Estimated MTP round
times differ by at most 0.7% across both prompt suites. No runtime changes
were needed after building the image.

The original symmetric 5% parity check has one exception: Docker's 1K
natural-prompt prefill is **10.7% faster**. Its cause is not isolated. All other
predeclared comparison metrics pass the 5% check. The separate no-regression
assessment passes; this is not an assertion that every raw throughput result
matches. Random-token decode averages range from 7.0% lower to 13.6% higher
in Docker as acceptance varies; those measurements are included below.

The image packages application source commit
`3b9bf5b0b043c8978b5a40cf3ffb113dc4724c2c`, containing both target-only and MTP
optimizations. It retains the native SM70 dependencies from Docker v3 and
compiles the new CUDA/Triton kernels inside the container. The 3,083 tracked
application files and symlinks match the host checkout. There is no host
source or binary mount. The model cache is mounted read-only and Docker has
its own persistent JIT volume.

Image: `sglang-v100:v100-qwen38-flash-next-v4` (local build).

Image ID: `sha256:aeccb5c5e485f80f41d9952c97c44ecc2edd2bf752f61ced1d198d58e205d830`.

The same image is also tagged `sglang-v100:latest`. The local container
`qwen38-flash-next-mtp-v4` is running at `http://127.0.0.1:8082/v1` after a
restart and four further response checks. This image has not been pushed to
a registry. [Service record](service_final.json),
[image metadata](image_inspect.json), [build log](build.log).

## Measured performance

Each table entry is the arithmetic mean of two different requests. MTP uses
1,024 output tokens and target-only uses 2,048. Full request rates, acceptance
counts and window minima appear in [the comparison summary](parity_summary.json).

| Natural MTP input | Host prefill tok/s | Docker prefill tok/s | Host decode tok/s | Docker decode tok/s | Decode change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 2,952 | 3,269 | 116.81 | 117.55 | +0.64% |
| 8,192 | 5,439 | 5,422 | 117.49 | 120.85 | +2.86% |
| 25,000 | 4,729 | 4,693 | 117.86 | 119.84 | +1.68% |
| 70,000 | 4,630 | 4,611 | 118.89 | 118.11 | -0.66% |

Docker's individual natural requests span **113.99–125.68 decode tok/s**.
The lowest Docker natural window is 107.31 tok/s. The host's range is
112.40–124.02, with a 105.59 tok/s minimum window. This deployment preserves
the host performance level; **consistent 120 tok/s is still not achieved**.

| Target-only input | Host prefill tok/s | Docker prefill tok/s | Host decode tok/s | Docker decode tok/s | Decode change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 2,784 | 2,704 | 74.284 | 74.265 | -0.027% |
| 25,000 | 4,889 | 4,872 | 73.555 | 73.548 | -0.010% |

Docker's 25K target-only runs individually reach 73.556 and 73.539 tok/s;
their minimum windows are 73.542 and 73.510. Its prefill is 0.4–2.9% lower
than the corresponding host means.

| Random MTP input | Host decode tok/s | Docker decode tok/s | Host accepted tokens/round | Docker accepted tokens/round | Host round ms | Docker round ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 129.66 | 129.59 | 2.791 | 2.791 | 21.533 | 21.536 |
| 8,192 | 149.20 | 138.80 | 3.246 | 3.011 | 21.750 | 21.690 |
| 25,000 | 141.06 | 160.22 | 3.072 | 3.501 | 21.773 | 21.854 |
| 70,000 | 128.23 | 140.58 | 2.822 | 3.114 | 22.008 | 22.153 |

At 8K random input, Docker's output throughput is 7.0% lower while its round
time is 0.3% lower; the accepted-token counts explain the throughput
difference arithmetically. This does not establish its underlying cause.
The earlier host work already observed varying long-context text/acceptance
between unchanged configurations. All random requests are retained, including
the slowest host 70K request (109.20 tok/s; minimum window 61.60).

[Docker natural requests](docker_mtp_natural_measurements.jsonl),
[host natural requests](host_mtp_natural_measurements.jsonl),
[Docker random requests](docker_mtp_random_measurements.jsonl),
[host random requests](host_mtp_random_measurements.jsonl),
[Docker target-only requests](docker_target_random_measurements.jsonl),
[host target-only requests](host_target_random_measurements.jsonl).

## Protocol

Both runtimes use four V100 SXM2 32 GB GPUs, TP4, NVFP4 experts, FP16
activations/recurrent state, E5M2 KV cache, a 262,144-token context capacity,
8,192-token prefill chunks, one running request, graph batch size one, and
memory fraction 0.80. MTP uses three draft steps and four verification tokens
from the same checkpoint. The launch script is baked into the image and is
also used on the host.

The unchanged clients from the [host report](../qwen38_nvfp4_v100_mtp_20260908/README.md)
run sequentially against both runtimes. At each input length there is one
discarded 256-output warmup followed by two measured requests using different
prompts. MTP covers 1K, 8K, 25K and 70K inputs with 1,024 output tokens, on both
random tokens and prose/code tutorials. Target-only covers 1K and 25K inputs
with 2,048 output tokens. All requests are greedy, ignore EOS, flush the
cache, and verify exact token counts and zero cached input tokens.

Prefill is input length divided by client time to first token. Decode excludes
the first observed streamed token batch. Window rates use at least 256 tokens
plus the final partial window. Estimated MTP round time is mean accepted
tokens per round divided by client decode throughput.

The comparison checks a 5% difference between per-context mean prefill/decode
rates on the natural and target-only workloads, and between estimated MTP
round times. All random-token output rates are reported; they are not used
alone to establish runtime parity because draft acceptance varies between
unchanged runs. This is a finite sequential comparison, not a statistical
confidence interval or a throughput guarantee for every prompt. Four short
responses with log probabilities run before and after each sweep and are
compared with the selected host candidate.

The server-generated `random_seed` differs between starts; the requests are
greedy and the client uses the same two prompt seeds in both runtimes. All
other server-info settings match, apart from package version labels and
runtime status/statistics. No GPU clocks or power limits were changed.

## Validation and provenance

- The image passes the existing V100 smoke script, covering native SM70
  imports, Marlin repack, sampling, QSA routing/prefill and NCCL 2.27.5.
- All **32 short-response checks** across the four sweeps match the saved
  optimized host's text, token IDs and token log probabilities exactly.
  Four more checks pass after restarting the final serving container.
- All 40 measured long requests and 20 warmups complete with exact token
  counts and zero cached input tokens. The summary recomputes decode rates
  from raw stream timestamps and counters.
- All 3,083 tracked application files/symlinks match the host. Docker uses
  the same Torch 2.9.1/CUDA 12.8, Triton 3.5.1, TileLang 0.1.8, TVM FFI 0.1.9,
  Transformers 5.8.1 and NCCL 2.27.5 versions. Native Marlin binary hashes
  differ because Docker retains its independently built v3 binaries;
  no host binary is copied or mounted into the image.
- The model is the cached RadixArk snapshot
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`, identical in both runtimes.
  The previously completed host kernel suite passed 123 targeted tests;
  this Docker validation uses real serving workloads plus the smoke checks.

The first target-only checker stopped after one correct short answer when
`/flush_cache` raced the scheduler's final request cleanup and returned 400.
The checker now supplies `timeout=30` to wait for an idle cache. The original
attempt is retained in [its log](interrupted_target_check/docker_target_validation.log);
the subsequent complete run passes before/after response checks. No image
or inference-runtime change was needed for this client sequencing fix.

[Response summary](response_summary.json), [smoke log](smoke.log),
[original comparison criteria](comparison_criteria.json),
[host environment](host_environment.json), [Docker environment](docker_environment.json),
[model provenance](model_provenance.json), [Docker mounts](docker_mounts.json).
Full server logs are retained as gzip files for the
[Docker MTP](docker_mtp_server.log.gz), [host MTP](host_mtp_server.log.gz),
[Docker target](docker_target_server.log.gz) and [host target](host_target_server.log.gz) runs.
Large JSON files use compact serialization and text logs have trailing
whitespace removed; recorded values are unchanged. The report covers one
request at a time, not concurrency-four behavior or a broad model quality suite.

## Reproduction

Build from the repository root:

```bash
docker build --network=host \
  --build-arg SGLANG_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -f docker/v100-qwen38-flash-next-v4.Dockerfile \
  -t sglang-v100:v100-qwen38-flash-next-v4 .
```

Use the `sglang-v100` host environment for the benchmark client. Each command
starts a server, runs the checks and benchmarks, then stops that server. Run
the commands sequentially on otherwise idle GPUs. The Docker image needs no
source mount; only the host client imports the checkout. Choose fresh tags
and container names for subsequent runs because evidence is never overwritten.

```bash
conda activate sglang-v100
export PYTHONPATH="$PWD/python"
python benchmark/qwen38_nvfp4_v100_docker_v4_20260908/validate.py \
  --runtime docker --mode mtp --tag docker_mtp \
  --output-dir /tmp/qwen38_docker_parity
python benchmark/qwen38_nvfp4_v100_docker_v4_20260908/validate.py \
  --runtime docker --mode target --tag docker_target \
  --output-dir /tmp/qwen38_docker_parity
python benchmark/qwen38_nvfp4_v100_docker_v4_20260908/validate.py \
  --runtime host --mode mtp --tag host_mtp \
  --output-dir /tmp/qwen38_docker_parity
python benchmark/qwen38_nvfp4_v100_docker_v4_20260908/validate.py \
  --runtime host --mode target --tag host_target \
  --output-dir /tmp/qwen38_docker_parity
python benchmark/qwen38_nvfp4_v100_docker_v4_20260908/summarize.py \
  --output-dir /tmp/qwen38_docker_parity
```
