# Qwen3.8 NVFP4 on V100: sustained 25K decode, 2026-09-07–08

The final candidate **clears 70 decode tok/s** in the measured 25K-input tests. Three 2K-output runs average **71.874–71.997 tok/s**; the slowest interval window is **71.824 tok/s**. An additional 8K-output run averages **71.833 tok/s**, with a minimum window of **71.711 tok/s**. Median 25K prefill is **4,876.3 tok/s**. Median sustained decode improves **13.9%** over the first optimization pass.

Measurements span September 7–8, 2026. These are host source measurements on four V100 SXM2 32 GB GPUs, TP4,
FP16 activations/Mamba state, NVFP4 experts, E5M2 KV cache, one request at a
time, CUDA graph batch size one. The context capacity remains 262,144 and
prefill chunks remain 8,192 tokens. No MTP, clock/power changes, or changes to
the installed Marlin binary were used.

## Sustained serving measurements

Each development point used one discarded 256-output-token warmup, then three
requests with **25,000 input and 2,048 output tokens**, seeds 20260830–20260832.
Inputs use the repository's random request sampler and exact token IDs;
generation is greedy with EOS ignored. Every request flushes the radix/Mamba
cache. Profiling and kernel benchmarks ran separately from throughput tests.

| Development point | Median prefill tok/s | Median decode tok/s | Slowest window tok/s |
| --- | ---: | ---: | ---: |
| First optimization pass | 4,872.1 | 63.206 | 63.092 |
| + Native dense GEMVs | 4,903.6 | 67.646 | 67.561 |
| + Shared-expert/GDN fusions | 4,922.1 | 69.662 | 69.558 |
| + Remove shared input copies | 4,906.1 | 70.036 | 69.897 |
| + Parallel QSA metadata / PLE GEMVs | 4,900.8 | 70.108 | 70.016 |
| Rejected HC block-size change | 4,928.9 | 70.070 | 69.924 |
| Stable router sort / PLE gate | 4,893.4 | 70.067 | 69.973 |
| Final: parallel QSA split merge | 4,876.3 | 71.977 | 71.824 |


The first row is the completed first optimization pass, not unmodified main.
The earlier [main comparison](../qwen38_nvfp4_v100_20260907/README.md) measured
60.278 tok/s at 25K input with **256** output tokens. That shorter protocol and
the README's historical Docker/MTP figures should not be mixed with these
sustained measurements. Prefill improvements remain small; the additional
work here mainly improves decode.

| Final seed | Prefill tok/s | Decode tok/s | Slowest window tok/s |
| --- | ---: | ---: | ---: |
| 20260830 | 4,876.3 | 71.997 | 71.968 |
| 20260831 | 4,851.0 | 71.977 | 71.960 |
| 20260832 | 4,893.7 | 71.874 | 71.824 |


Prefill throughput is input tokens divided by client TTFT. Decode excludes
the first token and is `(last_count - first_count) / (last_time - first_time)`.
Window rates use consecutive groups of 256 observed token intervals, with
the final 255-token interval group included. The saved token counters allow
readers to check stream batching and recompute these values. All final 2K- and 8K-output
runs delivered each token separately and reported zero cached input tokens. Results establish consistency for these
samples and this configuration; they are not a guarantee for every workload.

The additional **25,000-input / 8,192-output** request uses seed 20260832, which was the slowest seed in the earlier tests. It reaches approximately 33K total context, with **71.833 tok/s** overall decode and **71.711–71.947 tok/s** across all 32 windows, including the final partial window. Its prefill rate is 4,946.0 tok/s.

[Measurements and token timelines](measurements.jsonl), [summary](summary.json),
[final server configuration](qsa_merge_server_info.json) and
[source/binary hashes](provenance.json) are retained alongside this report.
These were sequential development comparisons, not randomized interleaved
trials or a statistical confidence interval.

## Same-protocol comparison with main

The final source was also rerun with the first report's protocol: one discarded
warmup then three cold requests at each input length, 256 greedy output tokens,
seed 20260830, and the same `bench_serving.py` driver. Each request ran in a
separate client process. The baseline values are the retained main measurements
from the first report; these are sequential measurements across the development
session. All nine final requests completed with exact input/output counts.

| Input tokens | Main prefill tok/s | Final prefill tok/s | Change | Main decode tok/s | Final decode tok/s | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 2,538.2 | 2,443.6 | -3.7% | 61.375 | 73.460 | +19.7% |
| 8,192 | 5,335.3 | 5,505.8 | +3.2% | 60.472 | 72.261 | +19.5% |
| 25,000 | 4,800.2 | 4,864.9 | +1.3% | 60.278 | 71.965 | +19.4% |

A separate five-run **1K recheck** measured a median prefill rate of
**2,425.1 tok/s** (range 2,209.4–2,448.7).
The final source does **not** establish a prefill improvement at 1K. The initial
comparison's median is 3.7% below main, while its 8K/25K medians are only 3.2%/1.3%
higher. Cold short-prompt TTFT is variable; the cause of the lower 1K result has
not been isolated. Decode gains are much larger and stable across these shapes.
[Short-request measurements](short_measurements.jsonl) retain all original and
recheck runs; [summary](short_summary.json) records the comparison. These results
do not demonstrate a 20% prefill gain.

## Implementation and profiling evidence

- **QSA split-attention merge:** replace serial maximum, sum and output
  loops over up to 160 splits with CUDA block reductions and 16-coordinate
  output tiles. The batch-one grid uses 96 CTAs rather than six. Preserve
  FP16 partial outputs, FP32 log-sum-exp weights, active-split masking and
  FP16 final output. Isolated batch-one latency falls from 23.69 to 4.84 us;
  batches two/four fall from 20.21/12.37 to 4.96/4.53 us. The in-model trace
  exposed this merge as roughly 34 us per attention layer before replacement;
  the final trace measures 5.80 us and confirms all 12 old merge calls are gone.
  [Profile aggregates](decode_profile_summary.json) retain the sampled kernels.
  Reduction order changes; independent FP64 references, the prior TileLang
  implementation, poisoned inactive buffers and changing-length graph replay
  validate the result. Set `SGLANG_SM70_QSA_COMBINE=0` before launch to use
  the previous combine kernel.
- **FP16 CUDA GEMVs:** aligned 128-bit loads, independent FP32 FMA accumulators,
  and tuned warp/subgroup assignments replace cuBLAS for the measured
  batch-one projection shapes. This covers attention, routers, shared experts,
  PLE projections and the TP-local 62,080-row LM head. Other shapes, dtypes,
  biases and layouts use the existing path. Representative isolated QKV
  latency fell from 32.55 to 26.98 us; the shared gate/up projection from
  6.21 to 4.02 us. The [shape sweep](gemv_tuning.jsonl) retains all variants. Additional measured
  candidates are recorded in [kernel experiments](kernel_experiments.json).
- **Shared-expert fusions:** combine gate/up projection with SiLU/multiply, and
  combine the scalar gate projection with sigmoid and output multiplication.
  Preserve the FP16 projection and sigmoid rounding boundaries. Isolated
  latencies fell from 6.90 to 4.10 us and 8.46 to 2.94 us respectively.
- **GDN projections:** one CUDA launch computes QKVZ and BA from their existing
  separate weights. Reuse the contiguous single-token Q/K/V prefix instead
  of concatenating it. Exact layer types, quantization methods, layouts and
  shapes guard this path so wrapped or transformed projections keep their
  original forward implementation.
- **Shared input lifetime:** the ModelOpt SM70 Marlin method declares that it
  preserves its input. Its specialized decode and Marlin fallback both
  allocate separate outputs, so shared experts can read that input on another
  stream without 48 redundant copies per token. Other methods retain copies.
- **QSA graph metadata:** distribute page-table tiles across GPU blocks for
  SM70 graph buckets of at most four rows. Only tile zero writes each row's
  scalar and pending-ring metadata. This preserves the entire capacity-sized
  table, including entries beyond the current sequence. An 8,192-page
  microbenchmark fell from 23.26 to 2.11 us; the serving configuration has
  16,384 pages at its 262K context capacity.
- **Router sort:** CUB's stable radix sort carries expert IDs as 16-bit
  values, so it sorts 16 logit bits rather than a 25-bit logit/ID key.
  Three six-bit passes preserve lower-ID ties, including signed zero.
  The isolated sweep improved from 4.77 to 4.27 us, with identical selected
  IDs and weights in its random/tie fixtures.
- **PLE FP16 gating:** enable the existing gate/value fusion for the measured
  SM70 FP16 path, preserving each half-precision rounding boundary. Precise
  FP32 exponential/division avoids two midpoint errors exposed by an
  exhaustive check of every finite FP16 gate value.

The preceding pass also optimized FP4 unpacking with Volta PRMT instructions,
CTA geometry, radix top-10 routing, prefill Marlin geometry and unused QSA
selection. Its [kernel/SASS evidence](../qwen38_nvfp4_v100_20260907/README.md)
is retained separately. No external Marlin source or binary was rebuilt.

Rejected experiments included wider MoE split-K (slower and changed rounding),
a single-CTA HC combine (slower), broadly compiling GEMV K as a constant
(shape-dependent regressions), and cooperative-grid HC fusion (only about
0.37 us saved per pair in its fixture, insufficient to justify the added
cooperative-launch constraint). A 512-thread HC down / 128-thread HC up
configuration improved isolated kernel latency but regressed serving, so it
was also reverted. Its 8K-output run fell to 69.83 tok/s in the slowest window.
These rejected changes are not enabled in the patch. A four-GPU all-reduce
sweep also retained the current one-shot push algorithm: it measured 3.35 us
versus at least 3.53 us for the tested pull configurations. The router/PLE
follow-up alone did not establish an additional serving gain; the table
retains that result rather than inferring one from its microbenchmarks.

## Correctness and reproduction

**63 targeted tests passed.** Coverage includes FP32 dequantized MoE references,
nonuniform scales, padded expert IDs, routing ties/signed zero, input
preservation, FP32 GEMV/HC references, alignment and dispatch guards, changing
input CUDA graph replay, QKV prefix aliasing, QSA routing/cache/MTP-seed guards,
page-table/ring metadata against independent arithmetic, and exhaustive
finite FP16 PLE gate values. QSA merge tests also compare against independent
FP64 references while poisoning inactive splits and changing lengths during
graph replay. The final sigmoid correction separately reran
both PLE tests successfully. These are 63 distinct passing cases across the
initial suite, corrected PLE rerun and new QSA merge tests; they were not all
rerun together after the final addition. [Validation logs](validation_results.json)
record the initial midpoint failure and successful corrected rerun.

Four deterministic response checks retain identical text for 4 of four prompts. Maximum common-prefix token log-probability difference is 0.137. [Response checks](response_checks.json) retain both texts. Whole-model numerics are not bitwise identical to main: FP16 reductions and attention accumulation order change. No perplexity or task-accuracy evaluation, MTP throughput test, or concurrency-above-one test was performed.

The reference launch script enables `SGLANG_SM70_DENSE_GEMV=1` and
`SGLANG_SM70_QWEN_FUSIONS=1` by default. Each can be set to zero before startup
for an A/B check; CUDA graphs capture that choice. Outside this model's launch
script these two paths remain opt-in. `SGLANG_SM70_HC_NATIVE=0` selects the
prior Triton HC path. Disabling these flags does not reconstruct the earlier
source snapshots, because the copy and metadata changes remain.

```bash
export PATH=/home/rah/miniconda3/envs/sglang-v100/bin:$PATH
export PYTHONPATH="$PWD/python"
export CUDA_HOME=/usr/local/cuda-12.8
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1
PORT=8082 bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh
```

From another shell with the same Python environment, after the server is ready:

```bash
PYTHONPATH=python python benchmark/qwen38_nvfp4_v100_70tps_20260907/bench_sustained.py \
  --tag candidate --output /tmp/qwen38_25k_2048.jsonl
PYTHONPATH=python python benchmark/qwen38_nvfp4_v100_70tps_20260907/bench_sustained.py \
  --tag candidate_8k --seed 20260832 --output-len 8192 --repeats 1 \
  --output /tmp/qwen38_25k_8192.jsonl
```

On an idle V100, the targeted checks can be reproduced with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python FLASHINFER_DISABLE_VERSION_CHECK=1 \
  python -m pytest -q \
  test/registered/unit/layers/moe/test_sm70_nvfp4_moe_decode.py \
  test/registered/unit/layers/attention/test_qwen_sparse_sm70_routing.py \
  test/manual/layers/test_sm70_hc_mix.py \
  test/manual/layers/test_sm70_dense_gemv.py \
  test/manual/layers/test_sm70_qwen_fusions.py \
  test/manual/layers/test_qsa_graph_metadata_tiles.py \
  test/manual/layers/test_sm70_ple_gate.py \
  test/manual/layers/test_sm70_qsa_combine.py
```

Use fresh output paths: the driver appends records. It reads actual HTTP chunks;
an early discarded experiment read one byte at a time, saturated the client
CPU on cumulative SSE text, and falsely reported falling throughput. Those
invalid measurements are excluded. Full exploratory sources, traces and logs
remain in `/tmp/qwen38_70_20260907`; the first-pass source snapshot is in its
`pass1_source` subdirectory. The current branch is
`perf/qwen38-nvfp4-v100-prefill-decode`, based on main `72ef2f5f367a`.
The original feature branch remains parked at `975262bbf396`.
The changes are local and have not been built or published in a Docker image.
