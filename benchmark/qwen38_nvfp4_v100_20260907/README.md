# Qwen3.8 Flash Next NVFP4: Volta kernel optimization, 2026-09-07

This report records the **first optimization pass**. The current branch also
includes a [sustained 25K decode follow-up](../qwen38_nvfp4_v100_70tps_20260907/README.md).
Its additional changes and measurements are documented there. The first-pass
source snapshot is retained at `/tmp/qwen38_70_20260907/pass1_source` on the
measurement host; the current working tree is newer than this report's candidate.

The first pass improved measured target-only prefill throughput by **2.2–4.2%**
and decode throughput by **4.8–5.2%** over current `main` on four V100s.
The reported 20% deficit against the root README was not reproduced with the
reference host launch: the unmodified baseline already reached 61.4 decode
tok/s at 1K input. The README's historical target-only figure is 60.2 tok/s,
measured with a different Docker image and a 25,000-token output request.
These results do not claim to close a reproduced 20% regression.

## Controlled serving comparison

One request at a time, exact token-ID input, 256 greedy output tokens with EOS
ignored, seed 20260830. Each shape had one discarded warmup followed by three
measurements; every request flushed the radix/Mamba cache. Values below are
medians. Prefill = input tokens / client TTFT; decode = 1000 / mean TPOT in ms.
These exclude neither scheduling nor HTTP time from TTFT. No profiler or other
GPU workload ran concurrently with the measured requests.

| Input tokens | Main prefill tok/s | Candidate prefill tok/s | Gain | Main decode tok/s | Candidate decode tok/s | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 2,538.2 | 2,645.2 | 4.2% | 61.375 | 64.565 | 5.2% |
| 8,192 | 5,335.3 | 5,512.7 | 3.3% | 60.472 | 63.412 | 4.9% |
| 25,000 | 4,800.2 | 4,907.5 | 2.2% | 60.278 | 63.189 | 4.8% |

All 18 measured requests completed with the requested token counts.
The baseline 1K prefill range was 2,416–2,638 tok/s, versus 2,622–2,649 for the
candidate; that small prefill improvement overlaps individual-run variation.
The decode ranges do not overlap. This is a sequential before/after experiment,
not randomized interleaved trials or a statistical confidence bound.

[Measurements](measurements.jsonl) retain the benchmark's latency and token
fields plus the derived rates. [Server configuration](server_config.json)
retains each process's reported configuration. The original full client JSONL,
server logs, traces, assembly and exploratory scripts remain in
`/tmp/qwen38_nvfp4_perf_20260907` on the measurement host.

## Changes and evidence

### Native CUDA hyperconnection projections

Qwen3.8 runs 97 hyperconnection mixes per forward. For batch-one FP16
`10240 -> 320 -> 10240 -> 2560`, the new down kernel uses aligned 128-bit loads
and eight independent FP32 accumulators per thread. The up kernel assigns a
hidden coordinate to each warp, with four eight-lane groups evaluating the HC
branches concurrently. Eight warps share a CTA, removing the old single-warp
CTA residency limit. SiLU and sigmoid/multiply/mean remain fused, and FP16
projection boundaries are preserved. The accumulation order changes slightly.

The native path is limited to the existing exact SM70 shape guard. Other
shapes keep their existing implementations. `SGLANG_SM70_HC_NATIVE=0` selects
the prior Triton path for comparisons; set it before starting the server so
CUDA graphs capture the intended implementation.

### NVFP4 expert decode and routing

The TP4 local expert width is only 160, with ten routes per token. Native
half2 kernels avoid padding each route into mostly empty tensor-core tiles.
The changes are in `python/sglang/jit_kernel/csrc/sm70_nvfp4_moe_decode.cu`:

- Reduce CTA size from 256 to 64 threads. At batch one the gate/up grid grows
  from 63 CTAs to 250, allowing work on all 80 SMs. The existing independent
  work items and reduction order are preserved.
- Expand packed FP4 even/odd nibbles once and use CUDA `__byte_perm` / Volta
  `PRMT` instructions to form four half2 values. In the isolated CUDA 12.8
  cubin comparison, gate/up static SASS instruction count falls from 1,816 to
  1,560; down falls from 992 to 856. This counts padding instructions too and
  is an instruction-footprint comparison, not a runtime speedup estimate.
  The selected cubin reports no register spills.
- For FP16 routing logits, replace ten dependent hierarchical argmax/merge
  iterations with a CUB block radix sort. A 25-bit key combines ordered FP16
  bits with expert ID; five-bit passes preserve lower-ID tie breaking,
  including signed zero. Softmax remains FP32. FP32 logits keep the old path.

Isolated graph replay of the four expert kernels improved from 33.23 to 30.87
microseconds at batch one, 49.71 to 45.45 at batch two, and 79.75 to 75.24 at
batch four. All tested scheduling/dequantization variants matched the old
kernel bit for bit. [Variant measurements](moe_tuning.jsonl) include all
retained comparisons. The final Python-entry-point microbenchmark uses a
separate fixture; its absolute times should not be mixed with this cubin A/B.

### Prefill GEMM geometry and unnecessary QSA work

A CTA/warp-geometry sweep of the installed C++ Marlin kernels found two useful
shape-specific choices, now selected by the existing SM70 dispatch mechanism:

| Stage | Shape | CTA geometry |
| --- | --- | --- |
| 1K gate/up | N=320, K=2560, route block=32 | `32x64x64x4x32x32x32` |
| Large-prefill down | N=2560, K=160, route block=64 | `64x256x32x4x64x64x32` |

Both use split-K 1. The 1K gate/up candidate reduced the synthetic kernel time
by about 13%; the 8K down candidate by about 15%. The 8K gate/up sweep found no
better geometry, so that stage retains its selector. User-supplied Marlin
geometry settings still take precedence. [Sweep data](marlin_tuning.jsonl)
includes numerical error against the default kernel: the selected 1K gate/up
changes rounding slightly; the selected 8K down result is bitwise equal.
No external Marlin source or binary was changed during this work.

The existing bounded dense QSA prefill route now also handles short prompts.
Below the selection budget the attention is already full causal attention.
For longer prompts the launch script already enabled dense attention through
8,192 tokens; this branch does not extend that configured limit. When dense
attention consumes the result, the compressed QSA indexer skips unused score,
top-k and expanded-index work, while still updating pending/compressed keys
for subsequent decode and chunked prefill. CPU scheduling metadata controls
both routes without GPU synchronization. Prefix-bearing chunks, speculative
verification, and non-SM70 hardware retain their existing selection paths.
MTP draft prefill also retains selection when it must seed the next decode.
`SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=0` keeps sparse attention everywhere.

## In-model profiler attribution

The separate TP0 8K trace contains one prefill and four decode steps. These
kernel means are diagnostic; CUPTI/CPU tracing perturbs timings and cache
behavior, so only the unprofiled serving table supports throughput claims.

| Decode kernel | Calls per token | Baseline mean us | Candidate mean us |
| --- | ---: | ---: | ---: |
| Router top-10 + softmax | 48 | 15.979 | 7.067 |
| NVFP4 gate/up partial | 48 | 22.955 | 21.706 |
| NVFP4 down partial | 48 | 15.383 | 14.172 |
| HC down + SiLU | 97 | 12.506 | 12.225 |
| HC up + sigmoid/mix | 97 | 12.515 | 10.876 |

The 48 prefill down GEMMs fall from 228.61 to 198.12 ms summed GPU time.
Gate/up remains about 333–336 ms. Full aggregates are in
[profile_kernels.json](profile_kernels.json).

Remaining opportunities are mainly outside packed FP4 arithmetic: the baseline
profile spends about 2.46 ms per decode token in 97 FP16-output cuBLAS GEMVs,
plus about 1.44 ms in FP32-output GEMVs and their reductions. Prefill still
spends about 71 ms in HC combine, 49 ms in sigmoid/multiply/mean, and 42 ms in
group normalization. Fusing these operations with adjacent projections and
improving dense GEMV reuse are concrete next experiments. They are unmeasured
opportunities, not additional gains claimed by this patch. The measured router
reduction explains more of this patch's decode gain than FP4 unpacking alone.

## Correctness and limits

The final targeted suite passed **34 tests** on V100, including independent
FP32 dequantized MoE references with nonuniform scales and invalid expert
padding, top-k ties/signed zero, HC FP32 references and changing-input CUDA
graph replay, unaligned-input fallback, and QSA routing/cache-update/MTP seed tests.

Four deterministic 128-token response checks retained identical generated text
for three prompts. The fourth shared 89 tokens before diverging. Maximum
common-prefix token log-probability difference was 0.116. FP16 reduction and
attention accumulation order can change; this is not bitwise whole-model
parity. [Response checks](response_checks.json) retain both texts. No perplexity
or task-accuracy evaluation was performed. MTP end-to-end throughput,
concurrency above one, and contexts beyond 25K input were not benchmarked.

## Reproduction

Hardware/software provenance is in [provenance.json](provenance.json).
Baseline: `72ef2f5f367a6c15eb21cf1b708ad3557b805305` on `main`. Candidate:
`perf/qwen38-nvfp4-v100-prefill-decode`, the accompanying working-tree changes.
The prior `feature/muse-glimmer-nvfp4-v100` branch remains at `975262bbf3`.

Use the same installed Marlin library in both checkouts. Start one server at a
time from the relevant checkout with the repository's reference script:

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

After readiness, run in another shell using the same Python environment:

```bash
PYTHONPATH=python python benchmark/qwen38_nvfp4_v100_20260907/bench_serving.py \
  --label candidate --output-dir /tmp/qwen38-repeat --port 8082
```

For a baseline checkout, invoke a saved copy of `bench_serving.py` with label
`baseline` and that checkout's `PYTHONPATH`. The script performs the discarded
warmup and three cold measurements for each shape. It refuses an existing
label to prevent accidental mixing of runs.

On an idle GPU, compare the kernel entry points with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python FLASHINFER_DISABLE_VERSION_CHECK=1 \
  python benchmark/qwen38_nvfp4_v100_20260907/bench_kernels.py --label candidate
```

The final candidate microbenchmarks are retained in
[final_kernel_benchmark.jsonl](final_kernel_benchmark.jsonl). The HC `native=0`
rows use Triton in the same process; `native=1` rows use the new CUDA kernels.
For old MoE/top-k measurements run the same script with baseline `PYTHONPATH`.
On baseline, both HC flag values select its original Triton implementation.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python FLASHINFER_DISABLE_VERSION_CHECK=1 \
  python -m pytest -q \
  test/registered/unit/layers/moe/test_sm70_nvfp4_moe_decode.py \
  test/manual/layers/test_sm70_hc_mix.py \
  test/registered/unit/layers/attention/test_qwen_sparse_sm70_routing.py
```

The changes are source-only; no Docker image was built or published.
