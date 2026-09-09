# Qwen3.8 NVFP4 MTP on four V100s: results through September 8

The interrupted experiment and final validation are complete. The selected
configuration is `mtp34_vector`: three draft steps and four verification
tokens. **Consistent 120 decode tok/s has not been achieved.** The target-only
optimization is committed as `ddea10a60f07a133a6524038b69883c8844c4c28`;
the changes here build on it. The three-token serving candidate, wider expert
weight loads and padded matrix products were evaluated and not selected.

| Input tokens | Selected MTP, natural decode tok/s | Selected MTP, random decode tok/s | Natural prefill tok/s |
| ---: | ---: | ---: | ---: |
| 1,000 | 115.99–118.02 | 122.05–139.03 | 2,971–3,157 |
| 8,192 | 116.24–118.12 | 121.62–137.93 | 5,455 |
| 25,000 | 108.44–119.06 | 148.08–172.02 | 4,681–4,687 |
| 70,000 | 113.88–116.93 | 138.90–176.80 | 4,625–4,627 |

These ranges contain two different requests per length, each with 1,024
output tokens. The lowest observed natural window is 105.36 tok/s; the lowest
random window is 84.30 tok/s. The initial MTP random sweep measured
61.49–103.35 tok/s. Median estimated round cost fell from 38.9 to 21.6 ms
(44.4%), calculated as mean accepted tokens per round divided by streamed
decode throughput. Acceptance and generated text vary, so this is not a
claim of identical-workload throughput gains or a latency confidence interval.

Target-only regression remains above 70 tok/s: **74.28–74.32 at 1K** and
**73.55–73.58 at 25K**, using 2,048 output tokens. The lowest 25K window is
73.53 tok/s and prefill is 4,916–4,923 tok/s. The earlier 8K-output sustained
25K run reached 71.83 tok/s before these additional kernel improvements;
the final candidate has not been rerun with 8K output tokens. Prefill gains
over main remain modest and the earlier short-prompt regression has not
been resolved by a fresh matched comparison.

[Machine-readable final summary](final_summary.json) includes all selected
measurements, the initial MTP random baseline and the three-token comparison.
The following sections retain the development evidence and rejected trials.

The measured MTP cycle fell from roughly 38–39 ms to about 21–22 ms after
capturing draft extend and specializing two/four-row FP16 projections and
hyperconnection mixing, followed by recurrent-attention tile tuning and the
sparse-attention accumulator rewrite. Output throughput still depends on how many drafts
the target accepts. Ordinary prose/code requests are included alongside the
repository's random-token benchmark to make that dependence visible.

## Correctness issue found during optimization

A short response with log probabilities returned a truncated answer and
subsequently left TP ranks disagreeing about whether the request had finished.
The full candidate also hung on a fresh short request. The asynchronous result
copy path replaced GPU sources with CPU destinations while the copy stream
could still be reading the GPU allocations. A deterministic allocator-reuse
probe changed both token IDs and acceptance counts to `-1`. Recording the
copy stream on each CUDA source preserved their values.

The result-copy fix now passes the allocator regression with and without log
probabilities. Four short responses before and after the long-context sweep
completed normally and returned identical token IDs to the MTP baseline.
The largest log-probability difference against that baseline was 0.131;
this is a targeted response check, not a full quality evaluation. Earlier
`mtp34_capture` and `mtp34_batch` measurements predate the copy fix and are
retained as development evidence, not final validated configurations.

[Allocator reproduction](copy_lifetime_probe.log),
[copy and QSA graph checks](copy_and_qsa_tests.log),
[response comparison](copyfixed_response_comparison.json).

## Implementation under evaluation

- Enable compressed QSA draft-extend capture in the V2 worker for the SM70
  Qwen4 MTP architecture. Unwrap the full-attention backend because this MTP
  model does not execute the hybrid backend's linear-attention layers.
  Both the parsed capability flag and the worker dispatch must enable it.
  The initial `mtp34_graph` experiment missed the capability flag and remained
  eager; its timing differences must not be attributed to graph capture.
- Share vectorized FP16 weight loads across two/four projection rows, with
  FP32 accumulation. Preserve the existing batch-one kernels and library
  fallbacks for other shapes. In particular, batch-one fusions must not accept
  the newly supported batched projection inputs.
- Fuse batched HC projection epilogues while preserving the FP16 projection
  boundaries. The isolated two/four-row HC mix drops from about 35.0/32.8 us
  to 22.4/26.7 us. Eight-row prototypes were slower and are not enabled.
- The recurrent-attention probe favors an eight-value tile over the current
  32-value tile: about 20.3 to 7.6 us for four tokens, with identical outputs
  and cached states in the probe. Four kernel/graph checks pass with bitwise-identical outputs and states.
  Serving reduces a four-token cycle by about 1.1 ms; this specialization
  is now enabled by default (`SGLANG_SM70_MTP_GDN=0` disables it).
- A 50-configuration four-GPU all-reduce probe selected the existing one-shot
  push algorithm at both two and four rows; no all-reduce setting changed.

[Projection and HC tests](batched_kernel_tests.log),
[QSA routing tests](qsa_cpu_regression.log),
[GDN probe](gdn_verify_summary.json),
[all-reduce measurements](allreduce_batch_probe.log).

## Measurement protocol

Four V100 SXM2 32 GB GPUs, TP4, FP16 activations and recurrent state, NVFP4
experts, E5M2 KV cache, one request at a time, 262,144-token context capacity,
8,192-token prefill chunks. MTP uses the same RadixArk checkpoint for target
and draft, EAGLE top-k one, initially three steps/four verification tokens.
The MTP expert weights are FP16. The installed Marlin binary was not rebuilt.

Input lengths: 1,000, 8,192, 25,000 and 70,000. Each length gets one discarded
256-output-token warmup and two 1,024-output-token requests, using seeds
20260830 and 20260831. These seeds select different prompts, not repeated
measurements of the same prompt. Requests are greedy, ignore EOS, flush the
cache and verify exact input/output counts with zero cached tokens.

Decode throughput uses `(last_count - first_count) / (last_time - first_time)`
from streamed HTTP responses; it excludes the first observed token batch.
Window rates accumulate at least 256 observed tokens, including the final
partial window. MTP streams can deliver several tokens at once; raw counters
and timestamps are retained. Prefill throughput is input length divided by
client TTFT. Profiling, kernel probes and serving benchmarks run separately.

`bench.py --tag TAG --output-dir DIR` runs the random-token protocol against
port 8082. `bench_natural.py` uses long prose/code tutorial requests padded
with reference text to the exact input lengths. Both retain the generated
text and per-request acceptance statistics. Server launch/configuration JSON
files record each development point. The final selection and summary appear
above; reproduction commands and final validation appear below.

## Validated four-token candidate

`mtp34_gdn` includes the copy-lifetime fix. Four short answers produce the same
IDs as the baseline. The table lists both requests at each context length;
minimum windows are the slowest observed windows across those two requests.
The two tasks in the natural sweep are a web-service performance guide and
a Python cache tutorial. This is a limited content check, not a quality suite.

| Input tokens | Random decode tok/s | Natural decode tok/s | Lowest random window | Lowest natural window |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 128.59 / 103.90 | 105.30 / 109.42 | 82.08 | 99.38 |
| 8,192 | 147.12 / 120.14 | 104.90 / 113.62 | 89.61 | 97.49 |
| 25,000 | 128.43 / 119.73 | 102.92 / 114.23 | 74.97 | 99.44 |
| 70,000 | 127.64 / 133.45 | 101.21 / 111.17 | 109.83 | 99.40 |

Natural tasks accept approximately 2.5–2.7 tokens per round. At the measured
23.5–24.3 ms per round, they remain below 120 tok/s. Reducing verification to
two tokens (`mtp12_gdn`) lowers round time to 18.8–19.3 ms but loses too much
acceptance: natural requests achieve only 90.7–95.0 tok/s. Three of its four
short responses also differ from the four-token baseline; this configuration
is not selected. Long-context outputs/acceptance can differ between runs of
unchanged configurations; the cause has not been isolated. Compare round cost
alongside output throughput instead of attributing every acceptance change
to a kernel speedup.

[Four-token response comparison](mtp34_gdn_response_comparison.json),
[two-token response comparison](mtp12_gdn_response_comparison.json),
[GDN graph regression checks](gdn_verify_tests.log),
[profile kernel summary](mtp34_gdn_kernel_summary.json).
Profiling adds overhead; its kernel durations and client throughput are not
substitutes for the unprofiled serving measurements.

## Further kernel probes

An 18-variant NVFP4 expert probe varied loop unrolling and CTA size at one,
two and four rows. Keeping the 16-element inner unroll was best. The fastest
four-row alternative improved the isolated MoE from approximately 75.2 to
73.5 us, too small to explain the remaining end-to-end gap. No expert-kernel
change was selected from this sweep. All outputs matched the existing kernel
bit for bit. The initial grouped Marlin comparison was slower at two rows
(156.4 versus 47.3 us); wider-K configurations cannot handle this model's
160-wide expert down projection and were excluded.

The shared-expert fusion probe combines projection, FP16 rounding, sigmoid
and output multiplication into one launch. Four rows improve from about
9.3 to 3.0 us; fusing gate/up projection with SwiGLU improves 7.7 to 6.1 us.
The experiment preserves FP16 boundaries and passes 54 kernel, graph,
dispatch and copy-lifetime checks. However, `mtp34_fused` does not reduce
serving round cost: natural requests take 23.6–24.4 ms per round versus
23.5–24.3 ms without it. Shared experts already overlap the routed expert
work. The fusion candidate was removed from production code; its isolated
probe and serving records are retained. Natural throughput is 102.8–116.7
tok/s, still below the requested consistency target.

The rejected shared-expert fusion also changes token IDs in two of the four
short checks, consistently before and after the long sweep. Its removal keeps
the previous candidate, whose four short responses match the baseline.

## Recurrent projection/layout candidate

`SGLANG_SM70_MTP_QKVZBA` is now enabled by default; set it to `0` to disable. It writes QKV, Z, B and A into
their consumed layouts directly from one CUDA projection, avoiding two
projection launches plus four copies in the two/four-row ratio-three GDN
path. The four-row isolated comparison is 47.8 versus 30.1 us; outputs match
the existing projection kernels bit for bit. The two-row comparison is
46.5 versus 30.4 us. Thirty-three projection, layout, graph and batch-one
regression checks pass. The resumed `mtp34_layout_resume` serving validation passes the four short
response checks before and after the two long-context sweeps.

A separate diagnostic times CPU submission of CUDA graphs without GPU
synchronization. It uses a temporary `sitecustomize.py` outside production
code and its throughput is not used as a release result. Its first startup
caught a missing M=1 guard during removal of the rejected shared fusion;
the guard was restored before measurements. The diagnostic was restarted.

The CPU submission diagnostic collected 1,000 replays per graph on each TP
rank. Median submission times are about 131 us for the large graph and 37/28
us for the other two: approximately 0.2 ms total per round. Graph submission
itself cannot account for the remaining several milliseconds needed to bring
ordinary prompts to 120 tok/s. See [submission timing](graph_submit_summary.json).

The first `mtp34_layout` sweep was interrupted when its benchmark client was
terminated across a session continuation. The server remained healthy and
completed all four follow-up short responses with the baseline IDs. This is
an interrupted measurement, not a confirmed kernel failure. The restarted
`mtp34_layout_resume` random sweep completed: 1K 130.95/113.67, 8K
153.00/135.43, 25K 140.09/158.57, and 70K 168.54/131.95 tok/s. Round cost is
22.7–23.6 ms, around 1 ms lower than the previous candidate. The natural sweep and target-only regression are complete. Acceptance varies
between these runs; the highest rates do not establish consistent 120 tok/s.

The final natural sweep for the layout candidate is below. Round cost remains
lower than the preceding candidate, but consistent 120 tok/s is still unmet.

| Input tokens | Natural decode tok/s (two tasks) | Lowest window |
| ---: | ---: | ---: |
| 1,000 | 112.33 / 110.24 | 104.86 |
| 8,192 | 104.53 / 111.32 | 101.62 |
| 25,000 | 108.92 / 117.07 | 103.01 |
| 70,000 | 107.07 / 122.88 | 103.57 |

[Short checks after the sweeps](mtp34_layout_resume_after_long_response_comparison.json).
Expert-weight reuse across token routes, standalone HC FP32 FMA substitution,
and alternative QSA CTA/split configurations remain outside production code.
The QSA accumulator rewrite was subsequently validated and selected below.

## Target-only regression

After the MTP layout candidate, two cold 25K-input/2,048-output requests
achieve 71.969 and 71.990 decode tok/s, with minimum windows of 71.930 and
71.971. Prefill is 4,898 and 4,923 tok/s. The two 1K/2,048 requests achieve
72.704 and 72.655, with minimum windows of 72.341 and 72.282. These checks
preserve the earlier sustained 70+ target-only result. The launch uses zero
speculative steps and omits all speculative server arguments.


## Sparse-attention accumulator rewrite

The CUDA QSA partial kernel kept per-head arrays even though each compute
warp owns only one query head. Indexing those thread-local arrays by the
runtime warp ID forced a 240-byte stack allocation per thread. Keeping just
the owned head's state removes that stack frame (64 registers, no spills).
The selected change keeps the existing CTA size, split count, E5M2 conversion
and arithmetic order. Compiler flags match production in the final probe.

| Verification rows | Before, us | After, us |
| ---: | ---: | ---: |
| 1 | 25.12 | 14.02 |
| 2 | 43.69 | 16.91 |
| 4 | 71.31 | 24.40 |

Outputs in that isolated comparison are bitwise identical. Six attention and
combine tests pass with changing cache mappings, selections, masking and
CUDA graph inputs. The subsequent tests also cover the production 2,051-wide
selection, including its extra tail slots. Serving `mtp34_qsa` reduces round
cost to approximately 22.2–22.9 ms. Both short-response suites preserve the
baseline token IDs and the previous log-probability differences.

| Input tokens | Random decode tok/s | Natural decode tok/s | Lowest natural window |
| ---: | ---: | ---: | ---: |
| 1,000 | 132.73 / 117.15 | 114.82 / 112.93 | 106.80 |
| 8,192 | 156.84 / 133.34 | 112.90 / 125.08 | 104.45 |
| 25,000 | 136.36 / 119.96 | 106.35 / 125.73 | 99.00 |
| 70,000 | 141.11 / 134.99 | 109.63 / 118.11 | 99.93 |

The target-only regression improves to 72.649/72.639 tok/s at 25K input and
2,048 output tokens, with both minimum windows above 72.62. Prefill measures
4,915/4,925 tok/s. At 1K input decode is 73.360/73.345 tok/s. All four short
responses match the preceding target-only candidate, including log probabilities.

[Production-flags kernel comparison](qsa_production_timing.jsonl),
[attention tests](qsa_scalar_tests.log),
[short checks after long requests](mtp34_qsa_after_long_response_comparison.json),
[target-only checks](target_qsa_regression_response_comparison.json).

## Consistency and HC gate experiments

A fixed-input CUDA graph reproduction confirms that radix-selected sparse
indices change order between replays while retaining the same selected set.
The resulting FP16 attention outputs change; canonical logical ordering
eliminates those differences in this reproduction. This establishes one
source of variability, not a complete explanation for all serving variation.
A guarded experiment sorted block indices inside the existing expansion
kernel. Two repetitions of each of two identical tasks preserved both output
text and acceptance histograms at 1K. Only one of the tasks repeated exactly
at 8K; neither did at 25K or 70K. Sorting did not resolve serving variability
and added about 0.1 ms per round. It was removed from production code.
See [fixed-input replay evidence](qsa_order_replay.json).

A second guarded experiment computes the HC injection gate during the earlier
mix projection, when its inputs are already available. This removes one gate
launch per mix/combine pair. The isolated chain improves by about 3 us at two
and four rows, with bitwise comparisons across three seeds and graph replays.
The combined gate/sorting serving candidate passed all four short response
checks before and after the sweeps. The gate fusion is now enabled by default
(`SGLANG_SM70_MTP_HC_GATE=0` disables it), with sorting removed. The combined
serving check with vector scale loads is reported below.

A 14-configuration router radix/CTA sweep keeps the existing 128-thread,
six-bit radix choice at one, two and four rows. Reusing expert weights across
verification tokens regresses when few experts are shared and saves only
about 5.6 us even with all ten shared; it is not selected. The standalone HC
FMA substitution is also not selected as a separate optimization.

## GPU graph timing and NVFP4 scale loads

A temporary CUDA-event diagnostic collected about 1,866 replays per graph
on each TP rank. Median GPU times were 17.68 ms for target verification,
2.34–2.41 ms for the draft loop, and 1.37–1.38 ms for draft extend. These
measurements locate the dominant cost in target verification; diagnostic
client throughput is not used as a release result. See
[GPU graph timing](graph_events_summary.json).

The expert kernel now loads eight adjacent scale bytes together and uses
CUDA byte-permutation instructions to form the original FP16 scale pairs.
The aligned path preserves arithmetic and scale ordering; unaligned views
retain scalar loads independently for each projection. The isolated four-row
MoE improves from about 75.96 to 72.35 us. Vectorizing activation loads as
well was not selected because it increased register use without a consistent
gain. See [scale-load measurements](moe_scale_timing.jsonl).

The production build passes 37 checks covering independent dequantized-weight
references, aligned/unaligned scales with changing graph inputs and routes,
HC gate/mix, and sparse attention including the 2,051-slot production shape.
See [production validation](vector_production_tests.log).

The unprofiled `mtp34_vector` sweep includes both the gate fusion and vector
scale loads. Natural requests take about 21.6–22.3 ms per round. Their
throughput remains below 120 tok/s at every tested length; higher random-token
rates do not establish the requested consistency.

| Input tokens | Random decode tok/s | Natural decode tok/s | Lowest natural window |
| ---: | ---: | ---: | ---: |
| 1,000 | 139.03 / 122.05 | 118.02 / 115.99 | 109.67 |
| 8,192 | 121.62 / 137.93 | 116.24 / 118.12 | 105.94 |
| 25,000 | 148.08 / 172.02 | 108.44 / 119.06 | 105.36 |
| 70,000 | 176.80 / 138.90 | 113.88 / 116.93 | 109.44 |

The slowest random-token window is 84.30 tok/s. All four short checks before
and after the sweeps retain the MTP baseline token IDs and the preceding
candidate's log-probability differences. The target-only regression improves
to 73.581/73.549 tok/s at 25K input and 2,048 output tokens; minimum windows
are 73.565/73.527 and prefill is 4,916/4,923 tok/s. At 1K input, decode is
74.323/74.280 tok/s. Its four short responses match the previous target-only
candidate exactly, including log probabilities.

[MTP checks after long requests](mtp34_vector_after_long_response_comparison.json),
[target-only checks](target_vector_regression_response_comparison.json),
[source hashes](mtp34_vector_source_provenance.json).

## Completed final experiments

Three-token verification (`mtp23_native`, two draft steps) passes its 24
isolated numerical/graph checks and reduces natural-request round time to
19.0–19.7 ms. Fewer drafts are accepted, so it does not improve throughput
consistently:

| Input tokens | Natural decode tok/s (two tasks) | Lowest window |
| ---: | ---: | ---: |
| 1,000 | 115.71 / 121.65 | 112.27 |
| 8,192 | 114.69 / 117.50 | 106.93 |
| 25,000 | 109.35 / 124.79 | 101.96 |
| 70,000 | 112.98 / 119.72 | 106.24 |

Three of its four short answers differ from the four-token MTP baseline,
both before and after long requests. This does not establish a quality
regression, but fails the response-equivalence check used to select this
optimization. It remains an isolated patch, outside the selected runtime.
See [three-token results](mtp23_native_natural_measurements.jsonl),
[response comparison](mtp23_native_response_comparison.json),
[post-sweep comparison](mtp23_native_after_long_response_comparison.json),
[kernel checks](three_rows_serving_tests.log) and
[candidate patch](three_rows_serving.patch).

Loading more adjacent expert weight words is slower at all four batch sizes:
the best wider four-row gate/up variant takes 71.35 us versus 67.71 us for the
existing load width. Padding four-row cuBLAS inputs to 8, 16 or 32 rows also
fails to improve any of the three measured projection shapes. These changes
are not selected. See [expert sweep](moe_wide_gate_timing.jsonl) and
[padded projection timings](hc_tensorcore_padding_timing.jsonl).

## Final validation and reproduction

The selected production source matches the saved `mtp34_vector` source
hashes, apart from a subsequent import-spacing cleanup and CI registration
of the new CPU dispatch test. **123 regression checks pass**, covering
projections, HC mix/gates, GDN outputs and cached states, sparse attention,
NVFP4 dequantization references, graph replay, batch-one fallbacks, routing
and asynchronous copy lifetimes. The six QSA dispatch checks also pass with
CUDA hidden. All four selected MTP short responses preserve baseline token
IDs before and after the long sweeps; the maximum log-probability difference
is 0.131. These limited checks do not replace a broader model quality suite.

The repository-wide CI registration check still reports 14 files already
missing registrations in the parent commit. The new dispatch test is
registered; the existing unrelated failures are recorded in
[validation details](final_validation.json). Python lint, formatting and
import checks pass for the changed runtime/test files. Existing formatting
violations in the two older CUDA translation units remain; all four new
CUDA headers pass the formatter check.

[Regression log](final_regression_tests.log),
[CPU dispatch log](final_cpu_tests.log),
[baseline short-response requests/results](mtp34_eos_base_correctness.json).
All experiment servers were stopped after the completed measurements.
Large archival JSON files use compact serialization and log lines have
trailing whitespace removed; recorded values are unchanged.

From the repository root, in the existing `sglang-v100` environment:

```bash
conda activate sglang-v100
export PYTHONPATH="$PWD/python"
export CUDA_HOME=/usr/local/cuda-12.8
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=2
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=8082 \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh \
  RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

With that server running, use another terminal in the same environment and
run the clients sequentially, with no kernel probes competing for the GPUs:

```bash
PYTHONPATH=python python benchmark/qwen38_nvfp4_v100_mtp_20260908/bench.py \
  --tag mtp_final_random --output-dir /tmp/qwen38_mtp_recheck
PYTHONPATH=python python benchmark/qwen38_nvfp4_v100_mtp_20260908/bench_natural.py \
  --tag mtp_final_natural --output-dir /tmp/qwen38_mtp_recheck
```

Each script refuses to overwrite an existing measurements file. For
target-only serving, omit the five speculative arguments from the launch.
The remaining 120 tok/s goal requires lower verification cost and/or more
reliable draft acceptance on ordinary prompts; the measured random-token
peaks do not meet that goal.
