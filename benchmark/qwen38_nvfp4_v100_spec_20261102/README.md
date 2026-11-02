# Qwen3.8 Flash Next NVFP4: PR integration and speculation experiments

This round integrates the two V100 contributions through merged
[PR #4](https://github.com/haohervchb/sglang-V100/pull/4) and adds opt-in QSA graph
metadata and greedy prompt-lookup/MTP experiments. Both speculative experiments
default to off. The [harness and reproduction instructions](../qwen38_spec/README.md)
keep throughput measurements separate from diagnostic profiling.

## Machine and measurement scope

- Four Tesla V100-SXM2 32 GB GPUs, TP4, NV2 links between every pair.
- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4`, revision
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- CUDA 12.8, driver 580.178.04, PyTorch 2.9.1+cu128, patched Triton 3.5.1.
  See [environment.json](environment.json) for the full serving environment.
- This host currently has about 126 GiB RAM. Do not assume the 256 GB host
  configuration in earlier reports. PLE remains on CPU.
- FP16 compute, FP8 E5M2 KV, 262,144 maximum context, one active request,
  MTP three steps/four verification tokens, multimodal enabled.
- Fixed token IDs at 1,000/8,192/25,000/70,000 context tokens; 512 output tokens.
  The synthetic code edit copies a repetitive Python file while changing its
  missing-key behavior. It is not a representative agent benchmark.
- Each request starts after an acknowledged cache flush. Decode throughput
  excludes the first streamed burst. Verification-counter differences measure
  observed cycle time and tokens per cycle, including the bonus token.
- No simulated acceptance. Warmups and profiler runs are excluded from throughput
  comparisons. Raw JSONL includes output text, hashes, token events and histograms.

## Contributed fixes

[PR #1](https://github.com/haohervchb/sglang-V100/pull/1), by
**Prethea-Phoenixia**, fixes SM70 warmup capacity. Warmup now uses the model
runner's resolved request capacity after hybrid-state budgeting. This also
handles an unspecified server argument correctly. The original author is
preserved in the integrated commit.

[PR #2](https://github.com/haohervchb/sglang-V100/pull/2), by **Horacio Vico
(@hvico)**, adds opt-in custom all-reduce for PCIe V100 systems with a bounded
message size. The original author is preserved. Integration adds these guards:

- Keep the real topology flag, so enabling the PCIe option on an NVLink host
  does not change its dispatch or message cap.
- Enable the exception only in v2; keep legacy topology eligibility unchanged.
- Require push-only dispatch on PCIe even when an explicit algorithm override
  requests pull. Oversized messages fall back to NCCL.
- Validate a positive, 16-byte-aligned PCIe cap and check every peer when trusting
  driver-reported P2P access.

The contributor's PCIe speed measurements remain contributor-reported. Our
four-GPU execution test forces the logical PCIe policy on physical NVLink:
12 dtype/size/graph cases, each with 16 repetitions and four reductions. It
validates dispatch and arithmetic, not actual PCIe transport performance.
Published Docker v4 predates these options; they require a source build.

## Upstream review

| Upstream change | Decision for this SM70 tree |
| --- | --- |
| [#41173](https://github.com/sgl-project/sglang/pull/41173), QSA graph metadata | Adapted behind `SGLANG_SM70_QSA_FUSED_GRAPH_METADATA=1`. Fuse row/layout preparation and compatible draft steps; preserve the existing tiled page-table work and fallback paths. |
| [#41175](https://github.com/sgl-project/sglang/pull/41175), NEXTN verify/draft preparation | Deferred. The newer worker/layout interfaces differ substantially from this tree. Use the measured phase costs before choosing a smaller backport. |
| [#40972](https://github.com/sgl-project/sglang/pull/40972), QSA KV preparation | Not applicable as written: it targets BF16/E4M3 and TRT-LLM, while this V100 path uses FP16/E5M2 and reads selected KV directly in its attention kernel. |
| [#41166](https://github.com/sgl-project/sglang/pull/41166), grouped small copies | Tested and rejected for this round. The tuple kernel fails compilation on the pinned Triton 3.5.1 stack for unit strides. The production hook and kernel were removed; failure logs are retained. |

The upstream B200 results combine optimizations and simulated acceptance. They
are not independent V100 speed measurements. No generic FP4 kernel rewrite,
expert offload, reduced draft vocabulary, or new PLE offload path is included.

## Initial MTP baseline

The pilot baseline is one measured repetition after warmup, with both new flags
off. Its launch was recorded at `2d06a902ae`; development continued in the same
checkout while that resident process ran. Treat it as a baseline snapshot,
not a controlled isolated-commit A/B test or an attribution of small QSA gains.

| Context | Prose decode tok/s | Code-edit decode tok/s | Prose tokens/cycle | Code tokens/cycle |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 107.6 | 172.3 | 2.33 | 3.76 |
| 8,192 | 112.2 | 174.1 | 2.46 | 3.84 |
| 25,000 | 107.7 | 175.4 | 2.35 | 3.84 |
| 70,000 | 110.6 | 168.8 | 2.44 | 3.76 |

Observed cycles were about 21.7–22.3 ms. Different prompts explain why these
numbers differ from the September prose benchmark. The copy-heavy code case
already approaches four tokens per cycle, leaving little acceptance headroom.

## Interleaved prompt-lookup comparison

The QSA flag was on for this whole run. Lookup off/on alternated for each
saved prompt, with the order reversed in the second measured repetition.
There were 16 warmup requests and 32 measured requests. These are medians of
two observations, not confidence intervals. All samples are retained.

| Context | Prose, lookup off/on tok/s | Code edit, lookup off/on tok/s |
| ---: | ---: | ---: |
| 1,000 | 109.8 / 109.8 | 175.7 / 160.5 |
| 8,192 | 112.8 / 115.1 | 176.8 / 163.9 |
| 25,000 | 86.8 / 76.7 | 175.9 / 163.3 |
| 70,000 | 110.0 / 68.8 | 113.5 / 57.9 |

**Do not enable this lookup prototype by default.** In the stable 1K–25K code
comparisons it lost 7–9% throughput: roughly 0.7–0.8 ms cheaper cycles did not
offset acceptance falling from about 3.76–3.83 to 3.32–3.46 tokens per cycle.
An exact prompt match can still be a poor proposal where the requested edit
changes the next tokens.

The longer runs also had substantial timing swings, including a lookup-off
70K code sample at 67.8 ms/cycle and a lookup-off 25K prose sample at 37.3 ms.
Other observations of those modes were near 22 ms. The cause is unresolved;
these rows do not justify a clean estimate of long-context optimization gains.
No slow samples were discarded from the table.

All eight measured code on/off pairs produced identical output text. Prose was
not bitwise repeatable even between the two lookup-off requests at 8K/25K/70K.
This round therefore does not establish deterministic full-output equivalence
for prose. The real verifier rejection tests and explicit correctness checks
remain the stronger checks on this prototype.

The QSA port passed metadata equivalence and end-to-end smoke checks, but this
run does not isolate its speed effect from restart and runtime variation. It
also remains off by default. The historical 120→140 tok/s stretch improvement
has **not** been demonstrated.

Raw evidence: [paired observations](candidate.jsonl.gz),
[summary with both values and text comparisons](candidate_summary.json),
[lookup hit counts](candidate_lookup_hits.json), and
[server arguments](candidate.jsonl.server.json.gz).

## Separate diagnostic profiles

CPU-only profiler sessions enabled the rank-zero CUDA-event probe without GPU
kernel tracing. Requests used the same saved 8K cases, with 128 output tokens.
Medians below include GPU stream gaps within each phase. Verify includes target
forward and recurrent-state commit, so these nested columns must not be added.

| Request | Rounds | Full cycle ms | Draft ms | Verify ms | Target forward ms | State commit ms | Draft extend ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Code, lookup off | 34 | 21.81 | 2.69 | 17.61 | 17.29 | 0.184 | 1.51 |
| Code, lookup on | 36 | 21.30 | 0.47 | 19.23 | 17.51 | 0.183 | 1.51 |
| Prose, lookup off | 59 | 21.72 | 2.67 | 17.51 | 17.18 | 0.184 | 1.51 |

Lookup skips substantial draft work but increases the verify interval that
contains its history transfer. The probe does not isolate that transfer as a
separate phase, so the 1.6 ms increase is an observation of the complete interval,
not a direct copy-kernel measurement.

A separate CPU+GPU trace of 64 code tokens covered 18 rounds. Its median cycle
was **26.65 ms**, demonstrating profiler overhead. Correlating CUDA submissions
with GPU completion gives a 489.40 ms decode window, 366.71 ms busy and 122.69 ms
idle (**25.1%**). This is the traced window, not an unprofiled GPU-idle estimate.

Kernel-duration sums in that window were 72.56 ms MoE, 24.37 ms GDN, 20.68 ms
all-reduce, 14.48 ms QSA compute, 0.305 ms QSA metadata and 24.57 ms copies.
Other kernels include substantial HC mixing and dense projections. These sums
can overlap across streams; they are not percentages of wall time.

See [profile summary](profile_summary.json) and [raw rank-zero traces and
event records](profiles/). The next bounded experiment should improve proposal
selection at edit sites and reduce history coordination before widening lookup
or reducing the draft vocabulary. The current data does not support promoting
the prototype.


## Validation and retained failures

The focused CPU checks for the two contributed fixes passed **39 tests** with
CUDA hidden. Prompt lookup passed **10 unit tests**, including verified-history
tracking and unsupported-request fallback, plus **four real GPU chain-verifier
tests**, one for each possible first-rejection position. QSA metadata passed
**34 GPU cases**, including padding, changing graph inputs, both compression
ratios and full-context page tables. The combined graph-test log also records
failures of the rejected small-copy experiment; it must not be read as a clean
pass for that removed implementation.

The baseline passed short arithmetic/geography, constrained JSON, sampled
request completion, retrieval of two keys at **262,000 input tokens**, and a
short request immediately afterward. These are correctness smoke checks, not a
sampling-distribution or broad model-quality evaluation.

Baseline multimodal results were **11/12** after installing the missing
`decord==0.6.0` video fallback in the serving environment. Image, streamed image,
multiple-image, chunked-prefill and video-order cases passed. The OCR case
consistently returned `V10 CHECK 4827` instead of `V100 CHECK 4827`, including a
separate repeat. The expected answer was not loosened.

The candidate also passed all six text/full-context checks and the same **11/12**
multimodal cases, with exactly the same OCR miss. The profile-analysis tests
passed **three cases**, including overlapping GPU intervals, async decode
boundaries and prevention of eager instrumentation imports.

## Target-only check

With both experimental flags off and no diagnostic probe, target-only decode
remained consistent over two repetitions of every case:

| Context | Prose tok/s | Code-edit tok/s |
| ---: | ---: | ---: |
| 1,000 | 74.68 | 74.69 |
| 8,192 | 73.85 | 73.86 |
| 25,000 | 73.55 | 73.57 |
| 70,000 | 72.80 | 72.81 |

These observations are consistent with the established target-only range, but
are not an isolated old/new commit regression comparison. All six target-only
correctness checks passed, including 262K retrieval and the subsequent short
request. See [raw observations](target.jsonl.gz), [summary](target_summary.json)
and [correctness responses](target_checks.json.gz).


An early diagnostic probe eagerly imported the serving stack before normal
server setup. That run had a large slowdown even with lookup disabled; its
interrupted warmups are excluded. The probe now attaches after the worker's
normal import, with a test preventing premature Torch/SGLang imports. Profiling
still adds substantial overhead, so trace timings cannot be used as unprofiled
throughput evidence.
