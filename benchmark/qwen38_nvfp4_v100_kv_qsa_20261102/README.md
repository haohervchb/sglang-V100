# Qwen3.8 Flash Next: QSA correctness and V100 optimization experiments

This round uses four V100-SXM2 32 GB GPUs with NVLink, TP4 and one active request.
Both target and native MTP use RadixArk revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`. DFlash was excluded at the user's request.

The host restarted on November 3 and removed the earlier `/tmp` scratch data.
Code, committed QSA validation logs and the rejected prototype source survived.
Final comparisons are being rerun into durable storage and archived after every
completed stage. Earlier pilot numbers are not used as final performance evidence.

## Correctness comes before the performance comparison

The existing QSA ring could overwrite a retained prefix index key during even
four-token MTP verification. We reproduced `[8, 5, 6, 7]` being read instead of
`[4, 5, 6, 7]` after publishing positions `[5, 6, 7, 8]`.

[PR #6](https://github.com/haohervchb/sglang-V100/pull/6) isolates the correction
on main. It adapts [andyl98's upstream #46125](https://github.com/sgl-project/sglang/pull/46125)
to the legacy API and FP16 V100 path. Allocation now preserves the whole verify
window plus preceding group members, all metadata paths use that allocation,
and capture publishes its actual draft-extend width. Unknown or oversized
geometry is still rejected.

The fix passed 35 GPU compressed-cache equivalence cases, 44 graph-metadata
cases, five CPU ring tests with 1,443 subtests, and five graph-producer/guard
integration cases. The separate main-based branch passed its 45 GPU cases.
Fourteen GPU chain-verifier tests cover every rejection position at widths 6/8.

All resumed runs use the fixed ring. The initial `baseline` is additionally excluded from
performance attribution: its isolated checkout lacked the ignored TurboMind
binary, causing the draft MoE to fall back. Later launches pin the same compiled
library, record its checksum, and require its successful-load marker before
measuring. Target and draft revisions are both pinned; the early draft `main`
reference was independently verified to resolve to the same revision.

## Measurement protocol

- Same saved token IDs at 1,000, 8,192, 25,000 and 70,000 input tokens; 512 outputs.
  Three measured repeats reverse the request order; warmups are excluded.
- Cold-cache requests wait for an acknowledged flush. Decode timing excludes
  the first streamed burst. Acceptance includes the bonus token.
- A separate pinned 20-case, first-turn SPEED-Bench screen covers coding,
  reasoning, writing, retrieval and four CJK-oriented multilingual cases.
  It permits natural EOS, caps output at 512 tokens, and repeats at temperatures
  0 and 0.7. This is neither a full SPEED-Bench score nor an agent-session replay.
- All MTP comparisons use a 0.83 memory fraction and the same 262,144
  context limit. The final target-only check uses 0.84 to admit 262K inputs. GPU clocks, power, temperature and memory are recorded.
- Profiling is diagnostic and is excluded from throughput summaries.

Full-context checks retrieve two keys from 262,000 input tokens and then run a
short request without flushing. Tool checks validate arguments and a subsequent
tool-result response without executing a tool. Multimodal tests include image
swaps, streaming, chunked prefill and reversed video order. The existing OCR
case reads `V100 CHECK 4827` as `V10 CHECK 4827`; that expectation remains unchanged.

## Implemented experiments

**FP16 KV:** adapt the direct-cache decode loader from
[dg1kjd/sglang-sxm2](https://github.com/dg1kjd/sglang-sxm2/blob/73abfe1dd7e7aae7a62b222a201e48f532fb298e/python/sglang/kernels/jit/csrc/sm70_longctx_decode.cu)
and extend it to prefill. E5M2 and FP16 use the same attention arithmetic, so the
dtype comparison does not silently select a different attention implementation.
Nineteen GPU tests cover masks, changing graph inputs, non-contiguous physical
cache mappings and full-context positions.

**MTP loading:** inspired by [1Cat #389](https://github.com/1CatAI/1Cat-vLLM/pull/389),
select shards from the checkpoint index for models declaring the exact weight
prefixes they consume. This checkpoint's 31 MTP tensors occupy three of 206
shards. The target retains the ordinary loader, and missing required shards fail.

**FR-Spec:** build a separate draft head from global token IDs, gather the owned
rows, then shard the result across TP4. The target head and vocabulary remain
intact. This addresses the shard-boundary issue identified by
[Penny Royal's implementation](https://github.com/jpezzulli/sglang-rtxpro6000/blob/120cbbe66205cf70de0b339b1bdcc3e014bb0e79/python/sglang/srt/speculative/hot_vocab.py)
without replicating the entire hot head on every GPU. Real TP4 tests verify rows,
gathered logits and remapped IDs; sampled-verifier tests retain outcomes absent
from the draft vocabulary. The 64K map's tokenizer and file checksums are pinned.

The initial 64K-map pilot reduced acceptance on the CJK screen. A separate
131,328-token map adds every individually
decoded CJK token and punctuation entry, then pads to TP4 alignment. Its rule
uses no evaluation prompts or outputs; vocabulary-entry coverage is not a
frequency-weighted language guarantee.

**Wider MTP:** `SGLANG_SM70_MTP_WIDE_MOE=1` permits the existing exact-shape MoE
kernel at 6/8 rows and keeps the stable router tie-breaking there. The default
dispatch remains at four rows. GPU tests compare batched outputs to single-row
execution and independently dequantized weights.

## Rejected grouped-QSA adapter

The grouped Page4 design in [1Cat #387](https://github.com/1CatAI/1Cat-vLLM/pull/387)
is valuable, but our adapter did not reproduce its advantage. The standalone
prototype uses standard WMMA, two three-head CTAs per eight-query group, a
deterministically sorted page plan, exact visibility masks, and a fallback for
duplicate selections or plan overflow. It passed eight correctness cases.

The initial microbenchmark, including planner cost, rejected the adapter.
Its transient timing logs were lost in the restart, so those timings are not
included in the final numerical comparison. This does not reject the upstream
custom kernel or its published results. The unused runtime modules were
removed; their source is retained in
[the prototype archive](rejected_grouped_prefill.tar.gz).

A fresh experiment also tried the existing native HC mixing kernels at six
and eight rows. Numerics matched the reference, but graph-replay times were
39.2/52.5 microseconds versus 34.5/38.2 microseconds for the current path.
It was not integrated. See [measurements](wide_hc_micro.json) and
[prototype source](rejected_wide_hc.tar.gz).

## Results

The completed four-token comparison uses three repeats of each saved prompt.

| Input tokens | Prose E5M2 / FP16 tok/s | Code edit E5M2 / FP16 tok/s |
| ---: | ---: | ---: |
| 1,000 | 107.8 / 119.0 | 172.4 / 185.8 |
| 8,192 | 110.7 / 115.2 | 174.3 / 183.5 |
| 25,000 | 108.6 / 113.2 | 173.9 / 182.7 |
| 70,000 | 104.1 / 111.4 | 170.9 / 179.8 |

FP16 preserves the 262K single-request context at memory fraction 0.83, but
reduces the allocated KV pool from 554,576 to 277,280 tokens. Higher concurrency
and long-session prefix retention are not established by these measurements.

### Wider verification and draft vocabulary

Eight-token MTP improves the saved copy-heavy code edits by 15.2% relative to
four-token FP16 MTP, but reduces prose throughput by 32.9%. On the 20-case screen,
its paired coding gains are 4.4% greedy / 8.1% sampled; the other domains lose
throughput. Six-token MTP was screened at 8K and likewise did not provide a
balanced improvement. Both remain opt-in experiments.

The expanded CJK map gives a 1.6% gain on the saved code edits and a 0.9% loss
on prose. On the small screen, greedy-domain changes range from -3.8% to +1.5%;
sampled changes range from -3.9% to +14.1%. This is mixed evidence with only two
repetitions per prompt. Keep the full vocabulary for the general profile.

### Diagnostic profile

The eight-token 8K code diagnostic, using CPU profiling plus CUDA events, measured
median cycle 37.20 ms: draft 9.05 ms, target forward 25.97 ms, verify 26.30 ms,
state commit 0.185 ms and draft extension 1.85 ms. Verify contains target forward
and state commit; these columns must not be added. GPU kernel tracing increased
the cycle to 41.58 ms. Its 20.7% traced idle fraction is not an unprofiled idle
estimate. Dense target GEMMs account for substantial traced work; that motivated
a separate wider-projection kernel screen.

See [profile summary](profiles/summary.json),
[CPU-launch-correlated kernel phases](profiles/phase_kernels.json), and the
retained rank-zero traces. Kernel categories are name-based heuristics, and
duration sums can overlap across streams.

### Recommended single-request profile

The current evidence favors FP16 KV with ordinary full-vocabulary MTP-3/4.
Run from this source build in the V100 environment with its built SM70 libraries:

```bash
KV_CACHE_DTYPE=auto PORT=8082 \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh \
  RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --mem-fraction-static 0.83 --enable-multimodal \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --speculative-draft-model-revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

### Final gates and limits

Target-only decode measured **72.9–74.9 tok/s** across the saved contexts, in
line with the established range. It passed all eight checks at memory fraction
0.84, including 262K retrieval, the following short request, oversized-input
rejection and recovery. The initial target-only 0.83 run resolved to only
259,328 KV slots and is excluded; its failed gate and logs are retained.

That rejection exposed a separate QSA edge case: a padded compression plan read
four members from a one-token dummy prefill. Short packed prefills now keep their
pending keys and skip compression until a full group exists. This passed 41 GPU
state cases, including six new short-prefill cases. The fix is also in PR #6.

Every MTP configuration passed all six text/full-context checks and tool-use
checks. The four main comparison profiles retained the same 11/12 multimodal
result; the shorter six-token screen did not repeat multimodal tests. Fresh
validation also passed 37 kernel/sampler cases, 13 CPU cases, three head-graph
cases, and the real TP4 row/logit/remapping test. See [validation](validation.json).

The isolated eight-row projection experiment was also rejected: nine of ten
shapes were slower than the current path. The one improvement saved about
2.5 microseconds for a 640-by-2560 projection, which does not justify a new
serving path. [Measurements](wide_projection_micro.json) and
[source](rejected_wide_projection.tar.gz) are retained.

These are concurrency-one results and a small prompt screen. They do not
establish higher-concurrency performance, deterministic long-form output, or
broad model accuracy. The general 140 tok/s prose stretch target was not
established. Full-vocabulary FP16 MTP-3/4 is the balanced recommendation; the
wider and pruned-vocabulary paths remain optional experiments.

### Serving follow-up: long-document memory pressure

On November 3, the v5 serving profile exhausted GPU memory in PLE's short
convolution during queued document processing. A 160 MiB allocation failed
with 110.69 MiB free on GPU 0. The standalone smoke checks above did not cover
this sustained workload.

The persistent serving launcher now uses 2,048-token prefill chunks and a
2,048-token prefill batch budget, retaining FP16 KV, native MTP-3/4 and the
262K context limit. It passed all eight text/full-context checks and 24
document retrieval requests from four concurrent clients. Minimum sampled
free memory was 2,200.7 MiB on GPU 0 and at least 2,614.4 MiB on the other GPUs.

The queue run recorded no prefix-cache hits, so it tests repeated prefill
memory pressure rather than prefix reuse. Other client traffic overlapped the
run; its timings are not a throughput comparison. The earlier 8,192-token
prefill performance figures do not describe this serving configuration. See
[incident and validation record](serving_oom_20261103.json) and the reusable
[queue check](../qwen38_spec/check_queue.py). PDF extraction and image-page
memory use are outside this synthetic text test.

### Reproduction and acknowledgements

The reusable [runner](../qwen38_spec/README.md) accepts the retained
[matrix](matrix.json); update its local paths for another checkout. Results are
archived in `resumed_20261103` as each stage completes. SPEED prompt text and
responses stay in the local run directory; published records retain stable IDs,
input/output hashes, token counts and timing events. The archived corpus from
the previous round supplies the identical synthetic fixed-context prompts.

Source and review: [correctness PR #6](https://github.com/haohervchb/sglang-V100/pull/6)
and [optimization PR #7](https://github.com/haohervchb/sglang-V100/pull/7).
Thanks were posted to [andyl98](https://github.com/sgl-project/sglang/pull/46125#issuecomment-6591734845),
[yangzhuxinyzx](https://github.com/1CatAI/1Cat-vLLM/pull/389#issuecomment-6591978383),
and [dg1kjd and jpezzulli](https://github.com/haohervchb/sglang-V100/pull/7#issuecomment-6592538493)
for the concrete fixes, kernels and reference implementations used in this work.
