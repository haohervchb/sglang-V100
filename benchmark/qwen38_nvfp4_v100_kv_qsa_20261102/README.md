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
- All corrected comparisons use a 0.83 memory fraction and the same 262,144
  context limit. GPU clocks, power, temperature and memory are recorded.
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

Corrected full-model measurements are still running. This section will be
replaced with the completed comparison and recommended profile.
