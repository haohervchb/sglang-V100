# Audited Docker V100 context-scaling benchmark

This directory contains the 2026-07-27 four-V100 Docker refresh and the
2026-07-28 Laguna DFlash follow-up used by the root README. The directory name
is retained so existing plot links remain stable. The Qwen tests used image
`sglang-v100:18878a5f0`
(`sha256:f9feb5340d56d29f8b8e42463daf0f79a429ef90a5b0e5d6ed63c6283103cc42`),
built from commit `18878a5f080ab6a1a160854126b8426438bcec21`. Laguna
used the rebuilt `sglang-v100:laguna-mmfix-20260727` image
(`sha256:b4c9a16f10a90e306ed2d7bacafd5865307210b62237f901aa6027a34fd5c302`),
which is also tagged `sglang-v100:latest`. The Laguna DFlash follow-up used
the equivalent `sglang-v100` Conda source at commit `19cac341d` and the
repaired Poolside draft revision
`f6b32f4fb7ef2fb2ad481bb4c05433a2bf8b0ed1`.

## Workload

- Targets: Qwen3.6-27B FP16, Qwen3.5-122B-A10B factory GPTQ-Int4, and
  Poolside Laguna S 2.1 118B-A8B INT4.
- Drafts: the matching z-lab block-16 DFlash checkpoints for Qwen and
  Poolside's repaired block-8 Laguna DFlash INT4 checkpoint. Laguna
  target-only is retained as the direct baseline.
- Hardware: four V100-SXM2-32GB GPUs with NVLink and TP4.
- Matrix: prompt lengths 1K through 25K in 2K steps, at 1, 2, and 4
  concurrent clients (39 cells/configuration, 156 total).
- Generation: greedy, exactly 256 output tokens/request, `ignore_eos=false`.
- Cache: `/flush_cache` before every cell; every recorded `cached_tokens` value
  is zero.
- Prompts: unique chat-formatted slices of this repository. All 91
  request-level prompt hashes match across the two Qwen runs, and all 91 match
  between Laguna target-only and Laguna DFlash. Laguna uses its corrected
  Mistral-family tokenizer and therefore has the same deterministic
  construction and exact token lengths as Qwen, but different token IDs and
  hashes.
- Repetitions: one audited trial per cell. Each length uses a different source
  slice, so the prompt-dependent acceptance curves are intentionally
  unsmoothed and jagged.

The harness retains all generated text, prompt/output hashes, finish reasons,
re-tokenized diversity, and maximum identical token/character runs. It aborts
instead of summarizing a cell that fails the output audit. All 156 cells and
all 364 request responses succeeded; every response contained 256
server-reported output tokens. No repeated-word, repeated-character,
replacement-character, or
stray-`9` corruption pattern was found. Two apparent letter-`9` matches in the
122B corpus were ordinary hexadecimal prompt nonces.

Five timing cells were rerun with identical prompt hashes after one-time
prefill/JIT-path stalls: 27B 3K/concurrency-2, 27B 3K/concurrency-4, 27B
1K/concurrency-4, 122B 1K/concurrency-4, and Laguna 1K/concurrency-4. Only the
immediate steady reruns are published. The most visible cold outliers were the
first 1K/concurrency-4 attempts (11.34 seconds for 27B, 10.88 seconds for 122B,
and 20.34 seconds for Laguna); their retained steady TTFTs are 1.005, 0.813,
and 0.858 seconds. No prompt, sampling, or output-length setting changed.

## Agent and multimodal correctness audit

Both Qwen models passed all seven live OpenAI-compatible API checks in the
same Docker image and their corresponding sweep configurations:

1. thinking enabled with separated, coherent reasoning and a correct answer;
2. thinking disabled with no reasoning leakage;
3. native tool call followed by a tool-result round trip;
4. streamed tool-call fragment assembly and JSON argument parsing;
5. image description;
6. image-conditioned separated reasoning; and
7. image-conditioned structured tool calling.

The multimodal fixture is
[`examples/assets/example_image.png`](../../examples/assets/example_image.png)
(SHA-256
`e06917184a00b14abd70cd8ea0ff5dca9abfbbad29f7b25c02f97133d4cd060e`).
Both models identified the person ironing on a yellow taxi/vehicle and emitted
the requested `report_scene` arguments. The raw requests, responses, separated
reasoning, streaming events, tool arguments, usage, hashes, and server settings
are retained in [27B](correctness/27b.json) and
[122B](correctness/122b.json).

Laguna target-only and Laguna DFlash each passed 7/7 checks:

1. thinking enabled with separated, coherent reasoning and the correct answer;
2. thinking disabled without reasoning leakage;
3. a correct native tool-call/result round trip;
4. streamed tool-call assembly with valid JSON arguments; and
5. three image request variants that each returned an explicit HTTP 400
   unsupported-multimodal error.

The Laguna checkpoint at revision
`67dbeda456e68139f281c40831f9d12049d8fc11` is text-only: it declares
`LagunaForCausalLM` and has neither `vision_config` nor `image_token_id`.
The first audit uncovered a real API bug: string-format chat conversion silently
dropped the image before validation, allowing plausible image-conditioned
hallucinations. The serving layer now rejects image, video, or audio parts
before template conversion for every text-only model. The rebuilt container
passed the full OpenAI serving-chat unit file (61 tests) and the live audit.
Laguna's raw evidence is retained in
[target-only](correctness/laguna.json) and
[DFlash](correctness/laguna_dflash.json).

Manual review found one isolated `.cw` suffix at the end of the 122B hidden
reasoning trace. It did not appear in the separated final answer, affect the
answer, recur in the other reasoning/tool/image cases, or resemble the former
systematic corruption. The 27B thinking test also demonstrated that a
512-token reasoning budget can be consumed by a verbose but coherent trace
before a final answer is emitted; the published correctness case therefore
uses a realistic 1024-token budget.

The servers used `--enable-multimodal`, `--reasoning-parser qwen3`, and
`--tool-call-parser qwen3_coder`. Optional TorchCodec/FFmpeg and FlashAttention
3 availability warnings appeared at startup, but neither is used by these
image requests or the selected `flash_attn_v100` backend. Qwen recurrent state
was explicitly kept in FP16 with `SGLANG_MAMBA_CONV_DTYPE=float16` and
`SGLANG_MAMBA_SSM_DTYPE=float16`.

Laguna intentionally omitted `--enable-multimodal` and used
`--reasoning-parser poolside_v1 --tool-call-parser poolside_v1`. Its final
target pool exposed 399,184 full-attention and 31,920 sliding-window token
slots and admitted four requests. Target-only left 5.25–5.77 GiB per GPU after
graph capture; block-8 DFlash added a 399,184-slot draft KV pool and left about
1.6 GiB on the tightest rank.

The Laguna DFlash sweep accepted 2.03–3.24 tokens per verify across its 39
cells. Relative to target-only, the median per-request decode ratio across the
13 context lengths was 1.41x at concurrency 1, 1.40x at concurrency 2, and
1.28x at concurrency 4. Median prefill-rate ratios were 0.99x at every
concurrency, so the measured gain is decode-side.

A separate block-16 tuning run sampled 1K, 9K, and 25K at concurrency 1 and 4.
Block 16 was 2.7% faster only for 1K/concurrency-1. Block 8 was faster in the
other five cells; at concurrency 4 it led by 15.2%, 13.6%, and 11.3%
respectively. The acceptance increase was too small to offset twice the verify
width on V100. Those six trials are retained in
[`laguna_dflash_block16_tuning`](results/laguna_dflash_block16_tuning).

Greedy target-only and DFlash text is not byte-identical: the native
multi-token split-K verification kernel and ordinary one-token Triton decode
reduce attention in different orders, so near-tied logits can choose different
tokens. None of the 91 complete 256-token hashes matched end-to-end, although
individual prefixes could match for hundreds of tokens. This is not an
unverified-draft path: each committed draft token is compared with target
top-1, and the SM70 fused accept/bonus kernel matched the eager reference in
2,000 randomized cases. Use target-only when bitwise greedy reproducibility is
a requirement.

## Metrics

- `median_request_decode_tps`: median per-request client-visible decode rate,
  excluding the first streamed chunk. This is the decode line in the plots.
- `aggregate_output_tps`: total completed output tokens divided by batch wall
  time. It is retained in the tables but is not plotted as "decode".
- `median_client_ttft_ms`: median request-start to first non-empty stream event.
- `effective_input_tps`: total prompt tokens divided by the latest first-token
  time. It includes scheduling and chunked-prefill effects and is not an
  isolated kernel microbenchmark.
- `weighted_accept_length`: total output tokens divided by total DFlash verify
  calls for the cell. It is N/A only for target-only Laguna.

## Reproduce

Start the corresponding four-request Docker server from the root README, then
run the benchmark with the `sglang-v100` Conda environment:

```bash
/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key qwen3.6-27b-fp16-dflash-docker \
  --tokenizer Qwen/Qwen3.6-27B \
  --output-dir benchmark/dflash_v100_20260716/results/27b
```

For 122B:

```bash
/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key qwen3.5-122b-a10b-gptq-int4-dflash-docker \
  --tokenizer Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --output-dir benchmark/dflash_v100_20260716/results/122b
```

For target-only Laguna:

```bash
/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key laguna-s-2.1-int4-target-only-docker \
  --tokenizer poolside/Laguna-S-2.1-INT4 \
  --output-dir benchmark/dflash_v100_20260716/results/laguna
```

For Laguna DFlash:

```bash
/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key laguna-s-2.1-int4-dflash \
  --tokenizer poolside/Laguna-S-2.1-INT4 \
  --output-dir benchmark/dflash_v100_20260716/results/laguna_dflash
```

Run the live correctness audit against each corresponding server:

```bash
/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/run_agent_multimodal_audit.py \
  --model Qwen/Qwen3.6-27B \
  --output benchmark/dflash_v100_20260716/correctness/27b.json

/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/run_agent_multimodal_audit.py \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --output benchmark/dflash_v100_20260716/correctness/122b.json

/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/run_agent_multimodal_audit.py \
  --model poolside/Laguna-S-2.1-INT4 \
  --multimodal-mode unsupported \
  --output benchmark/dflash_v100_20260716/correctness/laguna.json

/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/run_agent_multimodal_audit.py \
  --model poolside/Laguna-S-2.1-INT4 \
  --multimodal-mode unsupported \
  --output benchmark/dflash_v100_20260716/correctness/laguna_dflash.json
```

Regenerate the dependency-free SVG figures with:

```bash
/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/plot_results.py
```

Each result directory contains the complete JSONL audit record, summary JSON
and CSV, and captured server arguments.
