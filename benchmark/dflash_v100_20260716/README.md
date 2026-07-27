# Audited Docker DFlash context-scaling benchmark

This directory contains the 2026-07-27 four-V100 Docker refresh used by the
root README. The directory name is retained so existing plot links remain
stable. The tests used image `sglang-v100:18878a5f0`
(`sha256:f9feb5340d56d29f8b8e42463daf0f79a429ef90a5b0e5d6ed63c6283103cc42`),
built from commit `18878a5f080ab6a1a160854126b8426438bcec21`.

## Workload

- Targets: Qwen3.6-27B FP16 and Qwen3.5-122B-A10B factory GPTQ-Int4.
- Drafts: the matching z-lab block-16 DFlash checkpoints.
- Hardware: four V100-SXM2-32GB GPUs with NVLink and TP4.
- Matrix: prompt lengths 1K through 25K in 2K steps, at 1, 2, and 4
  concurrent clients (39 cells/model, 78 total).
- Generation: greedy, exactly 256 output tokens/request, `ignore_eos=false`.
- Cache: `/flush_cache` before every cell; every recorded `cached_tokens` value
  is zero.
- Prompts: unique chat-formatted slices of this repository. All 91
  request-level prompt hashes match across the two model runs.
- Repetitions: one audited trial per cell. Each length uses a different source
  slice, so the prompt-dependent acceptance curves are intentionally
  unsmoothed and jagged.

The harness retains all generated text, prompt/output hashes, finish reasons,
re-tokenized diversity, and maximum identical token/character runs. It aborts
instead of summarizing a cell that fails the output audit. All 78 cells and all
182 request responses succeeded; every response contained 256 server-reported
output tokens. No repeated-word, repeated-character, replacement-character, or
stray-`9` corruption pattern was found. Two apparent letter-`9` matches in the
122B corpus were ordinary hexadecimal prompt nonces.

Four timing cells were rerun with identical prompt hashes after one-time
prefill/JIT-path stalls: 27B 3K/concurrency-2, 27B 3K/concurrency-4, 27B
1K/concurrency-4, and 122B 1K/concurrency-4. Only the immediate steady reruns
are published. The most visible cold outliers were the first 1K/concurrency-4
attempts (11.34 seconds for 27B and 10.88 seconds for 122B); their retained
steady TTFTs are 1.005 and 0.813 seconds. No prompt, sampling, or output-length
setting changed.

## Agent and multimodal correctness audit

Both models passed all seven live OpenAI-compatible API checks inside the same
Docker image and server configuration used for the sweep:

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
  calls for the cell.

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
```

Regenerate the dependency-free SVG figures with:

```bash
/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/plot_results.py
```

Each result directory contains the complete JSONL audit record, summary JSON
and CSV, and captured server arguments.
