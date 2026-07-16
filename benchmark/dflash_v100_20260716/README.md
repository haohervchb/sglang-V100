# Audited DFlash context-scaling benchmark

This directory contains the 2026-07-16 four-V100 benchmark used by the root
README. It replaces an earlier invalid run whose apparent 300+ tok/s decode
rate was repeated punctuation produced after unsafe CUDA-graph padding.

## Workload

- Targets: Qwen3.6-27B FP16 and Qwen3.5-122B-A10B factory GPTQ-Int4.
- Drafts: the matching z-lab block-16 DFlash checkpoints.
- Hardware: four V100-SXM2-32GB GPUs with NVLink and TP4.
- Matrix: prompt lengths 1K through 25K in 2K steps, at 1, 2, and 4
  concurrent clients (39 cells/model, 78 total).
- Generation: greedy, up to 256 output tokens, `ignore_eos=false`.
- Cache: `/flush_cache` before every cell; all recorded `cached_tokens` values
  are zero.
- Prompts: unique chat-formatted slices of this repository. All 91
  request-level prompt hashes match across the two model runs.
- Repetitions: one audited trial per cell. The plots intentionally have no
  confidence band; small point-to-point changes should not be read as precise
  model rankings. Each length uses a different unique source slice, so the
  prompt-dependent acceptance curve is intentionally unsmoothed and jagged.

The harness stores generated text, prompt/output hashes, finish reasons,
re-tokenized diversity, and maximum identical token/character runs. It aborts
instead of summarizing a cell that fails the output audit. All 78 published
cells passed.

The first 122B 1K/concurrency-4 attempt hit a one-off 10.9-second first-token
stall while producing valid text. The identical prompt hashes were immediately
rerun after the prefill/JIT path was resident; the published steady result was
0.845 seconds. No other result was replaced.

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

Start the corresponding four-request server from the root README, then run:

```bash
python benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key qwen3.6-27b-fp16 \
  --tokenizer Qwen/Qwen3.6-27B \
  --output-dir benchmark/dflash_v100_20260716/results/27b
```

For 122B:

```bash
python benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key qwen3.5-122b-a10b-gptq-int4 \
  --tokenizer Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --output-dir benchmark/dflash_v100_20260716/results/122b
```

Regenerate the dependency-free SVG figures with:

```bash
python benchmark/dflash_v100_20260716/plot_results.py
```

Each result directory contains the complete JSONL audit record, summary JSON
and CSV, and captured server arguments.
