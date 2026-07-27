# V100 quick model comparison (2026-07-27)

This is a small, audited comparison of two DFlash configurations and
target-only Laguna S 2.1 on four V100-SXM2-32GB GPUs. It was collected from
merge commit `18878a5f080ab6a1a160854126b8426438bcec21`.

![V100 model comparison](plots/v100_model_comparison.svg)

A ready-to-upload [1600×1080 PNG](plots/v100_model_comparison.png) is included
for posts that do not accept SVG images.

## Results

Each cell is one cold-cache trial with an exact 1,000- or 25,000-token prompt
per request and up to 256 greedily decoded tokens. Decode is the median
per-request client-visible rate after the first streamed token. Aggregate
output rate includes prefill and every request in the batch. Effective input
rate is total prompt tokens divided by the time at which the last request
receives its first token.

| Model | Context | Clients | Decode | Aggregate output | Effective input | TTFT | DFlash accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B FP16 + DFlash | 1K | 1 | 101.4 tok/s | 90.7 tok/s | 3,265 tok/s | 0.306 s | 4.49 |
| Qwen3.6-27B FP16 + DFlash | 25K | 1 | 85.9 tok/s | 26.0 tok/s | 3,629 tok/s | 6.889 s | 4.13 |
| Qwen3.6-27B FP16 + DFlash | 1K | 2 | 71.6 tok/s | 115.1 tok/s | 3,478 tok/s | 0.569 s | 3.91 |
| Qwen3.6-27B FP16 + DFlash | 25K | 2 | 28.1 tok/s | 24.2 tok/s | 3,634 tok/s | 11.219 s | 2.15 |
| Qwen3.6-27B FP16 + DFlash | 1K | 4 | 59.4 tok/s | 152.7 tok/s | 2,950 tok/s | 1.354 s | 3.97 |
| Qwen3.6-27B FP16 + DFlash | 25K | 4 | 11.7 tok/s | 21.5 tok/s | 2,778 tok/s | 17.896 s | 1.59 |
| Qwen3.5-122B-A10B GPTQ-Int4 + DFlash | 1K | 1 | 109.0 tok/s | 97.3 tok/s | 3,419 tok/s | 0.292 s | 4.20 |
| Qwen3.5-122B-A10B GPTQ-Int4 + DFlash | 25K | 1 | 82.0 tok/s | 30.4 tok/s | 4,717 tok/s | 5.300 s | 3.41 |
| Qwen3.5-122B-A10B GPTQ-Int4 + DFlash | 1K | 2 | 98.2 tok/s | 157.7 tok/s | 3,628 tok/s | 0.546 s | 4.53 |
| Qwen3.5-122B-A10B GPTQ-Int4 + DFlash | 25K | 2 | 32.1 tok/s | 28.8 tok/s | 4,804 tok/s | 8.460 s | 1.90 |
| Qwen3.5-122B-A10B GPTQ-Int4 + DFlash | 1K | 4 | 79.2 tok/s | 208.5 tok/s | 3,602 tok/s | 1.108 s | 4.49 |
| Qwen3.5-122B-A10B GPTQ-Int4 + DFlash | 25K | 4 | 15.2 tok/s | 28.2 tok/s | 3,486 tok/s | 13.563 s | 1.62 |
| Laguna S 2.1 118B-A8B INT4, target-only | 1K | 1 | 47.9 tok/s | 45.7 tok/s | 3,596 tok/s | 0.278 s | n/a |
| Laguna S 2.1 118B-A8B INT4, target-only | 25K | 1 | 44.1 tok/s | 22.2 tok/s | 4,346 tok/s | 5.753 s | n/a |
| Laguna S 2.1 118B-A8B INT4, target-only | 1K | 2 | 37.4 tok/s | 70.7 tok/s | 3,718 tok/s | 0.430 s | n/a |
| Laguna S 2.1 118B-A8B INT4, target-only | 25K | 2 | 26.9 tok/s | 26.8 tok/s | 4,386 tok/s | 9.109 s | n/a |
| Laguna S 2.1 118B-A8B INT4, target-only | 1K | 4 | 36.5 tok/s | 131.1 tok/s | 4,847 tok/s | 0.823 s | n/a |
| Laguna S 2.1 118B-A8B INT4, target-only | 25K | 4 | 15.4 tok/s | 32.2 tok/s | 4,407 tok/s | 14.895 s | n/a |

The 122B GPTQ model is the fastest configuration here at 1K, despite its
larger total parameter count. At 25K with two or four clients, DFlash
acceptance falls to roughly 1.6–2.2 tokens per verification for both Qwen
models, and their per-request decode rates fall sharply. This is a
prompt-dependent speculative-decoding result, not an isolated kernel
microbenchmark.

Laguna is target-only in this comparison, so it is not an equal
speculative-decoding comparison. Its one-client decode rate changes from 47.9
tok/s at 1K to 44.1 tok/s at 25K, while its batched aggregate output reaches
131.1 tok/s at 1K and 32.2 tok/s at 25K.

## Method and correctness audit

- Hardware: 4× Tesla V100-SXM2-32GB, NVLink, driver 580.159.04, TP4.
- Runtime: the `sglang-v100` Conda environment and
  `flash_attn_v100`; Qwen DFlash uses block size 16.
- Workload: deterministic, unique slices of real repository source, rendered
  through each model's chat template at exactly 1K or 25K input tokens.
- Cache and sampling: `/flush_cache` before every cell, zero reported cached
  prompt tokens, temperature 0, and at most 256 output tokens per request.
- Audit: all 42 responses produced 256 tokens; all passed the retained-text
  diversity and repeated-token/run checks. No stray-`9` corruption was found.
- Comparability: the 14 request-level prompt hashes match exactly between the
  two Qwen models. Laguna uses its own tokenizer and chat template, so it uses
  the same deterministic construction and exact token counts but not identical
  token IDs.
- Tokenizer correctness: the harness now calls SGLang's tokenizer loader, so
  Laguna receives Transformers' `fix_mistral_regex=True` compatibility fix.
- Warm-up: a preliminary Laguna pass was discarded after exposing the missing
  tokenizer fix and a one-time 20.8-second 1K/concurrency-4 JIT stall. The
  published full pass used the corrected tokenizer after that path was
  resident; its other five cells reproduced the preliminary timings closely.

This is deliberately a quick comparison: there is one published trial per
cell and therefore no confidence interval. The raw streamed text, hashes,
request timings, server arguments, speculative counters, JSON/CSV summaries,
and audit fields are retained under [`results/`](results/).

## Reproduction

Start one server at a time using the matching command in the root
[README](../../README.md), with `--max-running-requests 4` and CUDA graph batch
sizes `1 2 4`. The Laguna benchmark used the target-only command, page size 16,
`--swa-full-tokens-ratio 0.08`, and
`--triton-attention-num-kv-splits 128`.

Run the matrix from the repository root:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key MODEL_KEY \
  --tokenizer TOKENIZER_ID \
  --output-dir benchmark/v100_quick_comparison_20260727/results/RESULT_NAME \
  --lengths 1000,25000 \
  --concurrency 1,2,4 \
  --output-tokens 256
```

The values used for this run were:

| Result | `MODEL_KEY` | `TOKENIZER_ID` |
| --- | --- | --- |
| `27b` | `qwen3.6-27b-fp16-dflash` | `Qwen/Qwen3.6-27B` |
| `122b` | `qwen3.5-122b-a10b-gptq-int4-dflash` | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` |
| `laguna` | `laguna-s-2.1-int4-target-only` | `poolside/Laguna-S-2.1-INT4` |

Regenerate the SVG without plotting dependencies:

```bash
/home/rah/miniconda3/envs/sglang-v100/bin/python \
  benchmark/v100_quick_comparison_20260727/plot_results.py
```

## Docker validation

A clean build of merge commit `18878a5f0` completed with:

```bash
DOCKER_BUILDKIT=1 docker build --network=host \
  -f docker/v100.Dockerfile \
  -t sglang-v100:18878a5f0 \
  -t sglang-v100:latest .
```

The resulting local image is
`sha256:f9feb5340d56d29f8b8e42463daf0f79a429ef90a5b0e5d6ed63c6283103cc42`
(20,871,303,979 bytes). Both the GPU/extension entrypoint check and an
end-to-end TP4 server test passed. The latter loaded
`Qwen/Qwen3.6-27B` from a read-only host cache, reached API-ready, and returned
a normal 32-token completion to `/generate`.

The empty named JIT volume took several minutes to compile the first hybrid
GDN/Triton paths; subsequent launches can reuse it. Startup also logged ignored
optional-import warnings for TorchCodec/FFmpeg and FA3. Those did not affect
this text-model inference test; audio or H100-specific multimedia paths were
not validated.
