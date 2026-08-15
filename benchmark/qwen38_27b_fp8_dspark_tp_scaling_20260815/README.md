# Qwen3.8-27B-FP8 DSpark TP2/TP4 context scaling

This benchmark compares two and four V100-SXM2-32GB GPUs serving
`Qwen/Qwen3.8-27B-FP8` with the `RadixArk/Qwen3.8-27B-DSpark` drafter.

## Benchmark settings

- Exact input lengths from 1,000 through 25,000 in 2,000-token increments.
- One cold-cache request per point and 256 greedy output tokens.
- FP8 target weights, FP16 target and draft KV caches, and DSpark block size 7.
- V100 FlashAttention backend, 4,096-token prefill chunks, and a 32,768-token
  server context and token pool.
- TP2 used GPUs 2 and 3; TP4 used all four GPUs.
- `SGLANG_SM70_FP8_PREFILL_BACKEND=turbomind` was used for the compact V100
  FP8 weight layout on both runs.

The README DSpark serve command was used with `--kv-cache-dtype auto`,
`--tensor-parallel-size {2,4}`, `--context-length 32768`, and
`--max-total-tokens 32768`. Each TP value represents a separate server run.
The benchmark command was:

```bash
python benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key qwen3.8-27b-fp8-fp16kv-dspark-tp{2,4} \
  --tokenizer Qwen/Qwen3.8-27B-FP8 \
  --output-dir benchmark/qwen38_27b_fp8_dspark_tp_scaling_20260815/tp{2,4} \
  --concurrency 1 \
  --output-tokens 256
```

## Results

| Context | TP2 prefill | TP4 prefill | TP2 decode | TP4 decode | TP2 accept | TP4 accept |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1,760.7 | 2,795.1 | 71.5 | 98.6 | 3.048 | 3.048 |
| 3K | 1,922.7 | 3,447.3 | 49.7 | 76.3 | 2.226 | 2.510 |
| 5K | 1,892.9 | 3,358.6 | 62.1 | 84.7 | 2.844 | 2.844 |
| 7K | 1,910.1 | 3,509.5 | 97.6 | 131.7 | 4.571 | 4.571 |
| 9K | 1,889.0 | 3,400.5 | 53.3 | 74.1 | 2.586 | 2.586 |
| 11K | 1,844.6 | 3,392.2 | 57.0 | 78.6 | 2.753 | 2.753 |
| 13K | 1,827.7 | 3,355.1 | 52.1 | 105.0 | 2.560 | 3.765 |
| 15K | 1,797.4 | 3,324.8 | 58.3 | 72.7 | 2.909 | 2.612 |
| 17K | 1,783.2 | 3,276.8 | 56.6 | 79.1 | 2.844 | 2.844 |
| 19K | 1,751.6 | 3,265.7 | 44.3 | 61.4 | 2.265 | 2.265 |
| 21K | 1,768.1 | 3,249.8 | 47.2 | 67.3 | 2.485 | 2.485 |
| 23K | 1,755.9 | 3,245.9 | 86.0 | 122.9 | 4.571 | 4.571 |
| 25K | 1,682.6 | 3,151.4 | 52.0 | 75.0 | 2.844 | 2.844 |

Rates are tokens per second. Prefill is exact input tokens divided by client
TTFT, and decode excludes TTFT. TP4/TP2 geometric-mean speedup across all 13
points is 1.81x for prefill and 1.43x for client-visible decode. Acceptance is
included because it is prompt-dependent and materially affects DSpark decode
throughput.

All 26 requests returned HTTP 200, used the exact requested input length,
generated 256 tokens, used zero cached tokens, and passed the repetition and
coherence audit. All prompt hashes matched across TP2 and TP4; 10 of the 13
greedy completions were byte-identical, and the other three passed their
independent output audits.

Raw requests, server configurations, CSV summaries, and JSON summaries are in
[`tp2`](tp2) and [`tp4`](tp4).
