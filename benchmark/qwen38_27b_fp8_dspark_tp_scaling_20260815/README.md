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
  --model-key qwen3.8-27b-fp8-fp16kv-dspark-tp{2,4}-optimized \
  --tokenizer Qwen/Qwen3.8-27B-FP8 \
  --output-dir benchmark/qwen38_27b_fp8_dspark_tp_scaling_20260815/tp{2,4} \
  --concurrency 1 \
  --output-tokens 256
```

## Results

| Context | TP2 prefill | TP4 prefill | TP2 decode | TP4 decode | TP2 accept | TP4 accept |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1,761.0 | 2,748.7 | 76.5 | 107.8 | 3.048 | 3.122 |
| 3K | 1,916.3 | 3,497.1 | 60.8 | 77.6 | 2.510 | 2.226 |
| 5K | 1,887.9 | 3,309.5 | 67.8 | 98.8 | 2.844 | 2.844 |
| 7K | 1,904.0 | 3,475.2 | 104.1 | 152.2 | 4.571 | 4.571 |
| 9K | 1,887.7 | 3,354.5 | 60.7 | 84.7 | 2.753 | 2.586 |
| 11K | 1,843.8 | 3,384.3 | 59.6 | 87.8 | 2.753 | 2.753 |
| 13K | 1,829.8 | 3,350.1 | 79.0 | 117.5 | 3.765 | 3.765 |
| 15K | 1,796.2 | 3,310.5 | 58.6 | 79.9 | 2.876 | 2.612 |
| 17K | 1,778.2 | 3,278.3 | 56.9 | 86.7 | 2.844 | 2.844 |
| 19K | 1,746.3 | 3,255.7 | 44.4 | 61.1 | 2.265 | 2.116 |
| 21K | 1,767.5 | 3,245.9 | 47.1 | 72.2 | 2.485 | 2.485 |
| 23K | 1,755.2 | 3,242.0 | 84.8 | 129.0 | 4.571 | 4.491 |
| 25K | 1,686.0 | 3,140.4 | 51.9 | 78.3 | 2.844 | 2.844 |

Rates are tokens per second. Prefill is exact input tokens divided by client
TTFT, and decode excludes TTFT. TP4/TP2 geometric-mean speedup across all 13
points is 1.81x for prefill and 1.44x for client-visible decode. Acceptance is
included because it is prompt-dependent and materially affects DSpark decode
throughput. Compared with the initial branch benchmark, geometric-mean decode
improved by 8.0% on TP2 and 8.9% on TP4 while prefill stayed within 0.5%.

All 26 requests returned HTTP 200, used the exact requested input length,
generated 256 tokens, used zero cached tokens, and passed the repetition and
coherence audit. All prompt hashes matched across TP2 and TP4; 8 of the 13
greedy completions were byte-identical, and the other five passed their
independent output audits.

Raw requests, server configurations, CSV summaries, and JSON summaries are in
[`tp2`](tp2) and [`tp4`](tp4).
