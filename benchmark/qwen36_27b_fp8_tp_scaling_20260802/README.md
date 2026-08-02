# Qwen3.6-27B-FP8 TP2/TP4 context scaling

This directory contains the audited 1K-to-25K DFlash context-scaling sweep run
on four V100-SXM2-32GB GPUs on 2026-08-02. The target checkpoint is
`Qwen/Qwen3.6-27B-FP8`; FP8 refers to the target weights. Both runs use FP16 KV
(`--kv-cache-dtype auto`) so the TP2 and TP4 comparison has the same validated
cache format.

## Benchmark envelope

- Exact input lengths: 1,000 through 25,000 in 2,000-token increments.
- One cold-cache request per point, concurrency one, 256 greedy output tokens.
- DFlash draft `z-lab/Qwen3.6-27B-DFlash`, block size 16, spec-v2 enabled.
- V100 FlashAttention backend, 4,096-token prefill chunks, 32,768-token server
  context and token pool.
- TP2 ran on GPUs 2 and 3; TP4 ran on all four GPUs.
- `SGLANG_SM70_FP8_PREFILL_BACKEND=turbomind` selects the compact V100 target
  layout. The automatic prefill bridge retains a second checkpoint-weight copy
  and cannot fit the target plus DFlash draft on two 32GB GPUs.
- Both runs set `NCCL_P2P_LEVEL=NVL`,
  `SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage`, the Mamba convolution and SSM dtypes to
  FP16, and enabled spec-v2 plus the overlap plan stream.

The server command was the README Qwen3.6 FP8 DFlash command with
`--kv-cache-dtype auto`, `--tensor-parallel-size {2,4}`,
`--context-length 32768`, `--max-total-tokens 32768`, and
`--max-running-requests 1`. The benchmark driver invocation was:

```bash
python benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key qwen3.6-27b-fp8-fp16kv-dflash-tp{2,4} \
  --tokenizer Qwen/Qwen3.6-27B-FP8 \
  --output-dir benchmark/qwen36_27b_fp8_tp_scaling_20260802/tp{2,4} \
  --concurrency 1 \
  --output-tokens 256
```

Each brace value above represents a separate server and benchmark run. The
driver flushes the radix cache before every measured request and retains the
full response for output auditing.

## Results

![TP2 versus TP4 context scaling](context_scaling.svg)

| Context | TP2 prefill | TP4 prefill | TP2 decode | TP4 decode | TP2 accept | TP4 accept |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1,754.8 | 2,773.5 | 115.9 | 154.0 | 5.333 | 5.333 |
| 3K | 1,741.0 | 3,483.8 | 72.6 | 94.8 | 3.507 | 3.507 |
| 5K | 1,888.5 | 3,334.4 | 95.9 | 127.3 | 4.741 | 4.741 |
| 7K | 1,847.6 | 3,493.7 | 71.4 | 93.3 | 3.556 | 3.556 |
| 9K | 1,890.1 | 3,366.6 | 92.8 | 121.5 | 4.655 | 4.655 |
| 11K | 1,802.5 | 3,351.4 | 110.1 | 146.1 | 5.565 | 5.565 |
| 13K | 1,797.6 | 3,352.9 | 67.2 | 88.1 | 3.413 | 3.413 |
| 15K | 1,771.1 | 3,332.5 | 88.0 | 111.4 | 4.491 | 4.267 |
| 17K | 1,757.0 | 3,273.2 | 79.4 | 105.7 | 4.063 | 4.063 |
| 19K | 1,730.4 | 3,261.7 | 118.3 | 157.6 | 6.095 | 6.095 |
| 21K | 1,746.7 | 3,253.7 | 67.5 | 85.7 | 3.507 | 3.368 |
| 23K | 1,739.4 | 3,244.8 | 74.6 | 101.3 | 3.938 | 4.000 |
| 25K | 1,680.7 | 3,127.5 | 90.1 | 126.2 | 4.830 | 4.923 |

Rates are tokens/second. Prefill is exact input tokens divided by client TTFT;
decode excludes TTFT. TP4/TP2 geometric-mean speedup is 1.84x for effective
prefill and 1.32x for client-visible decode across all 13 points. Decode is
intentionally shown alongside acceptance: prompt-dependent draft acceptance
causes the saw-tooth curve and prevents interpreting every point as pure GPU
scaling. These are single cold trials, not confidence intervals.

All 26 requests returned HTTP 200, consumed the exact requested input length,
generated 256 tokens, used zero cached tokens, and passed the benchmark's
repetition/coherence audit. All 13 prompt hashes match across TP2 and TP4.
Run `python audit_results.py` to reproduce the integrity checks and summary.

TP2 DFlash with E4M3 KV is not included: live end-to-end testing found corrupt
output even though the individual attention kernels matched FP16 references.
The V100 backend now fails fast for that unsafe combination. Use FP16 KV for
TP2; E4M3 DFlash remains available in its validated TP4 layout.

Raw and summarized results live under [`tp2`](tp2) and [`tp4`](tp4). Regenerate
the chart with `python plot_results.py`.
