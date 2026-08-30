# Qwen3.8 Flash Next QSA prefill-fix Docker benchmark

This benchmark validates the image published as
`geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v2` with
`RadixArk/Qwen3.8-Flash-Next-NVFP4` on 4x V100-SXM2-32GB (TP4), both target
only and with the checkpoint's built-in MTP layer.

- Image ID: `sha256:f3ff54e8b6c09c70a75d85ea23edac49c265a820df6e1077c5d3ca7844a6c861`
- Registry manifest: `sha256:f3af2ff8118bf810bd035a2acfd086f3f3de33d335455d41be208f419fe76648`
- Image creation time: `2026-08-29T16:31:40.616390095+10:00`
- Benchmark date: 2026-08-30
- Raw measured results: [`results.jsonl`](results.jsonl)
- All measured requests used exact token-ID input/output lengths, ignored EOS,
  used seed `20260830`, arrived at infinite request rate, and completed with no
  request errors.

## Results

### 1,000 input -> 25,000 output, concurrency 1

| Mode | Duration | Output throughput | TTFT | TPOT | Accept length | Relative throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target only | 416.03 s | 60.09 tok/s | 434.93 ms | 16.62 ms | - | 1.000x |
| MTP-3/4 | 283.86 s | 88.07 tok/s | 486.73 ms | 11.33 ms | 3.493 | 1.466x |

MTP improved sustained output throughput by 46.6% and reduced the 25K-output
request duration by 132.17 seconds.

### 8,192 input -> 1,024 output concurrency sweep

Output throughput is aggregate and includes prefill time in the benchmark
duration. TPOT is the mean per-request time per output token excluding the
first output token.

| Concurrency | Target output tok/s | MTP output tok/s | MTP delta | Target TTFT | MTP TTFT | Target TPOT | MTP TPOT | MTP accept length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 55.12 | 74.82 | +35.7% | 1,552.28 ms | 1,637.65 ms | 16.62 ms | 11.76 ms | 3.375 |
| 4 | 137.26 | 149.25 | +8.7% | 4,856.90 ms | 5,049.17 ms | 24.39 ms | 20.04 ms | 3.089 |
| 8 | 172.98 | 194.13 | +12.2% | 8,000.80 ms | 8,329.74 ms | 38.43 ms | 28.94 ms | 3.130 |
| 16 | 239.36 | 183.87 | -23.2% | 14,063.34 ms | 14,524.25 ms | 53.12 ms | 55.64 ms | 3.048 |

MTP helps through concurrency 8. At concurrency 16, its speculative work and
slow-request tail outweigh the accepted-token gain for this workload; target
only is faster.

## Server sizing

Both modes used:

- FP16 activations, ModelOpt NVFP4 weights, and E5M2 KV cache.
- TileLang V100 full attention, TileLang linear-attention prefill, and Triton
  linear-attention decode.
- `--context-length 262144`, `--chunked-prefill-size 8192`, and
  `SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192`.
- `--max-running-requests 16` and `--max-mamba-cache-size 80`. The
  `extra_buffer` strategy consumes five Mamba slots per live request, so 80
  slots are required for a true concurrency ceiling of 16.
- `--cuda-graph-max-bs 16 --cuda-graph-bs 1 2 4 8 16`, covering every
  requested load point.

Target-only used `--mem-fraction-static 0.85`, which produced 536,512 KV-token
slots. The MTP 1K->25K run also used 0.85. An initial MTP concurrency-16
warmup at 0.85 exhausted transient memory in the PLE `conv1d` prefill path
while requesting another 160 MiB. The concurrency matrix therefore used
`--mem-fraction-static 0.82`, retaining 221,616 KV-token slots (well above the
147,456 tokens required by 16 x (8,192 + 1,024)) while adding about 1 GiB of
transient headroom. The retried c16 warmup and measured c16 run both completed.

MTP added:

```text
--speculative-algorithm EAGLE
--speculative-draft-model-path RadixArk/Qwen3.8-Flash-Next-NVFP4
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

## Measurement protocol

Each mode was launched once with the complete graph list. A short same-shape
request warmed prefill/decode kernels before measurement. Every recorded point
then flushed the radix/Mamba caches. MTP speculative counters were reset with
an empty `/set_internal_state` update before every recorded MTP point.

The measured client command for concurrency `C` was:

```bash
HF_HUB_OFFLINE=1 \
/home/rah/miniconda3/envs/sglang-v100/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 8082 \
  --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen \
  --tokenizer RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --dataset-name random --num-prompts C \
  --random-input-len 8192 --random-output-len 1024 \
  --random-range-ratio 1.0 --max-concurrency C \
  --warmup-requests 0 --flush-cache --tokenize-prompt \
  --seed 20260830 --disable-tqdm \
  --output-file benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/results.jsonl
```

The long-decode command changed `--num-prompts`, `--max-concurrency`,
`--random-input-len`, and `--random-output-len` to `1`, `1`, `1000`, and
`25000`, respectively.
