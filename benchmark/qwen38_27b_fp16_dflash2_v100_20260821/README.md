# Qwen3.8-27B (FP16) DFlash2 Docker 1K/25K reference

Cold-cache reference for the non-quantized `Qwen/Qwen3.8-27B` checkpoint
served with FP16 KV and the DFlash2 drafter in the V100 container image.

## Settings

- 4x V100-SXM2-32GB, TP4, image `geesegeesegeese/sglang-v100:latest`.
- BF16/FP16 target weights, FP16 target and draft KV caches, DFlash2 block 8,
  draft window 2,048.
- 32,768-token context and token pool, 4,096-token prefill chunks, one request
  at a time, CUDA graphs at batch size 1, `--enable-nccl-nvls`.
- One cold-cache request per point, 256 greedy output tokens, concurrency 1,
  one-token warmup before the measured points.

The serve command mirrors the top-level README DFlash2 Docker block with
`--model-path Qwen/Qwen3.8-27B`, `--kv-cache-dtype auto`,
`--context-length 32768`, `--max-total-tokens 32768`, and
`--chunked-prefill-size 4096`.

```bash
python benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key qwen3.8-27b-fp16kv-dflash2-tp4-optimized \
  --tokenizer Qwen/Qwen3.8-27B \
  --output-dir benchmark/qwen38_27b_fp16_dflash2_v100_20260821 \
  --concurrency 1 --output-tokens 256 --lengths 1000,25000
```

## Results

| Context | Prefill | Decode | Accept |
| ---: | ---: | ---: | ---: |
| 1K | 2,093.8 tok/s | 86.7 tok/s | 3.71 |
| 25K | 2,991.9 tok/s | 68.6 tok/s | 3.16 |

Both requests returned HTTP 200, used the exact requested input length,
generated 256 tokens, used zero cached tokens, and passed the repetition and
coherence audit.

Raw requests, server configuration, and CSV/JSON summaries are in this
directory.
