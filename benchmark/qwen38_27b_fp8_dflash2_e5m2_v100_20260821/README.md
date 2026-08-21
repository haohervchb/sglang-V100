# Qwen3.8-27B-FP8 DFlash2 Docker 1K/25K reference

Two cold-cache points measuring the DFlash2-8 E5M2 route for
`Qwen/Qwen3.8-27B-FP8` in the published V100 container image.

## Settings

- 4x V100-SXM2-32GB, TP4, image `geesegeesegeese/sglang-v100:latest`
  (local tag `sglang-v100:latest`).
- FP8 target weights, E5M2 target KV, FP16 five-layer draft KV, DFlash2
  block 8, draft window 2,048.
- 262,144-token context and token pool, 8,192-token prefill chunks, one
  request at a time, CUDA graphs at batch size 1, `--enable-nccl-nvls`.
- One cold-cache request per point, 256 greedy output tokens, concurrency 1.
- A one-token warmup request primed kernels before the measured points; the
  harness flushes the cache before every point, so both measured requests saw
  zero cached tokens.

The serve command is the DFlash2 Docker block in the top-level README
(`--kv-cache-dtype fp8_e5m2`, `--speculative-algorithm DFLASH`,
`--speculative-dflash-block-size 8`, context 262,144).

```bash
python benchmark/dflash_v100_20260716/run_benchmark.py \
  --model-key qwen3.8-27b-fp8-e5m2kv-dflash2-tp4-optimized \
  --tokenizer Qwen/Qwen3.8-27B-FP8 \
  --output-dir benchmark/qwen38_27b_fp8_dflash2_e5m2_v100_20260821 \
  --concurrency 1 --output-tokens 256 --lengths 1000,25000
```

## Results

| Context | Prefill | Decode | Accept |
| ---: | ---: | ---: | ---: |
| 1K | 1,802.5 tok/s | 136.6 tok/s | 3.77 |
| 25K | 2,700.6 tok/s | 102.3 tok/s | 3.28 |

Both requests returned HTTP 200, used the exact requested input length,
generated 256 tokens, used zero cached tokens, and passed the repetition and
coherence audit.

Raw requests, server configuration, and CSV/JSON summaries are in this
directory.
