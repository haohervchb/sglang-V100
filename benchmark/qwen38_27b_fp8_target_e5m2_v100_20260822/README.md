# Qwen3.8-27B-FP8 target-only V100 acceptance sweep

This is the primary acceptance benchmark for the SM70 work in this tree. It
uses the real `Qwen/Qwen3.8-27B-FP8` checkpoint, FP16 activations, E5M2 KV,
TP4, and ordinary target-only decoding. Speculative decoding was absent from
both the environment and the server arguments.

## Settings

- 4x V100-SXM2-32GB with NVLink, one request at a time.
- `--attention-backend tilelang_fa_v100`.
- TileLang GDN for prefill and Triton GDN for decode.
- 8,192-token prefill chunks, 262,144-token context and token pool.
- One exact-length random-token request per point and 256 generated tokens.
- Prefill is input tokens divided by client TTFT. Decode is `1 / TPOT` and
  excludes the first generated token.

The complete server command is the
[Qwen3.8-27B-FP8 target-only command](../../README.md#qwen38-27b-fp8-target-only)
in the top-level README.

## Cold-cache protocol

Long shape warmup must finish before cache flushing. The benchmark client's
combined warmup/flush option can race the still-running warmup at long context,
receive HTTP 400 from `/flush_cache`, and then accidentally measure a cached
prefix. Each retained point therefore used this sequence:

1. Send one request with the exact measured input and output shape to compile
   and warm all kernels.
2. Retry `POST /flush_cache` until it returns HTTP 200.
3. Run `sglang.bench_serving` with one prompt, exact random lengths, concurrency
   one, `--warmup-requests 0`, and no internal flush.

The measured client invocation for a length `N` was equivalent to:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 8082 \
  --model Qwen/Qwen3.8-27B-FP8 \
  --dataset-name random --num-prompts 1 \
  --random-input-len N --random-output-len 256 \
  --random-range-ratio 1.0 --max-concurrency 1 \
  --warmup-requests 0 --disable-ignore-eos
```

## Results

| Input | TTFT | Prefill | TPOT | Decode |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 342.256 ms | 2,991.9 tok/s | 17.180 ms | 58.21 tok/s |
| 4,096 | 990.204 ms | **4,136.5 tok/s** | 17.356 ms | 57.62 tok/s |
| 25,000 | 6,731.904 ms | 3,713.7 tok/s | 19.703 ms | 50.75 tok/s |
| 70,000 | 23,488.233 ms | 2,980.2 tok/s | 25.760 ms | 38.82 tok/s |
| 128,000 | 54,322.900 ms | 2,356.3 tok/s | 33.289 ms | 30.04 tok/s |

Every request completed and generated exactly 256 tokens. The curve reaches
the requested approximately 4K tok/s short-prefill target at 4K. Long prefill
degrades gradually rather than showing a new host-side routing cliff. Decode
still falls materially with context and remains the main unfinished target.

## FP8-model attribution

An end-to-end GDN ablation on the same FP8 checkpoint compared Triton prefill
GDN with the native TileLang prefill path. The specialized mixed FP16/FP32
RMSNorm route was disabled in both cells.

| Input | Triton GDN TTFT | TileLang GDN TTFT | Improvement |
| ---: | ---: | ---: | ---: |
| 4,096 | 1.02745 s | 0.99365 s | 3.4% |
| 70,000 | 24.11098 s | 23.56628 s | 2.3% |

The mixed-dtype RMSNorm microbenchmark is deliberately not counted as a Qwen
FP8 model gain: the optimized contract requires an FP32 residual, while the
normal Qwen3.8 serve path observed here uses an FP16 residual.

At 70K, an Nsight Systems target-only decode trace attributed approximately:

| Decode bucket | Time/token | Share |
| --- | ---: | ---: |
| TurboMind FP8 projection GEMMs | 10.60 ms | 41.7% |
| Grouped TileLang attention and reduction | 10.35 ms | 40.8% |
| CUTLASS LM head | 0.99 ms | 3.9% |
| Custom all-reduce | 0.63 ms | 2.5% |
| Norms | 0.52 ms | 2.0% |
| GDN and convolution | 0.42 ms | 1.7% |

This is why decode work is focused on FP8 GEMM and long-context attention,
not on the much smaller GDN decode bucket.

## Decode experiments retained or rejected

Direct E5M2-to-FP16 bit expansion removes the dependent LUT load and is exact:
E5M2 and FP16 have compatible sign/exponent placement for this conversion.
The grouped attention microkernel measured 0.065/0.186/0.444/0.766 ms at
1K/25K/70K/128K. The old LUT route was about 0.963 ms at 128K. This conversion
is retained, although full-model TPOT was neutral within run noise because FP8
projection GEMMs are a co-bottleneck.

Two alternatives received implementation and measurement time before being
rejected:

- A scalar grouped-G6 decoder measured 0.876/2.272/3.872 ms at
  25K/70K/128K, versus 0.186/0.444/0.792 ms for the fused tensor-core path.
- A staged QK/PV design measured 0.710/1.801/3.041 ms at
  25K/70K/128K, 3.8--4.3x slower than the fused path after correctness was
  fixed.

Increasing the decode CTA target from 80 to 120 was also slower at 128K. The
current device-side partition count grows with context and caps at one 80-CTA
wave, matching this host's 80 SMs rather than copying 1Cat's 72-SM launch
geometry.
