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

## Earlier baseline

| Input | TTFT | Prefill | TPOT | Decode |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 342.256 ms | 2,991.9 tok/s | 17.180 ms | 58.21 tok/s |
| 4,096 | 990.204 ms | **4,136.5 tok/s** | 17.356 ms | 57.62 tok/s |
| 25,000 | 6,731.904 ms | 3,713.7 tok/s | 19.703 ms | 50.75 tok/s |
| 70,000 | 23,488.233 ms | 2,980.2 tok/s | 25.760 ms | 38.82 tok/s |
| 128,000 | 54,322.900 ms | 2,356.3 tok/s | 33.289 ms | 30.04 tok/s |

Every request completed and generated exactly 256 tokens. This earlier curve
reaches the requested approximately 4K tok/s short-prefill target at 4K. Long
prefill degrades gradually rather than showing a new host-side routing cliff.
Decode still falls materially with context and motivated the later decode work
below.

## Post-port validation

The later native GDN, exact logical-tail split-KV, QPN8 decoder, and fused
gate/up work used the same target-only server and cold-cache protocol. The
current default configuration produced:

| Input | TTFT | Prefill | TPOT | Decode |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 969.600 ms | 4,224.4 tok/s | 15.820 ms | 63.21 tok/s |
| 70,000 | 22,799.250 ms | 3,070.3 tok/s | 16.910 ms | 59.14 tok/s |
| 200,000 | 105,429.770 ms | 1,897.0 tok/s | 20.170 ms | 49.58 tok/s |

The FP8 decode change was isolated by disabling only
`SGLANG_SM70_FP8_QPN8_FASTDEC` and `SGLANG_SM70_FP8_QPN8_FUSED_GATE` in the
control server:

| Input | Control TPOT | Optimized TPOT | Latency reduction |
| ---: | ---: | ---: | ---: |
| 4,096 | 16.240 ms | 15.820 ms | 2.59% |
| 70,000 | 17.210 ms | 16.910 ms | 1.74% |

The paired gate/up kernel is bitwise equal to materializing QPN8 gate/up,
applying PyTorch FP16 SiLU, and multiplying by FP16 up. At the operator level
it saves roughly 4--20% for decode batches M=1--8. Short output projections
remain on the scalar decoder because the word-parallel decoder was slower for
that K=1,536 shape.

The long-context policy was tested rather than extrapolated from full chunks.
At Q=64/K=245,760, exact 64-way KV splitting measured 4.306 ms versus
40.675 ms unsplit (9.45x, max absolute difference 4.8e-7). A real 198K FP8
A/B, whose 1,392-token final chunk exercises this policy, reduced TTFT from
105.705 s to 104.249 s. Full 8K chunks deliberately stay unsplit because their
existing query/head grid already fills the GPU.

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
