# Qwen3.8-27B on V100

[Back to the main README](../../../README.md) · [Host setup](../../../README.md#install-on-the-host) · [Docker setup](../../../README.md#docker)

Activate `sglang-v100` before running host commands. Examples use four
V100 32 GB GPUs; each command specifies its listening port.

The commands below use the FP8 checkpoint. FP16 checkpoint measurements
are retained in the [DFlash2 report](../../../benchmark/qwen38_27b_fp16_dflash2_v100_20260821/README.md)
and [DSpark report](../../../benchmark/qwen38_27b_fp16_dspark_v100_20260821/README.md).

## Qwen3.8-27B-FP8 target-only

This is the reference command for ordinary, non-speculative decode. No
speculative environment switch or `--speculative-*` argument is present. The
SM70 CUDA read-once split-KV decode partial and the QPN8 W8A16 decode GEMMs are
enabled by default; set `SGLANG_V100_DECODE_CUDA=0` or
`SGLANG_SM70_FP8_DECODE_QPN8=0` to opt out. With the current CUDA partial and
QPN8 paths, target-only decode measures 59.1 tok/s at 70K and 49.6 tok/s at
200K, up from about 30 tok/s at 128K before. QPN8's word-parallel decoder and
fused gate/up SiLU path are also default-on; use
`SGLANG_SM70_FP8_QPN8_FASTDEC=0` and
`SGLANG_SM70_FP8_QPN8_FUSED_GATE=0` for a controlled rollback. The CUDA
partial covers TP1/TP2/TP4 (the GQA ratio is fixed at 6:1, so tensor-parallel
splits change only the per-rank KV-head count).

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --dtype float16 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.75 \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 8192 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 1 \
  --cuda-graph-bs 1 \
  --enable-nccl-nvls \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

## Qwen3.8-27B-FP8 with DFlash2

The published DFlash2 checkpoint uses block size 8: one anchor plus seven
proposed tokens. Its selector top-k of 16 is the number of candidates scored at
each proposal position, not the proposal length.

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --dtype float16 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.75 \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 8192 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 1 \
  --cuda-graph-bs 1 \
  --enable-nccl-nvls \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 \
  --speculative-dflash-block-size 8 \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-window-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

The V100 path keeps the target KV cache in E5M2 but uses FP16 for the much
smaller five-layer draft cache. Both the DFlash2 candidate selector and draft
forward are captured in the CUDA graph. Block 16 can be A/B tested by changing
only `--speculative-dflash-block-size 8` to `16`; this is outside the checkpoint's
published block-8 configuration and should be selected by measured output
throughput rather than acceptance length alone.

Bring-up results on four V100-SXM2-32GB GPUs, TP4, E5M2 target KV, CUDA graphs,
and 256 or 512 greedy output tokens:

| Input/workload | Context handling | Output throughput | Average commit length |
| --- | --- | ---: | ---: |
| Short prompt, warmed | Warm graph and kernels | 118.1 tok/s | 3.66 |
| 4K periodic synthetic prompt | Full prompt included in client time | 121.5 tok/s end-to-end | 7.11 |
| 70K periodic synthetic prompt | 69,952 prompt tokens reused | 79.2 tok/s end-to-end | 4.13 |
| 200K periodic synthetic prompt | Steady scheduler decode intervals | ~60 tok/s | 4.45 |
| 150K cold-cache prompt | 150,000 prompt tokens, fresh server, zero cached tokens | 134 tok/s | ~3.5 |
| 200K cold-cache prompt | 200,000 prompt tokens, fresh server, zero cached tokens | 112 tok/s | ~3.5 |

Acceptance is workload-dependent. The synthetic rows validate graph capture,
long-context KV operation, and selector stability; use the same prompt corpus
for block-8 versus block-16 comparisons. The cold-cache rows used
`--random-input-len 150000/200000`, one request, and 256 greedy output tokens
on a freshly restarted server.

### Serve DFlash2 in Docker

Build (or pull) the container image as shown in the
[Docker section](../../../README.md#docker), then launch the same DFlash2 workload in a
container. The image bakes in the V100 defaults (`NCCL_P2P_LEVEL=NVL`,
`SGLANG_MAMBA_CONV_DTYPE=float16`, `SGLANG_MAMBA_SSM_DTYPE=float16`); the
remaining flags are passed explicitly.

```bash
docker rm -f v100-dflash2 2>/dev/null

docker run --rm --name v100-dflash2 \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
  -e SGLANG_ENABLE_SPEC_V2=1 \
  -e SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
  sglang-v100:latest \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --dtype float16 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.75 \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 8192 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 1 \
  --cuda-graph-bs 1 \
  --enable-nccl-nvls \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 \
  --speculative-dflash-block-size 8 \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-window-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

`--network host` is required for TP4 NCCL on the tested single-node host.
Startup takes several minutes (weight load plus CUDA-graph capture), and the
first request after boot hits a one-time kernel/JIT warmup. The >128K prefill
path uses the native D256 split-D operator that is compiled into this image;
host and container were measured within 0.2% of each other at 131K and 200K.

## Qwen3.8-27B-FP8 with DSpark

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
NCCL_P2P_LEVEL=NVL \
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage \
SGLANG_MAMBA_CONV_DTYPE=float16 \
SGLANG_MAMBA_SSM_DTYPE=float16 \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --dtype float16 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8082 \
  --mem-fraction-static 0.75 \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --max-running-requests 1 \
  --chunked-prefill-size 8192 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 1 \
  --cuda-graph-bs 1 \
  --enable-nccl-nvls \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
  --speculative-dspark-block-size 7 \
  --speculative-draft-model-quantization unquant \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

`fp8_e5m2` is the optimized compact-KV route for this model on V100: cache
writes, long D=256 prefill, long decode, and DSpark target verification all
have native SM70 paths. Use `--kv-cache-dtype auto` for the faster FP16 KV
cache when its lower capacity is acceptable. `fp8_e4m3` remains a compatibility
option, but is not the preferred compact-KV format for Qwen3.8-27B-FP8.
For this single-request 262K configuration, keep `--max-total-tokens` equal to
the context length. Leaving it automatic can allocate target and draft KV pools
for over one million tokens, wasting the activation headroom required by the
optimized prefill kernels.

The following historical sweep was measured on V100-SXM2-32GB with FP8
weights, **FP16 KV**, DSpark block 7, one cold-cache request, and 256 greedy
output tokens. It is not an E5M2 result:

| Input | TP2 prefill | TP4 prefill | TP2 decode | TP4 decode |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,761 tok/s | 2,749 tok/s | 76.5 tok/s | 107.8 tok/s |
| 9K | 1,888 tok/s | 3,355 tok/s | 60.7 tok/s | 84.7 tok/s |
| 17K | 1,778 tok/s | 3,278 tok/s | 56.9 tok/s | 86.7 tok/s |
| 25K | 1,686 tok/s | 3,140 tok/s | 51.9 tok/s | 78.3 tok/s |

Across the complete 1K-to-25K sweep, TP4 is 1.81x faster for prefill and
1.44x faster for decode by geometric mean. See the
[full 13-point benchmark](../../../benchmark/qwen38_27b_fp8_dspark_tp_scaling_20260815/README.md).

## Native SM70 optimization status

The acceptance workload for these changes is the actual
`Qwen/Qwen3.8-27B-FP8` checkpoint with TP4, E5M2 KV, and speculative decoding
off. The post-port cold validation reaches 4,224 prefill tok/s and 63.2 decode
tok/s at 4K, 59.1 decode tok/s at 70K, and 49.6 decode tok/s at 200K. The SM70
CUDA read-once split-KV decode partial removed the old severe context decay;
the fused QPN8 gate/up path then reduced TPOT another 2.6% at 4K and 1.7% at
70K in controlled A/B runs. The full curve and profiler
breakdown are in the [FP8 target-only report](../../../benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/README.md).
The operator measurements below explain individual paths; they are not being
used as a substitute for that FP8 end-to-end result.

| Path | Measured shape | Result |
| --- | --- | ---: |
| Chunked GDN prefill | Qwen3.8 TP4, 2,048 tokens | 1.455 ms vs 2.017 ms previous native schedule (27.9% lower) |
| Chunked GDN prefill | Qwen3.8 TP4, 4,096 tokens | 2.199 ms vs 3.236 ms previous native schedule (32.0% lower) |
| Chunked GDN prefill | Qwen3.8 TP4, 8,192 tokens | 3.725 ms vs 5.674 ms previous native schedule (34.3% lower) |
| Exact D256 tail split-KV | Q=64, K=245,760, Hq/Hkv=6/1 | 4.306 ms vs 40.675 ms unsplit (9.45x) |
| Fused QPN8 gate/up + SiLU | M=1, K=5,120, N=8,704 | 0.059 ms vs 0.062 ms materialized gate/up, bitwise exact (about 1.04x) |
| Mixed FP16/FP32 Gemma RMSNorm | 4,096 x 5,120 | 0.315 ms vs 1.397 ms PyTorch (4.44x) |
| Experimental BFLA sparse attention | Q=4,096, K=32,768, D=256, 10% keep | about 9.8 ms including selection vs 23.9 ms dense (about 2.4x) |

The GDN prefill dispatcher chooses the direct recurrent kernel through 448
tokens and the tensor-core 64-token chunk kernel above that boundary. It
supports packed variable-length batches, row-strided mixed QKV, indexed FP32
state, direct output, and a column-group CTA schedule. Keep decode on Triton in
the reference command: the native fused recurrent decoder is correct, but the
existing Triton decoder remains faster for Qwen3.8 TP4's one-token shape.

The mixed-dtype Gemma residual/RMSNorm route is automatic on SM70 for at least
256 rows with hidden size 5,120, but only for the exact FP16 activation plus
FP32 residual contract. Qwen3.8-27B-FP8 normally keeps this residual in FP16,
so the 4.44x operator result is not claimed as a gain for the primary model.
Set `SGLANG_V100_GEMMA_RMSNORM=0` for an A/B rollback. BFLA is intentionally
disabled by default. An exact all-keep control can be enabled with
`SGLANG_V100_BFLA_PREFILL=1`; actually dropping blocks additionally requires
`SGLANG_V100_BFLA_ALLOW_APPROXIMATE=1` and a keep ratio such as
`SGLANG_V100_BFLA_KEEP_RATIO=0.1`. Sparse mode changes model semantics and must
pass retrieval and long-context quality evaluation before production use.

For strictly greedy, non-speculative requests, the opt-in
`SGLANG_V100_GREEDY_TP_TOP1=1` route exchanges only each TP rank's top candidate
instead of gathering full-vocabulary logits. It fails closed to the ordinary
logits path for sampling, speculative decoding, logprobs, penalties, grammar,
custom logits processors, or logits bias. It therefore does not affect the
official sampling benchmark.
