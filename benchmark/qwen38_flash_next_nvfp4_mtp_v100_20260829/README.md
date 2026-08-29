# Qwen3.8 Flash Next NVFP4 built-in MTP V100 benchmark

Cold-cache 1K and 25K serving measurements for the MTP module embedded in
`RadixArk/Qwen3.8-Flash-Next-NVFP4`.

## Settings

- 4x V100-SXM2-32GB with NVLink, TP4, one request at a time.
- The target and MTP draft source are both
  `RadixArk/Qwen3.8-Flash-Next-NVFP4`.
- EAGLE with three speculative steps, top-k one, and four draft tokens.
- FP16 activations and E5M2 target/draft KV caches.
- TileLang full-attention and GDN prefill; Triton GDN decode.
- 8,192-token prefill chunks and a 262,144-token context limit.
- One exact-length token-ID request per point and 256 forced greedy output
  tokens.

Prefill is the exact input length divided by client time to first token.
Decode is `1 / TPOT`, excluding the first generated token.

## Cold-cache protocol

Each point used the following sequence:

1. Send an exact-shape 256-output-token request to warm its kernels.
2. Flush the radix and Mamba caches with `POST /flush_cache?timeout=60`.
3. Reset the cumulative speculative counters with an empty
   `POST /set_internal_state` update.
4. Send one measured request with the same seed and shape.

The measured invocation for input length `N` was:

```bash
PYTHONPATH="$PWD/python" \
conda run --no-capture-output -n sglang-v100 \
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 8082 \
  --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen \
  --tokenizer RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --dataset-name random --num-prompts 1 \
  --random-input-len N --random-output-len 256 \
  --random-range-ratio 1.0 --max-concurrency 1 \
  --warmup-requests 0 --tokenize-prompt --seed 20260829
```

## Results

| Input | TTFT | Prefill | TPOT | Decode | Accept length |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 496.133 ms | 2,015.6 tok/s | 14.579 ms | 68.59 tok/s | 2.7875 |
| 25,000 | 5,541.488 ms | 4,511.4 tok/s | 13.094 ms | 76.37 tok/s | 3.2250 |

Both requests returned successfully with their exact requested 1,000/25,000
input tokens and 256 output tokens. The retained client output, including the
complete server configuration and per-request timing arrays, is in
[`results.jsonl`](results.jsonl).
