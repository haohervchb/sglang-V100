# Flash-Next v5.2: decode kernel bundle and 4K prefill

Four V100-SXM2 32 GB, NVLink, TP4, MTP-3/4, FP16 KV, eight slots, 262K context.
Base is v5.1 (`2a4c37ef61`). All arms ran sequentially on the same host with the
balanced profile; "fix" is the QSA indexer FP16-cache fix alone.

## Changes

Decode (`52996dfe53`):

- The fused QSA indexer prep stays active when the RoPE cache is FP16. It had
  been silently disabled, so every graph used a ~55-kernel eager chain.
- GDN verify BV tile for N > 1 rows; QSA graph metadata tiling above 8 rows;
  q/k norm off the indexer stream on target layers; topk=1 draft-chain
  constants; fused two-kernel NVFP4 MoE decode.

Prefill (`28749dcc1a`):

- The indexer prefill scorer splits each row block's key range across CTAs
  (bit-identical logits, 2.2-4.5x at 32K-62K compressed keys).
- From 8192 compressed keys, each TP rank scores a quarter of the rows and the
  ranks all-gather block ids. The indexer is replicated, so results are exact.
- Sequence lengths and the maximum RoPE position are read once per forward
  instead of once per QSA layer, removing ~22 device-to-host syncs per chunk.
- Grouped Page4 QSA prefill accepts any chunk up to 8192 rows.
- 4096-token chunks. `SGLANG_CHUNKED_PREFILL_CACHE_ALIGN=2048` ends a long
  final chunk on a 2K boundary so prefix-cache checkpoints stay where 2K chunks
  put them.

## Serving throughput (tok/s, two repeats, 8 requests per point)

| Workload | v5.1 | Fix | v5.2 |
| --- | ---: | ---: | ---: |
| Prose 1K/1024, C1 | 114.4 | 129.5 | 141.9 |
| Prose C4 | 250.0 | 273.3 | 338.7 |
| Prose C8 | 359.1 | 378.4 | 503.9 |
| Copy-edit 8K/512, C1 | 102.2 | 112.6 | 126.5 |
| Copy-edit C4 | 150.3 | 161.8 | 194.1 |

Acceptance is unchanged (prose 2.56/2.55/2.58, copy-edit 4.00). Output hashes
equal the fix arm for prose C1/C8 and copy-edit C1/C4 (64/64). Prose C4 is
not repeat-stable on the earlier builds (2/16), so it is not compared.

## Fresh-prompt prefill (time to first token, cache flushed)

| Prompt | Decode bundle, 2K chunks | v5.2 | Change |
| ---: | ---: | ---: | ---: |
| 8K | 2.06 s | 1.84 s | -11% |
| 25K | 6.08 s | 5.34 s | -12% |
| 70K | 17.53 s | 15.10 s | -14% |
| 160K | 43.0 s | 35.0 s | -19% |

Peak GPU memory during the 160K prefill is 29.9 GB of 32.5 GB with the KV pool
unchanged (277,280 tokens).

## Functional gates (v5.2)

Full checks 8/8 including 262K retrieval and oversized rejection, tool calls,
shared document at C8 (cached 30,720-32,384 tokens per answer) and multimodal
11/12, with only the known `V10 CHECK 4827` OCR miss that v5.1 also has.

Raw results, per-arm notes and the bisection history are in `raw/`.
