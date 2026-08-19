# V100 long-context lessons from 1Cat-vLLM 1.3.0

This note records the analysis and acceptance work used to transfer the useful
parts of 1Cat-vLLM's `v1.2.2..v1.3.0` SM70 campaign into SGLang-V100 and
TileLang-FA-V100. The compared release commits are:

- v1.2.2: `644d8a7cd05ed4ecd1cd188e3c05b4bbd074f504`
- v1.3.0: `6ada86ed64af6d1a7b3cb0f34df237fd86f06d48`

The main conclusion is that Volta long-context performance is controlled more
by operand dependency depth and launch decomposition than by nominal FLOPs or
HBM bandwidth. A dependent L2-hit global load is roughly two orders of
magnitude more expensive than one HMMA step. Consequently, fewer instructions
can still lose when they serialize the load/shared-memory/HMMA chain.

## What changed in 1Cat

The D=256 prefill campaign changed the actual dataflow rather than only tuning
tile constants:

1. The native kernel splits D=256 QK work into D64 ownership groups, uses
   Volta-specific tensor-core fragments, and reorganizes PV around D128
   ownership with double-buffered operands. FP32 online-softmax state is kept
   exact.
2. Long single-sequence chunks gather paged K/V once into reusable dense
   workspaces. This changes repeated page-indirected access inside attention
   into an O(K) copy followed by regular O(MK) access. At M=4096, the copy is
   amortized by thousands of query rows.
3. The split-KV3 path turns increasing K into three independent CTAs per
   query/head and merges the three FP32 softmax states exactly. Without it,
   longer K only lengthens a fixed CTA set.
4. Long decode uses native grouped XQA kernels, page-layout specializations,
   device-side active-partition control, and exact FP32 state reduction. This
   is fundamentally different from merely raising a Triton split count.
5. After attention moved out of the way, M=4096 AWQ projections became the
   bottleneck. Selected TP4 projections are expanded into one reusable FP16
   KxN workspace and immediately consumed by cuBLAS, while decode and partial
   chunks keep the compact TurboMind representation.

The sequence is an Amdahl's-law migration: optimize attention, profile again,
then optimize the projection path that becomes dominant.

## Accepted SGLang routes

### Dense-gather and exact native D256 prefill

The accepted route is deliberately narrow: FP16, B=1, Hq=6, Hkv=1, D=256,
M>=3920, causal full attention, and a sufficiently long total sequence. It
resolves paged K/V once into reusable dense workspaces, then calls the exact
1Cat split-D N32 CUDA kernel built from the v1.3.0 FlashAttention-V100 patches.
The normal E5M2 production chunk is M=8192; the accepted operator also covers
the M=15680 geometry that is best for FP16 KV in 1Cat. Non-64-aligned Q tails
use causal-safe padding; a full 4000-token prompt suffix-pads Q/K/V to 4032,
while a Q tail with an existing prefix is left-padded as in 1Cat. This preserves
the important part of the design: Volta-specific fragment ownership and its
deliberately scheduled global/shared/HMMA dependency graph.

For ordinary page-16 FP16 KV, admission starts at 8K. For the page-784 logical
FP16 workspace produced by the E5M2 bridge, the first 3920-token-or-larger chunk
is already dense and is admitted immediately. The TileLang N32 dense kernel is a
portable fallback if the optional native operator is absent, and unsupported
shapes retain the direct paged/Triton routes. The main rollback controls are
`SGLANG_V100_PREFILL_D256_GATHER=0` and
`SGLANG_V100_PREFILL_D256_NATIVE=0`.

Isolated adapter measurements on one Tesla V100-SXM2-32GB, including both K
and V gathers, were:

| Total K | Direct paged | Gathered dense | Speedup |
| ---: | ---: | ---: | ---: |
| 8K | 7.064 ms | 6.212 ms | 1.14x |
| 32K | 30.634 ms | 24.176 ms | 1.27x |
| 64K | 56.790 ms | 49.234 ms | 1.15x |

The standalone TileLang adapter measured 30.823 to 24.200 ms at 32K (1.27x)
with bitwise-identical output. After native integration, an interleaved 32K
comparison measured 17.833 ms for exact native dense versus 23.955 ms for the
TileLang dense fallback, another 1.34x kernel improvement. The native kernel's
FP16 output had relative L2 error about 3.46e-4 and maximum absolute error about
1.9e-6 against the existing implementation.

`SGLANG_V100_PREFILL_D256_SPLITKV3=1` remains an experimental switch. It is
off by default for the 80-SM host used here; the investigation is recorded
below rather than assuming that 1Cat's 72-SM wave geometry transfers unchanged.

### Native D256 long decode

For the same H6/Hkv1/D256 FP16 shape with page size 16 and B=1, SGLang now
uses the v1.3.0 XQA p256 specialization with dual CTAs, contiguous block-16
layout, and the split reducer. It constructs the native page table from the
Triton backend's existing logical token order and keeps launch capacity fixed
during CUDA graph replay while the device-side active-partition tensor tracks
the runtime sequence length.

All unsupported dtypes, head shapes, batches, sliding windows, sinks, logit
caps, or temperature transforms retain the existing Triton decode path.
`SGLANG_V100_LONG_DECODE_XQA=0` disables the route.

Isolated measurements were:

| K | Existing Triton | Native XQA p256 | Speedup |
| ---: | ---: | ---: | ---: |
| 32K | 0.325 ms | 0.117 ms | 2.77x |
| 64K | 0.588 ms | 0.230 ms | 2.56x |
| 128K | 1.119 ms | 0.445 ms | 2.51x |

An integrated 64K call including page-table construction measured about
0.245 ms. FP16 output differed from the scalar native reference by at most
4.77e-7 in the post-install smoke check.

### Bounded exact-dense block-FP8 prefill

Qwen3.8-27B-FP8 uses block-wise E4M3 weights with 128x128 scales. Its normal
decode and small-M projection path remains the compact TurboMind W8A16 kernel,
including the fused gate/up SiLU epilogue. For the measured TP4 full-chunk
projection shapes and M >= 3920, the backend instead dequantizes one projection
exactly into a single reusable 85 MiB FP16 KxN workspace and immediately calls
the FP16 GEMM. The same allocation is reused across projections and layers and
is safe under CUDA graph capture.

The admitted Qwen3.8 TP4 shapes are gate/up (K=5120, N=8704), down
(K=4352, N=5120), and output (K=1536, N=5120). Interleaved V100 measurements
at M=4096 were:

| Projection | Compact TurboMind | Exact dense | Speedup |
| --- | ---: | ---: | ---: |
| Gate/up | 6.166 ms | 4.492 ms | 1.37x |
| Down | 2.991 ms | 2.161 ms | 1.38x |
| Output | 1.118 ms | 0.823 ms | 1.36x |

The results were bitwise identical. `SGLANG_SM70_FP8_PREFILL_EXACT_DENSE=0`
disables this route, while `SGLANG_SM70_FP8_PREFILL_BACKEND=turbomind` forces
the compact path. This is the FP8-weight analogue of 1Cat's AWQ optimization:
at large M, removing inline unpack/dequantization from the GEMM dependency
chain matters more on Volta than retaining the compact representation for
every operation.

### Native E5M2 KV path for Qwen3.8

The optimized compact KV format is E5M2, kept separate from E4M3 despite both
being byte-addressed in the native API. Four pieces are required so compact KV
does not merely trade memory for a large conversion penalty:

1. A native SM70 cache writer quantizes fresh FP16 K/V directly to E5M2. It
   measured 22--24 microseconds versus 105--109 microseconds for the previous
   path and was byte-exact in the tested cases.
2. Full prefill chunks use 1Cat's vectorized physical-page resolver/converter
   to expand each active E5M2 K/V element once into reusable logical page-784
   FP16 workspaces, then run the native dense D256 kernel. The bridge alone
   measured about 0.110/0.185/0.363/0.729 ms at 32K/64K/128K/256K. This O(K)
   pass is amortized by the O(MK) attention operation.
3. Long G6/D256 decode calls the native p256 E5M2 XQA kernel directly. It
   measured about 0.176/0.287/0.485/0.822 ms at 32K/64K/128K/256K, with
   relative L2 error about 3.5e-4.
4. DSpark/DFlash target verification uses the same native independent-row
   E5M2 XQA route with fixed graph capacity and device-side active partition
   counts. Unsupported draft/extend shapes have a software E5M2 decoder in the
   SM70 Triton kernels rather than interpreting E5M2 bytes as E4M3.

`SGLANG_V100_FP8_PREFILL_SCRATCH=0` disables the prefill bridge. FP16 KV
(`--kv-cache-dtype auto`) is still faster when its 2x capacity cost is
acceptable; E5M2 is the optimized capacity-oriented choice.

### Bounded exact-dense AWQ prefill

The private SM70 TurboMind binding now exposes the v1.3.0 exact dequantizer.
All supported V100 AWQ layers use the compact TurboMind layout for decode and
partial chunks. Only the measured TP4, group-128 projection shapes receive a
shared 85 MiB FP16 workspace, and only exactly M=4096 expands into it before
`torch.mm`/cuBLAS. The original AWQ tensors are released after TurboMind
preparation, avoiding permanent compact-weight duplication.

The reconstructed KxN weight is bitwise identical to 1Cat's explicit exact
FP16 expansion, and the M=4096 result is bitwise identical to a direct GEMM
using that weight. On the tested K=1536, N=5120 projection:

| Path | Latency |
| --- | ---: |
| Compact TurboMind AWQ | 1.131 ms |
| Dequantize + cuBLAS | 0.798 ms |

This is a 1.42x operator speedup. `SGLANG_SM70_AWQ_PREFILL_EXACT_DENSE=0`
disables only the workspace route;
`SGLANG_SM70_AWQ_TURBOMIND=0` disables the complete SM70 AWQ backend.

## Explored but not enabled by default

Split-KV3 received two independent implementations and repeated warm,
interleaved measurements; it was not discarded after a single port attempt.

First, the faithful TileLang implementation launches three partial-attention
CTAs per query/head and performs the exact FP32 max/sum/output merge. It was
correct, but consistently 6.6--7.4% slower than the unsplit TileLang kernel on
this 80-SM V100 host:

| K | Unsplit TileLang | TileLang split-KV3 |
| ---: | ---: | ---: |
| 32K | 23.837 ms | 25.549 ms |
| 64K | 48.182 ms | 51.747 ms |
| 128K | 98.280 ms | 104.797 ms |

Second, the actual patched 1Cat CUDA split-D/split-KV3 operator was built and
integrated, including reusable partial-state workspaces and the native FP32
merge. Once warm and interleaved against the accepted native unsplit kernel,
it ranged from effectively neutral to about 1.3% slower. The crucial difference
from 1Cat's reported gain is hardware geometry: its 384 base CTAs form an
underfilled sixth wave on 72 SMs, while this host exposes 80 SMs and therefore
does not receive the same exact-wave benefit from 1152 CTAs. The implementation
is retained behind `SGLANG_V100_PREFILL_D256_SPLITKV3=1` for 72-SM systems and
future profiling, but is not the default on the measured machine.

The E5M2 decode path now retains a p256 graph/workspace envelope but selects
p256 or p1024 on device at 61,633 tokens, then uses two CTAs per partition from
81,921 tokens. SGLang's 16-token allocator pages cannot use 1Cat's p1024 shared
partition-page-ID array (64 IDs would exceed its 16-ID capacity), so page 16
uses tile-local page IDs with the same paired 16-byte E5M2 loads and contiguous
Hkv=1 address specialization. Large-page layouts retain the original cached-ID
route.

Isolated Hq=6/Hkv=1/D=256 E5M2 XQA measurements before and after the page-16
specialization were:

| Context | Previous page 16 | Adaptive page 16 | Improvement |
| ---: | ---: | ---: | ---: |
| 32K | 0.3604 ms | 0.2867 ms | 20.5% |
| 64K | 0.5156 ms | 0.3968 ms | 23.0% |
| 128K | 0.7537 ms | 0.4905 ms | 34.9% |
| 256K | 1.2257 ms | 0.8172 ms | 33.3% |

Randomized-page boundary checks at 16,384/16,385, 61,632/61,633,
81,920/81,921, 131,071, and 262,143 tokens had worst relative L2 error
4.78e-4 and worst absolute error 7.63e-6 against scalar paged decode.

The E5M2 chunk size was selected end to end rather than inherited from the
FP16-KV result. The primary production sweep used ordinary target-only q=1
decode: speculative decoding was absent both from the environment and from the
server arguments. With M=8192, one cold-cache request, 128 generated tokens,
and TP4, it measured:

| Input | Prefill | Target-only decode |
| ---: | ---: | ---: |
| 4K | 3,972 tok/s | 57.7 tok/s |
| 64K | 3,197 tok/s | 40.6 tok/s |
| 80K | 3,092 tok/s | 37.3 tok/s |
| 128K | 2,669 tok/s | 30.0 tok/s |
| 256K | 1,661 tok/s | 20.1 tok/s |

This reaches the approximately 4K tok/s short-context target and replaces the
old abrupt post-70K prefill drop with a gradual curve. The isolated attention
table above proves the adaptive page-16 decode kernel improvement, but the
full-model target-only table also shows its limit: at 256K, projection,
Mamba/GDN, collectives, sampling, and the remaining O(K) KV traffic dominate
enough that end-to-end decode is still about 20 tok/s. No speculative acceptance
rate is involved in those numbers.

As a secondary experiment on the same TP4 Qwen3.8-27B-FP8 + DSpark server and
identical audited prompts, M=8192 beat M=15680 by 17.2% at 64K, 21.0% at 80K,
and 28.0% at 128K. The DSpark measurements were:

| Input | Prefill | Client-visible decode | DSpark accept length |
| ---: | ---: | ---: | ---: |
| 4K | 3,832 tok/s | 113.0 tok/s | 3.200 |
| 64K | 3,234 tok/s | 47.8 tok/s | 2.462 |
| 80K | 3,033 tok/s | 44.7 tok/s | 2.560 |
| 128K | 2,633 tok/s | 38.4 tok/s | 2.909 |

DSpark decode throughput is prompt- and acceptance-dependent and must not be
compared directly with the target-only curve. A repeated target-only 4K request
after workspace warmup reached 3,968 tok/s.

## Installation and validation contract

The host installer and V100 Dockerfile pin 1Cat-vLLM v1.3.0, not the older
July source commit. They also build the exact D256 FA2 operators from pinned
FlashAttention-V100 source plus 1Cat's pipeline and split-KV3 patches. E4M3 and
E5M2 dispatch remain explicitly separate despite their common byte storage. A
clean smoke test produced finite outputs with maximum absolute errors of
4.77e-7 for FP16 XQA and 1.91e-6 for E4M3 XQA; E5M2 long-decode validation used
relative L2 error about 3.5e-4.

Regression tests cover the bounded gather policy, shuffled-page prefill
equivalence, native and fallback decode controls, E4M3/E5M2 separation, E5M2
cache writes and bridge shape contracts, speculative graph metadata, compact
FP8/AWQ decode, exact M=4096 projection output, and bounded reusable workspaces.

Real TP4 Qwen3.8-27B-FP8 serves with E5M2 KV completed the boundary-case
32,769-token request in both target-only and DSpark-7 configurations. The full DSpark command captured both
CUDA graphs, returned eight tokens with HTTP 200, then returned 64 more tokens
from a 32,768-token cached prefix while repeatedly exercising native E5M2
target verification. These are integration acceptance results, not controlled
before/after TTFT or TPOT claims; attention, projections, collectives, and graph
replay must still be profiled together because each accepted change moves the
next bottleneck.

The long-context E5M2 DSpark command uses `--chunked-prefill-size 8192` and
`--max-total-tokens 262144`. Without the explicit token-pool cap, the automatic
policy allocated 1,167,952 target and draft cache slots despite
`--max-running-requests 1`; during the 15680-chunk comparison this left only
35 MiB at the first large prefill and caused an activation OOM. With the cap,
idle headroom was 13.7 GiB per rank.
