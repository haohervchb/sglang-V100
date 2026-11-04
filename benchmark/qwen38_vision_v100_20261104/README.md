# Qwen3.8 Flash Next V100 vision optimization

On four V100 SXM2 32 GB GPUs with NVLink, a matched 32-image request fell from
**79.98 s to 12.12 s end to end (6.60× faster, 84.85% lower latency)**.
The selected profile retains four running requests, FP16 KV, native MTP-3/4,
262,144-token context, shared-prefix checkpoints and CPU pixel preprocessing.
Text throughput remains around 250 aggregate output tokens/s in this screen.

This is host release `v100-qwen38-flash-next-v5.4`, measured against
`d6a5a6727a` (v5.3.1 plus its validation record). No Docker image was rebuilt.
The checkpoint is `RadixArk/Qwen3.8-Flash-Next-NVFP4` at revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

## Measurements

Each image request uses 32 JPEGs at 768×1024, 24,675 prompt tokens and four
output tokens. The first image is blue and the last red. Each comparison uses
identical JPEG hashes and the same answer, with an acknowledged cache flush
before every request. Both profiles retain CPU preprocessing. Timing includes
HTTP processing, vision encoding, language prefill and generation.

| Configuration | First request | Second request | Mean |
| --- | ---: | ---: | ---: |
| Existing Triton vision attention | 80.03 s | 79.94 s | 79.98 s |
| SM70 Flash-V100 vision attention | 12.31 s | 11.92 s | 12.12 s |

These are two synthetic page-batch repetitions, not a measurement of arbitrary
PDF extraction, OCR quality or a 380 MB PDF. Two 3072×4096 images also returned
the correct answer in 34.68 s with the six-request candidate; that is a
correctness/stress check, without a matched old-backend speed comparison.
Large individual images still cost more than smaller pages with the same total
pixel count because attention works within each image.

Aggregate output throughput below includes prefill and queueing. The fixed
text screen uses eight approximately 1K-input requests, each generating exactly
1,024 tokens. SPEED uses 20 diverse prompts with natural EOS and a 512-token
cap, so its output lengths can differ. Each cell is one batch; small differences
are not evidence of a repeatable throughput gain.

| Profile | Fixed text, 4 clients | Fixed text, 8 clients | SPEED, 4 clients | SPEED, 8 clients |
| --- | ---: | ---: | ---: | ---: |
| Baseline: four active, MTP-3/4 | 253.15 | 248.49 | 221.22 | 224.49 |
| Selected: four active, MTP-3/4, faster vision | 250.44 | 249.70 | 217.34 | 226.00 |
| Candidate: six active, MTP-1/2, faster vision | 232.84 | 249.90 | 196.34 | 235.41 |

Units are output tokens/s. The six-request candidate gained 4.2% over the
selected profile on eight-client SPEED, but lost 7–10% at four clients and did
not improve fixed-length eight-client throughput or shared-document completion
time. Four active requests remains the deployed mixed-workload choice. The
six-request configuration passed its 48-request queue test; its measurements
are retained for future scheduler work.

## Bottleneck and implementation

An eight-page baseline trace spent 17.30 s in the Triton vision `_fwd_kernel`
within a 20.62 s GPU activity span. GPU 0 was busy for 19.44 s, with about 5.7%
idle time. Kernel-name attribution identifies vision attention as the dominant
cost; GPU busy percentage alone did not indicate efficient execution. The
frontend reported 166 ms total image preparation, including 159 ms processing.
Copies summed to 135 ms. These timings come from a profiled request and should
not be substituted for the unprofiled end-to-end comparison above.

`VisionFlashV100Attention` uses ai-bond's native SM70 variable-length attention
with `causal=False`. Qwen3.8's 72-wide heads are padded to 128; the original
`1/sqrt(72)` scale is preserved and the result is cropped. Image boundaries
remain separate. The backend is explicit and limited to FP16 on SM70 with
eager vision execution; language CUDA graphs remain enabled.

An isolated three-image, 3,072-patch/head-dimension-72 attention screen fell
from 238.56 ms to 3.29 ms, with relative L2 error 0.00038 against the existing
path. That isolated 72× result is not the full-model speedup. CPU preprocessing
thread experiments produced identical feature hashes; reducing the existing
28-thread default did not help, so no CPU thread policy changed.

## Validation and use

- Nine FP32-reference/image-isolation numerical tests passed, covering head
  dimensions 64/72/128, original and custom scales, ragged images,
  noncontiguous inputs and sequence-length metadata forms.
- Both candidates passed all eight text checks, including full-context
  retrieval and recovery after rejecting an oversized request, plus tool calls.
- Both retained the same 11/12 multimodal result. The known OCR case still
  reads `V10 CHECK 4827`; it is not reported as a pass.
- Eight shared-document answers passed, each reusing at least 30,720 tokens.
- The six-request candidate passed all 48 queued document requests, keeping
  at least 3,792.7 MiB free on GPU 0. This queue run is not attributed to the
  selected four-request profile; the latter retains the v5.3 language settings.

Install the optional [vision dependency](../../python/sglang/jit_kernel/third_party/flash_v100/README.md#optional-vision-attention),
then use `scripts/serve_qwen38_flash_next_nvfp4_v100_throughput.sh`.
`--mm-attention-backend triton_attn` selects the previous vision path.
The CPU preprocessing OOM fix remains enabled; this backend does not establish
an unlimited document-size guarantee.

Reproduce the page measurement against an idle test server:

```bash
python benchmark/qwen38_vision_v100_20261104/image_bench.py \
  --url http://127.0.0.1:8083 --output /tmp/qwen38-vision-run \
  --label candidate --repeats 2
```

The script flushes caches, so use a dedicated benchmark endpoint. The existing
`benchmark/qwen38_spec/throughput.py` harness produced the text/SPEED screens.
[summary.json](summary.json) contains the metrics and selection decision;
[metadata.json](metadata.json) records configuration and source/binary hashes.
Compressed evidence is under `raw/`; generated benchmark output text is omitted
from text/SPEED files while its hashes and timing metadata are retained. Full
profiler traces remain in the local run directory.

Thanks to D. Skryabin / ai-bond for the Flash-V100 kernels, and to the SGLang
and 1Cat contributors whose serving and grouped-QSA work this profile builds
on. The dependency's license and exact host compatibility patch are retained
with the [third-party notice](../../python/sglang/jit_kernel/third_party/flash_v100/README.md).
