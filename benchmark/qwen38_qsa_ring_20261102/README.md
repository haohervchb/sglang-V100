# QSA speculative pending-ring correctness

The existing four-slot ring can overwrite a retained prefix key during a
four-token verification window. With a committed prefix ending at position 4,
publishing draft positions `[5, 6, 7, 8]` makes compression read `[8, 5, 6, 7]`
instead of `[4, 5, 6, 7]`. The same issue affects the MTP draft-extend path.
[The reproduction](ring_bug_reproduction.json) uses the old slot builders.

This adapts [andyl98's upstream PR #46125](https://github.com/sgl-project/sglang/pull/46125),
commit `85cb8d0af76f3f9d65294b540da6aa0b654e517d`, to this older SGLang API and
FP16 V100 execution. The ring now holds the verification window plus the
`ratio - 1` preceding members a completed group may need. Its size rounds up
to a power of two: ratio 4 uses 4 slots for plain decode, 8 for MTP width 4,
and 16 for widths 6/8. Allocation follows the launch maximum. Metadata builders
and the runtime guard use that allocated size; the guard still refuses unknown
or oversized verification geometry.

A follow-up target-only gate also exposed an out-of-bounds read for one-to-three-token
prefills. The fixed-capacity compression plan has a padding entry even when no
group completed. Such chunks now preserve their pending keys and skip compression.
This also keeps an oversized-request rejection (which the scheduler represents
with a one-token dummy) from crashing the server. Six additional GPU cases cover
the fused and fallback compression paths.

The main-based fix changes no speculative width or serving default. Wider MTP
performance and FP16 KV experiments are separate work.

Validation on V100:

- 35 GPU tests compare compressed keys exactly against decode and prefill,
  including rejected drafts, drifting request alignments and draft extension.
- 10 GPU graph/page-table tests include changing inputs and 6/8-row metadata.
- Five CPU tests, including 1,443 subtests, exercise ring sizing, retained keys,
  slot-zero dumping and refusal of unsafe windows.
- Fourteen GPU verifier tests cover every rejection position at widths 6/8.
- The experimental fused metadata path, maintained separately from this
  main-based change, also passed the combined 44-case graph suite.

The corrected main-based build completed full-model validation on November 3:
six text checks passed, including retrieval from 262,000 input tokens and the
next short request; tool arguments and a tool-result continuation passed. Image
and video checks remained 11/12 with the same existing OCR miss (`V100` read as
`V10`). MTP-3/4 with E5M2 KV completed three repetitions at 1K/8K/25K/70K.
Launch settings, raw observations and validation responses are retained here.
The target-only follow-up passed eight checks, including 262K retrieval, clean
oversized-input rejection and the next short request, with FP16 KV at memory
fraction 0.84. The original 0.83 target-only launch had only 259K KV slots and
exposed the short-prefill bug through its rejected-request dummy.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python python -m pytest -q \
  test/registered/unit/mem_cache/test_qsa_pending_ring.py
PYTHONPATH=python python -m pytest -q \
  test/manual/layers/test_qsa_verify_ring.py \
  test/manual/layers/test_qsa_graph_metadata_tiles.py \
  test/manual/spec/test_wide_mtp_verify.py
```
