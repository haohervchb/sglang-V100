"""SGLANG_CHUNKED_PREFILL_CACHE_ALIGN: final prefill chunks end on cache boundaries."""

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_policy import _cache_aligned_final_chunk


def test_disabled_by_default():
    assert _cache_aligned_final_chunk(28672, 3000) == 3000


def test_cuts_long_final_chunk_at_last_boundary():
    with envs.SGLANG_CHUNKED_PREFILL_CACHE_ALIGN.override(2048):
        # 28672 + 3000 crosses 30720: run up to the boundary, tail next step.
        assert _cache_aligned_final_chunk(28672, 3000) == 2048
        # A 3.5K prompt from scratch matches what 2K chunks would checkpoint.
        assert _cache_aligned_final_chunk(0, 3500) == 2048


def test_keeps_chunks_that_need_no_cut():
    with envs.SGLANG_CHUNKED_PREFILL_CACHE_ALIGN.override(2048):
        # Ends exactly on a boundary.
        assert _cache_aligned_final_chunk(0, 4096) == 4096
        assert _cache_aligned_final_chunk(4096, 4096) == 4096
        # Short extends after a cache hit are never split.
        assert _cache_aligned_final_chunk(30700, 100) == 100
        assert _cache_aligned_final_chunk(30000, 2048) == 2048
