"""Regression tests for safe SWA location translation.

The translation result must never be cached by input ``data_ptr``. Scheduler
buffers are reused and mutated asynchronously between forwards, so a cached
gather can become stale or its temporary storage can be recycled while a
consumer is still running.

Run with:
    python -m pytest test/manual/core/test_swa_loc_translation_cache.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase


def _build_pool(
    kv_size: int = 32,
    kv_size_swa: int = 32,
    page_size: int = 1,
):
    device = get_device()
    num_layers = 8
    full_layer_ids = [0, 4]
    swa_layer_ids = [i for i in range(num_layers) if i not in set(full_layer_ids)]

    pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=4,
        head_dim=64,
        swa_attention_layer_ids=swa_layer_ids,
        full_attention_layer_ids=full_layer_ids,
        enable_kvcache_transpose=False,
        device=device,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=device,
        kvcache=pool,
        need_sort=False,
    )
    return pool, allocator, device


class TestFreshTranslation(CustomTestCase):
    def test_reused_input_storage_is_translated_again(self):
        pool, allocator, device = _build_pool()
        loc = allocator.alloc(4)
        self.assertIsNotNone(loc)

        first = pool.translate_loc_from_full_to_swa(loc)
        replacement = torch.tensor([8, 7, 6, 5], dtype=torch.int64, device=device)
        allocator.set_full_to_swa_mapping(loc, replacement)
        second = pool.translate_loc_from_full_to_swa(loc)

        self.assertIsNot(first, second)
        self.assertEqual(second.tolist(), replacement.tolist())

    def test_each_call_owns_a_live_result(self):
        pool, allocator, _ = _build_pool()
        loc = allocator.alloc(4)
        self.assertIsNotNone(loc)

        first = pool.translate_loc_from_full_to_swa(loc)
        second = pool.translate_loc_from_full_to_swa(loc)

        self.assertIsNot(first, second)
        self.assertEqual(first.tolist(), second.tolist())

    def test_views_at_different_offsets_translate_independently(self):
        pool, allocator, _ = _build_pool()
        loc = allocator.alloc(10)
        self.assertIsNotNone(loc)

        low = pool.translate_loc_from_full_to_swa(loc[:5])
        high = pool.translate_loc_from_full_to_swa(loc[5:10])

        self.assertFalse(torch.equal(low, high))


class TestInvalidationCompatibility(CustomTestCase):
    def test_concrete_noop_is_idempotent(self):
        pool, _, _ = _build_pool()
        pool.invalidate_loc_cache()
        pool.invalidate_loc_cache()

    def test_base_noop_does_not_raise(self):
        pool, _, _ = _build_pool()
        BaseSWAKVPool.invalidate_loc_cache(pool)


if __name__ == "__main__":
    unittest.main()
