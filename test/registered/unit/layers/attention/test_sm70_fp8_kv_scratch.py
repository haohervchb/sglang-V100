import pytest
import torch

from sglang.srt.layers.attention.tilelang_fa_v100 import _paged_adapter
from sglang.srt.layers.attention.triton_ops.fp8_sm70 import (
    dequantize_paged_kv_e4m3_sm70,
    store_paged_extend_kv_fp16_sm70,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 FP8 KV scratch kernels require an NVIDIA V100",
)


def test_batched_prefix_dequant_and_scaled_extend_overlay():
    torch.manual_seed(7)
    device = "cuda:0"
    page_size = 16
    heads = 1
    dim = 32
    batch_size = 2
    max_pages = 2

    k_source = (torch.randn(6, page_size, heads, dim, device=device) * 0.5).half()
    v_source = (torch.randn_like(k_source) * 0.5).half()
    k_cache = k_source.to(torch.float8_e4m3fn)
    v_cache = v_source.to(torch.float8_e4m3fn)
    block_table = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([18, 7], dtype=torch.int32, device=device)
    k_scratch = torch.full(
        (batch_size * max_pages, page_size, heads, dim),
        -999.0,
        dtype=torch.float16,
        device=device,
    )
    v_scratch = torch.full_like(k_scratch, -999.0)

    dequantize_paged_kv_e4m3_sm70(
        k_cache,
        v_cache,
        block_table,
        prefix_lens,
        k_scratch,
        v_scratch,
        max_seq_len=18,
    )

    for batch, prefix_len in enumerate((18, 7)):
        expected_k = torch.cat(
            [k_cache[int(page)] for page in block_table[batch].cpu()]
        )[:prefix_len]
        expected_v = torch.cat(
            [v_cache[int(page)] for page in block_table[batch].cpu()]
        )[:prefix_len]
        page_slice = slice(batch * max_pages, (batch + 1) * max_pages)
        actual_k = k_scratch[page_slice].flatten(0, 1)[:prefix_len]
        actual_v = v_scratch[page_slice].flatten(0, 1)[:prefix_len]
        torch.testing.assert_close(actual_k, expected_k.half(), rtol=0, atol=0)
        torch.testing.assert_close(actual_v, expected_v.half(), rtol=0, atol=0)

    qo_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    k_extend = torch.randn(5, heads, dim, dtype=torch.float16, device=device)
    v_extend = torch.randn_like(k_extend)
    store_paged_extend_kv_fp16_sm70(
        k_extend,
        v_extend,
        qo_indptr,
        prefix_lens,
        k_scratch,
        v_scratch,
        max_extend_len=3,
        k_scale=2.0,
        v_scale=0.5,
    )

    for batch, (start, end) in enumerate(((0, 3), (3, 5))):
        page_slice = slice(batch * max_pages, (batch + 1) * max_pages)
        token_slice = slice(
            int(prefix_lens[batch]),
            int(prefix_lens[batch]) + end - start,
        )
        actual_k = k_scratch[page_slice].flatten(0, 1)[token_slice]
        actual_v = v_scratch[page_slice].flatten(0, 1)[token_slice]
        torch.testing.assert_close(actual_k, k_extend[start:end] / 2.0, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, v_extend[start:end] / 0.5, rtol=0, atol=0)


def test_paged_adapter_applies_non_unit_kv_scales(monkeypatch):
    captured = {}

    def fake_get_paged_verify_kernels(**_kwargs):
        def partial(*args):
            captured["softmax_scale"] = args[-1]
            return torch.empty(1, device="cuda"), torch.empty(1, device="cuda")

        def combine(*_args):
            return torch.full((1, 1, 32), 4.0, dtype=torch.float16, device="cuda")

        return partial, combine, None

    monkeypatch.setattr(
        _paged_adapter,
        "get_paged_verify_kernels",
        fake_get_paged_verify_kernels,
    )
    q = torch.zeros((1, 1, 32), dtype=torch.float16, device="cuda")
    k_cache = torch.zeros((1, 16, 1, 32), dtype=torch.float16, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    out, _ = _paged_adapter.paged_forward(
        q,
        k_cache,
        v_cache,
        torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        torch.ones(1, dtype=torch.int32, device="cuda"),
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        softmax_scale=0.5,
        linear_verify=True,
        k_scale=2.0,
        v_scale=0.25,
    )

    assert captured["softmax_scale"] == 1.0
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
