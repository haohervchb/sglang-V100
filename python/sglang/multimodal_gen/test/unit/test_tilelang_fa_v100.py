# SPDX-License-Identifier: Apache-2.0
"""Contracts for MiniMax-H3's dense TileLang attention on Volta."""

import importlib.util
from itertools import pairwise

import pytest
import torch
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    backend_name_to_enum,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _has_tilelang_v100() -> bool:
    return bool(
        importlib.util.find_spec("tilelang")
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (7, 0)
    )


def test_tilelang_v100_backend_is_selectable_for_h3():
    backend = AttentionBackendEnum.TILELANG_FA_V100
    assert backend_name_to_enum("TILELANG_FA_V100") is backend
    assert backend in MiniMaxH3DiTArchConfig()._supported_attention_backends


@pytest.mark.skipif(
    not _has_tilelang_v100(), reason="requires TileLang and an NVIDIA V100"
)
def test_tilelang_v100_varlen_matches_sdpa():
    from sglang.multimodal_gen.runtime.layers.attention.backends.tilelang_fa_v100 import (
        tilelang_flash_attn_varlen,
    )

    torch.manual_seed(0)
    bounds = (0, 67, 196)
    heads, dim = 14, 128
    query = torch.randn(bounds[-1], heads, dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    scale = dim**-0.5

    actual = tilelang_flash_attn_varlen(
        query,
        key,
        value,
        cu_seqlens=cu_seqlens,
        max_seqlen=129,
        softmax_scale=scale,
    )
    expected = torch.empty_like(actual)
    for start, stop in pairwise(bounds):
        expected[start:stop] = torch.nn.functional.scaled_dot_product_attention(
            query[start:stop][None].transpose(1, 2),
            key[start:stop][None].transpose(1, 2),
            value[start:stop][None].transpose(1, 2),
            scale=scale,
        ).transpose(1, 2)[0]

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
