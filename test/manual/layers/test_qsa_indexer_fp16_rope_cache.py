"""Fused QSA indexer prep must stay enabled after attention RoPE runs in FP16.

The attention MRoPE's Triton path converts the shared cos/sin cache to the
activation dtype. The fused indexer kernels then read that FP16 cache; eager
RoPE rounds cos/sin to FP16 as well, so outputs must stay bit-identical.
"""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The FP16 Qwen3.8 indexer path is validated on V100",
)

HEADS, HEAD_DIM, ROTARY_DIM, RATIO = 4, 128, 64, 4


@pytest.fixture(scope="module")
def indexer():
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    torch.manual_seed(0)
    rope = get_rope(
        head_size=256,
        rotary_dim=256,
        max_position=262144,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        base=10000000,
        partial_rotary_factor=0.25,
        is_neox_style=True,
        dtype=torch.float16,
    ).cuda()
    config = types.SimpleNamespace(
        hidden_size=2560,
        indexer_n_heads=HEADS,
        indexer_kv_heads=1,
        indexer_head_dim=HEAD_DIM,
        indexer_budget=2048,
        indexer_compress_ratio=RATIO,
        rms_norm_eps=1e-6,
    )
    assert rope.cos_sin_cache.dtype == torch.float32
    module = QSAIndexer(config, layer_id=3, rotary_emb=rope).cuda()
    # Only the norms run in FP16; converting the whole module would also
    # convert the shared RoPE cache before the attention path does.
    module.q_layernorm.half()
    module.k_layernorm.half()
    module.q_layernorm.weight.data.normal_(0, 0.1)
    module.k_layernorm.weight.data.normal_(0, 0.1)
    module.original_fp32_cache = rope.cos_sin_cache.clone()
    return module


def _positions(tokens, multimodal, start):
    text = torch.arange(start, start + tokens, device="cuda")
    if not multimodal:
        return text
    # Image-like spans give the three MRoPE axes different coordinates.
    pos = text.unsqueeze(0).repeat(3, 1)
    pos[1] += torch.randint(0, 64, (tokens,), device="cuda")
    pos[2] += torch.randint(0, 64, (tokens,), device="cuda")
    return pos


def _convert_cache_like_attention(rope):
    q = torch.randn(4, 6 * 256, device="cuda", dtype=torch.float16)
    k = torch.randn(4, 256, device="cuda", dtype=torch.float16)
    rope(torch.zeros((3, 4), device="cuda", dtype=torch.long), q, k)
    assert rope.cos_sin_cache.dtype == torch.float16


def test_attention_rope_converts_shared_cache_and_guard_accepts_it(indexer):
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    rope = indexer.rotary_emb
    fp16_probe = torch.empty(1, device="cuda", dtype=torch.half)
    _convert_cache_like_attention(rope)
    assert indexer._use_fused_prep(fp16_probe)
    # bf16 storage with an FP16 cache stays on the eager path.
    assert not indexer._use_fused_prep(fp16_probe.bfloat16())
    assert QSAIndexer._use_fused_prep is not None


@pytest.mark.parametrize("tokens", [1, 4, 16, 2048])
@pytest.mark.parametrize("multimodal", [False, True])
def test_q_prep_fp16_cache_matches_eager(indexer, tokens, multimodal):
    from sglang.srt.layers.attention.qsa_indexer import qsa_index_q_norm_rope_store

    rope = indexer.rotary_emb
    _convert_cache_like_attention(rope)
    assert indexer._use_fused_prep(torch.empty(1, device="cuda", dtype=torch.half))

    qk = torch.randn(
        (tokens, (HEADS + 1) * HEAD_DIM), device="cuda", dtype=torch.float16
    )
    positions = _positions(tokens, multimodal, start=257000 - tokens)
    slots = 4096
    cache_loc = torch.randperm(slots, device="cuda")[:tokens]
    key_state = torch.zeros((slots, HEAD_DIM), device="cuda", dtype=torch.float16)
    rope_buffer = torch.zeros((slots, 3), device="cuda", dtype=torch.long)

    q = qsa_index_q_norm_rope_store(
        qk,
        positions,
        rope.cos_sin_cache,
        indexer._rope_axis_map(qk.device),
        indexer.q_layernorm.weight.data,
        cache_loc,
        key_state,
        rope_buffer,
        HEADS,
        ROTARY_DIM,
        indexer.q_layernorm.variance_epsilon,
        True,
        q_heads_padded=HEADS,
    )

    q_raw = qk[:, : HEADS * HEAD_DIM]
    expected = indexer.q_layernorm(q_raw.reshape(-1, HEAD_DIM)).reshape(
        -1, HEADS, HEAD_DIM
    )
    expected = indexer.apply_rope(positions, expected)
    assert torch.equal(q.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(key_state[cache_loc], qk[:, HEADS * HEAD_DIM :])
    expected_pos = positions if positions.ndim == 2 else positions.expand(3, -1)
    assert torch.equal(rope_buffer[cache_loc], expected_pos.transpose(0, 1))


@pytest.mark.parametrize("groups", [1, 3, 512])
@pytest.mark.parametrize("multimodal", [False, True])
def test_k_compress_fp16_cache_matches_eager(indexer, groups, multimodal):
    from sglang.srt.layers.attention.qsa.kernel import average_pool_qsa_keys

    rope = indexer.rotary_emb
    _convert_cache_like_attention(rope)
    pool = types.SimpleNamespace(index_state_dtype=torch.float16)
    slots = 8192
    keys = torch.randn((slots, HEAD_DIM), device="cuda", dtype=torch.float16)
    rope_buffer = torch.zeros((slots, 3), device="cuda", dtype=torch.long)
    member_slots = torch.randperm(slots, device="cuda")[: groups * RATIO]
    group_locs = member_slots.view(groups, RATIO)
    pos = _positions(groups * RATIO, multimodal, start=200000)
    pos = pos if pos.ndim == 2 else pos.expand(3, -1)
    rope_buffer[member_slots] = pos.transpose(0, 1)
    compressed = torch.zeros((4096, HEAD_DIM), device="cuda", dtype=torch.float16)
    write_locs = torch.randperm(4096, device="cuda")[:groups].to(torch.int32)

    pool.get_qsa_key_state_buffer = lambda layer_id: keys
    pool.qsa_rope_position_buffer = rope_buffer
    pool.get_qsa_compressed_k_buffer = lambda layer_id: compressed
    assert indexer._use_fused_compress(pool)
    indexer._fused_compress_store(pool, group_locs, write_locs)

    pooled = average_pool_qsa_keys(keys[group_locs.long()].unsqueeze(2))
    block_positions = indexer._rope_from_matrix(rope_buffer[group_locs[:, 0].long()])
    expected = indexer.normalize_compressed_keys(pooled, block_positions)
    actual = compressed[write_locs.long()].view(groups, 1, HEAD_DIM)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_fp32_and_fp16_caches_agree(indexer):
    from sglang.srt.layers.attention.qsa_indexer import qsa_index_q_norm_rope_store

    rope = indexer.rotary_emb
    _convert_cache_like_attention(rope)
    fp16_cache = rope.cos_sin_cache
    # The original FP32 cache, not an FP16 round trip, is the fused path's
    # previous input; both must produce identical rows.
    fp32_cache = indexer.original_fp32_cache
    assert fp32_cache.dtype == torch.float32
    assert not torch.equal(fp32_cache, fp16_cache.float())
    tokens = 64
    qk = torch.randn(
        (tokens, (HEADS + 1) * HEAD_DIM), device="cuda", dtype=torch.float16
    )
    # A [3, T] slice of a wider graph buffer, as CUDA-graph replays pass it.
    graph_positions = torch.zeros((3, tokens + 7), device="cuda", dtype=torch.long)
    graph_positions[:, :tokens] = _positions(tokens, True, start=1000)
    positions = graph_positions[:, :tokens]
    outputs = []
    for cache in (fp16_cache, fp32_cache):
        outputs.append(
            qsa_index_q_norm_rope_store(
                qk,
                positions,
                cache,
                indexer._rope_axis_map(qk.device),
                indexer.q_layernorm.weight.data,
                torch.arange(tokens, device="cuda"),
                torch.zeros((tokens, HEAD_DIM), device="cuda", dtype=torch.half),
                torch.zeros((tokens, 3), device="cuda", dtype=torch.long),
                HEADS,
                ROTARY_DIM,
                indexer.q_layernorm.variance_epsilon,
                True,
            )
        )
    assert torch.equal(outputs[0], outputs[1])
