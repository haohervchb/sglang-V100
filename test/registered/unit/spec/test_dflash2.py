from types import SimpleNamespace

import pytest
import torch
from sglang.srt.arg_groups.speculative_hook import _handle_dflash
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
    EntryClass,
    _get_dflash_layer_attention_params,
    _grouped_conv,
    _use_sm70_bf16_emulation,
)
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_sampling_correct_drafts_and_bonus,
    map_dflash_target_layer_ids_for_capture,
    parse_dflash_draft_config,
)


def test_dflash2_config_fields_are_parsed():
    config = parse_dflash_draft_config(
        draft_hf_config={
            "num_hidden_layers": 5,
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "selector_rank": 256,
                "selector_top_k": 16,
                "target_layer_ids": [5, 19, 33, 47, 61],
            },
        }
    )
    assert config.block_size == 8
    assert config.conv_kernel_size == 2
    assert config.conv_group_size == 16
    assert config.selector_rank == 256
    assert config.selector_top_k == 16
    assert config.target_layer_ids == [5, 19, 33, 47, 61]
    assert DFlash2DraftModel in EntryClass


def test_dflash2_explicit_noncausal_sliding_attention_is_bidirectional():
    config = SimpleNamespace(
        layer_types=["sliding_attention"] * 5,
        sliding_window=2048,
        is_causal=False,
    )
    window, attention_type = _get_dflash_layer_attention_params(config, 0)
    assert window == 2047
    assert attention_type == AttentionType.ENCODER_ONLY


def test_dflash2_qwen35_layer_output_ids_map_to_capture_boundaries():
    assert map_dflash_target_layer_ids_for_capture(
        target_model_type="qwen3_5_text",
        draft_architectures=["DFlash2DraftModel"],
        layer_ids=[5, 19, 33, 47, 61],
    ) == [6, 20, 34, 48, 62]
    assert map_dflash_target_layer_ids_for_capture(
        target_model_type="qwen3",
        draft_architectures=["DFlash2DraftModel"],
        layer_ids=[5, 19],
    ) == [5, 19]


def test_dflash2_enables_range_preserving_bf16_emulation_on_sm70(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))
    assert _use_sm70_bf16_emulation(
        SimpleNamespace(architectures=["DFlash2DraftModel"])
    )


def test_dflash2_uses_existing_dflash_cli_and_checkpoint_defaults(monkeypatch):
    from sglang.srt.utils import hf_transformers_utils

    monkeypatch.setattr(
        hf_transformers_utils,
        "get_config",
        lambda *args, **kwargs: SimpleNamespace(
            architectures=["DFlash2DraftModel"],
            quantization_config=None,
            num_hidden_layers=5,
            dflash_config={
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "selector_rank": 256,
                "selector_top_k": 16,
            },
        ),
    )
    server_args = SimpleNamespace(
        enable_dp_attention=False,
        pp_size=1,
        speculative_draft_model_path="z-lab/Qwen3.8-27B-DFlash2",
        trust_remote_code=True,
        speculative_draft_model_revision="main",
        json_model_override_args="{}",
        speculative_draft_model_quantization="fp8",
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_dflash_block_size=None,
        speculative_num_draft_tokens=None,
        speculative_draft_window_size=2048,
        max_running_requests=1,
        disable_overlap_schedule=False,
        enable_mixed_chunk=False,
        attention_backend="flash_attn_v100",
    )

    _handle_dflash(server_args)

    assert server_args.speculative_num_draft_tokens == 8
    assert server_args.speculative_draft_model_quantization is None


@pytest.mark.parametrize("field", ["conv", "selector"])
def test_dflash2_config_requires_paired_fields(field):
    dflash_config = (
        {"conv_kernel_size": 2} if field == "conv" else {"selector_rank": 256}
    )
    with pytest.raises(ValueError, match=field):
        parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": dflash_config,
            }
        )


def test_dflash2_unary_logit_transform():
    logits = torch.tensor([[-100.0, 0.0, 100.0]], dtype=torch.bfloat16)
    for fields in ({}, {"output_multiplier": 0.2, "final_logit_softcapping": 20.0}):
        config = parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": {
                    "selector_rank": 256,
                    "selector_top_k": 16,
                    **fields,
                },
            }
        )
        actual = DFlash2DraftModel._transform_unary_logits(
            SimpleNamespace(draft_config=config), logits
        )
        expected = logits.float() * config.output_multiplier
        if config.final_logit_softcapping is not None:
            expected = torch.tanh(expected / config.final_logit_softcapping)
            expected *= config.final_logit_softcapping
        torch.testing.assert_close(actual, expected)


def test_dflash2_grouped_conv_supports_runtime_block_sizes():
    torch.manual_seed(0)
    groups, group_size, taps = 3, 2, 2
    hidden_size = groups * group_size
    batch_size = 2

    for block_size in (5, 8, 16):
        hidden = torch.randn(batch_size * block_size, hidden_size)
        delta = torch.randn(batch_size * block_size, taps, groups)
        base = torch.randn(taps, hidden_size)
        actual = _grouped_conv(
            hidden, delta, base, block_size, groups, group_size, taps
        )

        expected = torch.empty_like(hidden)
        hidden_3d = hidden.view(batch_size, block_size, groups, group_size)
        delta_4d = delta.view(batch_size, block_size, taps, groups)
        base_3d = base.view(taps, groups, group_size)
        for batch in range(batch_size):
            for position in range(block_size):
                value = torch.zeros(groups, group_size)
                for tap in range(min(taps, position + 1)):
                    coefficient = base_3d[tap] + delta_4d[batch, position, tap, :, None]
                    value += coefficient * hidden_3d[batch, position - tap]
                expected[batch * block_size + position] = value.flatten()
        torch.testing.assert_close(actual, expected)


def test_dflash2_selector_greedy_row_is_deterministic_in_mixed_batch():
    selector = CandidateSelector(hidden_size=4, vocab_size=16, state_rank=2, top_k=4)
    torch.manual_seed(1)
    candidate_ids = torch.randint(0, 16, (2, 3, 4))
    scores = torch.randn(2, 3, 4, 4)
    uniforms = torch.tensor([[0.2, 0.7, 0.4], [0.8, 0.1, 0.6]])
    temperatures = torch.tensor([1.0, 0.7])
    greedy_mask = torch.tensor([True, False])

    mixed_tokens, mixed_q = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    assert torch.all((mixed_q[0] == 0) | (mixed_q[0] == 1))
    for row in range(2):
        tokens, q_rows = selector.sample_path(
            candidate_ids=candidate_ids[row : row + 1],
            scores=scores[row : row + 1],
            uniforms=uniforms[row : row + 1],
            temperatures=temperatures[row : row + 1],
            greedy_mask=greedy_mask[row : row + 1],
        )
        torch.testing.assert_close(mixed_tokens[row], tokens[0])
        torch.testing.assert_close(mixed_q[row], q_rows[0])


def test_dflash2_selector_rejects_quantized_target_head():
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.empty(8, 4, dtype=torch.int8)),
        candidate_selector=SimpleNamespace(top_k=4),
    )
    with pytest.raises(RuntimeError, match="requires a dense"):
        DFlash2DraftModel.compute_candidates(model, torch.randn(2, 4))


def test_dflash2_sparse_selector_probabilities_reach_lossless_verifier(monkeypatch):
    from sglang.srt.speculative import dflash_utils

    captured = []

    def fake_verify(**kwargs):
        captured.append(kwargs["draft_probs"].clone())
        kwargs["accept_token_num"].zero_()
        kwargs["accept_index"].zero_()
        kwargs["predicts"].zero_()

    monkeypatch.setattr(dflash_utils, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True)
    monkeypatch.setattr(dflash_utils, "chain_speculative_sampling_triton", fake_verify)

    candidate_ids = torch.tensor([[[1, 2], [3, 4]]])
    q_rows = torch.tensor([[[0.75, 0.25], [0.4, 0.6]]])
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1)),
        top_ks=torch.tensor([8], dtype=torch.int32),
        top_ps=torch.ones((1, 1)),
        min_ps=torch.zeros((1, 1)),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        sampling_seed=None,
    )
    compute_dflash_sampling_correct_drafts_and_bonus(
        candidates=torch.tensor([[7, 1, 4]]),
        next_token_logits=torch.zeros((3, 8)),
        sampling_info=sampling_info,
        threshold_single=1.0,
        threshold_acc=1.0,
        selector_candidate_ids=candidate_ids,
        selector_q_rows=q_rows,
    )

    expected = torch.zeros((1, 2, 8))
    expected[0, 0, 1] = 0.75
    expected[0, 0, 2] = 0.25
    expected[0, 1, 3] = 0.4
    expected[0, 1, 4] = 0.6
    torch.testing.assert_close(captured[0], expected)

    # The optimized path reuses a dense carrier but clears only the sparse
    # top-k entries. A second call with disjoint ids must not see stale q mass.
    second_ids = torch.tensor([[[0, 5], [6, 7]]])
    second_q = torch.tensor([[[0.2, 0.8], [0.9, 0.1]]])
    compute_dflash_sampling_correct_drafts_and_bonus(
        candidates=torch.tensor([[7, 5, 6]]),
        next_token_logits=torch.zeros((3, 8)),
        sampling_info=sampling_info,
        threshold_single=1.0,
        threshold_acc=1.0,
        selector_candidate_ids=second_ids,
        selector_q_rows=second_q,
    )
    second_expected = torch.zeros((1, 2, 8))
    second_expected.scatter_(-1, second_ids, second_q)
    torch.testing.assert_close(captured[1], second_expected)
