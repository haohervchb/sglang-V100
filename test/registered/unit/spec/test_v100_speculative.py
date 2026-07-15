from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.arg_groups.speculative_hook import _handle_dflash, _handle_eagle_family
from sglang.srt.layers.attention.flash_attn_v100_backend import (
    FlashAttnV100Backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.dflash import (
    _resolve_dflash_model_sliding_window,
    _resolve_dflash_rope_config,
    _resolve_dflash_sliding_window,
)
from sglang.srt.models.qwen3_5_mtp import _is_mtp_dynamically_unquantized
from sglang.srt.speculative.dflash_worker import (
    _resolve_dflash_draft_attention_backend,
)
from sglang.srt.speculative.draft_utils import DraftBackendFactory


class _ForwardMode:
    def __init__(self, *, target_verify=False, draft_extend=False):
        self._target_verify = target_verify
        self._draft_extend = draft_extend

    def is_target_verify(self):
        return self._target_verify

    def is_draft_extend(self, include_v2=False):
        return self._draft_extend


@pytest.mark.parametrize(
    "model_arch",
    [
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ],
)
def test_qwen35_uses_builtin_mtp_checkpoint(model_arch):
    server_args = SimpleNamespace(
        speculative_algorithm="EAGLE",
        enable_dp_attention=False,
        max_running_requests=4,
        speculative_eagle_topk=1,
        disable_overlap_schedule=True,
        enable_mixed_chunk=False,
        get_model_config=Mock(
            return_value=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[model_arch])
            )
        ),
        model_path="Qwen/model",
        revision=None,
        speculative_draft_model_path=None,
        speculative_draft_model_revision=None,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        attention_backend="flash_attn_v100",
        decode_attention_backend=None,
        prefill_attention_backend=None,
        page_size=16,
    )

    _handle_eagle_family(server_args)

    assert server_args.speculative_draft_model_path == server_args.model_path


def test_qwen35_factory_gptq_keeps_mtp_unquantized():
    quant_config = SimpleNamespace(dynamic={"-:.*mtp.*": {}})

    assert _is_mtp_dynamically_unquantized(quant_config, "mtp")
    assert _is_mtp_dynamically_unquantized(quant_config, "draft.mtp.layers.0")
    assert not _is_mtp_dynamically_unquantized(quant_config, "model.layers.0")


def test_dflash_does_not_inherit_gptq_target_quantization(monkeypatch):
    from sglang.srt.utils import hf_transformers_utils

    monkeypatch.setattr(
        hf_transformers_utils,
        "get_config",
        lambda *args, **kwargs: SimpleNamespace(quantization_config=None),
    )
    server_args = SimpleNamespace(
        enable_dp_attention=False,
        pp_size=1,
        speculative_draft_model_path="z-lab/dflash",
        trust_remote_code=False,
        speculative_draft_model_revision="main",
        json_model_override_args="{}",
        speculative_draft_model_quantization="gptq_marlin",
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_dflash_block_size=16,
        speculative_num_draft_tokens=None,
        speculative_draft_window_size=None,
        max_running_requests=4,
        disable_overlap_schedule=False,
        enable_mixed_chunk=False,
    )

    _handle_dflash(server_args)

    assert server_args.speculative_draft_model_quantization is None
    assert server_args.speculative_num_draft_tokens == 16


@pytest.mark.parametrize("kind", ["decode", "extend"])
def test_mtp_maps_flash_attn_v100_to_triton(kind):
    server_args = SimpleNamespace(
        speculative_draft_attention_backend=None,
        attention_backend="flash_attn_v100",
        decode_attention_backend=None,
        prefill_attention_backend=None,
        speculative_attention_mode="prefill",
    )
    factory = DraftBackendFactory(
        server_args,
        draft_model_runner=object(),
        topk=1,
        speculative_num_steps=3,
    )
    sentinel = object()
    if kind == "decode":
        factory._create_triton_decode_backend = Mock(return_value=sentinel)
        result = factory.create_decode_backend()
        factory._create_triton_decode_backend.assert_called_once_with()
    else:
        factory._create_triton_prefill_backend = Mock(return_value=sentinel)
        result = factory.create_draft_extend_backend()
        factory._create_triton_prefill_backend.assert_called_once_with()
    assert result is sentinel


def test_dflash_uses_triton_draft_attention_on_v100():
    assert _resolve_dflash_draft_attention_backend("flash_attn_v100") == "triton"


def test_dflash_reads_transformers_v5_rope_parameters():
    config = SimpleNamespace(
        rope_theta=None,
        rope_scaling={"rope_type": "default", "rope_theta": 10_000_000},
    )

    rope_theta, rope_scaling = _resolve_dflash_rope_config(config)

    assert rope_theta == 10_000_000
    assert rope_scaling == config.rope_scaling


def test_dflash_interleaved_sliding_window_layers():
    config = SimpleNamespace(
        num_hidden_layers=3,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=2048,
    )

    assert _resolve_dflash_model_sliding_window(config) == 2048
    assert _resolve_dflash_sliding_window(config, 0) == 2048
    assert _resolve_dflash_sliding_window(config, 2) == -1


@pytest.mark.parametrize("mode", ["target_verify", "draft_extend"])
def test_v100_speculative_extend_delegates_to_triton(mode):
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._triton = Mock()
    backend._triton.forward_extend.return_value = "triton-output"
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(
            target_verify=mode == "target_verify",
            draft_extend=mode == "draft_extend",
        )
    )

    output = backend.forward_extend(
        q="q",
        k="k",
        v="v",
        layer="layer",
        forward_batch=forward_batch,
        save_kv_cache=False,
    )

    assert output == "triton-output"
    backend._triton.forward_extend.assert_called_once_with(
        "q",
        "k",
        "v",
        "layer",
        forward_batch,
        save_kv_cache=False,
    )


def test_v100_spec_v2_metadata_delegates_to_triton():
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._triton = Mock()
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)

    backend.init_forward_metadata(forward_batch)

    backend._triton.init_forward_metadata.assert_called_once_with(forward_batch)
    assert backend.forward_metadata is None


def test_v100_verify_cuda_graph_buffers_delegate_to_triton():
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend._triton = Mock()
    backend._triton.get_verify_buffers_to_fill_after_draft.return_value = [
        "mask",
        None,
    ]

    assert backend.get_verify_buffers_to_fill_after_draft() == ["mask", None]
    backend.update_verify_buffers_to_fill_after_draft("spec-info", 4)

    backend._triton.get_verify_buffers_to_fill_after_draft.assert_called_once_with()
    backend._triton.update_verify_buffers_to_fill_after_draft.assert_called_once_with(
        "spec-info", 4
    )
