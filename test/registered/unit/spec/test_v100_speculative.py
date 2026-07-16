from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.arg_groups.speculative_hook import _handle_dflash, _handle_eagle_family
from sglang.srt.layers.attention.flash_attn_v100_backend import (
    FlashAttnV100Backend,
    _get_native_paged_attention_params,
    _should_skip_triton_prefill,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.managers.overlap_utils import decide_needs_cpu_seq_lens
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.dflash import (
    _get_dflash_layer_attention_params,
    _resolve_dflash_rope_config,
)
from sglang.srt.models.qwen3_5_mtp import _is_mtp_dynamically_unquantized
from sglang.srt.speculative.dflash_worker import (
    DFlashWorker,
    _resolve_dflash_draft_attention_backend,
)
from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
    parse_dflash_draft_config,
    resolve_dflash_verify_mask_policy,
)
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class _ForwardMode:
    def __init__(self, *, target_verify=False, draft_extend=False):
        self._target_verify = target_verify
        self._draft_extend = draft_extend

    def is_target_verify(self):
        return self._target_verify

    def is_decode_or_idle(self):
        return False

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


def test_dflash_uses_native_draft_attention_on_v100():
    assert (
        _resolve_dflash_draft_attention_backend("flash_attn_v100")
        == "flash_attn_v100"
    )


@pytest.mark.parametrize("enable_spec_v2", [True, False])
def test_dflash_worker_selection_tracks_spec_v2_env(enable_spec_v2):
    from sglang.srt.environ import envs

    with envs.SGLANG_ENABLE_SPEC_V2.override(enable_spec_v2):
        worker_cls = SpeculativeAlgorithm.DFLASH.create_worker(SimpleNamespace())

        assert SpeculativeAlgorithm.DFLASH.supports_spec_v2() is enable_spec_v2
        if enable_spec_v2:
            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

            assert worker_cls is DFlashWorkerV2
        else:
            assert worker_cls is DFlashWorker


def test_dflash_v2_reports_target_and_draft_attention_capabilities():
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    target_backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    draft_backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=target_backend)
    )
    worker.draft_model_runner = SimpleNamespace(attn_backend=draft_backend)
    server_args = SimpleNamespace(
        enable_two_batch_overlap=False,
        disable_piecewise_cuda_graph=True,
    )

    assert worker.spec_v2_attn_backends == (target_backend, draft_backend)
    assert not decide_needs_cpu_seq_lens(
        server_args, worker.spec_v2_attn_backends
    )

    worker.draft_model_runner.attn_backend = SimpleNamespace(
        needs_cpu_seq_lens=True
    )
    assert decide_needs_cpu_seq_lens(server_args, worker.spec_v2_attn_backends)


def test_dflash_draft_skips_irrelevant_sm70_prefill_warmup(monkeypatch):
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "cuda"
    runner.is_generation = True
    runner.is_draft_worker = True
    runner.spec_algorithm = SimpleNamespace(
        is_dflash=lambda: True,
        is_speculative=lambda: True,
    )
    runner._should_run_flashinfer_autotune = Mock(return_value=False)
    runner._flashinfer_autotune = Mock()
    runner._warmup_sm70_flashinfer_sampling = Mock()
    runner._warmup_prefill_kernels_extends = Mock()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))

    runner.kernel_warmup()

    runner._warmup_sm70_flashinfer_sampling.assert_called_once_with()
    runner._warmup_prefill_kernels_extends.assert_not_called()


def test_dflash_compact_length_preserves_cpu_host_mirror():
    worker = DFlashWorker.__new__(DFlashWorker)
    worker.draft_window_size = 4096
    worker.page_size = 16
    worker.device = "cuda"
    seq_lens_cpu = torch.tensor([100, 10_005], dtype=torch.int32)

    compact = worker._compute_compact_draft_seq_lens(seq_lens_cpu)

    assert compact.device.type == "cpu"
    assert compact.tolist() == [100, 4101]


@pytest.mark.parametrize(
    ("attn_type", "window", "expected"),
    [
        (AttentionType.DECODER, 2047, (True, 2047)),
        (AttentionType.DECODER, -1, (True, -1)),
        (AttentionType.ENCODER_ONLY, -1, (False, -1)),
    ],
)
def test_v100_native_attention_uses_per_layer_dflash_mask(
    attn_type, window, expected
):
    layer = SimpleNamespace(
        is_cross_attention=False,
        attn_type=attn_type,
        sliding_window_size=window,
    )

    assert _get_native_paged_attention_params(layer, True) == expected


def test_dflash_v100_triton_verify_skips_redundant_custom_mask():
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)

    assert resolve_dflash_verify_mask_policy(backend) == (
        "FlashAttnV100Backend",
        False,
    )


def test_dflash_reads_transformers_v5_rope_parameters():
    config = SimpleNamespace(
        rope_theta=None,
        rope_scaling={"rope_type": "default", "rope_theta": 10_000_000},
    )

    rope_theta, rope_scaling = _resolve_dflash_rope_config(config)

    assert rope_theta == 10_000_000
    assert rope_scaling == config.rope_scaling


@pytest.mark.parametrize(
    (
        "config",
        "expected_block_size",
        "expected_target_layers",
        "expected_mask_token_id",
        "expected_window",
    ),
    [
        (
            SimpleNamespace(
                num_hidden_layers=5,
                num_target_layers=64,
                block_size=16,
                dflash_config={
                    "mask_token_id": 248070,
                    "target_layer_ids": [1, 16, 31, 46, 61],
                },
                layer_types=["sliding_attention"] * 4 + ["full_attention"],
                sliding_window=2048,
            ),
            16,
            [1, 16, 31, 46, 61],
            248070,
            2047,
        ),
        (
            SimpleNamespace(
                num_hidden_layers=6,
                num_target_layers=48,
                dflash_config={
                    "block_size": 16,
                    "mask_token_id": 248077,
                    "target_layer_ids": [1, 7, 14, 20, 26, 32, 39, 45],
                },
                layer_types=["sliding_attention"] * 5 + ["full_attention"],
                sliding_window=4096,
            ),
            16,
            [1, 7, 14, 20, 26, 32, 39, 45],
            248077,
            4095,
        ),
    ],
)
def test_dflash_checkpoint_config_layouts(
    config,
    expected_block_size,
    expected_target_layers,
    expected_mask_token_id,
    expected_window,
):
    parsed = parse_dflash_draft_config(draft_hf_config=config)

    assert parsed.resolve_block_size() == expected_block_size
    assert parsed.resolve_target_layer_ids(
        target_num_layers=config.num_target_layers
    ) == expected_target_layers
    assert parsed.mask_token_id == expected_mask_token_id
    assert get_dflash_attention_sliding_window_size(config) == expected_window
    assert _get_dflash_layer_attention_params(config, 0) == (
        expected_window,
        AttentionType.DECODER,
    )
    assert _get_dflash_layer_attention_params(
        config, config.num_hidden_layers - 1
    ) == (-1, AttentionType.ENCODER_ONLY)


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

    assert get_dflash_attention_sliding_window_size(config) == 2047
    assert _get_dflash_layer_attention_params(config, 0) == (
        2047,
        AttentionType.DECODER,
    )
    assert _get_dflash_layer_attention_params(config, 2) == (
        -1,
        AttentionType.ENCODER_ONLY,
    )


@pytest.mark.parametrize(
    ("is_speculative", "expected_skip_prefill"),
    [(False, True), (True, False)],
)
def test_v100_triton_allocates_prefill_metadata_for_speculation(
    is_speculative, expected_skip_prefill
):
    model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_speculative=lambda: is_speculative)
    )

    assert _should_skip_triton_prefill(model_runner) is expected_skip_prefill


@pytest.mark.parametrize("mode", ["target_verify", "draft_extend"])
def test_v100_speculative_extend_delegates_to_triton(mode):
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: False,
            is_eagle=lambda: False,
        ),
        server_args=SimpleNamespace(speculative_eagle_topk=1),
    )
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


def test_v100_mtp_linear_verify_builds_native_causal_metadata():
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: False,
            is_eagle=lambda: True,
        ),
        server_args=SimpleNamespace(speculative_eagle_topk=1),
    )
    backend._triton = Mock()
    backend._build_extend_metadata = Mock(return_value="native-metadata")
    prefix_lens = torch.tensor([17, 33], dtype=torch.int32)
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(target_verify=True),
        req_pool_indices="req-pool-indices",
        seq_lens=prefix_lens,
        spec_info=SimpleNamespace(draft_token_num=8),
    )

    backend.init_forward_metadata(forward_batch)

    args = backend._build_extend_metadata.call_args.args
    assert args[0] == "req-pool-indices"
    assert args[1].tolist() == [25, 41]
    assert args[2].tolist() == [8, 8]
    assert args[3] is prefix_lens
    assert backend._build_extend_metadata.call_args.kwargs == {"causal": True}
    assert backend.forward_metadata == "native-metadata"
    backend._triton.init_forward_metadata.assert_not_called()


def test_v100_dflash_verify_builds_native_causal_metadata():
    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: True,
            is_eagle=lambda: False,
        ),
        server_args=SimpleNamespace(speculative_eagle_topk=1),
    )
    backend._triton = Mock()
    backend._build_extend_metadata = Mock(return_value="native-metadata")
    prefix_lens = torch.tensor([9000], dtype=torch.int32)
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(target_verify=True),
        req_pool_indices="req-pool-indices",
        seq_lens=prefix_lens,
        spec_info=SimpleNamespace(draft_token_num=16),
    )

    backend.init_forward_metadata(forward_batch)

    args = backend._build_extend_metadata.call_args.args
    assert args[0] == "req-pool-indices"
    assert args[1].tolist() == [9016]
    assert args[2].tolist() == [16]
    assert args[3] is prefix_lens
    assert backend._build_extend_metadata.call_args.kwargs == {"causal": True}
    assert backend.forward_metadata == "native-metadata"
    backend._triton.init_forward_metadata.assert_not_called()


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
