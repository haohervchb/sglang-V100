"""The real draft graph producer must publish a ring-checkable row bound."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import eagle_draft_extend_cuda_graph_runner as module


@pytest.mark.parametrize(
    "width,ring,admitted",
    [(4, 4, False), (4, 8, True), (6, 8, False), (6, 16, True), (8, 16, True)],
)
def test_captured_geometry_reaches_the_real_guard(monkeypatch, width, ring, admitted):
    runner = module.EAGLEDraftExtendCudaGraphRunner.__new__(
        module.EAGLEDraftExtendCudaGraphRunner
    )
    runner.num_tokens_per_bs = width
    runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2
    runner.require_mlp_tp_gather = False
    runner.require_attn_tp_gather = False
    runner.stream = None
    runner.padded_static_len = -1
    zeros = lambda *shape: torch.zeros(shape, dtype=torch.int32)
    runner.buffers = SimpleNamespace(
        input_ids=zeros(width),
        req_pool_indices=zeros(1),
        seq_lens=zeros(1),
        seq_lens_cpu=zeros(1),
        extend_seq_lens=zeros(1),
        out_cache_loc=zeros(width),
        positions=zeros(width),
        mrope_positions=zeros(3, width),
        hidden_states=None,
        num_correct_drafts=zeros(1),
        num_accept_tokens=zeros(1),
        next_token_logits_buffer=torch.zeros(width, 16),
        global_num_tokens_gpu=None,
        global_num_tokens_for_logprob_gpu=None,
    )
    runner.extend_seq_lens_cpu = [width]
    guard = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    guard.compress_ratio = 4
    guard.token_to_kv_pool = SimpleNamespace(qsa_ring_size=ring)
    seen = []

    def check(**kwargs):
        info = kwargs["spec_info"]
        seen.append(info.num_tokens_per_req)
        guard._require_chain_speculation(kwargs["forward_mode"], info)

    runner.draft_extend_attn_backend = SimpleNamespace(
        init_forward_metadata_capture_cuda_graph=check
    )
    runner.deepep_adapter = SimpleNamespace(capture=lambda **kwargs: None)
    runner.model_runner = SimpleNamespace(canary_manager=None, spec_algorithm=None)
    runner._create_graph = lambda: SimpleNamespace(pool=lambda: None)
    runner._capture_init = lambda fn: None
    runner._capture_graph = lambda *args: None
    monkeypatch.setattr(module, "forward_context", lambda *args: nullcontext())
    monkeypatch.setattr(module, "get_global_graph_memory_pool", lambda: None)
    monkeypatch.setattr(module, "set_global_graph_memory_pool", lambda pool: None)
    if admitted:
        runner.capture_one_batch_size(1, lambda: None)
    else:
        with pytest.raises(NotImplementedError, match="pending index-key ring"):
            runner.capture_one_batch_size(1, lambda: None)
    assert seen == [width]
