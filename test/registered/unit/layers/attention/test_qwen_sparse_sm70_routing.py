from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _FakeTensor:
    def __init__(self, shape, dtype, *, is_cuda=True):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.is_cuda = is_cuda
        self.device = torch.device("cuda:0")

    @property
    def ndim(self):
        return len(self.shape)


@pytest.mark.parametrize(
    "forward_mode",
    [
        ForwardMode.DECODE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.DRAFT_EXTEND_V2,
    ],
)
def test_sm70_qsa_decode_accepts_paged_decode_modes(monkeypatch, forward_mode):
    rows = 4
    q = _FakeTensor((rows, 6, 256), torch.float16)
    k = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    v = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    topk_indices = _FakeTensor((rows, 2048), torch.int32)
    metadata = SimpleNamespace(
        sequence_lengths=SimpleNamespace(numel=lambda: rows),
    )
    forward_batch = SimpleNamespace(forward_mode=forward_mode)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))

    assert QwenSparseAttnBackend._can_use_sm70_sparse_decode(
        q,
        k,
        v,
        forward_batch,
        metadata,
        topk_indices,
    )


def test_sm70_qsa_decode_rejects_ordinary_extend(monkeypatch):
    rows = 4
    q = _FakeTensor((rows, 6, 256), torch.float16)
    k = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    v = _FakeTensor((4096, 1, 256), torch.float8_e5m2)
    topk_indices = _FakeTensor((rows, 2048), torch.int32)
    metadata = SimpleNamespace(
        sequence_lengths=SimpleNamespace(numel=lambda: rows),
    )
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))

    assert not QwenSparseAttnBackend._can_use_sm70_sparse_decode(
        q,
        k,
        v,
        forward_batch,
        metadata,
        topk_indices,
    )
