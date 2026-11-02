from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa import graph_metadata as legacy
from sglang.srt.layers.attention.qsa import graph_metadata_fused as fused

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def metadata(rows, pages, ratio):
    def empty(*shape):
        return torch.full(shape, -999, dtype=torch.int32, device="cuda")

    return SimpleNamespace(
        sequence_lengths=empty(rows),
        row_req_pool_indices=empty(rows),
        indexer_metadata=SimpleNamespace(
            graph_prefix_lengths=empty(rows),
            graph_compressed_lengths=empty(rows),
            graph_write_locs=empty(rows),
            graph_compressed_page_table=empty(rows, pages),
            decode_logical_positions=empty(rows),
            pending_ring_slots=empty(rows),
            graph_ring_group_locs=empty(rows, ratio),
            compress_ratio=ratio,
        ),
    )


def tensors(m):
    return [m.sequence_lengths, m.row_req_pool_indices] + [
        x for x in vars(m.indexer_metadata).values() if isinstance(x, torch.Tensor)
    ]


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("bs,pad", [(1, 0), (1, 1), (4, 0), (4, 2), (4, 4)])
@pytest.mark.parametrize("ratio", [4, 8])
def test_fused_matches_legacy_with_live_graph_inputs(monkeypatch, mode, bs, pad, ratio):
    monkeypatch.delenv("SGLANG_SM70_QSA_FUSED_GRAPH_METADATA", raising=False)
    rows = bs if mode == 0 else bs * 4
    pages = 16384
    pool = SimpleNamespace(
        qsa_compress_ratio=ratio,
        qsa_ring_size=ratio * 2,
        qsa_compressed_page_size=16 // ratio,
    )
    table = torch.arange(5 * 262144, dtype=torch.int32, device="cuda").reshape(5, -1)
    seq = torch.tensor([8, 15, 16, 262136][:bs], dtype=torch.int32, device="cuda")
    req = torch.arange(1, bs + 1, device="cuda")
    extend = torch.tensor([4, 1, 3, 2][:bs], dtype=torch.int32, device="cuda")
    old, new = metadata(rows, pages, ratio), metadata(rows, pages, ratio)
    kw = dict(
        mode=mode,
        bs=bs,
        num_rows=rows,
        seq_lens=seq,
        req_pool_indices=req,
        extend_lens=extend if mode == 2 else None,
        extend_len=4 if mode == 1 else 0,
        num_padding=pad,
        req_to_token=table,
        pool=pool,
    )

    def run():
        legacy.launch_graph_metadata(metadata=old, **kw)
        fused.launch_graph_metadata(metadata=new, **kw)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        seq.add_(1)
        table.add_(16)
        graph.replay()
        for a, b in zip(tensors(old), tensors(new)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    if mode == 0:
        expected = seq.cpu()
        expected[bs - pad :] = 1
        torch.testing.assert_close(new.sequence_lengths.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("bs,pad", [(1, 0), (4, 0), (4, 2), (4, 4)])
def test_grouped_draft_steps(monkeypatch, bs, pad):
    monkeypatch.delenv("SGLANG_SM70_QSA_FUSED_GRAPH_METADATA", raising=False)
    ratio, pages = 4, 16384
    pool = SimpleNamespace(
        qsa_compress_ratio=ratio, qsa_ring_size=ratio * 2, qsa_compressed_page_size=4
    )
    table = torch.arange(5 * 262144, dtype=torch.int32, device="cuda").reshape(5, -1)
    seq = torch.tensor([7, 15, 70000, 262136][:bs], dtype=torch.int32, device="cuda")
    req = torch.arange(1, bs + 1, device="cuda")
    old = [metadata(bs, pages, ratio) for _ in range(2)]
    new = [metadata(bs, pages, ratio) for _ in range(2)]
    args = fused.prepare_draft_graph_metadata(new, table, pool)
    for _ in range(3):
        fused.launch_draft_graph_metadata(args, seq, req, bs, pad)
        for step, m in enumerate(old):
            legacy.launch_graph_metadata(
                mode=0,
                bs=bs,
                num_rows=bs,
                seq_lens=seq + step + 1,
                req_pool_indices=req,
                extend_lens=None,
                extend_len=0,
                num_padding=pad,
                metadata=m,
                req_to_token=table,
                pool=pool,
            )
        for a, b in zip(old, new):
            for x, y in zip(tensors(a), tensors(b)):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        seq.add_(1)
