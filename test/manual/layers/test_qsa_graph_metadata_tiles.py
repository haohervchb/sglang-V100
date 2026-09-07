"""Check parallel page-table construction against independent pool arithmetic."""

import pytest
import torch
import triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("pages", [129, 8192])
def test_page_tiles_and_graph_replay(rows, pages):
    from sglang.srt.layers.attention.qsa.graph_metadata import (
        _qsa_graph_row_metadata_kernel,
    )

    torch.manual_seed(19)
    ratio, full_page = 4, 32
    page_ids = torch.randint(1, 10000, (6, pages), device="cuda")
    pool = (
        page_ids[:, :, None] * full_page
        + torch.arange(full_page, device="cuda")[None, None, :]
    ).flatten(1)
    seq = torch.tensor([1, 4, 251, 2500][:rows], dtype=torch.int32, device="cuda")
    req = torch.arange(1, rows + 1, dtype=torch.int32, device="cuda")

    def buffers():
        return [
            torch.empty(rows, dtype=torch.int32, device="cuda"),
            torch.empty(rows, dtype=torch.int32, device="cuda"),
            torch.empty((rows, pages), dtype=torch.int32, device="cuda"),
            torch.empty(rows, dtype=torch.int32, device="cuda"),
            torch.empty(rows, dtype=torch.int64, device="cuda"),
            torch.empty((rows, ratio), dtype=torch.int32, device="cuda"),
        ]

    serial, tiled = buffers(), buffers()

    def run(outputs, tiles):
        _qsa_graph_row_metadata_kernel[(rows, tiles)](
            seq,
            req,
            *outputs,
            pool,
            pool.stride(0),
            pages,
            RATIO=ratio,
            FULL_PAGE=full_page,
            PAGE_BLOCK=128,
            num_warps=1,
        )

    run(serial, 1)
    run(tiled, triton.cdiv(pages, 128))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(tiled, triton.cdiv(pages, 128))
    for increment in [0, 1, 2, 3]:
        seq.add_(increment)
        req.copy_(torch.arange(rows, device="cuda", dtype=torch.int32) + increment % 2)
        graph.replay()
        run(serial, 1)
        current = seq - 1
        last = pool[req.long(), current.long()]
        expected = [
            seq // ratio,
            torch.where(seq % ratio == 0, last // ratio, 0),
            page_ids[req.long()],
            current,
            req.long() * ratio + current % ratio,
            req[:, None] * ratio
            + (
                current[:, None] - torch.arange(ratio - 1, -1, -1, device="cuda")
            ).clamp_min(0)
            % ratio,
        ]
        for actual, old, reference in zip(tiled, serial, expected):
            torch.testing.assert_close(actual, old, rtol=0, atol=0)
            torch.testing.assert_close(
                actual, reference.to(actual.dtype), rtol=0, atol=0
            )
