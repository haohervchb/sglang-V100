import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 sparse index ordering requires V100",
)


@pytest.mark.parametrize("rows", [1, 2, 4, 17])
def test_ordered_expansion_preserves_selection_tail_and_graph(rows, monkeypatch):
    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

    torch.manual_seed(123)
    blocks = torch.zeros(rows, 512, device="cuda", dtype=torch.int32)
    lengths = torch.full((rows,), 25000, device="cuda", dtype=torch.int32)
    positions = lengths - 1
    monkeypatch.setenv("SGLANG_SM70_MTP_QSA_ORDER", "1")

    def run():
        return expand_qsa_block_indices(blocks, positions, lengths, 4, 2048)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for length in [1, 3, 4, 31, 2047, 2048, 2049, 2051, 25003, 70001]:
        lengths.copy_(torch.arange(rows, device="cuda", dtype=torch.int32) + length)
        positions.copy_(lengths - 1)
        blocks.fill_(-1)
        expected = torch.full_like(output, -1)
        for row in range(rows):
            visible = length + row
            count = min(512, visible // 4)
            chosen = torch.randperm(visible // 4, device="cuda")[:count]
            blocks[row, :count].copy_(chosen)
            expanded = (
                chosen.sort().values[:, None] * 4 + torch.arange(4, device="cuda")
            ).flatten()
            tail = torch.arange(visible // 4 * 4, visible, device="cuda")
            selected = torch.cat((expanded, tail))
            expected[row, : len(selected)].copy_(selected)
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
