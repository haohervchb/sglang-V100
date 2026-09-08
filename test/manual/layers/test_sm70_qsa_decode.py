import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Native QSA decode requires V100",
)


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("topk", [2048, 2051])
def test_selected_cache_attention_and_changing_graph_inputs(batch, topk):
    from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import (
        sm70_cuda_qsa_decode,
    )

    torch.manual_seed(31 + batch)
    pool = 70032
    q = torch.randn(batch, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(pool, 1, 256, device="cuda").to(torch.float8_e5m2)
    v = torch.randn(pool, 1, 256, device="cuda").to(torch.float8_e5m2)
    table = torch.stack(
        [torch.randperm(pool, device="cuda", dtype=torch.int32) for _ in range(2)]
    )
    requests = torch.zeros(batch, device="cuda", dtype=torch.int32)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(batch, 1)
    lengths = torch.full((batch,), 25000, device="cuda", dtype=torch.int32)

    def run():
        return sm70_cuda_qsa_decode(q, k, v, table, requests, indices, lengths, 0.0625)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()

    for iteration, length in enumerate([1, 31, 32, 33, 1000, 2048, 25000, 70000]):
        q.normal_()
        lengths.copy_(torch.arange(batch, device="cuda", dtype=torch.int32) + length)
        requests.copy_(
            (torch.arange(batch, device="cuda", dtype=torch.int32) + iteration) % 2
        )
        # Both the physical cache mapping and selected logical positions change.
        table.copy_(table.roll(17, dims=1))
        indices.fill_(-1)
        expected = []
        for row in range(batch):
            count = min(length + row, topk)
            selected = torch.randperm(length + row, device="cuda")[:count].sort().values
            if count > 4:
                # Include masked selections inside active splits, not only padding.
                selected[3::37] = -1
            indices[row, :count].copy_(selected)
            valid = selected >= 0
            slots = table[(row + iteration) % 2, selected[valid]]
            keys, values = k[slots, 0].float(), v[slots, 0].float()
            scores = (q[row].float() @ keys.T) * 0.0625
            expected.append(scores.softmax(-1) @ values)
        graph.replay()
        reference = torch.stack(expected)
        torch.testing.assert_close(output.float(), reference, rtol=0.005, atol=0.001)
        assert (output.float() - reference).norm() / reference.norm() < 0.001
