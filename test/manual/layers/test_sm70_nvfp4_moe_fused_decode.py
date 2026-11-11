"""The fused two-kernel NVFP4 MoE decode must match the four-kernel path bit for bit."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The specialized NVFP4 MoE decode kernel requires an NVIDIA V100",
)

EXPERTS = 512


@pytest.fixture(scope="module")
def setup():
    from sglang.jit_kernel.sm70_nvfp4_moe_decode import (
        _load_sm70_nvfp4_moe_decode_ops,
    )

    ext = _load_sm70_nvfp4_moe_decode_ops()
    g = torch.Generator(device="cuda")
    g.manual_seed(1234)
    w13 = torch.randint(
        -(2**31), 2**31 - 1, (EXPERTS, 160, 640), dtype=torch.int32, device="cuda", generator=g
    )
    w2 = torch.randint(
        -(2**31), 2**31 - 1, (EXPERTS, 10, 5120), dtype=torch.int32, device="cuda", generator=g
    )
    # FP8-E4M3 scale bytes; global scales near 1e2 give O(1) outputs.
    s13 = torch.randint(0x60, 0x78, (EXPERTS, 160, 320), dtype=torch.uint8, device="cuda", generator=g)
    s2 = torch.randint(0x60, 0x78, (EXPERTS, 10, 2560), dtype=torch.uint8, device="cuda", generator=g)
    g13 = ((torch.rand(EXPERTS, device="cuda", generator=g) * 2 + 0.25) * 128).float()
    g2 = ((torch.rand(EXPERTS, device="cuda", generator=g) * 2 + 0.25) * 128).float()
    return ext, (w13, w2, s13, s2, g13, g2)


def _ids(rows, mode, gen):
    if mode == "distinct":
        return torch.stack(
            [torch.randperm(EXPERTS, device="cuda", generator=gen)[:10] for _ in range(rows)]
        ).int()
    pool = torch.randperm(EXPERTS, device="cuda", generator=gen)[: max(10, rows * 10 // 4)]
    ids = torch.stack(
        [pool[torch.randperm(pool.numel(), device="cuda", generator=gen)[:10]] for _ in range(rows)]
    ).int()
    if mode == "invalid":
        ids[0, 3] = -1
        ids[-1, 7] = EXPERTS
    return ids


def _legacy(ext, weights, x, ids, wts):
    routes = x.shape[0] * 10
    gp = torch.empty((40, routes, 320), dtype=torch.float32, device="cuda")
    act = torch.empty((routes, 160), dtype=torch.float16, device="cuda")
    dp = torch.empty((routes, 5, 2560), dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    ext.decode_legacy(x, *weights, ids.flatten(), wts.flatten(), gp, act, dp, out)
    return out


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 12, 16, 24, 32])
@pytest.mark.parametrize("mode", ["distinct", "shared", "invalid"])
def test_fused_matches_legacy_bits(setup, rows, mode):
    from sglang.jit_kernel.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode

    ext, weights = setup
    gen = torch.Generator(device="cuda")
    gen.manual_seed(rows * 31 + len(mode))
    x = torch.randn((rows, 2560), dtype=torch.float16, device="cuda", generator=gen)
    ids = _ids(rows, mode, gen)
    wts = torch.softmax(torch.randn((rows, 10), device="cuda", generator=gen), -1)
    expected = _legacy(ext, weights, x, ids, wts)
    actual = sm70_nvfp4_moe_decode(x, *weights, ids.flatten(), wts.flatten())
    assert (expected != 0).float().mean() > 0.5
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("rows", [4, 16, 32])
def test_fused_graph_replay_with_changing_routes(setup, rows):
    from sglang.jit_kernel.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode

    ext, weights = setup
    gen = torch.Generator(device="cuda")
    gen.manual_seed(900 + rows)
    x = torch.randn((rows, 2560), dtype=torch.float16, device="cuda", generator=gen)
    ids = _ids(rows, "distinct", gen).flatten().contiguous()
    wts = torch.softmax(torch.randn((rows, 10), device="cuda", generator=gen), -1).flatten()
    sm70_nvfp4_moe_decode(x, *weights, ids, wts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = sm70_nvfp4_moe_decode(x, *weights, ids, wts)
    for mode in ["shared", "distinct", "invalid"]:
        x.copy_(torch.randn(x.shape, dtype=x.dtype, device="cuda", generator=gen))
        ids.copy_(_ids(rows, mode, gen).flatten())
        wts.copy_(torch.softmax(torch.randn((rows, 10), device="cuda", generator=gen), -1).flatten())
        for _ in range(2):
            graph.replay()
            expected = _legacy(ext, weights, x, ids.view(rows, 10), wts.view(rows, 10))
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
