import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The Qwen fusion kernels require a V100",
)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_shared_gate_preserves_fp16_boundaries_and_replays(seed):
    from sglang.jit_kernel.sm70_qwen_fusions import gate

    torch.manual_seed(seed)
    x = torch.randn((1, 2560), device="cuda", dtype=torch.float16)
    weight = torch.randn_like(x) * 0.01
    value = torch.randn_like(x)
    gate(x, weight, value)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gate(x, weight, value)
    for scale in [1.0, -4.0, 0.0]:
        x.mul_(scale)
        graph.replay()
        projected = (x.float() @ weight.float().T).half()
        expected = torch.sigmoid(projected) * value
        torch.testing.assert_close(output, expected, rtol=0.002, atol=0.001)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_shared_gate_up_matches_fp32_reference(seed):
    from sglang.jit_kernel.sm70_qwen_fusions import gate_up

    torch.manual_seed(seed)
    x = torch.randn((1, 2560), device="cuda", dtype=torch.float16)
    weight = torch.randn((320, 2560), device="cuda", dtype=torch.float16) * 0.01
    output = gate_up(x, weight)
    projected = (x.float() @ weight.float().T).half().float()
    expected = (
        torch.nn.functional.silu(projected[:, :160]) * projected[:, 160:]
    ).half()
    torch.testing.assert_close(output, expected, rtol=0.002, atol=0.001)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gate_up(x, weight)
    x.neg_()
    graph.replay()
    torch.testing.assert_close(captured, gate_up(x, weight), rtol=0, atol=0)


def test_qkv_ba_projection_and_prefix_aliasing(monkeypatch):
    from sglang.jit_kernel.sm70_qwen_fusions import (
        qkv_ba,
        qkv_ba_supported,
        reuse_qkv_prefix,
    )

    monkeypatch.setenv("SGLANG_SM70_DENSE_GEMV", "1")
    monkeypatch.setenv("SGLANG_SM70_QWEN_FUSIONS", "1")
    torch.manual_seed(51)
    x = torch.randn((1, 2560), device="cuda", dtype=torch.float16)
    weight = torch.randn((4096, 2560), device="cuda", dtype=torch.float16) * 0.01
    tail = torch.randn((24, 2560), device="cuda", dtype=torch.float16) * 0.01
    assert qkv_ba_supported(x, weight, tail)
    assert not qkv_ba_supported(x, None, tail)
    assert not qkv_ba_supported(x.expand(2, -1), weight, tail)
    a, b = qkv_ba(x, weight, tail)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        a, b = qkv_ba(x, weight, tail)
    for scale in [1.0, -0.5]:
        x.mul_(scale)
        graph.replay()
        torch.testing.assert_close(
            a, (x.float() @ weight.float().T).half(), rtol=0.002, atol=0.001
        )
        torch.testing.assert_close(
            b, (x.float() @ tail.float().T).half(), rtol=0.002, atol=0.001
        )
    q, k, v, z = a.split([512, 512, 1536, 1536], dim=-1)
    z_before = z.clone()
    prefix = reuse_qkv_prefix(a, q, k, v)
    assert prefix.is_contiguous() and prefix.data_ptr() == a.data_ptr()
    torch.testing.assert_close(prefix, torch.cat([q, k, v], dim=-1), rtol=0, atol=0)
    assert reuse_qkv_prefix(a, q, k, v.clone()) is None
    prefix.zero_()
    torch.testing.assert_close(z, z_before, rtol=0, atol=0)
