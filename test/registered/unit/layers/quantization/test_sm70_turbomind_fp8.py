import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.sm70_turbomind_fp8 import (
    _load_sm70_turbomind_fp8_ops,
    apply_sm70_turbomind_fp8_fused_silu_and_mul,
    apply_sm70_turbomind_fp8_linear,
    prepare_sm70_turbomind_fp8_linear,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="TurboMind SM70 FP8 kernels require an NVIDIA V100",
)


class _DummyLinear(torch.nn.Module):
    def __init__(self, output_size: int, input_size: int, gated: bool):
        super().__init__()
        weight = (
            torch.randn((output_size, input_size), device="cuda", dtype=torch.float16)
            * 0.1
        ).to(torch.float8_e4m3fn)
        scales = (
            torch.rand(
                ((output_size + 127) // 128, (input_size + 127) // 128),
                device="cuda",
                dtype=torch.float32,
            )
            .mul_(0.05)
            .add_(0.95)
        )
        self.register_parameter("weight", torch.nn.Parameter(weight, False))
        self.register_parameter("weight_scale_inv", torch.nn.Parameter(scales, False))
        self.orig_dtype = torch.float16
        self.weight_block_size = [128, 128]
        self.output_size_per_partition = output_size
        self.input_size_per_partition = input_size
        self.logical_widths = (
            [output_size // 2, output_size // 2] if gated else [output_size]
        )
        self.prefix = (
            "model.layers.0.mlp.gate_up_proj"
            if gated
            else "model.layers.0.self_attn.qkv_proj"
        )


def _expanded_weight(layer: _DummyLinear) -> torch.Tensor:
    scales = layer.sm70_fp8_prefill_scales.repeat_interleave(128, 0).repeat_interleave(
        128, 1
    )
    weight = layer.sm70_fp8_prefill_weight.float()
    return (weight * scales[: weight.shape[0], : weight.shape[1]]).half()


@pytest.mark.parametrize("gated,output_size", [(False, 512), (True, 1024)])
def test_prefill_bridge_and_decode_kernel(monkeypatch, gated, output_size):
    monkeypatch.setenv("SGLANG_SM70_FP8_PREFILL_BACKEND", "fp16")
    monkeypatch.setenv("SGLANG_SM70_FP8_PREFILL_MIN_TOKENS", "64")
    torch.manual_seed(7)
    assert _load_sm70_turbomind_fp8_ops()

    layer = _DummyLinear(output_size, 512, gated)
    prepare_sm70_turbomind_fp8_linear(layer)
    weight = _expanded_weight(layer)
    prefill_input = torch.randn(64, 512, device="cuda", dtype=torch.float16)
    if gated:
        actual = apply_sm70_turbomind_fp8_fused_silu_and_mul(layer, prefill_input)
        projected = F.linear(prefill_input, weight)
        gate, up = projected.chunk(2, dim=-1)
        expected = F.silu(gate) * up
    else:
        bias = torch.randn(output_size, device="cuda", dtype=torch.float16)
        actual = apply_sm70_turbomind_fp8_linear(layer, prefill_input, bias)
        expected = F.linear(prefill_input, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    decode_input = torch.randn(1, 512, device="cuda", dtype=torch.float16)
    decode = (
        apply_sm70_turbomind_fp8_fused_silu_and_mul(layer, decode_input)
        if gated
        else apply_sm70_turbomind_fp8_linear(layer, decode_input, None)
    )
    assert torch.isfinite(decode).all()


def test_decode_workspace_is_safe_across_cuda_streams():
    torch.manual_seed(19)
    assert _load_sm70_turbomind_fp8_ops()
    size = 1024
    weight = (torch.randn((size, size), device="cuda", dtype=torch.float16) * 0.05).to(
        torch.float8_e4m3fn
    )
    scales = torch.ones((8, 8), device="cuda", dtype=torch.float32)
    packed, packed_scales, meta = torch.ops.sglang_sm70_turbomind.fp8_prepare(
        weight, scales, 128, False
    )
    k_ld, q_ld = int(meta[0]), int(meta[1])
    inputs = [
        torch.randn((64, size), device="cuda", dtype=torch.float16) for _ in range(2)
    ]
    baselines = []
    for input_ in inputs:
        output = torch.empty((64, size), device="cuda", dtype=torch.float16)
        torch.ops.sglang_sm70_turbomind.fp8_gemm(
            output,
            input_,
            packed,
            packed_scales,
            128,
            k_ld,
            q_ld,
            False,
        )
        baselines.append(output)
    torch.cuda.synchronize()

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = [
        [torch.empty((64, size), device="cuda", dtype=torch.float16) for _ in range(8)]
        for _ in streams
    ]
    for stream, input_, stream_outputs in zip(streams, inputs, outputs):
        with torch.cuda.stream(stream):
            for output in stream_outputs:
                torch.ops.sglang_sm70_turbomind.fp8_gemm(
                    output,
                    input_,
                    packed,
                    packed_scales,
                    128,
                    k_ld,
                    q_ld,
                    False,
                )
    torch.cuda.synchronize()

    for baseline, stream_outputs in zip(baselines, outputs):
        for output in stream_outputs:
            torch.testing.assert_close(output, baseline, rtol=0, atol=0)
