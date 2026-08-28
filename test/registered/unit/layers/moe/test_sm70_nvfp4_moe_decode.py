import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The specialized NVFP4 MoE decode kernel requires an NVIDIA V100",
)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_sm70_topk10_softmax_matches_torch(batch_size):
    from sglang.jit_kernel.sm70_nvfp4_moe_decode import sm70_topk10_softmax

    torch.manual_seed(100 + batch_size)
    logits = torch.randn(
        batch_size, 512, dtype=torch.float16, device="cuda"
    )
    weights, ids = sm70_topk10_softmax(logits)

    probabilities = torch.softmax(logits.float(), dim=-1)
    _, reference_ids = torch.topk(probabilities, 10, dim=-1)
    selected = torch.gather(probabilities, 1, ids.long())
    reference_weights = selected / selected.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(
        torch.sort(ids).values,
        torch.sort(reference_ids.int()).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(weights, reference_weights, rtol=2e-6, atol=3e-8)


def test_sm70_nvfp4_moe_four_rows_match_individual_rows():
    from sglang.jit_kernel.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode

    torch.manual_seed(11)
    # Ten experts are sufficient for this route-layout test. The serving path
    # supplies all 512; keeping the fixture small avoids a >300 MB allocation.
    w13 = torch.empty(
        (10, 160, 640), dtype=torch.int32, device="cuda"
    ).random_()
    w2 = torch.empty(
        (10, 10, 5120), dtype=torch.int32, device="cuda"
    ).random_()
    w13_scales = torch.full(
        (10, 160, 320), 0x70, dtype=torch.uint8, device="cuda"
    )
    w2_scales = torch.full(
        (10, 10, 2560), 0x70, dtype=torch.uint8, device="cuda"
    )
    w13_global = torch.full((10,), 100.0, dtype=torch.float32, device="cuda")
    w2_global = torch.full((10,), 100.0, dtype=torch.float32, device="cuda")
    hidden_states = torch.randn((4, 2560), dtype=torch.float16, device="cuda")
    ids = torch.arange(10, dtype=torch.int32, device="cuda").repeat(4, 1)
    weights = torch.softmax(torch.randn((4, 10), device="cuda"), dim=-1)

    batched = sm70_nvfp4_moe_decode(
        hidden_states,
        w13,
        w2,
        w13_scales,
        w2_scales,
        w13_global,
        w2_global,
        ids.flatten(),
        weights.flatten(),
    )
    individual = torch.cat(
        [
            sm70_nvfp4_moe_decode(
                hidden_states[row : row + 1],
                w13,
                w2,
                w13_scales,
                w2_scales,
                w13_global,
                w2_global,
                ids[row],
                weights[row],
            )
            for row in range(4)
        ]
    )

    assert torch.isfinite(batched).all()
    assert batched.abs().max() > 0
    torch.testing.assert_close(batched, individual, rtol=0, atol=0)
