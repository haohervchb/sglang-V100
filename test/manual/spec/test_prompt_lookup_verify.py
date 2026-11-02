"""Exercise the actual EAGLE chain verifier at every rejection position."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rejected_at", range(4))
def test_prompt_chain_target_verification(rejected_at):
    device = "cuda"
    seq = torch.tensor([100], device=device, dtype=torch.int32)
    mask, positions, indices, child, sibling, tokens = build_tree_kernel_efficient(
        torch.tensor([10], device=device, dtype=torch.int32),
        torch.tensor([[-1, 0, 1]], device=device),
        torch.tensor([[0, 1, 2]], device=device),
        torch.tensor([[11, 12, 13]], device=device),
        seq,
        100,
        1,
        3,
        4,
        TreeMaskMode.QLEN_ONLY,
    )
    verify = EagleVerifyInput(
        draft_token=tokens,
        custom_mask=mask,
        positions=positions,
        retrieve_index=indices,
        retrieve_next_token=child,
        retrieve_next_sibling=sibling,
        retrieve_cum_len=None,
        spec_steps=3,
        topk=1,
        draft_token_num=4,
        capture_hidden_mode=None,
        seq_lens_sum=None,
        seq_lens_cpu=None,
    )
    logits = torch.full((4, 128), -100.0, device=device)
    choices = [11, 12, 13, 99]
    choices[rejected_at] = 99
    logits[torch.arange(4, device=device), torch.tensor(choices, device=device)] = 100
    batch = SimpleNamespace(
        device=device,
        seq_lens=seq,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        sampling_info=SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            is_all_greedy=True,
        ),
    )
    predict, lengths, accepted = verify.sample(
        batch, SimpleNamespace(next_token_logits=logits), None
    )
    assert lengths.item() == rejected_at + 1
    assert predict[accepted[0, : rejected_at + 1]].tolist() == [11, 12, 13][
        :rejected_at
    ] + [99]
    assert positions.tolist() == [100, 101, 102, 103]
