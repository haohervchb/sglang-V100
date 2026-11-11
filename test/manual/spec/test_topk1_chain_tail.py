"""The topk=1 draft chain shortcut must reproduce the generic tree tail exactly."""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _generic_tail(steps, bs, gen):
    from sglang.srt.speculative.spec_utils import select_top_k_tokens

    topk_p = torch.rand((bs, 1), device="cuda", generator=gen)
    topk_index = torch.randint(0, 248320, (bs, 1), device="cuda", generator=gen)
    hidden = torch.randn((bs, 2560), device="cuda", generator=gen).half()
    score_list, token_list, parents_list = [], [], []
    scores = None
    draws = []
    for i in range(steps):
        input_ids, hidden, scores, tree_info = select_top_k_tokens(
            i, topk_p, topk_index, hidden, scores, 1
        )
        score_list.append(tree_info[0])
        token_list.append(tree_info[1])
        parents_list.append(tree_info[2])
        draws.append((input_ids.clone(), hidden.clone()))
        if i == steps - 1:
            break
        topk_index = torch.randint(0, 248320, (bs, 1), device="cuda", generator=gen)
        topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top = torch.sort(torch.topk(score_list, steps, dim=-1).indices).values
    draft_tokens = torch.gather(ss_token_list, index=top, dim=1)
    parent_list = torch.cat(parents_list[:-1], dim=1)
    return parent_list, top, draft_tokens, draws


@pytest.mark.parametrize("steps", [2, 3, 4])
@pytest.mark.parametrize("bs", [1, 2, 3, 8])
def test_chain_constants_match_generic_tree(steps, bs):
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    gen = torch.Generator(device="cuda")
    gen.manual_seed(steps * 10 + bs)
    parent_ref, top_ref, tokens_ref, draws = _generic_tail(steps, bs, gen)
    worker = types.SimpleNamespace(speculative_num_steps=steps)
    parent, top = EagleDraftWorker._chain_tree_constants(worker, bs, torch.device("cuda"))
    assert parent.dtype == parent_ref.dtype == torch.int64
    assert top.dtype == top_ref.dtype == torch.int64
    assert parent.is_contiguous() and top.is_contiguous()
    assert torch.equal(parent, parent_ref)
    assert torch.equal(top, top_ref)
    # Later steps of a topk=1 chain pass tokens and hidden states through unchanged.
    gen.manual_seed(steps * 10 + bs)
    torch.rand((bs, 1), device="cuda", generator=gen)
    first_tokens = torch.randint(0, 248320, (bs, 1), device="cuda", generator=gen)
    torch.randn((bs, 2560), device="cuda", generator=gen)
    later = [torch.randint(0, 248320, (bs, 1), device="cuda", generator=gen) for _ in range(steps - 1)]
    assert torch.equal(torch.cat([first_tokens, *later], dim=1), tokens_ref)
    for i in range(1, steps):
        assert torch.equal(draws[i][0], later[i - 1].flatten())
        assert torch.equal(draws[i][1], draws[0][1])
    # Larger batches reuse the cached buffer's leading rows.
    parent_big, _ = EagleDraftWorker._chain_tree_constants(worker, bs + 70, torch.device("cuda"))
    parent_small, _ = EagleDraftWorker._chain_tree_constants(worker, bs, torch.device("cuda"))
    assert torch.equal(parent_small, parent_ref) and parent_big.shape == (bs + 70, steps)
