from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl
from huggingface_hub import snapshot_download

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    patch_tensor_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, next_power_of_2

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


if _is_cuda:
    from sgl_kernel import fast_topk
elif _is_hip:
    from sgl_kernel import fast_topk
else:
    from sglang.srt.utils.common import fast_topk


logger = logging.getLogger(__name__)


# Simulate acceptance length for benchmarking purposes
SIMULATE_ACC_LEN = envs.SGLANG_SIMULATE_ACC_LEN.get()  # turn off if < 0
SIMULATE_ACC_METHOD = envs.SGLANG_SIMULATE_ACC_METHOD.get()

TREE_TRAVERSE_TIME_THRESHOLD = 1  # TODO: set this properly
TREE_SPEC_KERNEL_AVAILABLE = (
    _is_cuda or _is_musa
)  # This kernel is only available for CUDA and MUSA now


def record_stream_each(tensors, stream):
    """Call record_stream(stream) on each cuda tensor in `tensors`, skipping
    non-tensor / non-cuda entries. Tells the caching allocator that the
    tensors are also used on `stream`, so memory is not recycled while
    queued work is still in flight after Python refs drop.
    """
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            t.record_stream(stream)


def record_stream_for_v2_verify(batch, verify_input, fwd_stream):
    """Mark pre-prepare SB / verify_input GPU tensors as used on `fwd_stream`.

    Spec V2 mutates SB mid-forward (`prepare_for_v2_verify` rebinds
    `batch.input_ids` / `out_cache_loc`; `_draft_extend_for_decode` later
    replaces `batch.input_ids` again). Each rebind drops the only SB Python
    ref to the old tensor while the verify forward kernel may still be
    reading its memory on `fwd_stream`; `record_stream` tells the caching
    allocator to wait for `fwd_stream` before recycling the block.

    Covers pre-prepare tensors only; caller must also `record_stream_each`
    the post-prepare rebinds (new `batch.input_ids` / `out_cache_loc`).
    """
    candidates = [
        batch.seq_lens,
        batch.req_pool_indices,
        batch.input_ids,
        batch.out_cache_loc,
    ]
    if verify_input is not None:
        candidates.extend(
            [
                getattr(verify_input, attr, None)
                for attr in (
                    "draft_token",
                    "custom_mask",
                    "positions",
                    "retrieve_index",
                    "retrieve_next_token",
                    "retrieve_next_sibling",
                )
            ]
        )
    record_stream_each(candidates, fwd_stream)


def spec_need_hidden_states(server_args: Optional[ServerArgs] = None) -> bool:
    if server_args is None:
        server_args = get_global_server_args()

    # STANDALONE drafts don't consume `spec_info.hidden_states` (vanilla LLM).
    # multi_layer_eagle and DFLASH don't relay hidden_states through FutureMap.
    # TODO(lsyin): also skip when step == 1.
    if server_args.speculative_algorithm in ("STANDALONE", "DFLASH", "DSPARK"):
        return False
    return not server_args.enable_multi_layer_eagle


@triton.jit
def create_extend_after_decode_spec_info(
    accept_tokens,
    seq_lens,
    accept_lens,
    positions,
    bonus_tokens_ptr,
    bs_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, bs_upper)
    seq_length = tl.load(seq_lens + pid)
    # `accept_lens` includes the bonus token; load this req's value.
    accept_len = tl.load(accept_lens + pid)

    accept_len_cumsum = tl.sum(
        tl.load(accept_lens + offsets, mask=offsets < pid, other=0)
    )
    positions_ptr = positions + accept_len_cumsum
    mask = offsets < accept_len
    tl.store(positions_ptr + offsets, seq_length - accept_len + offsets, mask)

    accept_len_cumsum += accept_len - 1
    bonus_token = tl.load(accept_tokens + accept_len_cumsum)
    tl.store(bonus_tokens_ptr + pid, bonus_token)


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


def assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
):
    assign_req_to_token_pool[(batch_size,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        next_power_of_2(batch_size),
    )


@triton.jit
def assign_draft_cache_locs(
    req_pool_indices,
    req_to_token,
    seq_lens,
    extend_lens,
    num_new_pages_per_topk,
    out_cache_loc,
    source_cache_loc,
    target_cache_loc,
    last_page_lens_cumsum,
    duplicate_cache_len: tl.constexpr,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    if page_size == 1 or topk == 1:
        copy_len = topk * speculative_num_steps
        out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    else:
        bs_offset = tl.arange(0, bs_upper)
        copy_len = tl.load(extend_lens + pid)
        cum_copy_len = tl.sum(tl.load(extend_lens + bs_offset, mask=bs_offset < pid))
        out_cache_ptr = out_cache_loc + cum_copy_len

    # Part 1: Copy from out_cache_loc to req_to_token
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(out_cache_ptr + copy_offset, mask=mask)
        tl.store(token_pool + kv_start + copy_offset, data, mask=mask)
    # XXX (MUSA): Triton issue: chained boolean operators (A or B or C) are not supported.
    if (page_size != 1 and topk != 1) and duplicate_cache_len > 0:
        # Part 2: Copy indices into source_cache_loc and target_cache_loc
        # Expected output: src:[8,9,10,8,9,10...] tgt:[16,17,18,24,25,26...]
        prefix_len = tl.load(seq_lens + pid)
        last_page_len = prefix_len % page_size
        offsets = tl.arange(0, page_size)
        mask = offsets < last_page_len
        num_new_pages_per_topk_ = tl.load(num_new_pages_per_topk + pid)
        prefix_base = token_pool + prefix_len - last_page_len
        src_indices = tl.load(prefix_base + offsets, mask=mask)
        last_page_lens_cumsum_ = tl.load(last_page_lens_cumsum + pid)
        # Skip the first one since no copy is needed
        for topk_id in range(1, topk):
            tl.store(
                source_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                src_indices,
                mask=mask,
            )
            tgt_indices = tl.load(
                prefix_base + topk_id * num_new_pages_per_topk_ * page_size + offsets,
                mask=mask,
            )
            tl.store(
                target_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                tgt_indices,
                mask=mask,
            )
        # Part 3: Copy and remove the used indices for duplication
        # speculative_num_steps=5, page_size=4, num_new_pages_per_topk_=2, last_page_len=1
        #  - xxxxx .. | - xxxxx .. |
        #   topk=0        topk=1
        #  "-" means prefix tokens
        #  "x" means speculative draft tokens
        #  "." means padded tokens
        # we only want to copy the "x" part.
        iter_offset = tl.arange(0, iter_upper)
        for topk_id in range(topk):
            mask_upper = iter_offset < (speculative_num_steps + last_page_len)
            mask_lower = iter_offset >= last_page_len
            combined_mask = mask_upper & mask_lower
            indices = tl.load(
                prefix_base
                + topk_id * num_new_pages_per_topk_ * page_size
                + iter_offset,
                mask=combined_mask,
                other=0,
            )
            # Shift from previous batches
            ptr_offset = pid * speculative_num_steps * topk
            # Subtract last_page_len to fill the gap of duplicated last page tokens.
            # For example, token pool is (1, 2, 3, 4 ,5) and last page is 1,
            # we write 2, 3, 4 to the front of out_cache_loc.
            tl.store(
                out_cache_loc
                + ptr_offset
                + topk_id * speculative_num_steps
                - last_page_len
                + iter_offset,
                indices,
                mask=combined_mask,
            )


@triton.jit
def generate_draft_decode_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    pool_len: tl.constexpr,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    num_steps = tl.num_programs(axis=0)
    num_seqs = tl.num_programs(axis=1)
    topk = tl.num_programs(axis=2)

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(paged_kernel_lens + load_offset, mask=load_offset < bid, other=0)
    seq_len = tl.load(paged_kernel_lens + bid)
    cum_seq_len = tl.sum(seq_lens)

    # Update kv_indices
    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    extend_offset = tl.arange(0, iter_upper)
    if page_size == 1 or topk == 1:
        extend_data = tl.load(
            token_pool_ptr + seq_len + topk_id * num_steps + tl.arange(0, iter_upper),
            mask=extend_offset < iters,
        )
    else:
        prefix_len = seq_len
        last_page_len = prefix_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        prefix_base = seq_len // page_size * page_size
        start = (
            prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
        )
        extend_data = tl.load(
            token_pool_ptr + start + extend_offset,
            mask=extend_offset < iters,
        )

    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr
    bs_offset = tl.arange(0, num_tokens_upper)

    zid = bid * topk + topk_id
    if zid == 0:
        zid = num_seqs * topk
    positions = tl.load(positions + bs_offset, mask=bs_offset < zid, other=0)
    base = tl.sum(positions)
    tl.store(kv_indptr + zid, base + zid * iters)


@triton.jit
def align_evict_mask_to_page_size(
    seq_lens,
    evict_mask,
    page_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    t_range = tl.arange(0, BLOCK_SIZE)

    bid = tl.program_id(axis=0)
    seq_len = tl.load(seq_lens + bid)
    io_mask = t_range < num_draft_tokens
    mask_row = tl.load(
        evict_mask + bid * num_draft_tokens + t_range, mask=io_mask, other=0
    )

    num_trues = tl.sum(mask_row)
    num_false = num_draft_tokens - num_trues

    start = (seq_len + num_false - 1) // page_size * page_size - seq_len
    for i in range(max(start, 0), min(start + page_size, num_draft_tokens)):
        tl.store(evict_mask + bid * num_draft_tokens + i, False)


@triton.jit
def get_target_cache_loc(
    tgt_cache_loc,
    to_free_slots,
    num_correct_drafts,
    to_free_num_slots,
    out_cache_loc,
    num_verify_tokens: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
    bs_upper: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offset = tl.arange(0, num_verify_tokens_upper)
    bs_offset = tl.arange(0, bs_upper)

    # write the first part to tgt_cache_loc
    accept_len_all = tl.load(num_correct_drafts + bs_offset, mask=bs_offset < bid)
    tgt_cache_loc_start = tl.sum(accept_len_all) + bid
    copy_len = tl.load(num_correct_drafts + bid) + 1
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + offset, mask=offset < copy_len
    )
    tl.store(
        tgt_cache_loc + tgt_cache_loc_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )

    # write the second part to to_free_num_pages
    to_free_num_slots_all = tl.load(to_free_num_slots + bs_offset, mask=bs_offset < bid)
    to_free_num_slots_cur = tl.load(to_free_num_slots + bid)
    out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
    to_free_slots_start = tl.sum(to_free_num_slots_all)

    copy_len = to_free_num_slots_cur
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + out_cache_loc_start + offset,
        mask=offset < copy_len,
    )
    tl.store(
        to_free_slots + to_free_slots_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )


@torch.compile(dynamic=True, disable=_is_npu)
def get_src_tgt_cache_loc(
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    draft_token_num: int,
    page_size: int,
):
    src_cache_loc = out_cache_loc[accept_index]
    # zeros_like, not empty_like: any uncovered tail stays at slot 0 (padding)
    # instead of caching-allocator garbage.
    tgt_cache_loc = torch.zeros_like(src_cache_loc)
    extended_len = seq_lens + draft_token_num
    keep_len = torch.minimum(
        (seq_lens + num_correct_drafts + 1 + page_size - 1) // page_size * page_size,
        extended_len,
    )
    to_free_num_slots = extended_len - keep_len
    return src_cache_loc, tgt_cache_loc, to_free_num_slots


@triton.jit
def filter_finished_cache_loc_kernel(
    out_cache_loc,
    tgt_cache_loc,
    num_correct_drafts,
    num_accept_tokens_filter,
    bs_upper: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
):
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_upper)

    num_correct_drafts_all = tl.load(
        num_correct_drafts + bs_offset, mask=bs_offset < bid
    )
    old_start = tl.sum(num_correct_drafts_all) + bid

    num_accept_tokens_filter_all = tl.load(
        num_accept_tokens_filter + bs_offset, mask=bs_offset < bid
    )
    new_start = tl.sum(num_accept_tokens_filter_all)

    copy_len = tl.load(num_accept_tokens_filter + bid)
    copy_offset = tl.arange(0, num_verify_tokens_upper)
    value = tl.load(
        tgt_cache_loc + old_start + copy_offset, mask=copy_offset < copy_len
    )
    tl.store(
        out_cache_loc + new_start + copy_offset, value, mask=copy_offset < copy_len
    )


@torch.compile(dynamic=True, disable=_is_npu)
def create_num_accept_tokens_filter(
    num_correct_drafts: torch.Tensor,
    unfinished_index_device: torch.Tensor,
    seq_lens: torch.Tensor,
):
    num_accept_tokens_filter = torch.zeros_like(num_correct_drafts)
    num_accept_tokens_filter[unfinished_index_device] = (
        num_correct_drafts[unfinished_index_device] + 1
    )
    seq_lens.add_(num_correct_drafts + 1)
    return num_accept_tokens_filter


def _select_top_k_tokens_first(
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    topk: int,
):
    input_ids = topk_index.flatten()
    if hidden_states is not None:
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)

    tree_info = (
        topk_p.unsqueeze(1),  # (b, 1, topk)
        topk_index,  # (b, topk)
        torch.arange(-1, topk, dtype=torch.long, device=input_ids.device).expand(
            topk_p.shape[0], -1
        ),  # (b, topk + 1) — expand avoids the allocation of repeat
    )
    return input_ids, hidden_states, topk_p, tree_info


@torch.compile(dynamic=True, disable=_is_npu)
def _select_top_k_tokens_later(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    topk_sq = topk * topk

    expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)
    # (b, topk, 1) * (b, topk, topk) -> (b, topk, topk)

    topk_cs_p, topk_cs_index = fast_topk(
        expand_scores.flatten(start_dim=1), topk, dim=-1
    )  # (b, topk)

    topk_index = topk_index.view(-1, topk_sq)
    input_ids = torch.gather(topk_index, 1, topk_cs_index).flatten()

    if hidden_states is not None and hidden_states.shape[0] > 0:
        flat_cs = topk_cs_index.flatten()
        batch_offsets = torch.arange(
            0, hidden_states.shape[0], step=topk, device=flat_cs.device
        )
        selected_input_index = flat_cs // topk + batch_offsets.repeat_interleave(topk)
        hidden_states = hidden_states[selected_input_index]

    tree_info = (
        expand_scores,  # (b, topk, topk)
        topk_index,  # (b, topk * topk)
        topk_cs_index + (topk_sq * (i - 1) + topk),  # (b, topk)
    )
    return input_ids, hidden_states, topk_cs_p, tree_info


def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    if i == 0:
        return _select_top_k_tokens_first(topk_p, topk_index, hidden_states, topk)
    return _select_top_k_tokens_later(
        i, topk_p, topk_index, hidden_states, scores, topk
    )


def generate_simulated_accept_index(
    accept_index,
    predict,
    num_correct_drafts,
    bs,
    spec_steps,
    simulate_acc_len: float = SIMULATE_ACC_LEN,
    simulate_acc_method: str = SIMULATE_ACC_METHOD,
):
    assert simulate_acc_len > 0.0

    if simulate_acc_method == "multinomial":
        simulated_values = torch.normal(
            mean=simulate_acc_len,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # clamp simulated values to be between 1 and self.spec_steps
        simulated_values = torch.clamp(simulated_values, min=1.0, max=spec_steps + 1)
        simulate_acc_len = int(simulated_values.round().item())
    elif simulate_acc_method == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        simulate_acc_len = max(1.0, min(spec_steps + 1, simulate_acc_len))
        lower = int(simulate_acc_len // 1)
        upper = lower + 1 if lower < spec_steps + 1 else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            weight_upper = simulate_acc_len - lower
            weight_lower = 1.0 - weight_upper
            probs = torch.tensor([weight_lower, weight_upper], device="cpu")
            sampled_index = torch.multinomial(probs, num_samples=1)
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"Invalid simulate_acc_method: {SIMULATE_ACC_METHOD}")

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device="cuda"
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    num_correct_drafts.fill_(simulate_acc_len - 1)
    predict.fill_(100)  # some legit token id
    return sim_accept_index


def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
):
    """
    Traverse the tree constructed by the draft model to generate the logits mask.
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    )

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            is_accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            curr_token_id = draft_tokens[curr]
            if vocab_size and curr_token_id >= vocab_size:
                is_accepted = False
            else:
                # 32 boolean bitmask values are packed into 32-bit integers
                is_accepted = (
                    parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))
                ) != 0

        if is_accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(int(draft_tokens[curr]))
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        int(retrieve_next_token[curr]),
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            dfs(
                int(retrieve_next_sibling[curr]),
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


def generate_token_bitmask(
    reqs: List[Req],
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
):
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to
    1. figure out which tokens are accepted by the grammar.
    2. if so, what is the corresponding logit mask.
    """

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    allocate_token_bitmask = None
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    grammar = None
    for i, req in enumerate(reqs):
        if req.grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = req.grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            grammar = req.grammar
            s = time.perf_counter()
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                req.grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
                vocab_size=vocab_size,
            )
            tree_traverse_time = time.perf_counter() - s
            if tree_traverse_time > TREE_TRAVERSE_TIME_THRESHOLD:
                logger.warning(
                    f"Bit mask generation took {tree_traverse_time} seconds with "
                    f"grammar: {req.grammar}"
                )

    verify_input.grammar = grammar
    return allocate_token_bitmask


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        repo_id = os.path.dirname(token_map_path)
        file_name = os.path.basename(token_map_path)

        cache_dir = None
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope.utils.file_utils import get_model_cache_root

            cached_repo_path = os.path.join(get_model_cache_root(), repo_id)
            if os.path.exists(cached_repo_path):
                cache_dir = cached_repo_path

        if cache_dir is None:
            if envs.SGLANG_USE_MODELSCOPE.get():
                from modelscope.hub.snapshot_download import (
                    snapshot_download as download_func,
                )
            else:
                download_func = snapshot_download
            cache_dir = download_func(
                repo_id,
                ignore_patterns=["*.bin", "*.safetensors"],
            )

        token_map_path = os.path.join(cache_dir, file_name)
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int64)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with patch_tensor_parallel_group(tp_group):
        yield


# Disable torch.compile for this function because it will be
# even slower.
# @torch.compile(dynamic=True)
def get_last_loc_large_page_size_large_top_k(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    speculative_num_steps: int,
    topk: int,
    page_size: int,
):
    prefix_lens = seq_lens
    last_page_lens = prefix_lens % page_size
    num_new_pages_per_topk = (
        last_page_lens + speculative_num_steps + page_size - 1
    ) // page_size
    seq_lens = prefix_lens // page_size * page_size + num_new_pages_per_topk * (
        page_size * topk
    )
    extend_lens = seq_lens - prefix_lens
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )

    return (
        prefix_lens,
        seq_lens,
        last_loc,
        num_new_pages_per_topk,
        extend_lens,
        last_page_lens,
    )
    accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
    if _is_cpu:
        assign_extend_cache_locs_cpu(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts + 1,
            tgt_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
        )
    else:
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts + 1,
            tgt_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
    fill_accept_out_cache_loc_func(
        accept_index,
        batch.out_cache_loc,
        accept_out_cache_loc,
        size,
    )
    token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
        tgt_cache_loc, accept_out_cache_loc
    )


def prepare_mamba_track_for_verify(batch: ScheduleBatch) -> None:
    """Rebuild mamba track indices from reqs before a TARGET_VERIFY forward.

    Spec batches skip the refresh in prepare_for_decode, and filter/merge
    null these fields, so they must be rebuilt right before verify. Clearing
    the mask also keeps a stale extend-time mask from triggering in-forward
    tracking during TARGET_VERIFY; tracking is done in
    commit_mamba_states_after_verify instead.

    Lazy: gather the positions planned by mamba_lazy_spec_prepare. Runs
    inside forward isolation, so it must not mutate req/pool state.
    """
    if not mamba_extra_buffer_enabled():
        return
    track_positions = None
    if mamba_extra_buffer_lazy_enabled():
        track_positions = batch.mamba_lazy_spec_track_positions_cpu
        assert track_positions is not None and len(track_positions) == len(
            batch.reqs
        ), (
            "lazy spec verify without a track plan: mamba_lazy_spec_prepare "
            "must run in prepare_for_decode for every spec decode iteration"
        )
    set_mamba_track_indices_from_reqs(batch, track_positions)
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None


def _verify_commit_step_indices(
    *,
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    draft_token_num: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Step indices for a post-verify state commit: per req, the tree step of
    the last accepted node (reduces to accept_lens - 1 for topk == 1), and the
    mamba-track interval-crossing step (-1 = no crossing; None when tracking
    is off)."""
    bs = accept_lens.shape[0]
    if accept_index.is_cuda:
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_commit_track_indices,
        )

        track_interval = (
            get_exec().mamba.mamba_track_interval
            if batch.mamba_track_indices is not None
            else 0
        )
        return fused_commit_track_indices(
            accept_index.contiguous(),
            accept_lens,
            batch.seq_lens if track_interval > 0 else None,
            draft_token_num,
            track_interval,
        )
    accept_indices_offset = torch.arange(
        0,
        bs * draft_token_num,
        step=draft_token_num,
        dtype=accept_lens.dtype,
        device=accept_lens.device,
    )
    req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
    last_correct_step_indices = (
        accept_index[req_idx, (accept_lens - 1).to(torch.int64)] - accept_indices_offset
    )
    if batch.mamba_track_indices is None:
        return last_correct_step_indices, None
    seq_lens_pre_verify = batch.seq_lens
    seq_lens_post_verify = batch.seq_lens + accept_lens
    mamba_track_interval = get_exec().mamba.mamba_track_interval
    to_track_mask = (
        seq_lens_pre_verify // mamba_track_interval
        != seq_lens_post_verify // mamba_track_interval
    )
    tracking_point = seq_lens_post_verify // mamba_track_interval * mamba_track_interval
    to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0).to(
        torch.int64
    )
    candidate_track_steps = accept_index[req_idx, to_track_ith] - accept_indices_offset
    mamba_steps_to_track = torch.where(
        to_track_mask,
        candidate_track_steps,
        torch.full_like(candidate_track_steps, -1),
    )
    return last_correct_step_indices, mamba_steps_to_track


def commit_mamba_states_after_verify(
    target_worker: TpModelWorker,
    batch: ScheduleBatch,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
    draft_token_num: int,
) -> None:
    """Commit accepted per-step mamba states into the persistent caches.

    During TARGET_VERIFY, hybrid linear attention backends keep per-step
    states in intermediate caches instead of advancing the persistent
    conv/ssm caches. After acceptance, the state of each request's last
    accepted step is committed back, plus the interval-crossing state used
    for prefix-cache tracking (mamba extra_buffer mode).

    No-op for models without mamba-style state or backends without the
    commit hook.
    """
    model_runner = target_worker.model_runner
    if mambaish_config(model_runner.model_config) is None:
        return

    # ReplaySSM spec-verify path (Part B of #28511): the accepted drafts already
    # live in the per-slot circular ring (written during verify). Instead of
    # scattering an intermediate full SSM state into `temporal`, advance the
    # block-keyed cursors by the accepted count (the ring owns the SSM state; the
    # verify/flush kernel folds it into `temporal` periodically). The CONV state
    # still needs its usual accept-rollback, so we keep the conv-window scatter and
    # skip only the SSM scatter. GDN-only + linear-chain (topk<=1) -- the runtime
    # ring is allocated only then; KDA never allocates the cursors.
    req_pool = model_runner.req_to_token_pool
    mamba_pool = getattr(req_pool, "mamba_pool", None)

    # Fold-every-commit: replay the accepted prefix from the ring into
    # `temporal`; the same fold stores the interval-crossing state to the
    # track slot, so no SSM scatter or force-flush is needed here.
    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_spec_fold", False)
        and not getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
            commit_gdn_replayssm_fold_after_verify,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        last_correct_step_indices, mamba_steps_to_track = _verify_commit_step_indices(
            batch=batch,
            accept_index=accept_index,
            accept_lens=accept_lens,
            draft_token_num=draft_token_num,
        )
        commit_gdn_replayssm_fold_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,
        )
        return

    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_cache_base", None) is not None
        and not getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            commit_gdn_replayssm_spec,
        )
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        bs = accept_lens.shape[0]
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        # Advance the per-slot circular cursors by the accepted count (incl. the
        # bonus token). max_cache_len = ring length L = replayssm_d.shape[-2].
        commit_gdn_replayssm_spec(
            write_pos=mamba_pool.replayssm_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            num_accepted=accept_lens,  # [bs], includes the bonus token
            state_batch_indices=state_batch_indices,
            max_cache_len=spec_state.replayssm_d.shape[-2],
            max_spec_len=draft_token_num,
            null_block_id=-1,  # SGLang: valid slots >= 0, padding == -1
        )
        # Roll back / commit the conv state to the last accepted draft step
        # (same logic as the recurrent commit, but conv-only).
        last_correct_step_indices, _ = _verify_commit_step_indices(
            batch=batch,
            accept_index=accept_index,
            accept_lens=accept_lens,
            draft_token_num=draft_token_num,
        )
        fused_conv_window_scatter_with_mask(
            spec_state.conv[0],
            spec_state.intermediate_conv_window[0],
            state_batch_indices,
            last_correct_step_indices,
        )
        # NOTE: radix mamba prefix-caching (mamba_track / extra_buffer) would need
        # a device-side force-flush so `temporal` reflects the ring before a
        # snapshot; not wired for Part B (server_args forbids extra_buffer with
        # --enable-linear-replayssm-spec), so the per-track scatters are intentionally
        # skipped here.
        return

    # KDA ReplaySSM (fold-every-commit): KDA keeps its own recurrent verify kernel
    # for the OUTPUT, so we replay the accepted window into the fp32 checkpoint
    # (`temporal`) here on commit -- `temporal` is always the current committed
    # state. The draft window's raw inputs were written to the ring during verify
    # by the KDA backend. Gate on the fold flag + is_kda (the cursor tensors are
    # never allocated under fold, so they cannot serve as the signal).
    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_spec_fold", False)
        and getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
            commit_kda_replayssm_after_verify,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        bs = accept_lens.shape[0]
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        accept_indices_offset = torch.arange(
            0,
            bs * draft_token_num,
            step=draft_token_num,
            dtype=accept_lens.dtype,
            device=accept_lens.device,
        )
        req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
        last_correct_step_indices = (
            accept_index[req_idx, (accept_lens - 1).to(torch.int64)]
            - accept_indices_offset
        )
        # extra_buffer: the interval-crossing step whose state must snapshot into
        # the track ping-pong slot (mirrors the regular commit's
        # mamba_steps_to_track); commit_kda_replayssm_spec folds it in one pass, so
        # `temporal` stays current and no device-side force-flush is needed.
        mamba_track_indices = batch.mamba_track_indices
        mamba_steps_to_track = None
        if mamba_track_indices is not None:
            ti = get_exec().mamba.mamba_track_interval
            seq_pre = batch.seq_lens
            seq_post = batch.seq_lens + accept_lens
            to_track_mask = seq_pre // ti != seq_post // ti
            tracking_point = seq_post // ti * ti
            to_track_ith = torch.clamp(tracking_point - seq_pre - 1, min=0).to(
                torch.int64
            )
            candidate = accept_index[req_idx, to_track_ith] - accept_indices_offset
            mamba_steps_to_track = torch.where(
                to_track_mask, candidate, torch.full_like(candidate, -1)
            )
        commit_kda_replayssm_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,  # incl. bonus token
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,  # SGLang: valid slots >= 0, padding == -1
        )
        return

    attn_backend = model_runner.attn_backend

    bs = accept_lens.shape[0]
    # `accept_lens` already includes the bonus token (drafts + 1 per req).
    if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
        last_correct_step_indices, mamba_steps_to_track = _verify_commit_step_indices(
            batch=batch,
            accept_index=accept_index,
            accept_lens=accept_lens,
            draft_token_num=draft_token_num,
        )

        if hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            attn_backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=model_runner.model,
                req_pool_indices=batch.req_pool_indices[:bs],
            )


def spec_prepare_for_decode(batch: ScheduleBatch) -> None:
    """eagle/ngram share a stateless free function; dflash keeps stateful
    prep on its draft input -- the dispatcher routes.
    """
    if mamba_extra_buffer_lazy_enabled():
        # Scheduler phase (outside forward isolation).
        batch.mamba_lazy_spec_prepare(
            get_exec().mamba.mamba_track_interval,
            max_speculative_num_draft_tokens(),
        )
    if batch.spec_algorithm.is_dflash_family():
        batch.spec_info.prepare_for_decode(batch)
    else:
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_decode

        eagle_prepare_for_decode(batch)


def get_plan_stream(
    device: str,
) -> Tuple[Any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()
