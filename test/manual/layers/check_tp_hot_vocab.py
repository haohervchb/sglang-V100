"""Run with torchrun --standalone --nproc_per_node=4 for real TP collectives."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from sglang.srt.speculative.tp_hot_vocab import build_sharded_hot_head


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    size = dist.get_world_size()
    torch.manual_seed(1729)
    full = torch.randn(248320, 32, device="cuda", dtype=torch.float16)
    hot = torch.randperm(248320, device="cuda")[:65536]
    width = full.shape[0] // size
    original = full[rank * width : (rank + 1) * width].clone()
    layer = SimpleNamespace(
        org_vocab_size=full.shape[0],
        num_embeddings=full.shape[0],
        tp_size=size,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=rank * width, org_vocab_end_index=(rank + 1) * width
        ),
    )

    def reduce(tensor):
        dist.all_reduce(tensor)
        return tensor

    group = SimpleNamespace(world_size=size, rank_in_group=rank, all_reduce=reduce)
    selected = build_sharded_hot_head(original, hot, layer, group)
    expected = full[hot[rank * (65536 // size) : (rank + 1) * (65536 // size)]]
    assert torch.equal(selected, expected)
    assert torch.equal(original, full[rank * width : (rank + 1) * width])
    q = torch.randn(4, 32, device="cuda", dtype=torch.float16)
    local_logits = q @ selected.T
    gathered = [torch.empty_like(local_logits) for _ in range(size)]
    dist.all_gather(gathered, local_logits)
    logits = torch.cat(gathered, dim=1)
    torch.testing.assert_close(logits, q @ full[hot].T, atol=0.02, rtol=0.002)
    assert torch.equal(hot[logits.argmax(-1)], hot[(q @ full[hot].T).argmax(-1)])
    dist.barrier()
    if rank == 0:
        print(
            "TP4 hot-head rows, gathered logits and token remapping passed", flush=True
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
