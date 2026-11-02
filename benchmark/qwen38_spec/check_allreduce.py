"""torchrun --standalone --nproc-per-node=4 this_file.py

Exercise the PCIe dispatch policy on NVLink hardware. This validates the push
kernel and dispatch guards, NOT actual PCIe transport or performance.
"""

import os

import torch
import torch.distributed as dist

from sglang.jit_kernel.all_reduce import AllReduceAlgo
from sglang.jit_kernel.tests.test_custom_all_reduce import init_distributed, worker_test
from sglang.srt.distributed.device_communicators import custom_all_reduce_utils


def main():
    os.environ["SGLANG_CUSTOM_AR_ALLOW_PCIE"] = "1"
    os.environ["SGLANG_CUSTOM_AR_PCIE_MAX_BYTES"] = "131072"
    custom_all_reduce_utils.is_full_nvlink = lambda *args: False
    rank, device, cpu_group, nccl_group, comm = init_distributed()
    assert comm.pcie_only and comm.max_size == 131072
    torch.cuda.set_stream(torch.cuda.Stream())
    count = 0
    for dtype in (torch.float16, torch.float32):
        for byte_count in (4096, 16384, 131072):
            size = byte_count // torch.empty((), dtype=dtype).element_size()
            for graph in (False, True):
                # Even a pull override must resolve to push on this policy.
                error = worker_test(
                    device,
                    nccl_group,
                    comm,
                    size,
                    dtype,
                    graph,
                    AllReduceAlgo.TWO_SHOT_PULL,
                )
                if error:
                    raise error
                count += 1
    assert not comm.should_custom_ar(
        torch.empty(65544, device=device, dtype=torch.float16)
    )
    dist.barrier()
    if rank == 0:
        print(
            f"Passed {count} dtype/size/graph cases, 16 repetitions and 4 reductions each. Logical PCIe policy over physical NVLink.",
            flush=True,
        )
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
