# V100 PR integration, 2026-11-02

Integrates [#1](https://github.com/haohervchb/sglang-V100/pull/1) by
Prethea-Phoenixia and [#2](https://github.com/haohervchb/sglang-V100/pull/2) by
Horacio Vico, preserving both original commits and authors.

Warmup uses the resolved request-pool capacity. The PCIe opt-in keeps the true
NVLink topology, only changes v2 eligibility, enforces push-only dispatch even
with a conflicting explicit override, validates its byte cap, and falls back
to NCCL for larger messages. Trusting driver P2P reports now checks every peer.
The opt-in does not change behavior on an NVLink host or in the legacy backend.

Validation on four V100-SXM2 32 GB GPUs with NV2 links, CUDA 12.8, PyTorch
2.9.1+cu128 and patched Triton 3.5.1:

- **39 focused CPU tests passed**, with CUDA hidden. The distributed test matrix
  covers topology, opt-in/out, unknown topology, legacy eligibility, explicit
  algorithm overrides, caps and all-peer checking. Warmup tests cover resolved,
  unset and clamped request capacities.
- **12 GPU dtype/size/graph cases passed**, with 16 repetitions and four
  reductions per case: FP16/FP32, 4/16/128 KiB, eager/CUDA graph. The test forces
  logical PCIe eligibility on NVLink hardware, requests a pull override, checks
  the push result against NCCL, and checks oversized-message fallback.
- These GPU checks validate dispatch and arithmetic. Actual PCIe transport and
  the speed results in #2 remain validated by the contributor's PCIe machine.
- Black/isort and focused Ruff checks passed; no kernel math changed.

Reproduce from the repository root in the serving environment:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=python python -m pytest -q \
  test/registered/unit/distributed/test_custom_all_reduce_pcie.py \
  test/registered/unit/model_executor/test_sm70_warmup_capacity.py
PYTHONPATH=python torchrun --standalone --nproc-per-node=4 \
  benchmark/qwen38_spec/check_allreduce.py
```

Raw outputs: [CPU tests](pr_cpu_hidden.log.gz), [GPU checks](allreduce_gpu.log.gz).
The integration files are identical to the files used for those tests.
