#!/usr/bin/env bash
# Validate an already-built SGLang-V100 environment and warm lazy sampling JIT.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  printf '[smoke_v100] Do not source this file; run: bash %q\n' \
    "${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail

if [[ -n "${SGLANG_V100_PYTHON:-}" ]] && [[ -x "$SGLANG_V100_PYTHON" ]]; then
  PYTHON="$SGLANG_V100_PYTHON"
elif [[ -x "$HOME/miniconda3/envs/sglang-v100/bin/python" ]]; then
  PYTHON="$HOME/miniconda3/envs/sglang-v100/bin/python"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "sglang-v100" ]] && \
  [[ -x "${CONDA_PREFIX:-}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  printf '[smoke_v100] sglang-v100 Python was not found. Activate the environment first.\n' >&2
  exit 1
fi

FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PYTHON" - <<'PY'
import glob
import os
from pathlib import Path

import torch
import flashinfer.sampling as flashinfer_sampling
import flash_attn_v100_cuda
import sgl_kernel
from flashinfer.sampling import top_k_top_p_sampling_from_probs
from sglang.srt.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

expected = os.environ.get("SGLANG_V100_FLASHINFER_DIR")
sampling_path = Path(flashinfer_sampling.__file__).resolve()
if expected:
    expected_path = Path(expected).resolve()
    assert sampling_path.is_relative_to(expected_path), (
        f"wrong FlashInfer source: {sampling_path}; expected {expected_path}"
    )

assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 0)
assert "/sm70/" in sgl_kernel.common_ops.__file__.replace("\\", "/")
assert NCCLLibrary().ncclGetRawVersion() == 22705

loaded_common_ops = Path(sgl_kernel.common_ops.__file__).resolve()
assert loaded_common_ops.name == "common_ops.abi3.so", loaded_common_ops
common_ops = [
    Path(path).resolve()
    for path in glob.glob(
        str(Path(sgl_kernel.__file__).parent / "sm70" / "common_ops*.so")
    )
]
for stale_artifact in common_ops:
    if stale_artifact != loaded_common_ops:
        stale_artifact.unlink()
        print("Removed stale kernel artifact:", stale_artifact)
common_ops = list(loaded_common_ops.parent.glob("common_ops*.so"))
assert common_ops == [loaded_common_ops], f"stale common_ops variants remain: {common_ops}"

# FlashInfer builds this module lazily. Doing it here prevents the first
# non-greedy chat request from paying the cold compilation cost.
probs = torch.full((1, 128), 1.0 / 128, device="cuda", dtype=torch.float32)
top_k = torch.tensor([20], device="cuda", dtype=torch.int32)
top_p = torch.tensor([0.8], device="cuda", dtype=torch.float32)
top_k_top_p_sampling_from_probs(
    probs, top_k, top_p, filter_apply_order="joint"
)
torch.cuda.synchronize()

print("SGLang V100 environment is ready:", torch.__version__)
print("FlashInfer SM70 sampling:", sampling_path)
print("Native attention:", flash_attn_v100_cuda.__file__)
print("SM70 kernel:", sgl_kernel.common_ops.__file__)
print("NCCL:", NCCLLibrary().ncclGetVersion())
PY
