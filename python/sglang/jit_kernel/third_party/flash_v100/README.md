# Optional grouped Page4 QSA dependency

`scripts/setup_v100_grouped_qsa.py` builds only the grouped planner and attention
bindings from the Flash-V100 sources bundled in
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM/tree/d8f42b391af06410ab01bb71657ededf7dce7799/flash-attention-v100).
The installed binary is a local build artifact and is not committed.

The underlying Flash-V100 project is by D. Skryabin / ai-bond. The grouped Page4
work was contributed by yangzhuxinyzx in
[1Cat-vLLM #387](https://github.com/1CatAI/1Cat-vLLM/pull/387).
The BSD-3-Clause and upstream Apache-2.0 license texts are retained here.
SGLang's adapter uses the library through its planner/attention API and keeps
the existing native QSA path available.

The shared library must use the same PyTorch ABI as the serving environment.
Source and output hashes are written to the build directory's `metadata.json`.

The build applies `grouped_masked_kv.patch` to the copied sources. The sparse
planner pads categories with page-zero entries whose query masks are zero.
Unused cache entries can contain NaNs after graph capture; multiplying them by
zero attention probabilities still propagates NaNs. The patch guards K/V loads
with the union of the eight query masks and writes zeros for unused tokens in
shared memory. It also synchronizes mask publication before those loads.

The patched binding advertises `sglang_masked_sparse_kv`. The adapter requires
this capability and falls back to native QSA for older binaries. Rebuild with
the setup script after updating; the script records the patch hash and replaces
the installed library atomically. Poisoned padding, partial pages, and changing
request mappings are covered by `test/manual/layers/test_sm70_grouped_qsa.py`.

## Optional vision attention

`--mm-attention-backend fa2_v100` uses the separate `flash_attn_v100` Python
package from [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100).
The validated host build uses revision `d89800edf608d85744f3ab6188be5fd0736acf39`
with `vision_host_compat.patch`, which records the existing host's CUDA 12.8,
packaging, TorchBind optional-tensor and paged-layout compatibility changes.
These are separate from the grouped QSA patch above. The vision adapter uses
non-paged, bidirectional variable-length attention.

Qwen3.8 has 72-wide vision heads. The adapter zero-pads them to the supported
128-wide kernel, retains the original attention scale, and crops the output.
It requires SM70, FP16 and eager vision execution. Language CUDA graphs remain
enabled. Numerical tests cover ragged images, custom scaling, noncontiguous
inputs and bidirectional attention without mixing images.

Build in the serving environment, with its existing PyTorch installation:

```bash
conda activate sglang-v100
q38_repo="$PWD"  # Run from the SGLang-V100 checkout.
git clone https://github.com/ai-bond/flash-attention-v100.git flash-attention-v100-vision
git -C flash-attention-v100-vision checkout --detach d89800edf608d85744f3ab6188be5fd0736acf39
git -C flash-attention-v100-vision apply \
  "$q38_repo/python/sglang/jit_kernel/third_party/flash_v100/vision_host_compat.patch"
MAX_JOBS=2 NVCC_THREADS=2 python -m pip install --no-deps --no-build-isolation \
  ./flash-attention-v100-vision
PYTHONPATH=python python -m pytest -q test/manual/layers/test_sm70_vision_attention.py
```

The performance evidence records the measured binary and interface hashes.
Docker v5 builds this dependency from the same pinned source and compatibility
patch. `vision_container_build.patch` permits compilation with CUDA-enabled
PyTorch and the CUDA toolkit when no GPU is visible to the image builder;
it does not change kernel code. GPU execution is checked separately after the
image is built. The grouped-QSA source snapshot and metadata are retained under
`/opt/deps/qwen38-grouped-qsa-build`, and the vision checkout under
`/opt/deps/qwen38-vision`, including their licenses and applied patches.
The in-tree Triton vision path remains available with
`--mm-attention-backend triton_attn`.
