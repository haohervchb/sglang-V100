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
