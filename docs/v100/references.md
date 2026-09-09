# References and attribution

[Back to the main README](../../README.md)

The following upstream projects were used as code dependencies, algorithmic
references, performance references, or model assets for the V100 work in this
repository. The relationship column states which kind of use applies.

| Work in this repository | Upstream project | Relationship | License / revision |
| --- | --- | --- | --- |
| SGLang serving runtime and model integration | [sgl-project/sglang](https://github.com/sgl-project/sglang) | Framework this V100 fork is based on. | Apache-2.0 |
| TileLang attention, FP8-KV bridge, GDN, and fused normalization kernels | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | Kernel language, compiler, and runtime; host and Docker currently install `tilelang==0.1.8`. | MIT / `0.1.8` |
| SM70 long-context attention and FP8 optimization campaign | [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM/tree/6ada86ed64af6d1a7b3cb0f34df237fd86f06d48) | Performance and design reference for D=256 paged attention, dense-KV gathering, K-axis splitting, FP8 E5M2 KV conversion, and long-context decode. | Apache-2.0 / `6ada86e` |
| Chunked GDN / gated-delta-rule algorithm | [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA/tree/v0.1.2) | Algorithm and API reference for chunk-64 KKT solving, gating, recurrent-state propagation, and variable-length GDN prefill. | MIT / `v0.1.2` |
| GDN utility and correctness-reference operators | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | Source of the adapted FLA utilities already carried under `python/sglang/srt/layers/attention/fla`; also used as the numerical reference for the SM70 GDN path. | MIT |
| TurboMind GEMM and MoE kernel lineage | [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) | Original TurboMind project and kernel architecture. | Apache-2.0 |
| SM70 TurboMind FP8, AWQ, and FP16-MoE build source | [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM/tree/6ada86ed64af6d1a7b3cb0f34df237fd86f06d48/csrc/sm70_turbomind) | Pinned sparse source snapshot used to build the current SGLang TurboMind adapter; its embedded TurboMind sources derive from LMDeploy. | Apache-2.0 / `6ada86e` |
| SM70 Marlin GPTQ/AWQ dense and MoE kernels | [zhinianqin/marlin_v100](https://github.com/zhinianqin/marlin_v100/tree/6d72a49939701d26b15b617a4cd2423174adb2d1) | Native extension built by `scripts/setup_v100_marlin.sh`, with the compatibility and Qwen tuning patches in this repository. | Apache-2.0 / `6d72a49` |
| QPN8 SM70 W8A16 decode kernel | [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) | Kernel architecture adapted for Qwen3.8 block-wise scales; this repository adds its own word-parallel FP8 decoder and paired gate/up SiLU epilogue. | MIT |
| FlashInfer sampling and remaining SM70-compatible runtime operations | [haohervchb/flashinfer](https://github.com/haohervchb/flashinfer/tree/c3c40a7b90b792fc59f90f8f55c9e2de9c1b6833), derived from [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Pinned source dependency with this repository's reduced SM70 compatibility patch. | Apache-2.0 / `c3c40a7` |
| Tensor-core templates used by TurboMind and Marlin builds | [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) | Header/template build dependency. TurboMind uses `da5e086`; Marlin uses CUTLASS `v4.2.1`. | BSD-3-Clause |
| Qwen3.8 DFlash2 speculative decoding | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | Draft checkpoint, published block configuration, and model contract used by the DFlash2 integration and benchmarks. | Apache-2.0 / model revision `ac04198` |
| Qwen3.8 DSpark speculative decoding | [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) | Draft checkpoint and model configuration used by the DSpark serving path and benchmarks. | See model card |
