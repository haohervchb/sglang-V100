# Qwen3.8 Flash Next NVFP4 on 8x V100: 2xTP4 OpenCode and multimodal field report

This report records a real OpenCode and Windows RPA workload using the published
Qwen3.8 Flash Next V100 Docker image. It is an operational field report, not a
controlled model-quality benchmark.

## Deployment

- Host: 8x Tesla V100-SXM2-32GB.
- Checkpoint: `RadixArk/Qwen3.8-Flash-Next-NVFP4`.
- Image: `geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v4`.
- Image ID: `sha256:4373c93d636ff83fba0e06eb0c88c10edee07621d7ba505eef8f51d9c84601af`.
- Context length: 262,144 tokens.
- Topology: two independent TP4 replicas, pinned to GPUs 0-3 and 4-7.
- Each replica has an independent OpenAI-compatible endpoint.
- Multimodal input is enabled.

The image ID and checkpoint match the repository's
[multimodal validation](../qwen38_nvfp4_v100_multimodal_20260909/README.md).

## Dual-replica smoke result

An eight-request smoke test sent four requests to each TP4 replica:

- all eight requests returned HTTP 200;
- each request produced 128 completion tokens;
- observed TTFT was approximately 70-73 seconds;
- observed wall time was approximately 73-75 seconds;
- GPU memory remained stable at approximately 26.5-27.0 GiB per GPU;
- health checks remained HTTP 200 after the run;
- no cross-placement between the two four-GPU groups was observed.

These values describe this bounded smoke test. They are not directly comparable
with the repository's single-replica benchmarks because the workload and output
length differ.

## Long OpenCode workload

The model was used in a long OpenCode session on a legacy Delphi/VCL medical
application. The work included source inspection, SQL fixture construction,
Windows UI automation, and visual validation. No patient data is included in
this report.

The exported Qwen portion of the session contained:

- 307 assistant turns;
- 317 tool calls;
- a maximum observed request input of approximately 228,600 tokens;
- continued operation near that context size without a Qwen-side compaction.

The model preserved task state well and completed multi-stage work across source
inspection, controlled database fixtures, `pywinauto`, and RPA validation. It
also recovered from several command and fixture errors. Human review and
explicit safety gates remained important: PowerShell, VCL capture, and some
fixture operations required iterative correction.

These session counts are observations from an OpenCode export, not a standardized
agent benchmark or a claim of one-shot correctness.

## Multimodal canary

Vision was verified through the deployed OpenAI-compatible endpoint rather than
inferred from model metadata.

A single non-thinking image request produced:

- HTTP 200 in 1.5 seconds;
- 321 prompt tokens and 40 completion tokens;
- zero reasoning tokens;
- `finish_reason: stop`;
- correct extraction of all four requested values: title, invoice total, button
  label, and shape color.

The same endpoint was then used through OpenCode in a privacy-controlled RPA
workflow. OpenCode required the custom provider model to declare image input:

```json
"modalities": {
  "input": ["text", "image"],
  "output": ["text"]
}
```

Without this client-side metadata, OpenCode did not expose image attachment even
though the SGLang deployment was already multimodal.

The RPA bridge used `pywinauto` and `PrintWindow` captures. It completed 16
controlled UI-state observations across two application hosts. Evidence images
were restricted to test UI elements and contained no patient data.

## Reboot recovery

After a host reboot, both pre-existing containers had exited with code 255 at
the reboot boundary. All eight GPUs were free, and there were no current CUDA,
QSA, or NCCL errors.

The unchanged containers were started sequentially:

- TP4 replica A reached HTTP 200 readiness in 392 seconds;
- TP4 replica B reached HTTP 200 readiness in 360 seconds;
- both returned the expected model ID and `max_model_len: 262144`;
- GPU placement remained 0-3 for A and 4-7 for B.

No container recreation, image pull, runtime-argument change, or inference was
performed during this recovery check.

## TP8 follow-up

The two-TP4 topology is currently useful because it provides two isolated
replicas and eight-request capacity. A controlled TP8 experiment could still be
valuable for single-session long-context and multimodal workloads.

Guidance from the maintainer would be useful on:

1. whether TP8 is currently supported and expected to be stable for this
   checkpoint on eight NVLink-connected V100s;
2. the expected latency, decode-throughput, and concurrency trade-offs versus
   two TP4 replicas;
3. recommended TP8 values for memory fraction, total token pool, running
   requests, Mamba cache, CUDA graph sizes, chunked prefill, NCCL, and custom
   all-reduce;
4. a preferred benchmark protocol for producing results useful to the project.

The working TP4 containers can be preserved while a separate TP8 candidate is
tested.

## Disclosure

Drafting and analysis of the sanitized session and runtime reports were
AI-assisted. All commands, deployment states, and observations reported here
were reviewed by the human operator. No proprietary source code, credentials,
patient data, or private endpoint addresses are included.
