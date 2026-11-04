# Qwen3.8 V100 user service

`sglang-qwen38-8082.service` serves the validated FP16 KV / native MTP-3/4
profile at `http://127.0.0.1:8082/v1`, with model name `qwen`, TP4, multimodal
support, a 262,144-token context limit and one active request. The source
checkout and compiled SM70 libraries must already be installed. The unit uses
`~/sglang-V100` and the `sglang-v100` Conda environment; adjust those paths for
a different installation.

The serving profile processes prompts in 2,048-token chunks to leave room for
temporary PLE activations during long document sessions and queued requests.
The 8,192-token benchmark profile exhausted GPU memory during sustained use;
its prefill timing results do not describe this more conservative configuration.
Chunk size does not reduce the context limit. Documents still need to fit the
token limit after extraction; PDF file size is not a token-count limit.

For concurrent office agents, the v5.5 profile uses
`scripts/serve_qwen38_flash_next_nvfp4_v100_throughput.sh` as `ExecStart` in a
user-service override. It admits eight active requests, preserves shared-prefix
checkpoints and 262K context, and enables the native SM70 vision backend.
Install its optional [Flash-V100 vision dependency](../../python/sglang/jit_kernel/third_party/flash_v100/README.md#optional-vision-attention)
before selecting this launcher. Pixel preprocessing remains on CPU to avoid
the frontend competing with the model for GPU 0 memory. The vision encoder
and language model run on GPU. The v5.4 vision change reduced the matched
32-page request from 80.0 s to 12.1 s. The v5.5 profile raises eight-client text
throughput from 253 to 360 output tokens/s, with native verification up to
32 rows and 48 Mamba cache slots. See the [core audit and validation](../../benchmark/qwen38_concurrent_core_audit_20261104/README.md).

The optimized profile enables the shared-expert gate for MTP verification,
the measured 2K MoE routing/geometry settings, and grouped Page4 QSA prefill.
Set `SGLANG_SM70_MTP_SHARED_GATE`, `SGLANG_SM70_MOE_PREFILL_2K`, or
`SGLANG_SM70_QSA_GROUPED_PREFILL` to `0` to disable an individual change.

Grouped QSA uses an optional Flash-V100 extension built against this serving
environment's PyTorch ABI. Build from the `flash-attention-v100` directory
bundled in 1Cat-vLLM (validated source revision
`d8f42b391af06410ab01bb71657ededf7dce7799`):

```bash
python scripts/setup_v100_grouped_qsa.py \
  --source /path/to/1Cat-vLLM/flash-attention-v100
```

This installs `python/sglang/jit_kernel/_sm70_grouped_qsa_v100.so` without
replacing the existing FlashAttention package. The build directory retains
the source snapshot, license and hashes. The new route is limited to the
measured 2,048-row FP16/Page16 Qwen shape; other shapes retain native QSA.
If the extension is unavailable, a warning is logged and native QSA remains
active. Restart after installing the library. See the bundled
[third-party notice](../../python/sglang/jit_kernel/third_party/flash_v100/README.md).

```bash
mkdir -p ~/.config/systemd/user
cp scripts/systemd/sglang-qwen38-8082.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sglang-qwen38-8082.service
```

Model loading and graph warmup take several minutes. Readiness is indicated by
an HTTP 200 from `/health`, not just systemd's active state.

```bash
curl --fail http://127.0.0.1:8082/health
curl --fail http://127.0.0.1:8082/v1/models
journalctl --user -u sglang-qwen38-8082.service -f
systemctl --user restart sglang-qwen38-8082.service
systemctl --user stop sglang-qwen38-8082.service
```

The service restarts on failure. User lingering is required for startup before
login; inspect it with `loginctl show-user "$USER" -p Linger`. Native CUDA build
caches live under `~/.cache/sglang-v100/qwen38` so normal reboots preserve them.
