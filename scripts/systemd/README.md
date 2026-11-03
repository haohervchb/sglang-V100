# Qwen3.8 V100 user service

`sglang-qwen38-8082.service` serves the validated FP16 KV / native MTP-3/4
profile at `http://127.0.0.1:8082/v1`, with model name `qwen`, TP4, multimodal
support, a 262,144-token context limit and one active request. The source
checkout and compiled SM70 libraries must already be installed. The unit uses
`~/sglang-V100` and the `sglang-v100` Conda environment; adjust those paths for
a different installation.

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
