# Flash-Next standalone launchers

The v5.1 packaging update includes `/usr/local/bin/qwen38.sh` in the image.
It can be copied out with `docker run --rm --entrypoint cat IMAGE
/usr/local/bin/qwen38.sh` and run from an otherwise empty host directory.
No source checkout, host Conda environment or host CUDA compiler is required
for Docker serving. The image also exposes `serve-qwen38` as its foreground
server command. A prepared source checkout uses `bash scripts/serve_qwen38.sh`.

Startup and restarts stay manual. The host script selects the prepared serving
Python or respects `SGLANG_PYTHON`. Neither launcher installs or enables a
systemd service.

Validation covered:

- Extracting the launcher from the built image into an empty directory and
  matching it to the checked-in launcher.
- Running the extracted Docker launcher with `start --help`, then `status`
  and `stop`. The help request exits before model initialization.
- Correct host networking, shared IPC, port override and a cache path containing
  spaces, with restart policy `no`.
- Foreground host environment discovery and argument forwarding, using a
  capture fixture instead of loading the model.
- Exact equality of the new foreground command's final arguments and relevant
  environment with the validated v5 profile.
- Exact content hashes for all 3,101 model-runtime files in v5 and v5.1,
  excluding regenerated Python bytecode. No inference kernels or weights changed.
- Byte-for-byte preservation of the other model instructions in README.

This is launcher and packaging validation, not a new inference benchmark. The
model was not started for these checks. The previous [v5 GPU and full-model
validation](../qwen38_docker_v5_20261105/README.md) describes the unchanged
runtime. Raw checks and file hashes are retained in `raw/` and `metadata.json`.
