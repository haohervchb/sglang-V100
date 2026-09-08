"""Run the existing host benchmark protocol against a host or Docker server."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
PREVIOUS = REPO / "benchmark/qwen38_nvfp4_v100_mtp_20260908"
MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
URL = "http://127.0.0.1:8082"


def collect_environment(root, runtime, container_name):
    files = subprocess.check_output(
        ["git", "ls-files", "python/sglang"], cwd=REPO, text=True
    ).splitlines()
    expected = {}
    for name in files:
        path = REPO / name
        data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        expected[name] = hashlib.sha256(data).hexdigest()
    packages = [
        "torch",
        "triton",
        "tilelang",
        "flashinfer-python",
        "sglang-kernel",
        "apache-tvm-ffi",
        "transformers",
        "nvidia-nccl-cu12",
    ]
    if runtime == "docker":
        script = """
import hashlib, importlib.metadata, json, os, sys
from pathlib import Path
expected, packages = json.load(sys.stdin)
actual = {}
for name in expected:
    path = Path('/opt/sglang') / name
    data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
    actual[name] = hashlib.sha256(data).hexdigest()
print(json.dumps(dict(
    versions={k: importlib.metadata.version(k) for k in packages},
    native_binaries={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path('/opt/sglang/python/sglang/jit_kernel').glob('*.so')},
    files_checked=len(expected),
    mismatches=[k for k in actual if actual[k] != expected[k]],
    source_tree_sha256=hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(),
)))
"""
        result = subprocess.run(
            ["docker", "exec", "-i", container_name, "python", "-c", script],
            input=json.dumps([expected, packages]),
            text=True,
            capture_output=True,
            check=True,
        )
        info = json.loads(result.stdout)
        assert not info["mismatches"], info["mismatches"]
        inspected = json.loads(
            subprocess.check_output(["docker", "inspect", container_name], text=True)
        )[0]
        assert not any(m["Destination"].startswith("/opt") for m in inspected["Mounts"])
        (root / f"{container_name}_mounts.json").write_text(
            json.dumps(inspected["Mounts"]) + "\n"
        )
    else:
        info = dict(
            versions={k: importlib.metadata.version(k) for k in packages},
            native_binaries={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (REPO / "python/sglang/jit_kernel").glob("*.so")
            },
            source_tree_sha256=hashlib.sha256(
                json.dumps(expected, sort_keys=True).encode()
            ).hexdigest(),
            python=sys.version,
        )
    destination = root / f"{runtime}_environment.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        assert previous["source_tree_sha256"] == info["source_tree_sha256"]
    else:
        destination.write_text(json.dumps(info, indent=2) + "\n")


def response_checks(root, tag, mode):
    reference = PREVIOUS / (
        "mtp34_vector_correctness.json"
        if mode == "mtp"
        else "target_vector_regression_correctness.json"
    )
    rows = []
    for ref in json.loads(reference.read_text()):
        # An EOS response can reach the client before the scheduler retires
        # its last overlap batch. Let the server wait for an idle cache.
        requests.post(
            URL + "/flush_cache", params={"timeout": 30}, timeout=60
        ).raise_for_status()
        response = requests.post(URL + "/generate", json=ref["request"], timeout=300)
        response.raise_for_status()
        actual = response.json()
        assert actual["meta_info"]["completion_tokens"] > 10, actual
        a = actual["meta_info"]["output_token_logprobs"]
        b = ref["response"]["meta_info"]["output_token_logprobs"]
        identical = [x[1] for x in a] == [x[1] for x in b]
        rows.append(
            dict(
                prompt=ref["prompt"],
                request=ref["request"],
                response=actual,
                identical_ids=identical,
                identical_text=actual["text"] == ref["response"]["text"],
                max_logprob_error=(
                    max(abs(x[0] - y[0]) for x, y in zip(a, b, strict=True))
                    if identical
                    else None
                ),
            )
        )
        print("response", tag, ref["prompt"], "identical IDs", identical, flush=True)
    (root / f"{tag}_responses.json").write_text(json.dumps(rows) + "\n")
    return all(row["identical_ids"] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["host", "docker"], required=True)
    parser.add_argument("--mode", choices=["mtp", "target"], required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", default="sglang-v100:v100-qwen38-flash-next-v4")
    parser.add_argument("--container-name")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / f"{args.tag}_launch.json").exists():
        raise RuntimeError("Refusing to overwrite an existing run")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8082)) == 0:
            raise RuntimeError("Port 8082 is already occupied")

    spec = []
    if args.mode == "mtp":
        spec = [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            MODEL,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ]
    env = os.environ.copy()
    settings = dict(
        PYTHONPATH=str(REPO / "python"),
        HF_HUB_OFFLINE="1",
        PORT="8082",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        CUDA_HOME="/usr/local/cuda-12.8",
        CXX="/usr/bin/g++-12",
        TORCH_CUDA_ARCH_LIST="7.0",
        MAX_JOBS="2",
        SGLANG_SM70_DENSE_GEMV="1",
        SGLANG_SM70_QWEN_FUSIONS="1",
    )
    env.update(settings)
    env["PATH"] = str(Path(sys.executable).parent) + ":" + env["PATH"]
    name = args.container_name or f"qwen38-v4-{args.tag}"
    process = None
    server_log = None
    started = False
    try:
        if args.runtime == "docker":
            command = [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--gpus",
                "all",
                "--network",
                "host",
                "--ipc",
                "host",
                "--ulimit",
                "memlock=-1",
                "--ulimit",
                "stack=67108864",
                "-v",
                str(Path.home() / ".cache/huggingface")
                + ":/root/.cache/huggingface:ro",
                "-v",
                "sglang-v100-jit-v4:/root/sglang-v100-jit",
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "PORT=8082",
                "-e",
                "CUDA_VISIBLE_DEVICES=0,1,2,3",
                "-e",
                "TVM_FFI_CACHE_DIR=/root/sglang-v100-jit/tvm-ffi",
                "-e",
                "TORCH_EXTENSIONS_DIR=/root/sglang-v100-jit/torch_extensions",
                "-e",
                "SGLANG_V100_NVFP4_MOE_BUILD_DIR=/root/sglang-v100-jit/nvfp4_moe",
                "-e",
                "SGLANG_V100_DECODE_CUDA_BUILD_DIR=/root/sglang-v100-jit/longctx_decode",
                args.image,
                "bash",
                "/opt/sglang/scripts/serve_qwen38_flash_next_nvfp4_v100.sh",
                MODEL,
                *spec,
            ]
            container_id = subprocess.check_output(command, text=True).strip()
            started = True
            launch = dict(command=command, container_id=container_id, image=args.image)
            launch["image_id"] = subprocess.check_output(
                ["docker", "inspect", "--format", "{{.Image}}", name], text=True
            ).strip()
        else:
            command = [
                "bash",
                "scripts/serve_qwen38_flash_next_nvfp4_v100.sh",
                MODEL,
                *spec,
            ]
            server_log = (root / f"{args.tag}_server.log").open("w")
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started = True
            launch = dict(command=command, pid=process.pid, env=settings)
        launch.update(
            runtime=args.runtime,
            mode=args.mode,
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
        )
        (root / f"{args.tag}_launch.json").write_text(
            json.dumps(launch, indent=2) + "\n"
        )
        print("started", args.tag, launch, flush=True)
        for attempt in range(1200):
            if process is not None and process.poll() is not None:
                raise RuntimeError("Host server exited before becoming ready")
            if args.runtime == "docker" and attempt % 5 == 0:
                running = subprocess.check_output(
                    ["docker", "inspect", "--format", "{{.State.Running}}", name],
                    text=True,
                ).strip()
                if running != "true":
                    raise RuntimeError("Docker server exited before becoming ready")
            try:
                if requests.get(URL + "/health", timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Server startup timed out")
        print("ready", args.tag, flush=True)
        collect_environment(root, args.runtime, name)
        before = response_checks(root, args.tag + "_before", args.mode)
        clients = (
            ["bench.py", "bench_natural.py"] if args.mode == "mtp" else ["bench.py"]
        )
        for client in clients:
            tag = args.tag + ("_natural" if client == "bench_natural.py" else "_random")
            command = [
                sys.executable,
                str(PREVIOUS / client),
                "--tag",
                tag,
                "--output-dir",
                str(root),
            ]
            if args.mode == "target":
                command += ["--lengths", "1000", "25000", "--output-len", "2048"]
            print("benchmark", tag, flush=True)
            subprocess.run(command, cwd=REPO, env=env, check=True)
        after = response_checks(root, args.tag + "_after", args.mode)
        (root / f"{args.tag}_done.json").write_text(
            json.dumps(
                dict(
                    completed=True,
                    short_response_ids_match_before=before,
                    short_response_ids_match_after=after,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        if started and args.runtime == "docker":
            subprocess.run(["docker", "stop", "-t", "30", name], check=True)
            with (root / f"{args.tag}_server.log").open("w") as log:
                subprocess.run(
                    ["docker", "logs", name],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        elif process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            server_log.close()


if __name__ == "__main__":
    main()
