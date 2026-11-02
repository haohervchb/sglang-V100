"""Run process-owned TP4 comparisons and checkpoint each completed stage."""

import argparse
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]


def write_json(path, data):
    with path.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def run_client(args, output, *command):
    with output.open("x") as log:
        result = subprocess.run(
            [sys.executable, *command], cwd=REPO, stdout=log, stderr=subprocess.STDOUT
        )
        log.flush()
        os.fsync(log.fileno())
    return result.returncode


def archive_completed(root, destination, label):
    """Persist metrics immediately; keep SPEED text local, publish its hash."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(root.glob(label + "*")):
        if not path.is_file() or path.suffix not in (
            ".json",
            ".jsonl",
            ".log",
            ".csv",
            ".diff",
        ):
            continue
        data = path.read_bytes()
        if "_speed" in path.name and path.suffix == ".jsonl":
            records = []
            for line in data.decode().splitlines():
                row = json.loads(line)
                text = row.pop("output_text", "")
                row["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                records.append(json.dumps(row))
            data = ("\n".join(records) + "\n").encode()
        with gzip.GzipFile(
            filename=str(destination / (path.name + ".gz")), mode="wb", mtime=0
        ) as stream:
            stream.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config)
    for variant in config["variants"]:
        label = variant["name"]
        if (root / (label + ".done")).exists():
            continue
        try:
            requests.get(args.url + "/health", timeout=2)
        except requests.ConnectionError:
            pass
        else:
            raise RuntimeError("Refusing to start over an existing server")
        source = Path(variant.get("cwd", REPO))
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("SGLANG_SPEC_PROBE"):
                del env[key]
        env.update(variant.get("env", {}))
        log_path = root / (label + "_server.log")
        print("Launching", label, flush=True)
        with log_path.open("x") as log:
            process = subprocess.Popen(
                variant["command"],
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip()
            write_json(
                root / (label + "_launch.json"),
                dict(
                    pid=process.pid,
                    revision=revision,
                    source=str(source),
                    variant=variant,
                ),
            )
            (root / (label + "_launch.diff")).write_bytes(
                subprocess.check_output(["git", "diff", "HEAD"], cwd=source)
            )
            telemetry = None
            try:
                deadline = time.monotonic() + 2700
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{label} exited during startup")
                    try:
                        ready = requests.get(args.url + "/health", timeout=2).ok
                    except requests.RequestException:
                        ready = False
                    if ready:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{label} startup timeout")
                    time.sleep(5)
                server_log = log_path.read_text()
                for marker in variant.get("required_log_markers", []):
                    if marker not in server_log:
                        raise RuntimeError(f"Missing required backend marker: {marker}")
                with (root / (label + "_gpu.csv")).open("x") as metrics:
                    telemetry = subprocess.Popen(
                        [
                            "nvidia-smi",
                            "--query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu",
                            "--format=csv",
                            "-l",
                            "1",
                        ],
                        stdout=metrics,
                    )
                    base = [
                        "benchmark/qwen38_spec/bench.py",
                        "run",
                        "--url",
                        args.url,
                        "--corpus",
                        config["corpus"],
                        "--label",
                        label,
                        "--output",
                        str(root / (label + ".jsonl")),
                        "--repeats",
                        str(variant.get("repeats", 3)),
                    ]
                    if variant.get("cases"):
                        base += ["--cases", *variant["cases"]]
                    code = run_client(args, root / (label + "_bench.log"), *base)
                    telemetry.terminate()
                    telemetry.wait()
                    telemetry = None
                    if code:
                        raise RuntimeError(f"{label} benchmark failed")
                archive_completed(root, args.archive_dir, label)
                if variant.get("speed", True):
                    for temperature, suffix in [(0, "_speed"), (0.7, "_speed_sampled")]:
                        name = label + suffix
                        code = run_client(
                            args,
                            root / (name + ".log"),
                            "benchmark/qwen38_spec/bench.py",
                            "run",
                            "--url",
                            args.url,
                            "--corpus",
                            config["speed_corpus"],
                            "--label",
                            name,
                            "--output",
                            str(root / (name + ".jsonl")),
                            "--repeats",
                            "2",
                            "--warmups",
                            "0",
                            "--natural-eos",
                            "--temperature",
                            str(temperature),
                        )
                        if code:
                            raise RuntimeError(f"{name} failed")
                        archive_completed(root, args.archive_dir, label)
                checks = root / (label + "_checks.json")
                code = run_client(
                    args,
                    root / (label + "_checks.log"),
                    "benchmark/qwen38_spec/check.py",
                    "--url",
                    args.url,
                    "--output",
                    str(checks),
                    "--full-context",
                )
                if code:
                    raise RuntimeError(f"{label} full-context checks failed")
                code = run_client(
                    args,
                    root / (label + "_tools.log"),
                    "benchmark/qwen38_spec/check_tools.py",
                    "--url",
                    args.url,
                    "--output",
                    str(root / (label + "_tools.json")),
                )
                if code:
                    raise RuntimeError(f"{label} tool checks failed")
                if variant.get("multimodal", True):
                    code = run_client(
                        args,
                        root / (label + "_mm.log"),
                        "benchmark/qwen38_nvfp4_v100_multimodal_20260909/validate.py",
                        "--base-url",
                        args.url,
                        "--tag",
                        label + "_mm",
                        "--output-dir",
                        str(root),
                    )
                    results = json.loads(
                        (root / (label + "_mm_results.json")).read_text()
                    )
                    failures = [case for case in results["cases"] if not case["passed"]]
                    if failures and not (
                        len(failures) == 1
                        and failures[0]["name"] == "ocr"
                        and failures[0]["text"].strip() == "V10 CHECK 4827"
                    ):
                        raise RuntimeError(f"{label} has a new multimodal failure")
                archive_completed(root, args.archive_dir, label)
                if variant.get("profile"):
                    profile_dir = root / (label + "_profiles")
                    statuses = {}
                    for suffix, flags in [("cpu", ["--cpu-only"]), ("gpu", [])]:
                        statuses[suffix] = run_client(
                            args,
                            root / (label + "_profile_" + suffix + ".log"),
                            "benchmark/qwen38_spec/collect_profile.py",
                            "--url",
                            args.url,
                            "--corpus",
                            config["corpus"],
                            "--case",
                            "code_edit_8192",
                            "--output-dir",
                            str(profile_dir),
                            "--label",
                            suffix,
                            "--output-tokens",
                            "128" if flags else "64",
                            *flags,
                        )
                    write_json(root / (label + "_profile_status.json"), statuses)
                write_json(root / (label + ".done"), dict(completed=time.time()))
                print("Completed", label, flush=True)
            finally:
                if telemetry is not None:
                    telemetry.terminate()
                    telemetry.wait()
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
                log.flush()
                os.fsync(log.fileno())
                archive_completed(root, args.archive_dir, label)
                time.sleep(3)


if __name__ == "__main__":
    main()
