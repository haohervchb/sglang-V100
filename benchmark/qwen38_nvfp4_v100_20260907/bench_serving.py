"""Repeat cold-cache serving measurements against an already running server."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--model", default="RadixArk/Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1000, 8192, 25000])
    parser.add_argument("--output-length", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob(f"{args.label}_*")):
        parser.error("Use a new label or directory to avoid mixing benchmark runs")
    response = requests.get(
        f"http://{args.host}:{args.port}/get_server_info", timeout=30
    )
    response.raise_for_status()
    (args.output_dir / f"{args.label}_server_info.json").write_text(
        json.dumps(response.json(), indent=2) + "\n"
    )
    command = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--served-model-name",
        "qwen",
        "--tokenizer",
        args.model,
        "--dataset-name",
        "random",
        "--num-prompts",
        "1",
        "--random-range-ratio",
        "1.0",
        "--max-concurrency",
        "1",
        "--warmup-requests",
        "0",
        "--tokenize-prompt",
        "--seed",
        "20260830",
        "--disable-tqdm",
        "--random-output-len",
        str(args.output_length),
        "--flush-cache",
    ]
    env = dict(os.environ, FLASHINFER_DISABLE_VERSION_CHECK="1")
    for length in args.lengths:
        for repeat in range(args.repeats + 1):
            run = "warmup" if repeat == 0 else f"r{repeat}"
            output = args.output_dir / f"{args.label}_{run}.jsonl"
            print(f"{args.label}: input={length}, {run}", flush=True)
            with (args.output_dir / f"{args.label}_{length}_{run}.log").open(
                "w"
            ) as log:
                subprocess.run(
                    command
                    + ["--random-input-len", str(length), "--output-file", str(output)],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )


if __name__ == "__main__":
    main()
