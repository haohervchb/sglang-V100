"""Exercise queued long documents, retained caches and changing prefill sizes.

This sends one concurrent client per document length (four by default). It
never flushes caches between requests. NVML tracks device memory without
creating another CUDA context.
"""

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path

import pynvml
import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[18753, 23791, 70015, 95037],
        help="One independent client per document length",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(
        "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        revision="7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    )
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    lock = threading.Lock()
    stop_monitor = threading.Event()
    results = []
    memory_samples = []
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]
    start = time.monotonic()

    def monitor():
        while not stop_monitor.is_set():
            memory = [pynvml.nvmlDeviceGetMemoryInfo(h) for h in handles]
            memory_samples.append(
                dict(
                    seconds=time.monotonic() - start,
                    used_bytes=[m.used for m in memory],
                    free_bytes=[m.free for m in memory],
                )
            )
            stop_monitor.wait(0.1)

    def run_client(index, base_tokens):
        session = requests.Session()
        session.trust_env = False
        key = str(731946 + index * 121)
        marker = "DOCUMENT_BODY_INSERTION"
        rendered = tokenizer.apply_chat_template(
            [
                dict(
                    role="user",
                    content=(
                        f"Document {index}:\n{marker}\n"
                        "Return only this document's six-digit archive key."
                    ),
                )
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before, after = rendered.split(marker)
        prefix, suffix = encode(before), encode(after)
        note = encode(
            f"Document {index} reference: measure CPU time, memory use, storage "
            "capacity, and queueing delays in a repeatable workload.\n"
        )

        def padding(length):
            return (note * ((length + len(note) - 1) // len(note)))[:length]

        anchor = encode(f"\nThis document's archive key is {key}.\n")
        length = base_tokens - len(prefix) - len(suffix) - len(anchor)
        body = padding(length // 2) + anchor + padding(length - length // 2)
        for turn, growth in enumerate((0, 1, 1023, 2049, 4095, 8193)):
            body += padding(growth)
            ids = prefix + body + suffix
            request_start = time.monotonic()
            record = dict(
                client=index,
                turn=turn,
                prompt_tokens=len(ids),
                input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                expected=key,
            )
            try:
                response = session.post(
                    args.url + "/generate",
                    json=dict(
                        input_ids=ids,
                        sampling_params=dict(temperature=0, max_new_tokens=32),
                    ),
                    timeout=(15, 600),
                )
                record["status_code"] = response.status_code
                response.raise_for_status()
                data = response.json()
                record["response"] = data
                record["passed"] = data["text"].strip() == key
            except Exception as exc:
                record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            record["seconds"] = time.monotonic() - request_start
            with lock:
                results.append(record)
                with (args.output_dir / "requests.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"client={index} turn={turn} tokens={len(ids)} "
                    f"passed={record['passed']} seconds={record['seconds']:.2f}",
                    flush=True,
                )
            if not record["passed"]:
                break
        session.close()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(args.lengths)
        ) as executor:
            futures = [
                executor.submit(run_client, index, length)
                for index, length in enumerate(args.lengths)
            ]
            for future in futures:
                future.result()
    finally:
        stop_monitor.set()
        thread.join()
        pynvml.nvmlShutdown()
        (args.output_dir / "memory.json").write_text(json.dumps(memory_samples) + "\n")

    summary = dict(
        passed=len(results) == 6 * len(args.lengths)
        and all(r["passed"] for r in results),
        completed=len(results),
        expected=6 * len(args.lengths),
        seconds=time.monotonic() - start,
        min_free_mib=[
            round(min(s["free_bytes"][i] for s in memory_samples) / 2**20, 1)
            for i in range(4)
        ],
        cached_requests=sum(
            r.get("response", {}).get("meta_info", {}).get("cached_tokens", 0) > 0
            for r in results
        ),
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
