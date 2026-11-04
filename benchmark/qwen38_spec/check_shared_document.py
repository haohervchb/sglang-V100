"""Check concurrent questions sharing a long document and retained prefix state."""

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8083")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(
        "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        revision="7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    )

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    note = encode(
        "Document note: measure time, memory use, storage capacity and queueing delays.\n"
    )

    def pad(length):
        return (note * ((length + len(note) - 1) // len(note)))[:length]

    names = ["cedar", "maple", "birch", "elm", "ash", "oak", "pine", "willow"]
    keys = {name: str(731946 + 121 * i) for i, name in enumerate(names)}
    body = []
    for name in names:
        anchor = encode(f"\nThe archive key for {name} is {keys[name]}.\n")
        body += pad(4000 - len(anchor)) + anchor
    cases = []
    for name in names:
        rendered = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"Shared reference document:\nBODY_MARKER\nQuestion about {name}. QUESTION_PADDING\nReturn only the six-digit archive key for {name}.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before, after = rendered.split("BODY_MARKER")
        middle, suffix = after.split("QUESTION_PADDING")
        # Put the branch point well before the final tracked checkpoint. A hit
        # must therefore retain an intermediate shared-document snapshot.
        ids = encode(before) + body + encode(middle) + pad(384) + encode(suffix)
        cases.append({"name": name, "expected": keys[name], "input_ids": ids})

    def flush():
        response = requests.post(
            args.url + "/flush_cache", params={"timeout": 30}, timeout=120
        )
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError:
            data = None
        if data is False or (isinstance(data, dict) and data.get("success") is False):
            raise RuntimeError("Cache flush refused")

    def generate(case):
        start = time.perf_counter()
        response = requests.post(
            args.url + "/generate",
            json={
                "input_ids": case["input_ids"],
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=(30, 600),
        )
        response.raise_for_status()
        data = response.json()
        return {
            "name": case["name"],
            "expected": case["expected"],
            "input_tokens": len(case["input_ids"]),
            "input_sha256": hashlib.sha256(
                json.dumps(case["input_ids"]).encode()
            ).hexdigest(),
            "seconds": time.perf_counter() - start,
            "response": data,
            "passed": data["text"].strip() == case["expected"],
        }

    flush()
    prime = generate(cases[0])
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        shared = list(pool.map(generate, cases))
    shared_seconds = time.perf_counter() - start
    cold = []
    for case in [cases[0], cases[-1]]:
        flush()
        cold.append(generate(case))
    cached = [r["response"]["meta_info"].get("cached_tokens", 0) for r in shared]
    result = {
        "passed": all(r["passed"] for r in [prime, *shared, *cold])
        and min(cached) >= 30000,
        "concurrency": args.concurrency,
        "shared_seconds": shared_seconds,
        "cached_input_tokens": cached,
        "prime": prime,
        "shared": shared,
        "cold": cold,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["prime", "shared", "cold"]}
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
