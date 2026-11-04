"""Check independent agent continuations concurrently, then replay them cold."""

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
    parser.add_argument("--clients", type=int, default=8)
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
        "Reference entry: measure service latency, memory use and queueing delays.\n"
    )

    def pad(length):
        return (note * ((length + len(note) - 1) // len(note)))[:length]

    cases = []
    for client in range(args.clients):
        key = str(731946 + 121 * client)
        rendered = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"Independent agent {client}:\nBODY_MARKER\nWrite a detailed practical guide of at least 2000 words about investigating service performance. Start with the guide.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before, after = rendered.split("BODY_MARKER")
        prefix, suffix = encode(before), encode(after)
        anchor = encode(f"\nThe archive key for this agent is {key}.\n")
        length = 16384 - len(prefix) - len(suffix) - len(anchor)
        ids = prefix + pad(length // 2) + anchor + pad(length - length // 2) + suffix
        cases.append({"client": client, "expected": key, "input_ids": ids})

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

    def generate(case, tokens, ignore_eos):
        start = time.perf_counter()
        response = requests.post(
            args.url + "/generate",
            json={
                "input_ids": case["input_ids"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": tokens,
                    "ignore_eos": ignore_eos,
                },
            },
            timeout=(30, 900),
        )
        response.raise_for_status()
        data = response.json()
        return {
            "client": case["client"],
            "expected": case["expected"],
            "input_tokens": len(case["input_ids"]),
            "input_sha256": hashlib.sha256(
                json.dumps(case["input_ids"]).encode()
            ).hexdigest(),
            "seconds": time.perf_counter() - start,
            "response": data,
            "passed": ignore_eos or data["text"].strip() == case["expected"],
        }

    def batch(inputs, tokens, ignore_eos):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            rows = list(
                pool.map(lambda case: generate(case, tokens, ignore_eos), inputs)
            )
        return {"seconds": time.perf_counter() - start, "requests": rows}

    flush()
    warm = batch(cases, 512, True)
    continuation = tokenizer.apply_chat_template(
        [
            {"role": "assistant", "content": "ASSISTANT_MARKER"},
            {
                "role": "user",
                "content": "NEW_BODY_MARKER\nReturn only this agent's six-digit archive key.",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ).split("ASSISTANT_MARKER", 1)[1]
    before, after = continuation.split("NEW_BODY_MARKER")
    for case, row in zip(cases, warm["requests"]):
        output_ids = row["response"]["output_ids"]
        assert len(output_ids) == 512
        case["input_ids"] = (
            case["input_ids"] + output_ids + encode(before) + pad(2048) + encode(after)
        )
    cached = batch(cases, 32, False)
    flush()
    cold = batch(cases, 32, False)
    result = {
        "passed": all(
            r["passed"] for phase in [cached, cold] for r in phase["requests"]
        ),
        "clients": args.clients,
        "concurrency": args.concurrency,
        "cached_input_tokens": [
            r["response"]["meta_info"].get("cached_tokens", 0)
            for r in cached["requests"]
        ],
        "warm": warm,
        "cached": cached,
        "cold": cold,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["warm", "cached", "cold"]}
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
