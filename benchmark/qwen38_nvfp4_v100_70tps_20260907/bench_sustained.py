"""Cold 25K prefill followed by sustained greedy decode, with token timelines."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer

from sglang.benchmark.datasets.random import sample_random_requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    parser.add_argument("--model", default="RadixArk/Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--tag", default="candidate")
    parser.add_argument("--input-len", type=int, default=25000)
    parser.add_argument("--output-len", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    requests.get(args.url + "/health", timeout=10).raise_for_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schedule = [(args.seed, 256)] + [
        (args.seed + i, args.output_len) for i in range(args.repeats)
    ]
    for index, (seed, output_len) in enumerate(schedule):
        random.seed(seed)
        np.random.seed(seed)
        prompt = sample_random_requests(
            args.input_len, output_len, 1, 1.0, tokenizer, "", return_text=False
        )[0].prompt
        requests.post(args.url + "/flush_cache", timeout=60).raise_for_status()
        body = dict(
            input_ids=prompt,
            sampling_params=dict(
                temperature=0, max_new_tokens=output_len, ignore_eos=True
            ),
            stream=True,
        )
        start = time.perf_counter()
        times, counts = [], []
        with requests.post(
            args.url + "/generate", json=body, stream=True, timeout=600
        ) as response:
            response.raise_for_status()
            # Read actual HTTP chunks. Byte-at-a-time reads can saturate the
            # client CPU as SGLang's cumulative text responses grow.
            for line in response.iter_lines(chunk_size=None):
                if not line.startswith(b"data: "):
                    continue
                if line[6:] == b"[DONE]":
                    break
                item = json.loads(line[6:])
                count = item["meta_info"]["completion_tokens"]
                if not counts or count > counts[-1]:
                    times.append(time.perf_counter() - start)
                    counts.append(count)
        assert counts[-1] == output_len, (counts[-1], output_len)
        assert item["meta_info"]["prompt_tokens"] == len(prompt)
        windows, window_counts = [], []
        for i in range(0, len(counts) - 1, 256):
            j = min(i + 256, len(counts) - 1)
            window_counts.append(counts[j] - counts[i])
            windows.append(window_counts[-1] / (times[j] - times[i]))
        row = dict(
            tag=args.tag,
            warmup=index == 0,
            seed=seed,
            input_len=len(prompt),
            output_len=output_len,
            ttft_s=times[0],
            prefill_tok_s=len(prompt) / times[0],
            decode_tok_s=(counts[-1] - counts[0]) / (times[-1] - times[0]),
            window_tok_s=windows,
            window_token_counts=window_counts,
            times_s=times,
            completion_tokens=counts,
            server_cached_tokens=item["meta_info"].get("cached_tokens"),
            server_prompt_tokens=item["meta_info"]["prompt_tokens"],
            server_completion_tokens=item["meta_info"]["completion_tokens"],
        )
        with args.output.open("a") as output:
            output.write(json.dumps(row) + "\n")
        print(
            args.tag,
            seed,
            output_len,
            f"prefill={row['prefill_tok_s']:.2f}",
            f"decode={row['decode_tok_s']:.3f}",
            f"slowest_window={min(windows):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
