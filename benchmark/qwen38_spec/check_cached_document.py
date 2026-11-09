"""Check a long MTP continuation with a real prefix hit and a cold replay."""

import argparse
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8083")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(
        "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        revision="7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    )
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    note = encode(
        "Reference entry: measure CPU time, memory use, storage capacity, and queueing delays in a repeatable workload.\n"
    )
    pad = lambda count: (note * ((count + len(note) - 1) // len(note)))[:count]
    marker = "DOCUMENT_BODY_INSERTION"
    rendered = tokenizer.apply_chat_template(
        [
            dict(
                role="user",
                content=marker
                + "\nWrite at least 1000 words explaining how to investigate service performance. Start immediately with the guide.",
            )
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    before, after = rendered.split(marker)
    prefix, suffix = encode(before), encode(after)
    key = encode("\nArchive key for cedar is 731946.\n")
    count = 32000 - len(prefix) - len(suffix) - len(key)
    ids = prefix + pad(count // 2) + key + pad(count - count // 2) + suffix
    session = requests.Session()
    session.trust_env = False
    session.post(
        args.url + "/flush_cache", params={"timeout": 30}, timeout=120
    ).raise_for_status()

    def generate(tokens, limit):
        response = session.post(
            args.url + "/generate",
            json=dict(
                input_ids=tokens,
                sampling_params=dict(temperature=0, max_new_tokens=limit),
            ),
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    first = generate(ids, 512)
    assert first["meta_info"]["completion_tokens"] >= 256, first["meta_info"]
    continuation_marker = "PREVIOUS_ASSISTANT_CONTENT"
    new_body = "ADDITIONAL_BODY"
    rendered = tokenizer.apply_chat_template(
        [
            dict(role="assistant", content=continuation_marker),
            dict(
                role="user",
                content=new_body
                + "\nWhat is the archive key for cedar? Return only the six digits.",
            ),
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    continuation = rendered.split(continuation_marker, 1)[1]
    before, after = continuation.split(new_body)
    next_ids = ids + first["output_ids"] + encode(before) + pad(4096) + encode(after)
    cached = generate(next_ids, 32)
    cached_tokens = cached["meta_info"].get("cached_tokens", 0)
    assert cached_tokens >= 32000, cached["meta_info"]
    assert cached["text"].strip() == "731946", cached["text"]
    session.post(
        args.url + "/flush_cache", params={"timeout": 30}, timeout=120
    ).raise_for_status()
    cold = generate(next_ids, 32)
    assert cold["meta_info"].get("cached_tokens", 0) == 0
    assert cold["text"].strip() == "731946", cold["text"]
    result = dict(
        passed=True,
        cached_tokens=cached_tokens,
        prompt_tokens=len(next_ids),
        first=first,
        cached=cached,
        cold=cold,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Cached document and cold replay passed:",
        cached_tokens,
        "cached tokens",
        flush=True,
    )


if __name__ == "__main__":
    main()
