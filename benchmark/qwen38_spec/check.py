"""Small correctness gates against a running server; saves every request/result."""

import argparse
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8082")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-context", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(
        "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        revision="7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    )
    results = []

    def chat(text):
        ids = tokenizer.apply_chat_template(
            [dict(role="user", content=text)],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return ids if isinstance(ids, list) else ids["input_ids"]

    def check(name, ids, expected=None, sampling=None, flush=True):
        if flush:
            requests.post(
                args.url + "/flush_cache", params={"timeout": 30}, timeout=120
            ).raise_for_status()
        params = dict(temperature=0, max_new_tokens=128)
        params.update(sampling or {})
        payload = dict(
            input_ids=ids,
            sampling_params=params,
            return_logprob=True,
            logprob_start_len=-1,
        )
        response = requests.post(args.url + "/generate", json=payload, timeout=1800)
        response.raise_for_status()
        data = response.json()
        passed = expected is None or data["text"].strip() == expected
        results.append(
            dict(
                name=name,
                prompt_tokens=len(ids),
                expected=expected,
                sampling=params,
                response=data,
                passed=passed,
            )
        )
        args.output.write_text(
            json.dumps(
                dict(
                    passed=all(r["passed"] for r in results),
                    complete=False,
                    expected_cases=8 if args.full_context else 4,
                    cases=results,
                ),
                indent=2,
            )
            + "\n"
        )
        print(name, passed, repr(data["text"][:150]), flush=True)

    check("arithmetic", chat("Compute 17 * 23. Respond with only the integer."), "391")
    check(
        "geography",
        chat("What is the capital of France? Respond with only the city name."),
        "Paris",
    )
    check(
        "json_grammar",
        chat('Return a JSON object with one key "answer" set to 391.'),
        sampling=dict(
            json_schema=json.dumps(
                dict(
                    type="object",
                    properties=dict(answer=dict(type="integer", const=391)),
                    required=["answer"],
                    additionalProperties=False,
                )
            )
        ),
    )
    grammar = json.loads(results[-1]["response"]["text"])
    results[-1]["passed"] = grammar == {"answer": 391}
    check(
        "sampled",
        chat("Explain why a bounded cache needs an eviction policy, in two sentences."),
        sampling=dict(temperature=0.7, top_p=0.9, sampling_seed=42),
    )
    if args.full_context:
        marker = "CONTEXT_INSERTION_MARKER"
        rendered = tokenizer.apply_chat_template(
            [
                dict(
                    role="user",
                    content=marker
                    + "\nReturn the archive keys for cedar and maple in that order, separated by one space. Return only the two keys.",
                )
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before, after = rendered.split(marker)
        enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
        prefix, suffix = enc(before), enc(after)
        note = enc(
            "Reference entry: measure CPU time, memory use, storage capacity, and queueing delays in a repeatable workload.\n"
        )
        keys = (
            enc("\nArchive key for cedar is 731946.\n"),
            enc("\nArchive key for maple is 284613.\n"),
        )
        remaining = 262000 - len(prefix) - len(suffix) - sum(map(len, keys))
        n = remaining // 3
        pad = lambda length: (note * ((length + len(note) - 1) // len(note)))[:length]
        ids = (
            prefix
            + pad(n)
            + keys[0]
            + pad(n)
            + keys[1]
            + pad(remaining - 2 * n)
            + suffix
        )
        check(
            "full_context_retrieval",
            ids,
            "731946 284613",
            sampling=dict(max_new_tokens=32),
        )
        check(
            "after_full_context",
            chat("Compute 17 * 23. Respond with only the integer."),
            "391",
            flush=False,
        )
        info_response = requests.get(args.url + "/get_server_info", timeout=30)
        info_response.raise_for_status()
        limit = info_response.json()["max_req_input_len"]
        response = requests.post(
            args.url + "/generate",
            json=dict(
                input_ids=[0] * (limit + 1),
                sampling_params=dict(temperature=0, max_new_tokens=1),
            ),
            timeout=120,
        )
        data = response.json()
        reason = data.get("meta_info", {}).get("finish_reason", {})
        rejected = response.status_code == 400 or (
            isinstance(reason, dict)
            and reason.get("type") == "abort"
            and reason.get("status_code") == 400
        )
        results.append(
            dict(
                name="oversized_input_rejected",
                prompt_tokens=limit + 1,
                resolved_limit=limit,
                status_code=response.status_code,
                response=data,
                passed=rejected,
            )
        )
        print("oversized_input_rejected", rejected, flush=True)
        check(
            "after_rejected_input",
            chat("Compute 17 * 23. Respond with only the integer."),
            "391",
            flush=False,
        )
    args.output.write_text(
        json.dumps(
            dict(
                passed=all(r["passed"] for r in results),
                complete=True,
                expected_cases=8 if args.full_context else 4,
                cases=results,
            ),
            indent=2,
        )
        + "\n"
    )
    if not all(r["passed"] for r in results):
        raise SystemExit("Correctness gate failed; see saved responses")


if __name__ == "__main__":
    main()
