"""Fixed-corpus, cold-cache speculative decoding comparisons.

Generate the corpus once, then reuse it for every variant. Timing comes from
streamed cumulative token counts, never from text length or token re-encoding.
Server lifecycle is explicit so TP4 variants can share the same four GPUs.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import requests


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def make_corpus(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    source = "\n\n".join(
        f"def read_item_{i}(store, key):\n"
        '    """Read an item from the store, returning None when absent."""\n'
        "    if key not in store:\n"
        "        return None\n"
        "    value = store[key]\n"
        "    return value\n"
        for i in range(12)
    )
    tasks = {
        "prose": "Write a detailed practical guide of at least 2000 words explaining how to investigate a slow web service. Cover reproducible measurements, CPU and memory costs, databases, concurrency, and ways to verify improvements. Give concrete examples.",
        "code_edit": "Rewrite the following Python file, changing every function to raise KeyError(key) when the key is absent. Preserve all other code and docstrings exactly. Output the entire file, without explanation.\n```python\n"
        + source
        + "\n```",
    }
    note = tokenizer.encode(
        "Reference note: Services have finite CPU, memory, storage and network capacity. Measure queueing separately from processing; compare repeatable workloads.\n",
        add_special_tokens=False,
    )
    rows = []
    for length in args.lengths:
        for name, task in tasks.items():
            # Split a unique marker inside the user message so padding never
            # relies on hard-coded chat-template token offsets.
            marker = "BENCHMARK_PADDING_INSERTION_20261102"
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": marker + "\n\nTask: " + task}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            before, after = rendered.split(marker)
            prefix = tokenizer.encode(before, add_special_tokens=False)
            suffix = tokenizer.encode(after, add_special_tokens=False)
            padding = length - len(prefix) - len(suffix)
            if padding < 0:
                raise ValueError(
                    f"{name} needs {len(prefix) + len(suffix)} tokens, requested {length}"
                )
            ids = (
                prefix
                + (note * ((padding + len(note) - 1) // len(note)))[:padding]
                + suffix
            )
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            rows.append(
                dict(
                    id=f"{name}_{length}", kind=name, input_ids=ids, input_sha256=digest
                )
            )
    with Path(args.output).open("x") as f:
        json.dump(dict(tokenizer=args.tokenizer, cases=rows), f)
        f.write("\n")


def measure(url, case, output_tokens, temperature=0, sampling_seed=42, lookup=None):
    response = requests.post(url + "/flush_cache", params={"timeout": 30}, timeout=120)
    response.raise_for_status()
    # Some scheduler endpoints return HTTP 200 even when they refuse a flush.
    if response.text.strip() not in ("", "null"):
        try:
            body = response.json()
        except ValueError:
            body = None
        if body is False or (isinstance(body, dict) and body.get("success") is False):
            raise RuntimeError(f"Cache flush refused: {body}")
    start = time.perf_counter()
    events = []
    final = None
    request_id = uuid.uuid4().hex
    with requests.post(
        url + "/generate",
        json=dict(
            input_ids=case["input_ids"],
            rid=request_id,
            stream=True,
            sampling_params=dict(
                temperature=temperature,
                sampling_seed=sampling_seed,
                max_new_tokens=output_tokens,
                ignore_eos=True,
                custom_params={"prompt_lookup": lookup} if lookup is not None else None,
            ),
        ),
        stream=True,
        timeout=(30, 1800),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=None):
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            item = json.loads(line[6:])
            if "error" in item:
                raise RuntimeError(item)
            final = item
            meta = item["meta_info"]
            count = meta["completion_tokens"]
            if events and count < events[-1]["tokens"]:
                raise RuntimeError("Non-monotonic completion counter")
            if count and (not events or count > events[-1]["tokens"]):
                events.append(
                    dict(
                        seconds=time.perf_counter() - start,
                        tokens=count,
                        verify_ct=meta.get("spec_verify_ct", 0),
                    )
                )
    if final is None or len(events) < 2:
        raise RuntimeError(
            "Incomplete stream or too few token events for decode timing"
        )
    meta = final["meta_info"]
    if meta["completion_tokens"] != output_tokens or meta["prompt_tokens"] != len(
        case["input_ids"]
    ):
        raise RuntimeError(f"Unexpected token counts: {meta}")
    if meta.get("cached_tokens", 0) or meta.get("retraction_count", 0):
        raise RuntimeError(f"Cache hit or retraction invalidates this sample: {meta}")
    first, last = events[0], events[-1]
    elapsed = last["seconds"] - first["seconds"]
    rounds = last["verify_ct"] - first["verify_ct"]
    return dict(
        case_id=case["id"],
        request_id=request_id,
        prompt_lookup=lookup,
        kind=case["kind"],
        input_sha256=case["input_sha256"],
        input_tokens=len(case["input_ids"]),
        output_tokens=output_tokens,
        temperature=temperature,
        sampling_seed=sampling_seed,
        ttft_s=first["seconds"],
        end_to_end_s=last["seconds"],
        end_to_end_tok_s=output_tokens / last["seconds"],
        decode_tok_s=(last["tokens"] - first["tokens"]) / elapsed,
        observed_cycle_ms=1000 * elapsed / rounds if rounds else None,
        # Includes the bonus token; use the server's histogram as well since
        # the final round may be truncated at max_new_tokens.
        observed_tokens_per_cycle=(
            (last["tokens"] - first["tokens"]) / rounds if rounds else None
        ),
        meta_info=meta,
        events=events,
        output_text=final.get("text", ""),
    )


def run(args):
    corpus = json.loads(Path(args.corpus).read_text())
    cases = corpus["cases"]
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
        if {c["id"] for c in cases} != set(args.cases):
            raise ValueError("Unknown corpus case")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as out:
        info = requests.get(args.url + "/get_server_info", timeout=120)
        info.raise_for_status()
        write_json(str(destination) + ".server.json", info.json())
        for repeat in range(-args.warmups, args.repeats):
            # Reverse each repeat to balance context-length drift within a run.
            for case in cases if repeat % 2 == 0 else reversed(cases):
                n = args.output_tokens if repeat >= 0 else min(128, args.output_tokens)
                variants = [False, True] if args.lookup_ab else [None]
                if repeat % 2:
                    variants.reverse()
                for lookup in variants:
                    row = measure(
                        args.url, case, n, args.temperature, args.seed, lookup
                    )
                    row.update(
                        label=args.label,
                        repeat=repeat,
                        warmup=repeat < 0,
                        diagnostic=args.diagnostic,
                    )
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "label",
                                    "case_id",
                                    "prompt_lookup",
                                    "repeat",
                                    "decode_tok_s",
                                    "observed_cycle_ms",
                                    "observed_tokens_per_cycle",
                                )
                            }
                        ),
                        flush=True,
                    )


def sweep(args):
    """Run command arrays in balanced blocks; never stop an unrelated server."""
    config = json.loads(Path(args.config).read_text())
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config)
    for block, label in enumerate(config["order"]):
        variant = config["variants"][label]
        prefix = root / f"{block:02d}_{label}"
        env = os.environ.copy()
        env.update(variant.get("env", {}))
        try:
            requests.get(args.url + "/health", timeout=1)
        except requests.ConnectionError:
            pass
        else:
            raise RuntimeError(f"A server already listens at {args.url}")
        with Path(str(prefix) + ".log").open("x") as log:
            process = subprocess.Popen(
                variant["command"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_json(
            str(prefix) + ".launch.json",
            dict(
                pid=process.pid,
                variant=variant,
                revision=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                diff_sha256=hashlib.sha256(
                    subprocess.check_output(["git", "diff", "HEAD"])
                ).hexdigest(),
            ),
        )
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; see {prefix}.log")
                try:
                    ready = requests.get(args.url + "/health", timeout=1).ok
                except requests.RequestException:
                    ready = False
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Server not ready; see {prefix}.log")
                time.sleep(1)
            args.label, args.output = label, str(prefix) + ".jsonl"
            run(args)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    corpus = commands.add_parser("corpus")
    corpus.add_argument("--tokenizer", default="RadixArk/Qwen3.8-Flash-Next-NVFP4")
    corpus.add_argument(
        "--lengths", nargs="+", type=int, default=[1000, 8192, 25000, 70000]
    )
    corpus.add_argument("--output", required=True)
    corpus.set_defaults(func=make_corpus)
    for name, func in (("run", run), ("sweep", sweep)):
        p = commands.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument("--url", default="http://127.0.0.1:8082")
        p.add_argument("--corpus", required=True)
        p.add_argument("--cases", nargs="+")
        p.add_argument("--output-tokens", type=int, default=512)
        p.add_argument("--repeats", type=int, default=2)
        p.add_argument("--warmups", type=int, default=1)
        p.add_argument("--temperature", type=float, default=0)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--diagnostic", action="store_true")
        p.add_argument(
            "--lookup-ab",
            action="store_true",
            help="Alternate lookup off/on; requires an experiment-enabled server",
        )
        if name == "run":
            p.add_argument("--label", required=True)
            p.add_argument("--output", required=True)
        else:
            p.add_argument("--config", required=True)
            p.add_argument("--output-dir", required=True)
            p.add_argument("--startup-timeout", type=int, default=1800)
    args = parser.parse_args()
    if hasattr(args, "repeats") and (
        args.repeats < 1 or args.warmups < 0 or args.output_tokens < 2
    ):
        parser.error("Need repeats >= 1, warmups >= 0 and output-tokens >= 2")
    args.func(args)


if __name__ == "__main__":
    main()
