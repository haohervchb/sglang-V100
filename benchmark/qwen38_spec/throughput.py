"""Measure completed output tokens / wall time with concurrent independent agents.

Cold runs use a unique early prompt prefix per request. Cached runs prime each
request separately. Both use the same saved corpus and fixed output lengths.
Per-request decode rates are diagnostics, never summed to claim throughput.
"""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import uuid
from pathlib import Path

import pynvml
import requests


def make_corpus(source, destination, count):
    from transformers import AutoTokenizer

    original = json.loads(source.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        original["tokenizer"], revision=original.get("tokenizer_revision")
    )
    cases = []
    for case in original["cases"]:
        # Insert immediately after the user-role header, before any shared body.
        text = tokenizer.decode(case["input_ids"], skip_special_tokens=False)
        marker = "<|im_start|>user\n"
        if text.count(marker) != 1:
            raise ValueError(f"Unexpected template: {case['id']}")
        position = text.index(marker) + len(marker)
        for agent in range(count):
            prefix = f"Independent session {agent:04d}.\n"
            ids = tokenizer.encode(
                text[:position] + prefix + text[position:], add_special_tokens=False
            )
            cases.append(
                {
                    "id": f"{case['id']}_agent{agent}",
                    "source_id": case["id"],
                    "agent": agent,
                    "input_ids": ids,
                    "input_sha256": hashlib.sha256(
                        json.dumps(ids).encode()
                    ).hexdigest(),
                }
            )
    with destination.open("x") as stream:
        json.dump(
            {
                "tokenizer": original["tokenizer"],
                "tokenizer_revision": original.get("tokenizer_revision"),
                "cases": cases,
            },
            stream,
        )


def generate(url, case, output_tokens, ignore_eos=True):
    start = time.perf_counter()
    start_unix_s = time.time()
    request_id = uuid.uuid4().hex
    events = []
    final = None
    with requests.post(
        url + "/generate",
        json={
            "input_ids": case["input_ids"],
            "rid": request_id,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "sampling_seed": 42,
                "max_new_tokens": output_tokens,
                "ignore_eos": ignore_eos,
            },
        },
        stream=True,
        timeout=(30, 900),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=None):
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            final = json.loads(line[6:])
            if "error" in final:
                raise RuntimeError(final)
            count = final["meta_info"]["completion_tokens"]
            if events and count < events[-1]["tokens"]:
                raise RuntimeError("Non-monotonic completion count")
            if count and (not events or count > events[-1]["tokens"]):
                events.append({"seconds": time.perf_counter() - start, "tokens": count})
    if not final or not events:
        raise RuntimeError("Incomplete response")
    meta = final["meta_info"]
    finish = meta.get("finish_reason")
    if not isinstance(finish, dict) or finish.get("type") not in ("stop", "length"):
        raise RuntimeError(f"Unsuccessful or incomplete generation: {finish}")
    if (ignore_eos and meta["completion_tokens"] != output_tokens) or meta[
        "completion_tokens"
    ] > output_tokens:
        raise RuntimeError(f"Incomplete output: {meta}")
    if meta["prompt_tokens"] != len(case["input_ids"]):
        raise RuntimeError(f"Unexpected prompt length: {meta}")
    output_text = final.get("text", "")
    if (
        case.get("source_id", "").startswith(("prose_", "code_edit_"))
        and meta["completion_tokens"] >= 32
        and len(set(output_text.strip())) < 8
    ):
        raise RuntimeError("Degenerate output for a long-form benchmark prompt")
    return {
        "case_id": case["id"],
        "request_id": request_id,
        "start_unix_s": start_unix_s,
        "input_sha256": case["input_sha256"],
        "input_tokens": len(case["input_ids"]),
        "output_tokens": meta["completion_tokens"],
        "output_budget": output_tokens,
        "ignore_eos": ignore_eos,
        "ttft_s": events[0]["seconds"],
        "e2e_s": events[-1]["seconds"],
        "decode_tok_s": (events[-1]["tokens"] - events[0]["tokens"])
        / (events[-1]["seconds"] - events[0]["seconds"])
        if len(events) > 1
        else None,
        "meta_info": meta,
        "output_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
        "output_text": output_text,
        "events": events,
    }


def run(args):
    cases = json.loads(args.corpus.read_text())["cases"]
    selected = [c for c in cases if c["source_id"] == args.case][: args.requests]
    if len(selected) != args.requests:
        raise ValueError("Not enough independent prompts in corpus")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "benchmark_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "start_unix_s": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    response = requests.get(args.url + "/get_server_info", timeout=30)
    response.raise_for_status()
    (args.output / "server.json").write_text(json.dumps(response.json(), indent=2))
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]
    all_runs = []
    try:
        if args.warmup_tokens:
            # Real multi-request prefill shapes can compile lazily even after
            # decode graph capture. Keep their first-use costs outside timing.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(args.concurrency)
            ) as pool:
                warmup = list(
                    pool.map(
                        lambda case: generate(
                            args.url,
                            case,
                            args.warmup_tokens,
                            ignore_eos=not args.natural_eos,
                        ),
                        selected,
                    )
                )
            (args.output / "warmup.json").write_text(json.dumps(warmup) + "\n")
        for repeat in range(args.repeats):
            order = (
                args.concurrency
                if repeat % 2 == 0
                else list(reversed(args.concurrency))
            )
            for concurrency in order:
                response = requests.post(
                    args.url + "/flush_cache", params={"timeout": 30}, timeout=120
                )
                response.raise_for_status()
                if response.text.strip() not in ("", "null"):
                    try:
                        body = response.json()
                    except ValueError:
                        body = None
                    if body is False or (
                        isinstance(body, dict) and body.get("success") is False
                    ):
                        raise RuntimeError("Cache flush refused")
                if args.cached:
                    for case in selected:
                        # Hybrid recurrent-state caching needs a completed
                        # tracking interval, not a single-token warmup.
                        generate(args.url, case, 512)
                stop = threading.Event()
                samples = []

                def monitor(stop=stop, samples=samples):
                    while not stop.is_set():
                        samples.append(
                            {
                                "unix_s": time.time(),
                                "free_mib": [
                                    pynvml.nvmlDeviceGetMemoryInfo(h).free / 2**20
                                    for h in handles
                                ],
                                "gpu_util": [
                                    pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                                    for h in handles
                                ],
                            }
                        )
                        stop.wait(0.2)

                thread = threading.Thread(target=monitor, daemon=True)
                thread.start()
                start = time.perf_counter()
                results, errors = [], []
                try:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=concurrency
                    ) as pool:
                        futures = [
                            pool.submit(
                                generate,
                                args.url,
                                case,
                                args.output_tokens,
                                not args.natural_eos,
                            )
                            for case in selected
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                results.append(future.result())
                            except Exception as exc:  # noqa: BLE001 - persist failures, then fail the run
                                errors.append(repr(exc))
                    elapsed = time.perf_counter() - start
                finally:
                    stop.set()
                    thread.join()
                decode_rates = [
                    r["decode_tok_s"] for r in results if r["decode_tok_s"] is not None
                ]
                row = {
                    "label": args.label,
                    "case": args.case,
                    "repeat": repeat,
                    "concurrency": concurrency,
                    "cached": args.cached,
                    "requested": len(selected),
                    "completed": len(results),
                    "errors": errors,
                    "seconds": elapsed,
                    "output_tok_s": sum(r["output_tokens"] for r in results) / elapsed,
                    "median_ttft_s": statistics.median(r["ttft_s"] for r in results)
                    if results
                    else None,
                    "median_e2e_s": statistics.median(r["e2e_s"] for r in results)
                    if results
                    else None,
                    "median_decode_tok_s": statistics.median(decode_rates)
                    if decode_rates
                    else None,
                    "cached_input_fraction": sum(
                        r["meta_info"].get("cached_tokens", 0) for r in results
                    )
                    / sum(r["input_tokens"] for r in results)
                    if results
                    else 0,
                    "retractions": sum(
                        r["meta_info"].get(
                            "num_retractions", r["meta_info"].get("retraction_count", 0)
                        )
                        for r in results
                    ),
                    "min_free_mib": [
                        min(s["free_mib"][i] for s in samples) for i in range(4)
                    ],
                    "requests": results,
                }
                with (args.output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                (args.output / f"memory-r{repeat}-c{concurrency}.json").write_text(
                    json.dumps(samples)
                )
                all_runs.append({k: v for k, v in row.items() if k != "requests"})
                (args.output / "summary.json").write_text(
                    json.dumps(all_runs, indent=2) + "\n"
                )
                print(json.dumps(all_runs[-1]), flush=True)
                if errors:
                    raise RuntimeError(errors)
    finally:
        pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-corpus")
    make.add_argument("--source", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--requests", type=int, default=8)
    bench = commands.add_parser("run")
    bench.add_argument("--url", default="http://127.0.0.1:8083")
    bench.add_argument("--corpus", type=Path, required=True)
    bench.add_argument("--case", default="prose_1000")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--label", required=True)
    bench.add_argument("--requests", type=int, default=8)
    bench.add_argument("--output-tokens", type=int, default=1024)
    bench.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--warmup-tokens", type=int, default=128)
    bench.add_argument("--natural-eos", action="store_true")
    bench.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    if args.command == "make-corpus":
        make_corpus(args.source, args.output, args.requests)
    else:
        run(args)


if __name__ == "__main__":
    main()
