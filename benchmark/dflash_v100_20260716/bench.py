#!/usr/bin/env python3
"""Cold-prompt DFlash concurrency/context benchmark for SGLang.

Uses unique chat-formatted slices of real repository source, flushes the radix
cache before every trial, and records streaming client-side timings plus SGLang
speculative metadata. K means 1,000 tokens in the report.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


DEFAULT_LENGTHS = list(range(1_000, 25_001, 2_000))
DEFAULT_CONCURRENCY = [1, 2, 4]
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "third_party",
}
TASKS = [
    "Identify the most important correctness and reliability risks in this code and explain concrete fixes.",
    "Trace the major data flow through this code, then propose a practical optimization that preserves behavior.",
    "Review this code as a senior maintainer. Prioritize bugs, unsafe assumptions, and missing tests.",
    "Produce an implementation plan for improving this code, including likely regressions and validation steps.",
]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def stable_int(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def load_source_corpus(repo: Path, tokenizer) -> list[int]:
    chunks: list[str] = []
    total_chars = 0
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        rel = path.relative_to(repo)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        chunks.append(f"\n\n# FILE: {rel}\n{text}")
        total_chars += len(text)
        if total_chars >= 1_500_000:
            break
    if not chunks:
        raise RuntimeError(f"No source corpus found under {repo}")
    old_model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = 1_000_000_000
    try:
        token_ids = tokenizer.encode("".join(chunks), add_special_tokens=False)
    finally:
        tokenizer.model_max_length = old_model_max_length
    if len(token_ids) < 100_000:
        raise RuntimeError(f"Source corpus is too small: {len(token_ids)} tokens")
    return token_ids


def circular_slice(values: list[int], start: int, length: int) -> list[int]:
    start %= len(values)
    if length <= len(values) - start:
        return values[start : start + length]
    out = values[start:]
    while len(out) < length:
        take = min(len(values), length - len(out))
        out.extend(values[:take])
    return out


def build_prompt_ids(
    tokenizer,
    corpus_ids: list[int],
    target_len: int,
    model_key: str,
    repeat: int,
    concurrency: int,
    request_idx: int,
) -> list[int]:
    nonce = hashlib.sha256(
        f"{model_key}:{repeat}:{target_len}:{concurrency}:{request_idx}".encode()
    ).hexdigest()[:16]
    placeholder = f"\nZXQCORPUSMARKER{nonce.upper()}QXZ\n"
    task = TASKS[stable_int(nonce) % len(TASKS)]
    messages = [
        {
            "role": "system",
            "content": (
                f"Benchmark session {nonce}. You are a coding agent performing "
                "a careful repository review. Be precise and evidence-driven."
            ),
        },
        {
            "role": "user",
            "content": (
                "The following is a slice of a real software repository.\n"
                f"{placeholder}"
                f"\nTask: {task}\n"
                "Write a technically detailed response of at least 350 words. "
                "Use specific examples from the supplied code and do not stop early."
            ),
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if rendered.count(placeholder) != 1:
        raise RuntimeError("Chat template did not preserve the corpus placeholder")
    prefix_text, suffix_text = rendered.split(placeholder)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    fixed_len = len(prefix_ids) + len(suffix_ids)
    body_len = target_len - fixed_len
    if body_len <= 0:
        raise ValueError(
            f"Target {target_len} is smaller than chat overhead {fixed_len}"
        )
    offset = stable_int(model_key, repeat, target_len, concurrency, request_idx)
    body_ids = circular_slice(corpus_ids, offset, body_len)
    prompt_ids = prefix_ids + body_ids + suffix_ids
    assert len(prompt_ids) == target_len
    return prompt_ids


async def process_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt_ids: list[int],
    output_tokens: int,
    request_idx: int,
) -> dict[str, Any]:
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    start = time.perf_counter()
    first_at: float | None = None
    first_chunk_tokens = 0
    completion_tokens = 0
    meta: dict[str, Any] = {}
    error = ""
    status = 0
    try:
        async with session.post(url, json=payload) as response:
            status = response.status
            if response.status != 200:
                error = await response.text()
            else:
                async for raw in response.content:
                    for line in raw.decode("utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        value = line[5:].strip()
                        if value == "[DONE]":
                            continue
                        data = json.loads(value)
                        current_meta = data.get("meta_info") or {}
                        if current_meta:
                            meta.update(current_meta)
                        current_tokens = int(
                            current_meta.get("completion_tokens", completion_tokens)
                        )
                        if data.get("text") and current_tokens > 0:
                            completion_tokens = current_tokens
                            if first_at is None:
                                first_at = time.perf_counter()
                                first_chunk_tokens = current_tokens
    except Exception as exc:  # preserve partial trial output for diagnosis
        error = repr(exc)
    end = time.perf_counter()
    success = (
        status == 200
        and not error
        and first_at is not None
        and completion_tokens == output_tokens
    )
    ttft = (first_at - start) if first_at is not None else None
    decode_seconds = (end - first_at) if first_at is not None else None
    post_first_tokens = max(0, completion_tokens - first_chunk_tokens)
    return {
        "request_idx": request_idx,
        "success": success,
        "status": status,
        "error": error,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": completion_tokens,
        "first_chunk_tokens": first_chunk_tokens,
        "start_time": start,
        "first_token_time": first_at,
        "end_time": end,
        "ttft_s": ttft,
        "e2e_s": end - start,
        "decode_s": decode_seconds,
        "post_first_tokens": post_first_tokens,
        "request_decode_tps": (
            post_first_tokens / decode_seconds
            if decode_seconds and decode_seconds > 0
            else None
        ),
        "cached_tokens": int(meta.get("cached_tokens", -1)),
        "spec_verify_ct": int(meta.get("spec_verify_ct", 0)),
        "spec_accept_length": meta.get("spec_accept_length"),
    }


async def flush_cache(session: aiohttp.ClientSession, base_url: str) -> None:
    async with session.post(f"{base_url}/flush_cache") as response:
        text = await response.text()
        if response.status != 200:
            raise RuntimeError(f"flush_cache failed ({response.status}): {text}")


def aggregate_trial(requests: list[dict[str, Any]]) -> dict[str, Any]:
    if not requests or not all(item["success"] for item in requests):
        errors = [item["error"] for item in requests if not item["success"]]
        raise RuntimeError(f"Trial request failure: {errors}")
    batch_start = min(item["start_time"] for item in requests)
    last_first = max(item["first_token_time"] for item in requests)
    first_first = min(item["first_token_time"] for item in requests)
    batch_end = max(item["end_time"] for item in requests)
    total_input = sum(item["prompt_tokens"] for item in requests)
    total_output = sum(item["completion_tokens"] for item in requests)
    post_first = sum(item["post_first_tokens"] for item in requests)
    verify_ct = sum(item["spec_verify_ct"] for item in requests)
    prefill_window = last_first - batch_start
    decode_window = batch_end - first_first
    wall = batch_end - batch_start
    return {
        "batch_wall_s": wall,
        "median_ttft_ms": statistics.median(item["ttft_s"] for item in requests)
        * 1000,
        "max_ttft_ms": max(item["ttft_s"] for item in requests) * 1000,
        "effective_prefill_tps": total_input / prefill_window,
        "aggregate_decode_tps": post_first / decode_window,
        "median_request_decode_tps": statistics.median(
            item["request_decode_tps"] for item in requests
        ),
        "e2e_output_tps": total_output / wall,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "weighted_accept_length": total_output / verify_ct if verify_ct else None,
        "max_cached_tokens": max(item["cached_tokens"] for item in requests),
        "first_chunk_tokens_total": sum(
            item["first_chunk_tokens"] for item in requests
        ),
    }


async def run_trial(
    session: aiohttp.ClientSession,
    args,
    tokenizer,
    corpus_ids: list[int],
    prompt_len: int,
    concurrency: int,
    repeat: int,
    output_tokens: int,
) -> dict[str, Any]:
    await flush_cache(session, args.base_url)
    await asyncio.sleep(args.post_flush_delay)
    prompts = [
        build_prompt_ids(
            tokenizer,
            corpus_ids,
            prompt_len,
            args.model_key,
            repeat,
            concurrency,
            idx,
        )
        for idx in range(concurrency)
    ]
    requests = await asyncio.gather(
        *[
            process_streaming_request(
                session,
                f"{args.base_url}/generate",
                prompt,
                output_tokens,
                idx,
            )
            for idx, prompt in enumerate(prompts)
        ]
    )
    aggregate = aggregate_trial(requests)
    return {
        "model_key": args.model_key,
        "prompt_len": prompt_len,
        "concurrency": concurrency,
        "repeat": repeat,
        "output_tokens_per_request": output_tokens,
        "aggregate": aggregate,
        "requests": requests,
    }


def q(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["prompt_len"], row["concurrency"]), []).append(
            row["aggregate"]
        )
    metrics = [
        "median_ttft_ms",
        "max_ttft_ms",
        "effective_prefill_tps",
        "aggregate_decode_tps",
        "median_request_decode_tps",
        "e2e_output_tps",
        "weighted_accept_length",
        "batch_wall_s",
    ]
    summary: list[dict[str, Any]] = []
    for (prompt_len, concurrency), trials in sorted(grouped.items()):
        item: dict[str, Any] = {
            "prompt_len": prompt_len,
            "concurrency": concurrency,
            "trials": len(trials),
            "max_cached_tokens": max(t["max_cached_tokens"] for t in trials),
        }
        for metric in metrics:
            values = [float(t[metric]) for t in trials if t[metric] is not None]
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
            item[f"{metric}_q25"] = q(values, 0.25)
            item[f"{metric}_q75"] = q(values, 0.75)
        summary.append(item)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    return summary


async def main_async(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "trials.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    corpus_ids = load_source_corpus(Path(args.repo), tokenizer)
    print(json.dumps({"event": "corpus_ready", "tokens": len(corpus_ids)}), flush=True)

    existing: list[dict[str, Any]] = []
    completed: set[tuple[int, int, int]] = set()
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing.append(row)
            completed.add((row["prompt_len"], row["concurrency"], row["repeat"]))

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.get(f"{args.base_url}/get_server_info") as response:
            if response.status != 200:
                raise RuntimeError(f"Server info failed: {await response.text()}")
            server_info = await response.json()
        (output_dir / "server_info.json").write_text(
            json.dumps(server_info, indent=2, default=str)
        )

        if not args.skip_warmup:
            for warm_len in (min(args.lengths), max(args.lengths)):
                print(
                    json.dumps(
                        {
                            "event": "warmup",
                            "prompt_len": warm_len,
                            "concurrency": max(args.concurrency),
                        }
                    ),
                    flush=True,
                )
                await run_trial(
                    session,
                    args,
                    tokenizer,
                    corpus_ids,
                    warm_len,
                    max(args.concurrency),
                    -1 - warm_len,
                    args.warmup_output_tokens,
                )

        with raw_path.open("a") as raw_file:
            for repeat in range(args.repeats):
                cells = [
                    (prompt_len, concurrency)
                    for prompt_len in args.lengths
                    for concurrency in args.concurrency
                ]
                random.Random(stable_int(args.model_key, repeat)).shuffle(cells)
                for prompt_len, concurrency in cells:
                    key = (prompt_len, concurrency, repeat)
                    if key in completed:
                        continue
                    started = time.perf_counter()
                    row = await run_trial(
                        session,
                        args,
                        tokenizer,
                        corpus_ids,
                        prompt_len,
                        concurrency,
                        repeat,
                        args.output_tokens,
                    )
                    raw_file.write(json.dumps(row) + "\n")
                    raw_file.flush()
                    existing.append(row)
                    aggregate = row["aggregate"]
                    print(
                        json.dumps(
                            {
                                "event": "trial",
                                "model": args.model_key,
                                "repeat": repeat,
                                "prompt_len": prompt_len,
                                "concurrency": concurrency,
                                "ttft_ms": round(aggregate["median_ttft_ms"], 1),
                                "prefill_tps": round(
                                    aggregate["effective_prefill_tps"], 1
                                ),
                                "decode_tps": round(
                                    aggregate["aggregate_decode_tps"], 1
                                ),
                                "accept": round(
                                    aggregate["weighted_accept_length"], 3
                                ),
                                "cached": aggregate["max_cached_tokens"],
                                "elapsed_s": round(time.perf_counter() - started, 2),
                            }
                        ),
                        flush=True,
                    )
                    await asyncio.sleep(args.cooldown)

    summary = write_summary(existing, output_dir)
    print(
        json.dumps(
            {
                "event": "complete",
                "trials": len(existing),
                "cells": len(summary),
                "output_dir": str(output_dir),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--repo", default="/home/rah/sglang-V100")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", type=parse_int_list, default=DEFAULT_LENGTHS)
    parser.add_argument(
        "--concurrency", type=parse_int_list, default=DEFAULT_CONCURRENCY
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--warmup-output-tokens", type=int, default=64)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--post-flush-delay", type=float, default=0.15)
    parser.add_argument("--cooldown", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
