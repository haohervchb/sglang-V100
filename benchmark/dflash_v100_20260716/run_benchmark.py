#!/usr/bin/env python3
"""Audited DFlash context/concurrency benchmark for the V100 fork.

The harness uses cold, unique prompts built from repository source. It records
the prompt and output hashes, retains generated text for manual inspection, and
refuses to summarize a trial when output repetition suggests state corruption.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import csv
import hashlib
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


@dataclass
class Prompt:
    input_ids: list[int]
    sha256: str
    nonce: str
    corpus_offset: int
    task: str


def build_prompt(
    tokenizer,
    corpus_ids: list[int],
    target_len: int,
    repeat: int,
    concurrency: int,
    request_idx: int,
) -> Prompt:
    # Deliberately independent of the model name so both models see the same
    # workload definition at every matrix cell.
    prompt_key = f"{repeat}:{target_len}:{concurrency}:{request_idx}"
    nonce = hashlib.sha256(prompt_key.encode()).hexdigest()[:16]
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
                f"{placeholder}\nTask: {task}\n"
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
    body_len = target_len - len(prefix_ids) - len(suffix_ids)
    if body_len <= 0:
        raise ValueError(f"Target {target_len} is smaller than chat overhead")
    corpus_offset = stable_int(repeat, target_len, concurrency, request_idx)
    body_ids = circular_slice(corpus_ids, corpus_offset, body_len)
    input_ids = prefix_ids + body_ids + suffix_ids
    if len(input_ids) != target_len:
        raise AssertionError((len(input_ids), target_len))
    token_bytes = b"".join(token.to_bytes(4, "little") for token in input_ids)
    return Prompt(
        input_ids=input_ids,
        sha256=sha256_bytes(token_bytes),
        nonce=nonce,
        corpus_offset=corpus_offset % len(corpus_ids),
        task=task,
    )


def max_run(values: list[Any]) -> int:
    best = current = 0
    previous: Any = object()
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        best = max(best, current)
    return best


def audit_output(tokenizer, text: str, completion_tokens: int) -> dict[str, Any]:
    output_ids = tokenizer.encode(text, add_special_tokens=False)
    unique_tokens = len(set(output_ids))
    unique_chars = len(set(text))
    reasons: list[str] = []
    token_run = max_run(output_ids)
    char_run = max_run(list(text))
    if completion_tokens >= 32 and len(output_ids) <= max(2, completion_tokens // 4):
        reasons.append("decoded output is implausibly short for reported token count")
    if len(output_ids) >= 16 and unique_tokens <= 2:
        reasons.append("output has at most two unique re-tokenized tokens")
    if token_run >= 16:
        reasons.append(f"identical output-token run is {token_run}")
    if char_run >= 64:
        reasons.append(f"identical character run is {char_run}")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "text_sha256": sha256_bytes(text.encode()),
        "text": text,
        "text_chars": len(text),
        "retokenized_tokens": len(output_ids),
        "unique_retokenized_tokens": unique_tokens,
        "unique_chars": unique_chars,
        "max_identical_token_run": token_run,
        "max_identical_char_run": char_run,
    }


async def process_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: Prompt,
    tokenizer,
    output_tokens: int,
    request_idx: int,
) -> dict[str, Any]:
    payload = {
        "input_ids": prompt.input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "ignore_eos": False,
        },
        "stream": True,
    }
    start = time.perf_counter()
    first_at: float | None = None
    first_chunk_tokens = 0
    completion_tokens = 0
    meta: dict[str, Any] = {}
    final_text = ""
    error = ""
    status = 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    line_buffer = ""

    def consume_line(line: str) -> None:
        nonlocal first_at, first_chunk_tokens, completion_tokens, final_text
        line = line.strip()
        if not line.startswith("data:"):
            return
        value = line[5:].strip()
        if not value or value == "[DONE]":
            return
        data = json.loads(value)
        current_meta = data.get("meta_info") or {}
        if current_meta:
            meta.update(current_meta)
        current_tokens = int(current_meta.get("completion_tokens", completion_tokens))
        text = data.get("text") or ""
        if text:
            # SGLang's default stream is cumulative. Retain a fallback for an
            # incremental stream so the audit still sees the complete output.
            if text.startswith(final_text):
                final_text = text
            elif not final_text.endswith(text):
                final_text += text
        completion_tokens = max(completion_tokens, current_tokens)
        if current_tokens > 0 and first_at is None:
            first_at = time.perf_counter()
            first_chunk_tokens = current_tokens

    try:
        async with session.post(url, json=payload) as response:
            status = response.status
            if response.status != 200:
                error = await response.text()
            else:
                async for raw in response.content.iter_any():
                    line_buffer += decoder.decode(raw)
                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        consume_line(line)
                line_buffer += decoder.decode(b"", final=True)
                if line_buffer.strip():
                    consume_line(line_buffer)
    except Exception as exc:
        error = repr(exc)
    end = time.perf_counter()

    output_audit = audit_output(tokenizer, final_text, completion_tokens)
    finish_reason = meta.get("finish_reason")
    finish_type = (
        finish_reason.get("type")
        if isinstance(finish_reason, dict)
        else finish_reason
    )
    success = (
        status == 200
        and not error
        and first_at is not None
        and completion_tokens >= min(64, output_tokens)
        and output_audit["valid"]
    )
    ttft = first_at - start if first_at is not None else None
    decode_seconds = end - first_at if first_at is not None else None
    post_first_tokens = max(0, completion_tokens - first_chunk_tokens)
    server_forward_start = meta.get("forward_entry_time")
    server_prefill_end = meta.get("prefill_finished_time")
    return {
        "request_idx": request_idx,
        "success": success,
        "status": status,
        "error": error,
        "prompt_tokens": len(prompt.input_ids),
        "prompt_sha256": prompt.sha256,
        "prompt_nonce": prompt.nonce,
        "corpus_offset": prompt.corpus_offset,
        "task": prompt.task,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_type,
        "first_chunk_tokens": first_chunk_tokens,
        "client_ttft_s": ttft,
        "client_e2e_s": end - start,
        "client_decode_s": decode_seconds,
        "client_request_decode_tps": (
            post_first_tokens / decode_seconds
            if decode_seconds and decode_seconds > 0
            else None
        ),
        "server_decode_tps": meta.get("decode_throughput"),
        "server_queue_s": meta.get("queue_time"),
        "server_prefill_s": (
            server_prefill_end - server_forward_start
            if server_prefill_end is not None and server_forward_start is not None
            else None
        ),
        "cached_tokens": int(meta.get("cached_tokens", -1)),
        "spec_verify_ct": int(meta.get("spec_verify_ct", 0)),
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "start_time": start,
        "first_token_time": first_at,
        "end_time": end,
        "output_audit": output_audit,
    }


async def flush_cache(session: aiohttp.ClientSession, base_url: str) -> None:
    async with session.post(f"{base_url}/flush_cache") as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"flush_cache failed ({response.status}): {body}")


def aggregate_trial(requests: list[dict[str, Any]]) -> dict[str, Any]:
    if not requests or not all(item["success"] for item in requests):
        failures = [
            {
                "request_idx": item["request_idx"],
                "error": item["error"],
                "audit": item["output_audit"]["reasons"],
            }
            for item in requests
            if not item["success"]
        ]
        raise RuntimeError(f"Trial failed output audit: {failures}")
    batch_start = min(item["start_time"] for item in requests)
    first_first = min(item["first_token_time"] for item in requests)
    last_first = max(item["first_token_time"] for item in requests)
    batch_end = max(item["end_time"] for item in requests)
    total_input = sum(item["prompt_tokens"] for item in requests)
    total_output = sum(item["completion_tokens"] for item in requests)
    verify_ct = sum(item["spec_verify_ct"] for item in requests)
    decode_window = batch_end - first_first
    request_decode_tps = [
        item["client_request_decode_tps"]
        for item in requests
        if item["client_request_decode_tps"] is not None
    ]
    server_decode_tps = [
        item["server_decode_tps"]
        for item in requests
        if item["server_decode_tps"] is not None
    ]
    return {
        "median_client_ttft_ms": statistics.median(
            item["client_ttft_s"] for item in requests
        )
        * 1000,
        "max_client_ttft_ms": max(item["client_ttft_s"] for item in requests)
        * 1000,
        "effective_input_tps": total_input / (last_first - batch_start),
        "median_request_decode_tps": statistics.median(request_decode_tps),
        "median_server_decode_tps": (
            statistics.median(server_decode_tps) if server_decode_tps else None
        ),
        "aggregate_output_tps": total_output / (batch_end - batch_start),
        "decode_window_output_tps": total_output / decode_window,
        "weighted_accept_length": total_output / verify_ct if verify_ct else None,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "batch_wall_s": batch_end - batch_start,
        "max_cached_tokens": max(item["cached_tokens"] for item in requests),
    }


async def run_trial(
    session: aiohttp.ClientSession,
    args,
    tokenizer,
    corpus_ids: list[int],
    prompt_len: int,
    concurrency: int,
    repeat: int,
) -> dict[str, Any]:
    await flush_cache(session, args.base_url)
    await asyncio.sleep(args.post_flush_delay)
    prompts = [
        build_prompt(tokenizer, corpus_ids, prompt_len, repeat, concurrency, idx)
        for idx in range(concurrency)
    ]
    requests = await asyncio.gather(
        *[
            process_streaming_request(
                session,
                f"{args.base_url}/generate",
                prompt,
                tokenizer,
                args.output_tokens,
                idx,
            )
            for idx, prompt in enumerate(prompts)
        ]
    )
    return {
        "model_key": args.model_key,
        "prompt_len": prompt_len,
        "concurrency": concurrency,
        "repeat": repeat,
        "output_tokens_per_request": args.output_tokens,
        "aggregate": aggregate_trial(requests),
        "requests": requests,
    }


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["prompt_len"], row["concurrency"]), []).append(
            row["aggregate"]
        )
    metrics = [
        "median_client_ttft_ms",
        "max_client_ttft_ms",
        "effective_input_tps",
        "median_request_decode_tps",
        "median_server_decode_tps",
        "aggregate_output_tps",
        "decode_window_output_tps",
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
            if not values:
                continue
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
            item[f"{metric}_q25"] = quantile(values, 0.25)
            item[f"{metric}_q75"] = quantile(values, 0.75)
        summary.append(item)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if summary:
        with (output_dir / "summary.csv").open("w", newline="") as file:
            fieldnames = sorted({key for row in summary for key in row})
            writer = csv.DictWriter(
                file, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(summary)


async def main_async(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "trials.jsonl"
    tokenizer = get_tokenizer(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )
    corpus_ids = load_source_corpus(Path(args.repo), tokenizer)
    print(json.dumps({"event": "corpus_ready", "tokens": len(corpus_ids)}), flush=True)

    rows: list[dict[str, Any]] = []
    completed: set[tuple[int, int, int]] = set()
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            completed.add((row["prompt_len"], row["concurrency"], row["repeat"]))

    if args.replace_existing:
        replace_keys = {
            (length, concurrency, repeat)
            for repeat in range(args.repeats)
            for length in args.lengths
            for concurrency in args.concurrency
        }
        rows = [
            row
            for row in rows
            if (row["prompt_len"], row["concurrency"], row["repeat"])
            not in replace_keys
        ]
        completed.difference_update(replace_keys)
        raw_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        async with session.get(f"{args.base_url}/get_server_info") as response:
            if response.status != 200:
                raise RuntimeError(f"Server info failed: {await response.text()}")
            server_info = await response.json()
        (output_dir / "server_info.json").write_text(
            json.dumps(server_info, indent=2, default=str)
        )

        with raw_path.open("a") as raw_file:
            for repeat in range(args.repeats):
                cells = [
                    (length, concurrency)
                    for length in args.lengths
                    for concurrency in args.concurrency
                ]
                random.Random(stable_int("cell-order", repeat)).shuffle(cells)
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
                    )
                    raw_file.write(json.dumps(row) + "\n")
                    raw_file.flush()
                    rows.append(row)
                    aggregate = row["aggregate"]
                    accept_length = aggregate["weighted_accept_length"]
                    print(
                        json.dumps(
                            {
                                "event": "trial",
                                "model": args.model_key,
                                "repeat": repeat,
                                "prompt_len": prompt_len,
                                "concurrency": concurrency,
                                "ttft_ms": round(
                                    aggregate["median_client_ttft_ms"], 1
                                ),
                                "input_tps": round(
                                    aggregate["effective_input_tps"], 1
                                ),
                                "request_decode_tps": round(
                                    aggregate["median_request_decode_tps"], 1
                                ),
                                "aggregate_output_tps": round(
                                    aggregate["aggregate_output_tps"], 1
                                ),
                                "accept": (
                                    round(accept_length, 3)
                                    if accept_length is not None
                                    else None
                                ),
                                "elapsed_s": round(time.perf_counter() - started, 2),
                            }
                        ),
                        flush=True,
                    )
                    await asyncio.sleep(args.cooldown)

    write_summary(rows, output_dir)
    print(
        json.dumps(
            {
                "event": "complete",
                "trials": len(rows),
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
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Rerun and replace the selected matrix cells in an existing result set.",
    )
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--post-flush-delay", type=float, default=0.15)
    parser.add_argument("--cooldown", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
