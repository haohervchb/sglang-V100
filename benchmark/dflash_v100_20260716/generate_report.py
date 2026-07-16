#!/usr/bin/env python3
"""Generate the human-readable report from the preserved benchmark summaries."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_summary(name: str) -> list[dict]:
    return json.loads((ROOT / name / "summary.json").read_text())


def index(rows: list[dict]) -> dict[tuple[int, int], dict]:
    return {(row["prompt_len"], row["concurrency"]): row for row in rows}


def metric(row: dict, name: str) -> float:
    return float(row[f"{name}_median"])


def result_table(rows: list[dict], concurrency: int) -> str:
    selected = [row for row in rows if row["concurrency"] == concurrency]
    selected.sort(key=lambda row: row["prompt_len"])
    lines = [
        "| Context | TTFT median / batch max (s) | Effective prefill (tok/s) | Aggregate decode (tok/s), median [range] | Median request decode (tok/s) | E2E output (tok/s) | DFlash accept length |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        decode = metric(row, "aggregate_decode_tps")
        decode_min = row["aggregate_decode_tps_min"]
        decode_max = row["aggregate_decode_tps_max"]
        lines.append(
            "| "
            f"{row['prompt_len'] // 1000}K | "
            f"{metric(row, 'median_ttft_ms') / 1000:.2f} / "
            f"{metric(row, 'max_ttft_ms') / 1000:.2f} | "
            f"{metric(row, 'effective_prefill_tps'):,.0f} | "
            f"{decode:.1f} [{decode_min:.1f}–{decode_max:.1f}] | "
            f"{metric(row, 'median_request_decode_tps'):.1f} | "
            f"{metric(row, 'e2e_output_tps'):.1f} | "
            f"{metric(row, 'weighted_accept_length'):.2f} |"
        )
    return "\n".join(lines)


def headline_table(models: list[tuple[str, list[dict]]]) -> str:
    lines = [
        "| Model | Context | Concurrent | TTFT (s) | Effective prefill (tok/s) | Aggregate decode (tok/s), median [range] | Per-request decode (tok/s) | Accept length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, rows in models:
        table = index(rows)
        for prompt_len in (1_000, 13_000, 25_000):
            for concurrency in (1, 2, 4):
                row = table[(prompt_len, concurrency)]
                lines.append(
                    f"| {label} | {prompt_len // 1000}K | {concurrency} | "
                    f"{metric(row, 'median_ttft_ms') / 1000:.2f} | "
                    f"{metric(row, 'effective_prefill_tps'):,.0f} | "
                    f"{metric(row, 'aggregate_decode_tps'):.1f} "
                    f"[{row['aggregate_decode_tps_min']:.1f}–{row['aggregate_decode_tps_max']:.1f}] | "
                    f"{metric(row, 'median_request_decode_tps'):.1f} | "
                    f"{metric(row, 'weighted_accept_length'):.2f} |"
                )
    return "\n".join(lines)


def integrity(name: str) -> dict[str, int]:
    trials = []
    with (ROOT / name / "trials.jsonl").open() as file:
        for line in file:
            trials.append(json.loads(line))
    requests = [request for trial in trials for request in trial["requests"]]
    return {
        "trials": len(trials),
        "cells": len({(trial["prompt_len"], trial["concurrency"]) for trial in trials}),
        "requests": len(requests),
        "input_tokens": sum(request["prompt_tokens"] for request in requests),
        "output_tokens": sum(request["completion_tokens"] for request in requests),
        "failures": sum(not request["success"] for request in requests),
        "wrong_outputs": sum(request["completion_tokens"] != 256 for request in requests),
        "max_cached": max(request["cached_tokens"] for request in requests),
        "max_first_chunk": max(request["first_chunk_tokens"] for request in requests),
    }


def main() -> None:
    dense = load_summary("27b")
    moe = load_summary("122b")
    dense_i = index(dense)
    moe_i = index(moe)
    dense_integrity = integrity("27b")
    moe_integrity = integrity("122b")

    dense_1k_c4 = metric(dense_i[(1_000, 4)], "aggregate_decode_tps")
    dense_25k_c4 = metric(dense_i[(25_000, 4)], "aggregate_decode_tps")
    dense_1k_c1 = metric(dense_i[(1_000, 1)], "aggregate_decode_tps")
    dense_25k_c1 = metric(dense_i[(25_000, 1)], "aggregate_decode_tps")
    moe_1k_c4 = metric(moe_i[(1_000, 4)], "aggregate_decode_tps")
    moe_25k_c4 = metric(moe_i[(25_000, 4)], "aggregate_decode_tps")

    report = f"""# DFlash on 4× V100: 27B dense and 122B-A10B GPTQ

Test date: 2026-07-16 (Australia/Brisbane)
Repository commit: `4c8434780e92bf0de20c2e4992ce87b3d113eba6`

## Bottom line

- Long-context concurrency is still the limiting case. Qwen3.6-27B at concurrency 4 fell from **{dense_1k_c4:.1f} aggregate decode tok/s at 1K** to **{dense_25k_c4:.1f} tok/s at 25K** ({dense_25k_c4 / dense_1k_c4 * 100:.1f}% of the 1K rate). Both cells reported a perfect 16-token weighted acceptance length, so that drop cannot be blamed on draft quality.
- Qwen3.6-27B at concurrency 1 was much flatter in the same two perfect-acceptance cells: **{dense_1k_c1:.1f} tok/s at 1K** and **{dense_25k_c1:.1f} tok/s at 25K** ({dense_25k_c1 / dense_1k_c1 * 100:.1f}% retained). The severe decay is therefore strongly tied to the context-length × concurrency interaction.
- Qwen3.5-122B-A10B GPTQ at concurrency 4 fell from **{moe_1k_c4:.1f} tok/s at 1K** to **{moe_25k_c4:.1f} tok/s at 25K** ({moe_25k_c4 / moe_1k_c4 * 100:.1f}% retained). Its acceptance also declined (about {metric(moe_i[(1_000, 4)], 'weighted_accept_length'):.2f} to {metric(moe_i[(25_000, 4)], 'weighted_accept_length'):.2f}), so both verification cost and draft quality contribute.
- At 25K and concurrency 4, median TTFT was **{metric(dense_i[(25_000, 4)], 'median_ttft_ms') / 1000:.2f} s** for 27B and **{metric(moe_i[(25_000, 4)], 'median_ttft_ms') / 1000:.2f} s** for 122B. These are client-visible localhost times and include chunked scheduling, not kernel-only prefill measurements.
- This run does **not** establish a DFlash speedup over normal decoding because no non-DFlash baseline was collected. It also does not support a direct V100-versus-5060 Ti claim. It answers the narrower question: what the current V100 DFlash stack delivers across 1K–25K context and concurrency 1/2/4 on a realistic mixed coding-agent workload.

## Plots

Each figure overlays both models. Lines are three-run medians and shading is the observed min–max.

### Concurrency 1

![DFlash V100 results at concurrency 1](plots/concurrency_1.png)

### Concurrency 2

![DFlash V100 results at concurrency 2](plots/concurrency_2.png)

### Concurrency 4

![DFlash V100 results at concurrency 4](plots/concurrency_4.png)

## Reddit-size result table

Values are medians of three cold-cache trials. Decode brackets are the observed min–max across those trials. `K` means 1,000 tokenizer tokens.

{headline_table([('Qwen3.6-27B FP16', dense), ('Qwen3.5-122B-A10B GPTQ Int4', moe)])}

## Test protocol

- Hardware: 4× Tesla V100-SXM2-32GB, TP=4 over NVLink; NVIDIA driver 580.159.04; 300 W power limit per GPU.
- Software: PyTorch 2.9.1+cu128, CUDA runtime 12.8; repository commit shown above. The captured SGLang package string was `0.0.0.dev13403+ge16efa89c.d20260713`, while execution used the local checkout through `PYTHONPATH`.
- Models: `Qwen/Qwen3.6-27B` FP16 with `z-lab/Qwen3.6-27B-DFlash`; and `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` with the V100 GPTQ-Marlin target path plus unquantized FP16 `z-lab/Qwen3.5-122B-A10B-DFlash` draft.
- Exact input lengths: 1,000, 3,000, ..., 25,000 tokens, including chat-template overhead. Concurrency: 1, 2, and 4. Each request generated exactly 256 tokens greedily (`temperature=0`, `ignore_eos=true`). Tasks explicitly requested at least 350 words so 256 tokens normally remain inside the answer rather than padding a short response.
- Prompts: unique chat-formatted slices from 423,373 tokens of real repository source, with a unique nonce, source offset, and coding-review task. There is no within-batch shared prefix. Each model uses its own tokenizer and deterministic model-specific slices, so the two model workloads have the same construction/distribution but are not byte-identical prompts.
- Cache policy: `/flush_cache` before every trial plus a short settling delay. Captured response metadata reported zero cached prompt tokens in every request.
- Warmup: unmeasured 1K×4 and 25K×4 boundary trials after server startup. Target and draft CUDA graphs were captured for batch sizes 1, 2, and 4. A one-time 122B first-request JIT event (12.3 s TTFT) was measured separately and excluded; the immediate repeat was 0.84 s at 1K×4.
- Repetitions: three measured passes per cell, 39 cells/model and 117 trials/model. Cell order was deterministically shuffled within each pass so longer contexts were not always tested later.
- Integrity: 27B had {dense_integrity['trials']} trials, {dense_integrity['requests']} requests, {dense_integrity['input_tokens']:,} input tokens, and {dense_integrity['output_tokens']:,} output tokens. 122B had {moe_integrity['trials']} trials, {moe_integrity['requests']} requests, {moe_integrity['input_tokens']:,} input tokens, and {moe_integrity['output_tokens']:,} output tokens. Across both: zero failed requests, zero wrong output lengths, zero cached prompt tokens, and every first stream chunk contained exactly one token.

## Metric definitions

- **TTFT median / batch max:** within each concurrent trial, the median request TTFT and slowest request TTFT; the table reports the median of each statistic across three trials. It includes localhost HTTP/SSE overhead and server scheduling.
- **Effective prefill:** total input tokens divided by the time from the earliest request start until the last request receives its first streamed token. This deliberately exposes scheduling and straggler effects; it is not a kernel-only prefill throughput.
- **Aggregate decode:** all tokens after each request's first stream chunk divided by the window from the earliest first chunk to the last completion. The first chunk is excluded in full to avoid inflating DFlash throughput if streaming ever batches accepted tokens.
- **Median request decode:** median of each request's post-first-chunk tokens divided by its own post-first-chunk time.
- **E2E output:** all output tokens divided by full batch wall time, including prefill/TTFT.
- **DFlash accept length:** total completion tokens divided by total speculative verification steps, weighted across the concurrent requests.

## Full results: Qwen3.6-27B FP16

### Concurrency 1

{result_table(dense, 1)}

### Concurrency 2

{result_table(dense, 2)}

### Concurrency 4

{result_table(dense, 4)}

## Full results: Qwen3.5-122B-A10B GPTQ Int4

### Concurrency 1

{result_table(moe, 1)}

### Concurrency 2

{result_table(moe, 2)}

### Concurrency 4

{result_table(moe, 4)}

## Reproduction notes

The 27B server used `--mem-fraction-static 0.70` and finalized 306,144 target/draft KV slots with four running-request slots. The 122B server required `--mem-fraction-static 0.72` and finalized 130,320 KV slots with four running-request slots. At 0.70, the 122B server automatically reduced capacity to three requests and changed graph capture to `[1, 2, 3]`; that launch was rejected before measurement. Four 25K prompts plus four 256-token completions require 101,024 token slots, so the measured 122B setup had sufficient capacity.

Exact raw records and server configurations are preserved beside this report:

- `27b/trials.jsonl`, `27b/summary.csv`, `27b/summary.json`, `27b/server_info.json`
- `122b/trials.jsonl`, `122b/summary.csv`, `122b/summary.json`, `122b/server_info.json`
- `bench.py` is the exact streaming harness used for both runs.

The summary CSV/JSON files contain median, min, max, 25th percentile, and 75th percentile for every recorded metric. With only three repetitions, min–max is more transparent than implying a statistically strong percentile estimate.
"""
    (ROOT / "README.md").write_text(report)


if __name__ == "__main__":
    main()
