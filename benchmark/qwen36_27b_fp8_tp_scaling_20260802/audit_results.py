#!/usr/bin/env python3
"""Audit and summarize the Qwen3.6-27B-FP8 TP2/TP4 scaling sweep."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_CONTEXTS = list(range(1_000, 25_001, 2_000))
EXPECTED_OUTPUT_TOKENS = 256


def load_trials(name: str) -> list[dict]:
    path = ROOT / name / "trials.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row["prompt_len"])
    assert [row["prompt_len"] for row in rows] == EXPECTED_CONTEXTS, name
    return rows


def audit_trial(row: dict) -> None:
    prompt_len = row["prompt_len"]
    assert row["concurrency"] == 1, prompt_len
    assert row["repeat"] == 0, prompt_len
    assert row["output_tokens_per_request"] == EXPECTED_OUTPUT_TOKENS, prompt_len

    aggregate = row["aggregate"]
    assert aggregate["max_cached_tokens"] == 0, prompt_len
    assert aggregate["total_input_tokens"] == prompt_len, prompt_len
    assert aggregate["total_output_tokens"] == EXPECTED_OUTPUT_TOKENS, prompt_len

    assert len(row["requests"]) == 1, prompt_len
    request = row["requests"][0]
    assert request["success"] and request["status"] == 200, prompt_len
    assert request["error"] == "", prompt_len
    assert request["prompt_tokens"] == prompt_len, prompt_len
    assert request["completion_tokens"] == EXPECTED_OUTPUT_TOKENS, prompt_len
    assert request["finish_reason"] == "length", prompt_len
    assert request["cached_tokens"] == 0, prompt_len

    output_audit = request["output_audit"]
    assert output_audit["valid"], (prompt_len, output_audit["reasons"])
    assert output_audit["reasons"] == [], prompt_len
    assert output_audit["retokenized_tokens"] == EXPECTED_OUTPUT_TOKENS, prompt_len
    assert output_audit["text"], prompt_len
    assert hashlib.sha256(output_audit["text"].encode()).hexdigest() == output_audit[
        "text_sha256"
    ], prompt_len


def geomean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def main() -> None:
    trials = {name: load_trials(name) for name in ("tp2", "tp4")}
    for rows in trials.values():
        for row in rows:
            audit_trial(row)

    prompt_matches = 0
    output_matches = 0
    for tp2, tp4 in zip(trials["tp2"], trials["tp4"], strict=True):
        assert tp2["prompt_len"] == tp4["prompt_len"]
        prompt_matches += (
            tp2["requests"][0]["prompt_sha256"]
            == tp4["requests"][0]["prompt_sha256"]
        )
        output_matches += (
            tp2["requests"][0]["output_audit"]["text_sha256"]
            == tp4["requests"][0]["output_audit"]["text_sha256"]
        )
    assert prompt_matches == len(EXPECTED_CONTEXTS)

    print(f"Audited {sum(map(len, trials.values()))} successful cold-cache trials")
    print(f"Matched prompts: {prompt_matches}/{len(EXPECTED_CONTEXTS)}")
    print(
        "Byte-identical completions: "
        f"{output_matches}/{len(EXPECTED_CONTEXTS)} "
        "(each non-identical completion passed its independent output audit)"
    )

    for metric, label in (
        ("effective_input_tps", "effective prefill"),
        ("median_request_decode_tps", "client-visible decode"),
    ):
        ratios = [
            tp4["aggregate"][metric] / tp2["aggregate"][metric]
            for tp2, tp4 in zip(trials["tp2"], trials["tp4"], strict=True)
        ]
        print(f"TP4/TP2 {label} geometric mean: {geomean(ratios):.3f}x")
        print(f"TP4/TP2 {label} geometric mean (3K-25K): {geomean(ratios[1:]):.3f}x")


if __name__ == "__main__":
    main()
