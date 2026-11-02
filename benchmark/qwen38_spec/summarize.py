"""Summarize completed observations, retaining variation and backend exclusions."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(rows):
    values = {}
    for key in [
        "decode_tok_s",
        "observed_cycle_ms",
        "observed_tokens_per_cycle",
        "ttft_s",
    ]:
        numbers = [row[key] for row in rows if row.get(key) is not None]
        values[key] = (
            dict(median=statistics.median(numbers), min=min(numbers), max=max(numbers))
            if numbers
            else None
        )
    values.update(
        observations=len(rows),
        output_hashes=len(
            {
                hashlib.sha256(row.get("output_text", "").encode()).hexdigest()
                for row in rows
            }
        ),
        actual_output_tokens=[row["meta_info"]["completion_tokens"] for row in rows],
    )
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = {}
    for path in sorted(args.directory.glob("*.jsonl")):
        lines = path.read_text().splitlines(keepends=True)
        rows = [json.loads(line) for line in lines if line.endswith("\n")]
        rows = [
            row for row in rows if not row.get("warmup") and not row.get("diagnostic")
        ]
        if not rows or "case_id" not in rows[0]:
            continue
        label = path.stem
        root_label = label.replace("_speed_sampled", "").replace("_speed", "")
        log = args.directory / (root_label + "_server.log")
        excluded = (
            log.exists() and "SM70 FP16 MoE operators unavailable" in log.read_text()
        )
        entry = dict(
            excluded=excluded,
            exclusion_reason="MTP silently fell back without the local TurboMind binary"
            if excluded
            else None,
            cases={
                case: summarize([r for r in rows if r["case_id"] == case])
                for case in sorted({r["case_id"] for r in rows})
            },
        )
        if "_speed" in label:
            entry["domains"] = {
                kind: summarize([r for r in rows if r["kind"] == kind])
                for kind in sorted({r["kind"] for r in rows})
            }
        result[label] = entry
    Path(args.output).write_text(json.dumps(result, indent=2))
    print("Summarized:", ", ".join(result))


if __name__ == "__main__":
    main()
