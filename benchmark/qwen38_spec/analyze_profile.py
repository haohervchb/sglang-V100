"""Summarize diagnostic CUDA events and Chrome traces without double counting overlap."""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def union_us(intervals):
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def category(name):
    name = name.lower()
    for label, patterns in (
        ("all_reduce", ("allreduce", "all_reduce", "nccl", "one_shot", "twoshot")),
        ("moe", ("moe", "expert", "marlin")),
        ("gdn", ("gated_delta", "gdn", "chunk_delta", "causal_conv", "recurrent")),
        (
            "qsa_metadata",
            (
                "qsa_graph",
                "qsa_draft_graph",
                "expand_topk",
                "sparse_kv",
                "sparse_valid",
            ),
        ),
        ("qsa_compute", ("qsa", "sparse_attn", "sparse_decode", "indexer")),
        ("copies", ("memcpy", "copy", "memset")),
    ):
        if any(pattern in name for pattern in patterns):
            return label
    return "other"


def analyze_trace(trace):
    # Kernel and memory events, grouped by device; CPU launch/API durations
    # must never be included in GPU busy time.
    devices = defaultdict(list)
    sums = defaultdict(float)
    counts = defaultdict(int)
    kernel_times = defaultdict(float)
    for e in trace["traceEvents"]:
        if e.get("ph") != "X" or e.get("cat") not in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        ):
            continue
        device = str(e.get("args", {}).get("device", e.get("pid")))
        devices[device].append((e["ts"], e["ts"] + e["dur"]))
        label = category(e.get("name", ""))
        sums[label] += e["dur"]
        counts[label] += 1
        kernel_times[e.get("name", "")] += e["dur"]
    result = {}
    for device, intervals in devices.items():
        span = max(e for _, e in intervals) - min(s for s, _ in intervals)
        busy = union_us(intervals)
        result[device] = dict(
            span_ms=span / 1000,
            busy_ms=busy / 1000,
            idle_ms=(span - busy) / 1000,
            idle_fraction=(span - busy) / span if span else 0,
        )
    return dict(
        devices=result,
        kernel_sum_ms={k: v / 1000 for k, v in sums.items()},
        kernel_counts=dict(counts),
        top_kernels_ms=dict(
            sorted(
                ((k, v / 1000) for k, v in kernel_times.items()), key=lambda x: -x[1]
            )[:20]
        ),
        note="Kernel sums can overlap; idle is the union over all recorded device streams, within first-to-last GPU activity. Categories use kernel names, not layer attribution.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for path in args.inputs:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            requests = defaultdict(list)
            for row in rows:
                requests[",".join(row["request_ids"])].append(row)

            def summarize(group):
                values = defaultdict(list)
                for row in group:
                    values["cycle_gpu_ms"].append(row["cycle_gpu_ms"])
                    values["cpu_submit_ms"].append(row["cpu_submit_ms"])
                    for name, value in row["phases_ms"].items():
                        values[name].append(value)
                return dict(
                    rounds=len(group),
                    median_ms={k: statistics.median(v) for k, v in values.items()},
                )

            result[str(path)] = dict(
                diagnostic=True,
                **summarize(rows),
                by_request={rid: summarize(group) for rid, group in requests.items()},
            )
        else:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt") as f:
                result[str(path)] = analyze_trace(json.load(f))
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
