"""Attribute audit traces and correlate DCGM samples with request phases."""

import argparse
import gc
import gzip
import heapq
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_profile import union_us


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def kernel_category(name):
    lower = name.lower()
    for category, patterns in [
        ("collectives", ["nccl", "all_reduce", "allreduce", "all_gather"]),
        ("moe", ["marlin_moe", "sm70_nvfp4_moe", "turbomind::gemm", "_moe_", "topk10"]),
        ("dense_projections", ["gemm", "gemv", "dot_kernel"]),
        (
            "gdn",
            [
                "gating_delta",
                "gated_delta",
                "recurrent",
                "chunk_delta",
                "causal_conv",
                "sm70_qwen_gdn",
            ],
        ),
        ("ple", ["ple_embedding", "short_conv_state"]),
        ("hyperconnections", ["sm70_hc", "hc_", "mean_mul_sigmoid_view"]),
        (
            "qsa",
            [
                "qsa",
                "sparse",
                "gqa",
                "indexer",
                "rotary",
                "flash_attention_grouped_verify",
            ],
        ),
        ("copies", ["memcpy", "memset", "copy_kernel", "copy_"]),
        ("norm_pointwise", ["norm", "elementwise", "reduce_kernel", "sigmoid", "silu"]),
    ]:
        if any(pattern in lower for pattern in patterns):
            return category
    return "other"


def semantic_category(path, kernel):
    category = kernel_category(kernel)
    if category == "collectives":
        return category
    for label in reversed(path):
        if label.startswith("core/ple"):
            return "ple"
        if label.startswith("core/hc_"):
            return "hyperconnections"
        if label == "core/moe_total":
            return "moe"
        if label == "core/gdn_total":
            return "gdn_including_projections"
        if label == "core/qsa_total":
            return "qsa_including_projections"
    return category


def trace_summary(path):
    with gzip.open(path, "rt") as stream:
        trace = json.load(stream)
    scopes = defaultdict(list)
    launches = defaultdict(list)
    gpu = []
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        name, cat = event.get("name", ""), event.get("cat", "")
        thread = (event.get("pid"), event.get("tid"))
        if name.startswith(("core/", "spec_cycle/")):
            scopes[thread].append(event)
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in event.get(
            "args", {}
        ):
            launches[thread].append(event)
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append(event)

    correlation = {}
    for thread, rows in launches.items():
        ranges = sorted(scopes[thread], key=lambda e: e["ts"])
        active, endings, cursor = {}, [], 0
        for event in sorted(rows, key=lambda e: e["ts"]):
            stamp = event["ts"]
            while cursor < len(ranges) and ranges[cursor]["ts"] <= stamp:
                scope = ranges[cursor]
                active[cursor] = scope
                heapq.heappush(endings, (scope["ts"] + scope["dur"], cursor))
                cursor += 1
            while endings and endings[0][0] <= stamp:
                _, index = heapq.heappop(endings)
                active.pop(index, None)
            names = [
                s["name"]
                for s in sorted(active.values(), key=lambda s: (s["ts"], -s["dur"]))
            ]
            correlation[event["args"]["correlation"]] = names

    groups = defaultdict(list)
    matched = Counter()
    for event in gpu:
        names = correlation.get(event.get("args", {}).get("correlation"), [])
        if any(name.startswith("spec_cycle/") for name in names) or any(
            name.startswith("core/") and ("decode" in name or "verify" in name)
            for name in names
        ):
            stage = "decode"
        elif any(
            name.startswith("core/target_") or name.startswith("core/draft_")
            for name in names
        ):
            stage = "prefill"
        else:
            stage = "outside_forward"
        matched[stage] += 1
        groups[stage].append((event, names))

    output = {
        "events": len(trace["traceEvents"]),
        "kernel_count_by_stage": dict(matched),
        "stages": {},
    }
    for stage, entries in groups.items():
        sums, semantics, names, counts = Counter(), Counter(), Counter(), Counter()
        intervals = []
        for event, scopes_path in entries:
            duration = event["dur"]
            label = kernel_category(event["name"])
            sums[label] += duration / 1000
            semantics[semantic_category(scopes_path, event["name"])] += duration / 1000
            names[event["name"]] += duration / 1000
            counts[label] += 1
            intervals.append((event["ts"], event["ts"] + duration))
        if not intervals:
            continue
        start, end = min(t[0] for t in intervals), max(t[1] for t in intervals)
        # Include every GPU operation inside this stage's correlated bounds.
        # This captures unannotated metadata operations between model forwards.
        window = [
            (max(start, e["ts"]), min(end, e["ts"] + e["dur"]))
            for e in gpu
            if e["ts"] < end and e["ts"] + e["dur"] > start
        ]
        busy = union_us(window)
        output["stages"][stage] = dict(
            span_ms=(end - start) / 1000,
            busy_ms=busy / 1000,
            traced_idle_fraction=1 - busy / (end - start) if end > start else 0,
            kernel_sum_ms=dict(sums),
            semantic_kernel_sum_ms=dict(semantics),
            kernel_counts=dict(counts),
            top_kernels_ms=dict(names.most_common(20)),
        )
    output["notes"] = [
        "Traced timings include profiler overhead; idle is not an untraced utilization estimate.",
        "Semantic attribution uses CUDA launch correlation and enclosing CPU ranges.",
        "Kernel duration sums can overlap across streams; they are not additive wall time.",
        "Decode CUDA graph kernels use name-based categories; generic kernels can remain other.",
    ]
    return output


def summarize(run):
    capture = run / "capture"
    rows = read_rows(capture / "requests.jsonl")
    result = {
        "requests": [
            {
                k: r[k]
                for k in [
                    "case_id",
                    "label",
                    "input_tokens",
                    "output_tokens",
                    "ttft_s",
                    "end_to_end_s",
                    "decode_tok_s",
                    "observed_cycle_ms",
                    "observed_tokens_per_cycle",
                ]
            }
            for r in rows
        ]
    }
    counters = []
    for path in capture.glob("dcgm_*.jsonl"):
        samples = []
        for row in read_rows(path):
            fields = row["raw"].split()
            if len(fields) < 2 or fields[0] != "GPU":
                continue
            if len(fields[2:]) != len(row["fields"]):
                continue
            values = {}
            for key, value in zip(row["fields"], fields[2:]):
                try:
                    value = float(value)
                    if abs(value) < 1e18:
                        values[str(key)] = value
                except ValueError:
                    pass
            samples.append(
                dict(unix_s=row["unix_s"], gpu=int(fields[1]), values=values)
            )
        for request in [r for r in rows if r["label"] == path.stem]:
            for phase, begin, end in [
                (
                    "prefill",
                    request["start_unix_s"],
                    request["start_unix_s"] + request["ttft_s"],
                ),
                (
                    "decode",
                    request["start_unix_s"] + request["ttft_s"],
                    request["start_unix_s"] + request["end_to_end_s"],
                ),
            ]:
                for gpu in range(4):
                    # Host-engine updates and CLI reads each run at ~1 Hz.
                    # Received values may span the previous phase for nearly
                    # two seconds, so discard that boundary conservatively.
                    chosen = [
                        s
                        for s in samples
                        if s["gpu"] == gpu and begin + 2.2 <= s["unix_s"] <= end - 0.2
                    ]
                    metric_values = defaultdict(list)
                    for sample in chosen:
                        for field, value in sample["values"].items():
                            metric_values[field].append(value)
                    counters.append(
                        dict(
                            case_id=request["case_id"],
                            pass_name=request["label"],
                            phase=phase,
                            gpu=gpu,
                            samples=len(chosen),
                            mean={
                                k: statistics.mean(v) for k, v in metric_values.items()
                            },
                        )
                    )
    result["dcgm"] = counters
    result["dcgm_fields"] = {
        "1001": "graphics_engine_active",
        "1002": "sm_active",
        "1003": "sm_occupancy",
        "1004": "tensor_pipe_active",
        "1005": "dram_active",
        "1009": "pcie_tx_bytes_s",
        "1010": "pcie_rx_bytes_s",
        "1011": "nvlink_tx_bytes_s",
        "1012": "nvlink_rx_bytes_s",
    }
    result["dcgm_notes"] = [
        "SM/occupancy and Tensor Core counters were collected in separate passes to avoid V100 multiplexing.",
        "These are time-averaged activity fractions, not achieved FLOP/s or DRAM bytes/s.",
        "Short windows with no complete sampling interval are left empty.",
    ]

    nvml = read_rows(capture / "nvml.jsonl")
    telemetry = []
    for request in rows:
        if request["label"].startswith("trace") or request["label"] == "warmup":
            continue
        begin = request["start_unix_s"]
        for phase, left, right in [
            ("prefill", begin, begin + request["ttft_s"]),
            ("decode", begin + request["ttft_s"], begin + request["end_to_end_s"]),
        ]:
            selected = [s for s in nvml if left + 0.2 <= s["unix_s"] <= right]
            if not selected:
                continue
            for gpu in range(4):
                values = [s["devices"][gpu] for s in selected]
                telemetry.append(
                    dict(
                        case_id=request["case_id"],
                        label=request["label"],
                        phase=phase,
                        gpu=gpu,
                        median_sm_mhz=statistics.median(v["sm_mhz"] for v in values),
                        mean_power_w=statistics.mean(v["power_w"] for v in values),
                        min_free_mib=min(v["free_bytes"] for v in values) / 2**20,
                        throttle_reasons=sorted(
                            {v["throttle_reasons"] for v in values}
                        ),
                    )
                )
    result["telemetry"] = telemetry
    result["traces"] = {}
    for path in sorted(capture.glob("trace_gpu*/*TP-*.trace.json.gz")):
        print("Analyzing", path.name, flush=True)
        result["traces"][str(path.relative_to(run))] = trace_summary(path)
        gc.collect()

    phases = defaultdict(list)
    request_labels = {r["request_id"]: r["label"] + "_" + r["case_id"] for r in rows}
    for path in (run / "events").glob("cycles_*.jsonl"):
        for row in read_rows(path):
            label = request_labels.get(row["request_ids"][0], "unknown")
            phases[label].append(row)
    result["cycles"] = {}
    for label, values in phases.items():
        fields = defaultdict(list)
        for row in values:
            fields["cycle_gpu_ms"].append(row["cycle_gpu_ms"])
            fields["cpu_submit_ms"].append(row["cpu_submit_ms"])
            for k, v in row["phases_ms"].items():
                fields[k].append(v)
        result["cycles"][label] = dict(
            rounds=len(values),
            median_ms={k: statistics.median(v) for k, v in fields.items()},
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
