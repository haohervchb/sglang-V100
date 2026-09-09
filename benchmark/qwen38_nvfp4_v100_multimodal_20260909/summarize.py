"""Compare the multimodal sweeps with the pre-optimization main baseline."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    tags = ["main_target", "host_mtp", "docker_mm_target", "docker_mm_mtp"]
    reports = {
        tag: json.loads((root / f"{tag}_results.json").read_text()) for tag in tags
    }
    baseline = reports["main_target"]
    assert len(baseline["cases"]) == 12
    result = {
        "baseline_failures": [c["name"] for c in baseline["cases"] if not c["passed"]],
        "all_ground_truth_checks_passed": all(r["passed"] for r in reports.values()),
        "comparisons": {},
    }
    for tag, report in reports.items():
        assert report["fixtures"] == baseline["fixtures"], tag
        assert len(report["cases"]) == 12 and report["health_status"] == 200, tag
        rows = []
        for expected, actual in zip(baseline["cases"], report["cases"], strict=True):
            assert actual["name"] == expected["name"]
            assert actual["request"] == expected["request"], actual["name"]
            assert actual["status_code"] == 200 and actual["finish_reason"] == "stop"
            rows.append(
                {
                    "case": actual["name"],
                    "ground_truth_passed": actual["passed"],
                    "baseline_ground_truth_passed": expected["passed"],
                    "text_matches_baseline": actual["text"] == expected["text"],
                    "no_regression": actual["passed"]
                    or (not expected["passed"] and actual["text"] == expected["text"]),
                }
            )
        result["comparisons"][tag] = rows
    result["regression_gate_passed"] = all(
        c["no_regression"] for rows in result["comparisons"].values() for c in rows
    )
    result["text_throughput_recheck"] = []
    previous = Path(__file__).resolve().parent.parent / (
        "qwen38_nvfp4_v100_docker_v4_20260908/host_mtp_natural_measurements.jsonl"
    )
    old = [json.loads(line) for line in previous.read_text().splitlines()]
    new = [
        json.loads(line)
        for line in (root / "docker_mm_mtp_perf_measurements.jsonl")
        .read_text()
        .splitlines()
    ]
    for length in (1000, 25000):
        a, b = [
            [r for r in rows if not r["warmup"] and r["input_len"] == length]
            for rows in (old, new)
        ]
        assert len(a) == len(b) == 2
        for row in a + b:
            assert row["output_len"] == row["completion_tokens"][-1] == 1024
            assert row["meta_info"]["cached_tokens"] == 0
        av, bv = [statistics.mean(r["decode_tok_s"] for r in rows) for rows in (a, b)]
        result["text_throughput_recheck"].append(
            {
                "input_len": length,
                "previous_host_decode_tok_s": av,
                "docker_decode_tok_s": bv,
                "change_percent": 100 * (bv / av - 1),
                "no_regression_over_five_percent": bv >= av * 0.95,
            }
        )
    result["regression_gate_passed"] &= all(
        row["no_regression_over_five_percent"]
        for row in result["text_throughput_recheck"]
    )
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))
    raise SystemExit(0 if result["regression_gate_passed"] else 1)


if __name__ == "__main__":
    main()
