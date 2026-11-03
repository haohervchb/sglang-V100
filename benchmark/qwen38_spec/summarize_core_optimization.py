"""Compare fixed-output request medians and retain acceptance alongside speed."""

import argparse
import json
import statistics
from pathlib import Path


def rows(path):
    return [
        r for r in map(json.loads, path.read_text().splitlines()) if not r.get("warmup")
    ]


def compare(base, candidate):
    result = []
    for name in sorted({r["case_id"] for r in candidate}):
        a = [r for r in base if r["case_id"] == name]
        b = [r for r in candidate if r["case_id"] == name]
        if not a or not b:
            continue
        for field in ["input_sha256", "temperature", "output_tokens"]:
            assert len({r[field] for r in a + b}) == 1, (name, field)
        stats = {}
        for field in [
            "ttft_s",
            "end_to_end_s",
            "decode_tok_s",
            "observed_cycle_ms",
            "observed_tokens_per_cycle",
            "actual_output_tokens",
        ]:
            av = [r[field] for r in a if r.get(field) is not None]
            bv = [r[field] for r in b if r.get(field) is not None]
            if av and bv:
                x, y = statistics.median(av), statistics.median(bv)
                stats[field] = dict(
                    baseline=x, candidate=y, relative_change_pct=100 * (y / x - 1)
                )
        result.append(
            dict(
                case=name,
                baseline_samples=len(a),
                candidate_samples=len(b),
                metrics=stats,
            )
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.run
    base = rows(root / "baseline_post.jsonl")
    candidate = [
        r for r in rows(root / "grouped_ab.jsonl") if r["label"] == "gate_moe_grouped"
    ]
    pairs = compare(base, candidate)
    result = dict(
        fixed_output=pairs,
        baseline_drift=compare(rows(root / "baseline_pre.jsonl"), base),
    )
    baseline_total = sum(p["metrics"]["end_to_end_s"]["baseline"] for p in pairs)
    candidate_total = sum(p["metrics"]["end_to_end_s"]["candidate"] for p in pairs)
    result["equal_case_latency_sum"] = dict(
        baseline_s=baseline_total,
        candidate_s=candidate_total,
        latency_reduction_pct=100 * (1 - candidate_total / baseline_total),
        note="Sum of one median request per case; not aggregate serving throughput.",
    )
    for suffix in ["speed", "speed_sampled"]:
        result[suffix] = compare(
            rows(root / f"baseline_{suffix}.jsonl"),
            rows(root / f"candidate_{suffix}.jsonl"),
        )
    result["speed_notes"] = (
        "Natural-EOS outputs can differ in length. These latency figures are "
        "not fixed-work comparisons; inspect actual_output_tokens and acceptance."
    )
    result["checks"] = json.loads(
        (root / "candidate_validation_status.json").read_text()
    )
    result["queue"] = json.loads((root / "candidate_queue/summary.json").read_text())
    cached = json.loads((root / "candidate_cached_document.json").read_text())
    result["prefix_reuse"] = dict(
        passed=cached["passed"], cached_tokens=cached["cached_tokens"]
    )
    result["multimodal"] = {}
    for label in ["baseline", "candidate"]:
        mm = json.loads((root / f"multimodal/{label}_results.json").read_text())
        result["multimodal"][label] = dict(
            passed=sum(c["passed"] for c in mm["cases"]),
            total=len(mm["cases"]),
            failed=[c["name"] for c in mm["cases"] if not c["passed"]],
        )
    assert set(result["multimodal"]["candidate"]["failed"]) <= set(
        result["multimodal"]["baseline"]["failed"]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for p in pairs:
        print(p["case"], p["metrics"], flush=True)
    print(result["equal_case_latency_sum"], flush=True)


if __name__ == "__main__":
    main()
