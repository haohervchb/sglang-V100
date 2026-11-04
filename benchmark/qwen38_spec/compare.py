"""Pair per-prompt medians rather than comparing unrelated headline rates."""

import argparse
import json
import math
import statistics
from pathlib import Path


def observations(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if not r.get("warmup") and not r.get("diagnostic")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", default="e5m2_4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {}
    exclusions_path = args.directory / "exclusions.json"
    exclusions = (
        json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
    )
    if args.baseline in exclusions:
        raise ValueError("The selected baseline is excluded")
    for path in sorted(args.directory.glob("*.jsonl")):
        label = path.stem
        suffix = (
            "_speed_sampled"
            if label.endswith("_speed_sampled")
            else "_speed"
            if label.endswith("_speed")
            else ""
        )
        control_path = args.directory / (args.baseline + suffix + ".jsonl")
        root_label = label[: -len(suffix)] if suffix else label
        if root_label in exclusions:
            continue
        if path == control_path or not control_path.exists():
            continue
        rows, control = observations(path), observations(control_path)
        if not rows:
            continue
        pairs = []
        for case in sorted({r["case_id"] for r in rows}):
            a = [r for r in control if r["case_id"] == case and r.get("decode_tok_s")]
            b = [r for r in rows if r["case_id"] == case and r.get("decode_tok_s")]
            if not a or not b:
                continue
            for key in ["input_sha256", "temperature", "output_tokens"]:
                if {r[key] for r in a + b} != {a[0][key]}:
                    raise ValueError(f"Unmatched {key} in {case}")
            base = statistics.median(r["decode_tok_s"] for r in a)
            candidate = statistics.median(r["decode_tok_s"] for r in b)
            pairs.append(
                dict(
                    case=case,
                    kind=b[0]["kind"],
                    control=base,
                    candidate=candidate,
                    ratio=candidate / base,
                )
            )
        if pairs:
            result[label] = dict(
                control=control_path.stem,
                pairs=pairs,
                geometric_mean_ratio=math.exp(
                    statistics.mean(math.log(p["ratio"]) for p in pairs)
                ),
                domains={
                    kind: math.exp(
                        statistics.mean(
                            math.log(p["ratio"]) for p in pairs if p["kind"] == kind
                        )
                    )
                    for kind in sorted({p["kind"] for p in pairs})
                },
            )
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
