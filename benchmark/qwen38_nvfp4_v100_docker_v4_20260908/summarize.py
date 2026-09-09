"""Compare complete Docker and host sweeps without dropping slow requests."""

import argparse
import json
import statistics
from pathlib import Path


def read_rows(root, tag):
    rows = [
        json.loads(line)
        for line in (root / f"{tag}_measurements.jsonl").read_text().splitlines()
    ]
    for row in rows:
        counts, times, meta = row["completion_tokens"], row["times_s"], row["meta_info"]
        assert counts[-1] == row["output_len"]
        assert meta["prompt_tokens"] == row["input_len"] and meta["cached_tokens"] == 0
        rate = (counts[-1] - counts[0]) / (times[-1] - times[0])
        assert abs(rate - row["decode_tok_s"]) < 1e-8
        if "spec_accept_length" in meta:
            row["round_ms"] = 1000 * meta["spec_accept_length"] / rate
    return [row for row in rows if not row["warmup"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    root = args.output_dir
    summary = dict(tolerance_fraction=args.tolerance, comparisons=[], checks=[])
    for mode, suffix in [("mtp", "natural"), ("mtp", "random"), ("target", "random")]:
        host = read_rows(root, f"host_{mode}_{suffix}")
        docker = read_rows(root, f"docker_{mode}_{suffix}")
        expected_lengths = (
            {1000, 8192, 25000, 70000} if mode == "mtp" else {1000, 25000}
        )
        expected_output = 1024 if mode == "mtp" else 2048
        for rows in (host, docker):
            assert {r["input_len"] for r in rows} == expected_lengths
            assert len(rows) == 2 * len(expected_lengths)
            assert all(r["output_len"] == expected_output for r in rows)
        assert [(r["input_len"], r["seed"], r["output_len"]) for r in host] == [
            (r["input_len"], r["seed"], r["output_len"]) for r in docker
        ]
        for length in sorted({r["input_len"] for r in host}):
            a, b = [r for r in host if r["input_len"] == length], [
                r for r in docker if r["input_len"] == length
            ]
            assert len(a) == len(b) == 2
            row = dict(
                mode=mode,
                scenario=suffix,
                input_len=length,
                host={},
                docker={},
                changes_percent={},
            )
            metrics = ["prefill_tok_s", "decode_tok_s"] + (
                ["round_ms"] if mode == "mtp" else []
            )
            for metric in metrics:
                av, bv = [r[metric] for r in a], [r[metric] for r in b]
                row["host"][metric] = av
                row["docker"][metric] = bv
                delta = statistics.mean(bv) / statistics.mean(av) - 1
                row["changes_percent"][metric] = 100 * delta
                # Random output rates depend strongly on accepted drafts. Report
                # every rate, but use round cost for its runtime comparison.
                if not (
                    mode == "mtp" and suffix == "random" and metric == "decode_tok_s"
                ):
                    summary["checks"].append(
                        dict(
                            name=f"{mode}_{suffix}_{length}_{metric}",
                            passed=abs(delta) <= args.tolerance,
                            delta_percent=100 * delta,
                            regression=(
                                delta > args.tolerance
                                if metric == "round_ms"
                                else delta < -args.tolerance
                            ),
                        )
                    )
            row["host"]["lowest_window_tok_s"] = min(
                w["tok_s"] for r in a for w in r["windows"]
            )
            row["docker"]["lowest_window_tok_s"] = min(
                w["tok_s"] for r in b for w in r["windows"]
            )
            summary["comparisons"].append(row)
            print(
                mode,
                suffix,
                length,
                {k: round(v, 2) for k, v in row["changes_percent"].items()},
            )

    config_fields = [
        "model_path",
        "dtype",
        "quantization",
        "kv_cache_dtype",
        "tp_size",
        "context_length",
        "max_running_requests",
        "mem_fraction_static",
        "chunked_prefill_size",
        "cuda_graph_bs",
        "mamba_scheduler_strategy",
        "mamba_full_memory_ratio",
        "attention_backend",
        "linear_attn_prefill_backend",
        "linear_attn_decode_backend",
        "speculative_algorithm",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
    ]
    for mode in ["mtp", "target"]:
        configs = [
            json.loads((root / f"{runtime}_{mode}_random_server_info.json").read_text())
            for runtime in ["host", "docker"]
        ]
        assert all(all(k in config for k in config_fields) for config in configs)
        differences = [
            k for k in config_fields if configs[0].get(k) != configs[1].get(k)
        ]
        summary["checks"].append(
            dict(
                name=f"{mode}_server_config",
                passed=not differences,
                differences=differences,
            )
        )
        for runtime in ["host", "docker"]:
            status = json.loads((root / f"{runtime}_{mode}_done.json").read_text())
            summary["checks"].append(
                dict(
                    name=f"{runtime}_{mode}_completed_and_responses",
                    passed=all(status.values()),
                    details=status,
                )
            )
    host_env = json.loads((root / "host_environment.json").read_text())
    docker_env = json.loads((root / "docker_environment.json").read_text())
    summary["checks"].append(
        dict(
            name="application_source_matches",
            passed=(
                host_env["source_tree_sha256"] == docker_env["source_tree_sha256"]
                and not docker_env["mismatches"]
            ),
            files_checked=docker_env["files_checked"],
        )
    )
    summary["passed"] = all(c["passed"] for c in summary["checks"])
    # Preserve the originally specified symmetric check. Also distinguish
    # faster Docker measurements from regressions; do not relabel an
    # out-of-tolerance improvement as an exact parity pass.
    summary["no_regression_over_tolerance"] = all(
        not c["regression"] if "regression" in c else c["passed"]
        for c in summary["checks"]
    )
    (root / "parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PARITY", "PASS" if summary["passed"] else "REQUIRES INVESTIGATION")
    print("NO REGRESSIONS", summary["no_regression_over_tolerance"])
    for check in summary["checks"]:
        if not check["passed"]:
            print(check)


if __name__ == "__main__":
    main()
