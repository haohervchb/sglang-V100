"""Interleave prefill-only candidates in a dedicated server; MTP graph is fixed."""

import argparse
import json
import time
from pathlib import Path

import requests
from bench import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8083")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument(
        "--compact-experiment",
        action="store_true",
        help="Reproduce the archived compact-state candidate after applying its patch",
    )
    args = parser.parse_args()
    if args.compact_experiment:
        source = (
            Path(__file__).resolve().parents[2]
            / "python/sglang/jit_kernel/csrc/sm70_longctx_decode.cu"
        )
        if "sm70_qsa_prefill_compact" not in source.read_text():
            parser.error(
                "Apply the archived rejected_compact.patch before using --compact-experiment"
            )
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = {c["id"]: c for c in json.loads(args.corpus.read_text())["cases"]}
    selected = [
        "prose_1000",
        "prose_8192",
        "prose_25000",
        "prose_70000",
        "code_edit_8192",
    ]
    modes = (
        [("gate_moe", True, False, False), ("gate_moe_grouped", True, False, True)]
        if args.grouped
        else [
            ("gate_only", False, False, False),
            ("gate_moe", True, False, False),
            (
                "gate_qsa" if args.compact_experiment else "gate_grouped",
                False,
                args.compact_experiment,
                not args.compact_experiment,
            ),
            ("combined", True, args.compact_experiment, not args.compact_experiment),
        ]
    )
    start = time.monotonic()
    while True:
        try:
            if requests.get(args.url + "/health", timeout=3).status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.monotonic() - start > 1500:
            raise TimeoutError("Experiment server startup timeout")
        time.sleep(5)
    info = requests.get(args.url + "/get_server_info", timeout=30)
    info.raise_for_status()
    args.output.with_suffix(".server.json").write_text(
        json.dumps(info.json(), indent=2) + "\n"
    )
    with args.output.open("x") as stream:
        for repeat in range(-1, args.repeats):
            for name in selected if repeat % 2 == 0 else reversed(selected):
                offset = (max(repeat, 0) // 2) % len(modes)
                order = modes[offset:] + modes[:offset]
                if repeat % 2:
                    order = list(reversed(order))
                for label, moe, qsa, grouped in order:
                    temp = args.control.with_suffix(".tmp")
                    temp.write_text(json.dumps(dict(moe=moe, qsa=qsa, grouped=grouped)))
                    temp.replace(args.control)
                    row = measure(args.url, cases[name], 32 if repeat < 0 else 512)
                    row.update(
                        label=label,
                        repeat=repeat,
                        warmup=repeat < 0,
                        moe_2k=moe,
                        qsa_compact=qsa,
                        qsa_grouped=grouped,
                        shared_gate=True,
                    )
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in [
                                    "case_id",
                                    "label",
                                    "repeat",
                                    "ttft_s",
                                    "decode_tok_s",
                                    "observed_cycle_ms",
                                    "end_to_end_s",
                                ]
                            }
                        ),
                        flush=True,
                    )
        for name in ["prose_160000"]:
            for label, moe, qsa, grouped in [modes[0], modes[-1], modes[-1], modes[0]]:
                temp = args.control.with_suffix(".tmp")
                temp.write_text(json.dumps(dict(moe=moe, qsa=qsa, grouped=grouped)))
                temp.replace(args.control)
                row = measure(args.url, cases[name], 512)
                row.update(
                    label=label,
                    repeat=0,
                    warmup=False,
                    moe_2k=moe,
                    qsa_compact=qsa,
                    qsa_grouped=grouped,
                    shared_gate=True,
                )
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in [
                                "case_id",
                                "label",
                                "ttft_s",
                                "decode_tok_s",
                                "end_to_end_s",
                            ]
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
