"""Collect a separate diagnostic request, Chrome traces and optional phase events."""

import argparse
import json
from pathlib import Path

import requests
from bench import measure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8082")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--lookup", choices=["on", "off"], default="off")
    args = p.parse_args()
    case = next(
        c for c in json.loads(args.corpus.read_text())["cases"] if c["id"] == args.case
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.label}.json"
    if output.exists():
        raise FileExistsError(output)
    # Warm outside the profile. The recorded request is cold-cache again.
    measure(args.url, case, min(64, args.output_tokens), lookup=args.lookup == "on")
    response = requests.post(
        args.url + "/start_profile",
        json=dict(
            output_dir=str(args.output_dir.resolve()),
            activities=["CPU", "GPU"],
            with_stack=False,
            record_shapes=False,
            profile_prefix=args.label,
        ),
        timeout=120,
    )
    response.raise_for_status()
    try:
        row = measure(args.url, case, args.output_tokens, lookup=args.lookup == "on")
        row.update(diagnostic=True, label=args.label)
        output.write_text(json.dumps(row, indent=2) + "\n")
    finally:
        response = requests.post(args.url + "/stop_profile", timeout=300)
        response.raise_for_status()
    # The next forward lets the event probe drain the last completed round.
    response = requests.post(
        args.url + "/generate",
        json=dict(
            text="2 + 2 =", sampling_params=dict(temperature=0, max_new_tokens=1)
        ),
        timeout=120,
    )
    response.raise_for_status()
    print(output)


if __name__ == "__main__":
    main()
