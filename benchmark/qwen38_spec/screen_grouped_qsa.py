"""Time grouped Page4 planning plus attention on locally captured QSA inputs."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from hardware_calibration import timed_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import (
        sm70_cuda_qsa_prefill,
    )
    from sglang.srt.layers.attention.tilelang_fa_v100._grouped_qsa import (
        grouped_qsa_prefill,
        load_grouped_qsa_ops,
    )

    assert load_grouped_qsa_ops() is not None
    os.environ["SGLANG_SM70_QSA_COMPACT_PREFILL"] = "0"
    os.environ["SGLANG_SM70_QSA_GROUPED_PREFILL"] = "0"
    results = []
    for path in sorted(args.captures.glob("*.pt")):
        d = torch.load(path, weights_only=True, map_location="cpu")
        tensors = {
            key: value.cuda()
            for key, value in d.items()
            if isinstance(value, torch.Tensor)
        }
        params = [
            tensors[k] for k in ["q", "k", "v", "table", "req", "indices", "lens"]
        ] + [d["scale"]]
        reference = sm70_cuda_qsa_prefill(*params)
        torch.testing.assert_close(reference, tensors["output"], rtol=0, atol=0)
        output = grouped_qsa_prefill(*params)
        torch.testing.assert_close(
            output.float(), reference.float(), rtol=0.005, atol=0.001
        )
        error = (
            (output.float() - reference.float()).norm() / reference.float().norm()
        ).item()
        assert error < 0.001
        base = timed_graph(
            lambda _, a=params: sm70_cuda_qsa_prefill(*a), operations=2, replays=3
        )
        grouped = timed_graph(
            lambda _, a=params: grouped_qsa_prefill(*a), operations=2, replays=3
        )
        row = dict(
            case=path.stem,
            baseline_ms=base,
            grouped_ms=grouped,
            speedup=base / grouped,
            relative_l2=error,
        )
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(row, flush=True)
        del tensors, params, reference, output, d
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
