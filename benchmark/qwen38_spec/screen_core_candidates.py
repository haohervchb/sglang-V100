"""Validate production dispatch and time isolated gate/MoE/QSA candidates."""

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
    parser.add_argument("--compact-experiment", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(gate=[], moe=[], qsa=[], complete=False)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    from sglang.jit_kernel.sm70_qwen_fusions import gate

    torch.manual_seed(731)
    for rows in [1, 2, 4]:
        x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
        weight = torch.randn(1, 2560, device="cuda", dtype=torch.float16) * 0.01
        value = torch.randn_like(x)
        reference = torch.sigmoid(torch.nn.functional.linear(x, weight)) * value
        output = gate(x, weight, value)
        torch.testing.assert_close(output, reference, rtol=0.002, atol=0.001)
        base = timed_graph(
            lambda _, a=x, b=weight, v=value: (
                torch.sigmoid(torch.nn.functional.linear(a, b)) * v
            ),
            operations=32,
            replays=16,
        )
        fused = timed_graph(
            lambda _, a=x, b=weight, v=value: gate(a, b, v), operations=32, replays=16
        )
        row = dict(
            rows=rows,
            torch_ms=base,
            fused_ms=fused,
            speedup=base / fused,
            max_abs_error=(output - reference).abs().max().item(),
        )
        result["gate"].append(row)
        save()
        print("gate", row, flush=True)
    del x, weight, value, reference, output

    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe

    for path in sorted(args.captures.glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=True)
        case = {
            key: (value.cuda() if isinstance(value, torch.Tensor) else value)
            for key, value in data.items()
        }
        del data
        os.environ["SGLANG_SM70_MOE_PREFILL_2K"] = "0"
        reference = fused_marlin_moe(**case)
        os.environ["SGLANG_SM70_MOE_PREFILL_2K"] = "1"
        output = fused_marlin_moe(**case)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            os.environ["SGLANG_SM70_MOE_PREFILL_2K"] = "0"
            old = fused_marlin_moe(**case)
            os.environ["SGLANG_SM70_MOE_PREFILL_2K"] = "1"
            new = fused_marlin_moe(**case)
        original_x = case["hidden_states"].clone()
        for scale in [1.0, -0.5, 0.0]:
            case["hidden_states"].copy_(original_x * scale)
            graph.replay()
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        case["hidden_states"].copy_(original_x)
        os.environ["SGLANG_SM70_MOE_PREFILL_2K"] = "0"
        base = timed_graph(
            lambda _, params=case: fused_marlin_moe(**params), operations=2, replays=4
        )
        os.environ["SGLANG_SM70_MOE_PREFILL_2K"] = "1"
        tuned = timed_graph(
            lambda _, params=case: fused_marlin_moe(**params), operations=2, replays=4
        )
        row = dict(
            layer=path.stem,
            baseline_ms=base,
            tuned_ms=tuned,
            speedup=base / tuned,
            bit_equal=True,
        )
        result["moe"].append(row)
        save()
        print("moe", row, flush=True)
        del case, reference, output, old, new, graph, original_x
        gc.collect()
        torch.cuda.empty_cache()

    if not args.compact_experiment:
        result["complete"] = True
        save()
        return
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/jit_kernel/csrc/sm70_longctx_decode.cu"
    )
    if "sm70_qsa_prefill_compact" not in source.read_text():
        raise RuntimeError(
            "Apply the archived compact-state patch before screening that experiment"
        )
    from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import (
        sm70_cuda_qsa_prefill,
    )

    for prefix in [25000, 70000]:
        rows, topk = 2048, 2048
        for dtype in [torch.float16, torch.float8_e5m2]:
            length = prefix + rows
            q = torch.randn(rows, 6, 256, device="cuda", dtype=torch.float16)
            k = torch.randn(length, 1, 256, device="cuda").to(dtype)
            v = torch.randn(length, 1, 256, device="cuda").to(dtype)
            table = torch.randperm(length, device="cuda", dtype=torch.int32)[None]
            req = torch.zeros(1, device="cuda", dtype=torch.int32)
            lens = torch.tensor([length], device="cuda", dtype=torch.int32)
            for pattern in ["shared", "shifted_groups"]:
                shift = (
                    torch.zeros(rows, device="cuda", dtype=torch.int32)
                    if pattern == "shared"
                    else torch.arange(rows, device="cuda", dtype=torch.int32) // 4 * 64
                )
                indices = (
                    (
                        torch.arange(topk, device="cuda", dtype=torch.int32)[None, :]
                        + shift[:, None]
                    )
                    % prefix
                ).contiguous()

                def run():
                    return sm70_cuda_qsa_prefill(
                        q, k, v, table, req, indices, lens, 0.0625
                    )

                os.environ["SGLANG_SM70_QSA_COMPACT_PREFILL"] = "0"
                reference = run()
                base = timed_graph(lambda _: run(), operations=2, replays=4)
                os.environ["SGLANG_SM70_QSA_COMPACT_PREFILL"] = "1"
                output = run()
                torch.testing.assert_close(
                    output.view(torch.int16),
                    reference.view(torch.int16),
                    rtol=0,
                    atol=0,
                )
                compact = timed_graph(lambda _: run(), operations=2, replays=4)
                row = dict(
                    prefix=prefix,
                    rows=rows,
                    dtype=str(dtype),
                    pattern=pattern,
                    baseline_ms=base,
                    compact_ms=compact,
                    speedup=base / compact,
                    bit_equal=True,
                )
                result["qsa"].append(row)
                save()
                print("qsa", row, flush=True)
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
