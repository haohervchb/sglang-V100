"""Screen existing SM70 MoE geometries on locally captured model tensors.

The benchmark clones the Python orchestration with one explicit routing-block
override. It uses the same native kernels, activation and reduction as serving;
it does not edit the production selector or rebuild the Marlin extension.
"""

import argparse
import ast
import copy
import gc
import importlib
import itertools
import json
import os
import random
import statistics
from pathlib import Path

import torch
from hardware_calibration import timed_graph


def prepare_function():
    module = importlib.import_module(
        "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe"
    )
    marlin = importlib.import_module("sglang.jit_kernel.moe_wna16_marlin")
    marlin._sm70_marlin_user_tuning = True
    tree = ast.parse(Path(module.__file__).read_text())
    node = copy.deepcopy(
        next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "fused_marlin_moe"
        )
    )
    node.decorator_list = []
    found = 0
    for i, statement in enumerate(node.body):
        if (
            isinstance(statement, ast.For)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "block_size_m"
        ):
            replacement = ast.parse("block_size_m = _TUNE_BLOCK").body[0]
            node.body[i] = ast.copy_location(replacement, statement)
            found += 1
    assert found == 1, "Routing heuristic changed; recheck the benchmark adapter"
    control = {}
    original_gemm = module.moe_wna16_marlin_gemm
    names = [
        "SM70_MARLIN_MOE_CTA_GEOMETRY",
        "SM70_MARLIN_MOE_SPLIT_K",
        "SM70_MARLIN_MOE_METADATA_CACHE",
    ]

    def configured_gemm(*args, **kwargs):
        stage = "up" if kwargs["top_k"] == 10 else "down"
        geometry = control.get(stage)
        values = [geometry, "1" if geometry else None, control.get("metadata")]
        for key, value in zip(names, values):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return original_gemm(*args, **kwargs)

    namespace = dict(vars(module))
    namespace.update(_TUNE_BLOCK=48, moe_wna16_marlin_gemm=configured_gemm)
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(code, "<moe-prefill-benchmark>", "exec"), namespace)
    return namespace["fused_marlin_moe"], namespace, control, module.fused_marlin_moe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    function, namespace, control, serving_function = prepare_function()
    up = [
        "32x64x64x4x32x32x32",
        "32x64x128x4x32x64x32",
        "64x64x32x4x32x32x32",
        "64x64x64x4x64x32x32",
    ]
    down = ["32x256x32x4x32x64x32", "64x256x32x4x64x64x32", "64x128x32x4x64x64x16"]
    variants = [dict(block=b, up=None, down=None, metadata=None) for b in [32, 48, 64]]
    variants += [
        dict(block=b, up=u, down=d, metadata="vector_words")
        for b, u, d in itertools.product([32, 48, 64], up, down)
    ]
    variants += [dict(block=48, up=None, down=None, metadata="vector_words")]
    baseline = dict(block=48, up=None, down=None, metadata=None)
    records = []
    for path in sorted(args.captures.glob("*.pt")):
        cpu = torch.load(path, map_location="cpu", weights_only=True)
        kwargs = {
            k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in cpu.items()
        }
        del cpu
        assert kwargs["hidden_states"].shape == (2048, 2560)
        for name in [
            "SM70_MARLIN_MOE_CTA_GEOMETRY",
            "SM70_MARLIN_MOE_SPLIT_K",
            "SM70_MARLIN_MOE_METADATA_CACHE",
        ]:
            os.environ.pop(name, None)
        reference = serving_function(**kwargs).clone()
        namespace["_TUNE_BLOCK"] = 48
        control.clear()
        control.update(baseline)
        torch.testing.assert_close(function(**kwargs), reference, rtol=0, atol=0)
        order = variants.copy()
        random.Random(731 + int(path.stem[-2:])).shuffle(order)
        # Repeated controls expose clock/thermal drift rather than relying on
        # a single fastest measurement at the start of a sweep.
        schedule = []
        for i, variant in enumerate(order):
            if i % 8 == 0:
                schedule.append((True, baseline))
            schedule.append((False, variant))
        schedule.append((True, baseline))
        for is_control, variant in schedule:
            control.clear()
            control.update(variant)
            namespace["_TUNE_BLOCK"] = variant["block"]
            row = dict(layer=path.stem, control=is_control, variant=variant)
            try:
                output = function(**kwargs)
                torch.cuda.synchronize()
                error = output.float() - reference.float()
                row["relative_l2"] = (
                    torch.linalg.vector_norm(error)
                    / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
                ).item()
                row["max_abs_error"] = error.abs().max().item()
                torch.testing.assert_close(output, reference, rtol=0.005, atol=0.002)
                assert row["relative_l2"] < 0.0015
                row["passed"] = True
                row["ms"] = timed_graph(
                    lambda _, case_kwargs=kwargs: function(**case_kwargs),
                    operations=2,
                    replays=3,
                    samples=3,
                )
                del output, error
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                if "illegal memory" in str(exc).lower():
                    raise
            records.append(row)
            with args.output.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k != "error"}), flush=True
            )
        del kwargs, reference
        gc.collect()
        torch.cuda.empty_cache()

    baseline_times = {
        layer: statistics.median(
            r["ms"]
            for r in records
            if r["layer"] == layer and r["control"] and r["passed"]
        )
        for layer in {r["layer"] for r in records}
    }
    summary = []
    for variant in variants:
        rows = [r for r in records if not r["control"] and r["variant"] == variant]
        if len(rows) != len(baseline_times) or not all(r["passed"] for r in rows):
            continue
        gains = [baseline_times[r["layer"]] / r["ms"] for r in rows]
        summary.append(
            dict(
                variant=variant,
                speedup=statistics.geometric_mean(gains),
                min_speedup=min(gains),
                max_relative_l2=max(r["relative_l2"] for r in rows),
            )
        )
    summary.sort(key=lambda row: -row["speedup"])
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(dict(baseline_ms=baseline_times, ranked=summary), indent=2) + "\n"
    )
    print("BEST", json.dumps(summary[:5]), flush=True)


if __name__ == "__main__":
    main()
