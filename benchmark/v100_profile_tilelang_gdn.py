#!/usr/bin/env python3
# ruff: noqa: B023, BLE001, C408
"""Profile independent TileLang GDN schedules for the Qwen3.8 TP4 shape."""

from __future__ import annotations

import argparse
import os
import statistics
from itertools import product

import torch
from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.linear.kernels.gdn_chunked_tilelang import (
    _get_chunk_forward,
    _get_kkt_inverse,
    _get_packed_gate_cumsum,
    _get_packed_qk_norm,
    _get_packed_v_copy,
    packed_chunked_gdn_sm70,
    packed_recurrent_gdn_sm70,
)


def timed(
    fn, state, initial_state, *, label: str, warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        state.copy_(initial_state)
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        state.copy_(initial_state)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.cuda.nvtx.range(label):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 1024, 1280, 1536, 2048, 4096, 8192],
    )
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument(
        "--only", choices=("all", "recurrent", "chunked"), default="all"
    )
    parser.add_argument("--stage-profile", action="store_true")
    parser.add_argument("--compare-chunk-configs", action="store_true")
    parser.add_argument("--value-blocks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--k-reuse-modes", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("This profile requires an NVIDIA V100 (SM70).")

    q_heads = 4
    value_heads = 12
    key_dim = value_dim = 128
    mixed_dim = 2 * q_heads * key_dim + value_heads * value_dim
    torch.manual_seed(41)
    a_log = torch.randn(value_heads, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.float16) * 0.1
    state_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    initial_state = (
        torch.randn(
            1,
            value_heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.01
    )
    state = initial_state.clone()

    for tokens in args.tokens:
        mixed_qkv = (
            torch.randn(tokens, mixed_dim, device="cuda", dtype=torch.float16) * 0.1
        ).contiguous()
        gate_a = (
            torch.randn(tokens, value_heads, device="cuda", dtype=torch.float16) * 0.1
        ).contiguous()
        gate_b = torch.randn_like(gate_a).mul_(0.1).contiguous()
        cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        common = dict(
            mixed_qkv=mixed_qkv,
            gate_a=gate_a,
            gate_b=gate_b,
            q_heads=q_heads,
            value_heads=value_heads,
            a_log=a_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            state=state,
            state_indices=state_indices,
            cu_seqlens=cu_seqlens,
        )

        print(f"tokens={tokens}")
        if args.stage_profile:
            chunk_indices = prepare_chunk_indices(cu_seqlens, 64).to(torch.int32)
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, 64).to(torch.int32)
            qk_kernel = _get_packed_qk_norm(q_heads, value_heads)
            v_kernel = _get_packed_v_copy(q_heads, value_heads)
            gate_kernel = _get_packed_gate_cumsum(value_heads, 1)

            q, k = qk_kernel(mixed_qkv)
            v = v_kernel(mixed_qkv)
            gate, beta = gate_kernel(
                gate_a, gate_b, a_log, dt_bias, cu_seqlens, chunk_indices
            )
            inverse_kernel = _get_kkt_inverse(q_heads, value_heads, 1)
            inverse = inverse_kernel(k, beta, cu_seqlens, chunk_indices)
            checkpoints = torch.empty(
                (1, 0, value_heads, value_dim, key_dim),
                device="cuda",
                dtype=torch.float16,
            )
            output_workspace = torch.empty_like(v)

            stage_calls = (
                ("qk_norm", lambda: qk_kernel(mixed_qkv)),
                ("v_copy", lambda: v_kernel(mixed_qkv)),
                (
                    "gate_cumsum",
                    lambda: gate_kernel(
                        gate_a,
                        gate_b,
                        a_log,
                        dt_bias,
                        cu_seqlens,
                        chunk_indices,
                    ),
                ),
                (
                    "kkt_inverse",
                    lambda: inverse_kernel(k, beta, cu_seqlens, chunk_indices),
                ),
            )
            for label, call in stage_calls:
                latency = timed(
                    call,
                    state,
                    initial_state,
                    label=f"gdn_{label}_t{tokens}",
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                print(f"  stage {label}: {latency:.4f} ms")

            baseline_output = None
            for value_block, k_reuse_mode in product(
                args.value_blocks, args.k_reuse_modes
            ):
                try:
                    forward_kernel = _get_chunk_forward(
                        q_heads,
                        value_heads,
                        1,
                        state.shape[0],
                        False,
                        True,
                        value_block=value_block,
                        k_reuse_mode=k_reuse_mode,
                    )
                except Exception as error:
                    detail = str(error).splitlines()[0]
                    print(
                        "  stage chunk_forward "
                        f"value_block={value_block} k_reuse={k_reuse_mode}: "
                        f"ERROR {type(error).__name__}: {detail}"
                    )
                    continue

                def run_forward():
                    return forward_kernel(
                        q,
                        k,
                        v,
                        inverse,
                        gate,
                        beta,
                        state,
                        state_indices,
                        cu_seqlens,
                        chunk_offsets,
                        key_dim**-0.5,
                        output_workspace,
                        checkpoints,
                    )

                latency = timed(
                    run_forward,
                    state,
                    initial_state,
                    label=(
                        f"gdn_chunk_forward_t{tokens}_vb{value_block}_kr{k_reuse_mode}"
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                state.copy_(initial_state)
                run_forward()
                torch.cuda.synchronize()
                if baseline_output is None:
                    baseline_output = output_workspace.clone()
                    max_abs = 0.0
                else:
                    max_abs = (
                        (output_workspace.float() - baseline_output.float())
                        .abs()
                        .max()
                        .item()
                    )
                print(
                    "  stage chunk_forward "
                    f"value_block={value_block} k_reuse={k_reuse_mode}: "
                    f"{latency:.4f} ms max_abs={max_abs:.3e}"
                )

                if value_block == 16 and k_reuse_mode == 0:
                    packed_forward_kernel = _get_chunk_forward(
                        q_heads,
                        value_heads,
                        1,
                        state.shape[0],
                        False,
                        True,
                        value_block=value_block,
                        packed_v=True,
                    )

                    def run_packed_forward():
                        return packed_forward_kernel(
                            q,
                            k,
                            mixed_qkv,
                            inverse,
                            gate,
                            beta,
                            state,
                            state_indices,
                            cu_seqlens,
                            chunk_offsets,
                            key_dim**-0.5,
                            output_workspace,
                            checkpoints,
                        )

                    latency = timed(
                        run_packed_forward,
                        state,
                        initial_state,
                        label=f"gdn_chunk_forward_packed_t{tokens}_vb{value_block}",
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    state.copy_(initial_state)
                    run_packed_forward()
                    torch.cuda.synchronize()
                    max_abs = (
                        (output_workspace.float() - baseline_output.float())
                        .abs()
                        .max()
                        .item()
                    )
                    print(
                        "  stage chunk_forward packed-v value_block=16: "
                        f"{latency:.4f} ms max_abs={max_abs:.3e}"
                    )
            continue

        if args.only in ("all", "recurrent"):
            for groups in args.groups:
                os.environ["SGLANG_V100_GDN_COLUMN_GROUPS_PER_BLOCK"] = str(groups)
                latency = timed(
                    lambda: packed_recurrent_gdn_sm70(**common),
                    state,
                    initial_state,
                    label=f"gdn_recurrent_t{tokens}_g{groups}",
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                print(f"  recurrent groups={groups}: {latency:.4f} ms")

        os.environ.pop("SGLANG_V100_GDN_COLUMN_GROUPS_PER_BLOCK", None)
        os.environ["SGLANG_V100_GDN_FULL_TILELANG"] = "1"
        if args.only in ("all", "chunked"):
            configs = (
                (
                    ("compact-v/vb16/kr0", "0", "16", "0"),
                    ("packed-v/vb16/kr0", "1", "16", "0"),
                    ("packed-v/vb32/kr3", "1", "32", "3"),
                )
                if args.compare_chunk_configs
                else (("production", "1", "32", "3"),)
            )
            baseline_output = None
            baseline_state = None
            for label, packed_v, value_block, k_reuse in configs:
                os.environ["SGLANG_V100_GDN_PACKED_V_DIRECT"] = packed_v
                os.environ["SGLANG_V100_GDN_VALUE_BLOCK"] = value_block
                os.environ["SGLANG_V100_GDN_K_REUSE_MODE"] = k_reuse
                call = lambda: packed_chunked_gdn_sm70(**common)
                chunked = timed(
                    call,
                    state,
                    initial_state,
                    label=f"gdn_chunked_t{tokens}_{label}",
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                state.copy_(initial_state)
                output, _ = call()
                torch.cuda.synchronize()
                if baseline_output is None:
                    baseline_output = output.clone()
                    baseline_state = state.clone()
                    max_output_abs = max_state_abs = 0.0
                else:
                    max_output_abs = (
                        (output.float() - baseline_output.float()).abs().max().item()
                    )
                    max_state_abs = (
                        (state.float() - baseline_state.float()).abs().max().item()
                    )
                print(
                    f"  chunked {label}: {chunked:.4f} ms "
                    f"output_abs={max_output_abs:.3e} "
                    f"state_abs={max_state_abs:.3e}"
                )


if __name__ == "__main__":
    main()
