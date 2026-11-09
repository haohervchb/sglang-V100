"""Local V100 calibration; isolated kernels are not model throughput estimates."""

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import pynvml
import torch


def timed_graph(operation, operations=8, replays=10, samples=5):
    for index in range(3):
        operation(index)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    # PyTorch's implicit capture stream can be cached on the first GPU used.
    # Pin the capture stream to this device so events time the same workload.
    capture_stream = torch.cuda.Stream(device=torch.cuda.current_device())
    with torch.cuda.graph(graph, stream=capture_stream):
        for index in range(operations):
            operation(index)
    graph.replay()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / replays / operations)
    return statistics.median(timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counter-smoke", action="store_true")
    args = parser.parse_args()
    if args.counter_smoke:
        x = torch.ones(1024 * 1024, device="cuda")
        x.add_(1)
        torch.cuda.synchronize()
        return

    result = dict(devices=[], projections=[], peer_copies=[])
    pynvml.nvmlInit()
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            source = torch.ones(256 * 1024 * 1024, dtype=torch.float32, device=device)
            destination = torch.empty_like(source)
            copy_ms = timed_graph(lambda _, dst=destination, src=source: dst.copy_(src))
            del source, destination
            a = torch.randn((8192, 8192), dtype=torch.float16, device=device)
            b = torch.randn_like(a)
            c = torch.empty_like(a)
            gemm_ms = timed_graph(
                lambda _, left=a, right=b, out=c: torch.mm(left, right, out=out),
                replays=3,
            )
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            sm_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            host = torch.ones(64 * 1024 * 1024, dtype=torch.float32, pin_memory=True)
            target = torch.empty(host.shape, dtype=host.dtype, device=device)
            host_ms = timed_graph(
                lambda _, src=host, dst=target: dst.copy_(src, non_blocking=True),
                operations=4,
                replays=2,
                samples=3,
            )
            row = dict(
                device=device,
                name=torch.cuda.get_device_name(device),
                copy_ms=copy_ms,
                copy_read_write_gb_s=2 * 1024**3 / (copy_ms / 1000) / 1e9,
                gemm_8192_ms=gemm_ms,
                gemm_8192_tflops=2 * 8192**3 / (gemm_ms / 1000) / 1e12,
                pinned_host_to_device_gb_s=256 * 1024**2 / (host_ms / 1000) / 1e9,
                sm_mhz_after_gemm=sm_mhz,
            )
            # Reject misplaced event/capture streams instead of publishing
            # physically impossible V100 calibration results.
            assert 0 < row["copy_read_write_gb_s"] < 2000, row
            assert 0 < row["gemm_8192_tflops"] < 160, row
            result["devices"].append(row)
            print(json.dumps(row), flush=True)
            del a, b, c, host, target
            gc.collect()
            torch.cuda.empty_cache()

    from sglang.jit_kernel.sm70_dense_gemv import linear, supported

    torch.cuda.set_device(0)
    shapes = [
        (4096, 2560),
        (3584, 2560),
        (2560, 1536),
        (320, 2560),
        (2560, 160),
        (512, 2560),
        (640, 2560),
        (10240, 2560),
        (2560, 2560),
        (62080, 2560),
    ]
    for n, k in shapes:
        # Rotate weights beyond V100's L2 capacity where practical. These are
        # minimum logical weight reads, not hardware DRAM byte counters.
        ring = min(32, max(2, math.ceil(16 * 1024**2 / (2 * n * k))))
        weights = [
            torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.01
            for _ in range(ring)
        ]
        for m in (1, 4, 2048):
            if m == 2048 and n > 10240:
                continue  # Prefill selects final-token logits, not a 2048-row head.
            x = torch.randn((m, k), device="cuda", dtype=torch.float16)
            methods = {
                "torch": lambda i, inputs=x, pool=weights, size=ring: (
                    torch.nn.functional.linear(inputs, pool[i % size])
                )
            }
            if supported(x, weights[0]):
                methods["native"] = lambda i, inputs=x, pool=weights, size=ring: linear(
                    inputs, pool[i % size]
                )
            for name, operation in methods.items():
                ms = timed_graph(operation, operations=max(8, ring), replays=5)
                row = dict(
                    m=m,
                    n=n,
                    k=k,
                    method=name,
                    weight_ring=ring,
                    ms=ms,
                    tflops=2 * m * n * k / (ms / 1000) / 1e12,
                    weight_read_equivalent_gb_s=2 * n * k / (ms / 1000) / 1e9,
                )
                result["projections"].append(row)
            print(json.dumps(result["projections"][-len(methods) :]), flush=True)
            del operation, methods, x
        del weights
        gc.collect()
        torch.cuda.empty_cache()

    for source_device, destination_device in ((0, 1), (0, 2), (0, 3)):
        size = 128 * 1024 * 1024
        source = torch.ones(size, dtype=torch.float32, device=source_device)
        destination = torch.empty(size, dtype=torch.float32, device=destination_device)
        for _ in range(3):
            destination.copy_(source, non_blocking=True)
        torch.cuda.synchronize(source_device)
        torch.cuda.synchronize(destination_device)
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(20):
                destination.copy_(source, non_blocking=True)
            torch.cuda.synchronize(source_device)
            torch.cuda.synchronize(destination_device)
            samples.append(time.perf_counter() - start)
        result["peer_copies"].append(
            dict(
                source=source_device,
                destination=destination_device,
                one_way_gb_s=20 * size * 4 / statistics.median(samples) / 1e9,
            )
        )
        del source, destination

    result["notes"] = [
        "Copy GB/s counts both HBM reads and writes; peer GB/s is one-way payload.",
        "Projection timings use CUDA graphs with rotating synthetic weights.",
        "Equivalent weight GB/s is logical bytes/time and is not measured DRAM throughput.",
        "Calibration cannot be applied as an end-to-end model FLOP utilization score.",
    ]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
