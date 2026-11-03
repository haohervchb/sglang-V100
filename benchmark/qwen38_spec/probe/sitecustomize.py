"""Diagnostic-only phase timing. Enable with SGLANG_SPEC_PROBE_DIR and PYTHONPATH.

CUDA events are queried later, never synchronized in the measured round. Timed
runs must be labelled diagnostic; they are not throughput release evidence.
"""

import os


def install_probe(eagle):
    global active, pending, recorded, destination, limit
    import atexit
    import functools
    import json
    import time
    from collections import deque
    from pathlib import Path

    import torch

    from sglang.srt.managers.tp_worker import TpModelWorker

    EagleDraftWorker = eagle.EagleDraftWorker
    EAGLEWorkerV2 = eagle.EAGLEWorkerV2

    active = None
    pending = deque()
    recorded = 0
    destination = None
    limit = int(os.environ.get("SGLANG_SPEC_PROBE_ROUNDS", "512"))

    def event():
        result = torch.cuda.Event(enable_timing=True)
        result.record()
        return result

    def drain():
        if destination is None or not pending or not pending[0]["end"].query():
            return
        with destination.open("a") as f:
            while pending and pending[0]["end"].query():
                row = pending.popleft()
                output = {
                    k: row[k]
                    for k in (
                        "round",
                        "cpu_start_s",
                        "cpu_submit_ms",
                        "batch_size",
                        "request_ids",
                    )
                }
                output["cycle_gpu_ms"] = row["start"].elapsed_time(row["end"])
                output["phases_ms"] = {
                    k: a.elapsed_time(b) for k, a, b in row["phases"]
                }
                f.write(json.dumps(output) + "\n")

    atexit.register(drain)

    def phase(cls, method, name, verify_only=False):
        original = getattr(cls, method)

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            if active is None or (verify_only and not kwargs.get("is_verify")):
                return original(self, *args, **kwargs)
            start = event()
            with torch.profiler.record_function("spec_cycle/" + name):
                result = original(self, *args, **kwargs)
            active["phases"].append((name, start, event()))
            return result

        setattr(cls, method, wrapped)

    phase(EagleDraftWorker, "draft", "draft")
    phase(EAGLEWorkerV2, "verify", "verify")
    phase(EAGLEWorkerV2, "_mamba_verify_update", "state_commit")
    phase(TpModelWorker, "forward_batch_generation", "target", verify_only=True)
    phase(EagleDraftWorker, "_draft_extend_for_decode", "draft_extend")
    original_round = EAGLEWorkerV2.forward_batch_generation

    @functools.wraps(original_round)
    def measured_round(self, batch, *args, **kwargs):
        global active, recorded, destination
        if self.tp_rank != 0:
            return original_round(self, batch, *args, **kwargs)
        drain()
        if (
            recorded >= limit
            or (
                os.environ.get("SGLANG_SPEC_PROBE_ON_PROFILE", "0") == "1"
                and not torch.autograd._profiler_enabled()
            )
            or batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
            or batch.forward_mode.is_idle()
        ):
            return original_round(self, batch, *args, **kwargs)
        if destination is None:
            root = Path(os.environ["SGLANG_SPEC_PROBE_DIR"])
            root.mkdir(parents=True, exist_ok=True)
            destination = root / f"cycles_tp0_{os.getpid()}.jsonl"
        start_time = time.perf_counter()
        active = dict(
            round=recorded,
            cpu_start_s=start_time,
            batch_size=len(batch.reqs),
            request_ids=[req.rid for req in batch.reqs],
            start=event(),
            phases=[],
        )
        try:
            return original_round(self, batch, *args, **kwargs)
        finally:
            active["end"] = event()
            active["cpu_submit_ms"] = 1000 * (time.perf_counter() - start_time)
            pending.append(active)
            active = None
            recorded += 1

    EAGLEWorkerV2.forward_batch_generation = measured_round


if os.environ.get("SGLANG_SPEC_PROBE_DIR"):
    # Wait for the worker's normal import. Importing Torch/SGLang here eagerly
    # can freeze environment-dependent backend choices before server setup.
    import importlib.abc
    import importlib.machinery
    import sys

    class ProbeLoader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            install_probe(module)

    class ProbeFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "sglang.srt.speculative.eagle_worker_v2":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = ProbeLoader(spec.loader)
                sys.meta_path.remove(self)
            return spec

    sys.meta_path.insert(0, ProbeFinder())

if os.environ.get("SGLANG_CORE_AUDIT_DIR"):
    from qwen38_core_probe import enable

    enable()
