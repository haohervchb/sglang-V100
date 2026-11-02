"""Diagnostic-only phase timing. Enable with SGLANG_SPEC_PROBE_DIR and PYTHONPATH.

CUDA events are queried later, never synchronized in the measured round. Timed
runs must be labelled diagnostic; they are not throughput release evidence.
"""

import os

if os.environ.get("SGLANG_SPEC_PROBE_DIR"):
    import atexit
    import functools
    import json
    import time
    from collections import deque
    from pathlib import Path

    import torch

    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2

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
