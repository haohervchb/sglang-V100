"""Opt-in local diagnostic: time CPU submission without synchronizing the GPU."""

import os

if os.environ.get("QWEN38_PROFILE_GRAPH_SUBMIT") == "1":
    import json
    import time
    from pathlib import Path

    import torch

    _original_replay = torch.cuda.CUDAGraph.replay
    _replay_times = {}
    _sample_count = 0

    def _timed_replay(graph):
        global _sample_count
        start = time.perf_counter_ns()
        ret = _original_replay(graph)
        elapsed = time.perf_counter_ns() - start
        _replay_times.setdefault(id(graph), []).append(elapsed / 1000)
        _sample_count += 1
        if _sample_count % 200 == 0:
            Path(
                "/tmp/qwen38_mtp_20260908/graph_submit_%d.json" % os.getpid()
            ).write_text(json.dumps(_replay_times))
        return ret

    torch.cuda.CUDAGraph.replay = _timed_replay
