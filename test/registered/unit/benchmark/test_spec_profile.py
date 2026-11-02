import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

path = Path(__file__).resolve().parents[4] / "benchmark/qwen38_spec/analyze_profile.py"
spec = importlib.util.spec_from_file_location("spec_profile", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_probe_does_not_import_torch_before_server_setup(tmp_path):
    env = os.environ.copy()
    env["SGLANG_SPEC_PROBE_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(path.parent / "probe")
    subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; assert "torch" not in sys.modules; assert "sglang" not in sys.modules',
        ],
        env=env,
        check=True,
        timeout=30,
    )


def test_idle_time_merges_stream_overlap_and_excludes_cpu_events():
    def event(start, duration, category="kernel", device=0):
        return dict(
            ph="X",
            cat=category,
            ts=start,
            dur=duration,
            name="test",
            args=dict(device=device),
        )

    result = module.analyze_trace(
        dict(
            traceEvents=[
                event(0, 10),
                event(5, 10),
                event(20, 5),
                event(0, 999, "cpu_op"),
            ]
        )
    )
    assert result["devices"]["0"]["busy_ms"] == 0.020
    assert result["devices"]["0"]["idle_ms"] == 0.005
    assert result["devices"]["0"]["idle_fraction"] == 0.2


def test_decode_window_follows_async_gpu_completion_and_excludes_prefill():
    def event(name, category, start, duration, correlation=0):
        return dict(
            ph="X",
            name=name,
            cat=category,
            ts=start,
            dur=duration,
            args=dict(device=0, correlation=correlation),
        )

    result = module.analyze_trace(
        dict(
            traceEvents=[
                event("prefill", "kernel", 0, 100),
                event("spec_cycle/draft", "user_annotation", 110, 10),
                event("spec_cycle/verify", "user_annotation", 120, 10),
                event("cudaGraphLaunch", "cuda_runtime", 115, 1, 1),
                event("cudaGraphLaunch", "cuda_runtime", 125, 1, 2),
                event("draft", "kernel", 150, 10, 1),
                event("memcpy", "gpu_memcpy", 175, 5, 3),
                event("verify", "kernel", 200, 10, 2),
                event("later_request", "kernel", 400, 10),
            ]
        )
    )
    window = result["decode_window"]["devices"]["0"]
    assert window["span_ms"] == 0.060
    assert window["busy_ms"] == 0.025
    assert window["idle_ms"] == 0.035
