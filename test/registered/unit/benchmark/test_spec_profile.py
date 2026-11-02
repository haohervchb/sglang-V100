import importlib.util
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

path = Path(__file__).resolve().parents[4] / "benchmark/qwen38_spec/analyze_profile.py"
spec = importlib.util.spec_from_file_location("spec_profile", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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
