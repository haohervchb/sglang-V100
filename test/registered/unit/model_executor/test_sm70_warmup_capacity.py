from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


@pytest.mark.parametrize(
    "requested,resolved,expected",
    [(2, 1, [1]), (8, 2, [1, 2]), (None, 1, [1]), (1, 1, [1])],
)
def test_prefill_warmup_fits_resolved_pool(monkeypatch, requested, resolved, expected):
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "cuda"
    runner.is_generation = True
    runner.is_draft_worker = False
    runner.server_args = SimpleNamespace(max_running_requests=requested)
    runner.max_running_requests = resolved
    runner._should_run_flashinfer_autotune = Mock(return_value=False)
    runner._warmup_sm70_flashinfer_sampling = Mock()
    runner._warmup_prefill_kernels_extends = Mock()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))
    runner.kernel_warmup()
    assert runner._warmup_prefill_kernels_extends.call_args_list == [
        call(bs) for bs in expected
    ]
