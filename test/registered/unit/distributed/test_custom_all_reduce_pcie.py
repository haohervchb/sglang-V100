"""Dispatch and topology guards; actual PCIe timings require a PCIe-only host."""

from unittest.mock import Mock

import pytest
import torch

from sglang.jit_kernel.all_reduce import AllReduceAlgo
from sglang.srt.distributed.device_communicators import custom_all_reduce_utils as utils
from sglang.srt.distributed.device_communicators import custom_all_reduce_v2 as v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


@pytest.mark.parametrize("nvlink", [True, False])
@pytest.mark.parametrize("opt_in", [True, False])
@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("p2p", [True, False])
def test_topology_is_preserved(monkeypatch, nvlink, opt_in, legacy, p2p):
    monkeypatch.setenv("SGLANG_CUSTOM_AR_ALLOW_PCIE", str(int(opt_in)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(utils.dist, "get_backend", lambda g: "gloo")
    monkeypatch.setattr(utils.dist, "get_rank", lambda **kw: 0)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda **kw: 4)
    monkeypatch.setattr(utils, "in_the_same_node_as", lambda *a, **kw: [True] * 4)
    monkeypatch.setattr(utils.dist, "all_gather", lambda *a, **kw: None)
    monkeypatch.setattr(utils, "is_full_nvlink", lambda *a: nvlink)
    check = Mock(return_value=p2p)
    monkeypatch.setattr(utils, "can_p2p", check)
    monkeypatch.setattr(utils, "_is_hip", False)
    result = utils.can_use_custom_all_reduce_with_nvlink(
        object(),
        torch.device("cuda:0"),
        [2, 4, 8],
        "test",
        allow_pcie=opt_in and not legacy,
    )
    eligible = nvlink or (opt_in and not legacy)
    assert result is (nvlink if eligible and p2p else None)
    assert check.call_count == int(eligible)


@pytest.mark.parametrize("nvlink", [True, False, None])
@pytest.mark.parametrize("opt_in", [True, False])
def test_v2_initialization(monkeypatch, nvlink, opt_in):
    monkeypatch.setenv("SGLANG_CUSTOM_AR_ALLOW_PCIE", str(int(opt_in)))
    monkeypatch.setenv("SGLANG_CUSTOM_AR_PCIE_MAX_BYTES", "131072")
    monkeypatch.setattr(v2, "_custom_all_reduce_v2_nvlink", lambda **kw: nvlink)
    monkeypatch.setattr(v2, "_maybe_init_config", lambda: None)
    monkeypatch.setattr(
        v2, "THRESHOLD_2_SHOT_MAP", {4: v2.ModeConfig(384 * 1024, 256 * 1024)}
    )
    monkeypatch.setattr(v2.dist, "get_rank", lambda **kw: 0)
    monkeypatch.setattr(v2.dist, "get_world_size", lambda **kw: 4)
    factory = Mock()
    monkeypatch.setattr(v2, "get_custom_all_reduce_cls", lambda: factory)
    monkeypatch.setattr(v2.CustomAllReduceV2, "_post_init_obj", lambda self: None)
    eligible = nvlink is not None and (nvlink or opt_in)
    assert v2.can_use_custom_all_reduce_v2(object(), torch.device("cuda:0")) == eligible
    comm = v2.CustomAllReduceV2(object(), torch.device("cuda:0"))
    assert comm.disabled == (not eligible)
    assert factory.call_count == int(eligible)
    if eligible:
        assert comm.pcie_only == (not nvlink)
        assert comm.max_size == (16 * 1024 * 1024 if nvlink else 131072)
        assert comm.max_push_size == (384 * 1024 if nvlink else 131072)


@pytest.mark.parametrize("override", list(AllReduceAlgo))
@pytest.mark.parametrize("env_override", ["1stage", "2stage", ""])
def test_pcie_cannot_select_pull(monkeypatch, override, env_override):
    monkeypatch.setenv("SGLANG_CUSTOM_ALLREDUCE_ALGO", env_override)
    comm = v2.CustomAllReduceV2.__new__(v2.CustomAllReduceV2)
    comm.disabled = False
    comm.pcie_only = True
    comm.max_size = 131072
    comm.override_algo = override
    x = torch.empty(65536, dtype=torch.float16)
    assert comm.should_custom_ar(x)
    assert comm._determine_algo(x) == AllReduceAlgo.ONE_SHOT_PUSH
    too_large = torch.empty(65544, dtype=torch.float16)
    assert not comm.should_custom_ar(too_large)
    with pytest.raises(ValueError, match="exceeds"):
        comm._determine_algo(too_large)


@pytest.mark.parametrize("byte_cap", [0, -16, 17])
def test_invalid_pcie_byte_cap(monkeypatch, byte_cap):
    monkeypatch.setenv("SGLANG_CUSTOM_AR_ALLOW_PCIE", "1")
    monkeypatch.setenv("SGLANG_CUSTOM_AR_PCIE_MAX_BYTES", str(byte_cap))
    monkeypatch.setattr(v2, "_custom_all_reduce_v2_nvlink", lambda **kw: False)
    monkeypatch.setattr(v2, "_maybe_init_config", lambda: None)
    monkeypatch.setattr(
        v2, "THRESHOLD_2_SHOT_MAP", {4: v2.ModeConfig(384 * 1024, 256 * 1024)}
    )
    monkeypatch.setattr(v2.dist, "get_rank", lambda **kw: 0)
    monkeypatch.setattr(v2.dist, "get_world_size", lambda **kw: 4)
    with pytest.raises(ValueError, match="positive multiple of 16"):
        v2.CustomAllReduceV2(object(), torch.device("cuda:0"))
