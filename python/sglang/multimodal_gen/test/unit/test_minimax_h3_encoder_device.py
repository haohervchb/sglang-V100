# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import torch
from sglang.multimodal_gen.runtime.models.encoders import minimax_h3_qwen3vl
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
)


def _encoder_with_parameter_on(device: torch.device) -> MiniMaxH3Qwen3VLEncoder:
    encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.register_parameter(
        "offloaded", torch.nn.Parameter(torch.zeros(1, device=device))
    )
    return encoder


def test_device_uses_compute_device_with_cpu_offloaded_parameters():
    encoder = _encoder_with_parameter_on(torch.device("cpu"))
    compute_device = torch.device("cuda", 3)

    with mock.patch.object(
        minimax_h3_qwen3vl,
        "get_local_torch_device",
        return_value=compute_device,
    ):
        assert encoder.device == compute_device

    assert next(encoder.parameters()).device.type == "cpu"


def test_device_supports_cpu_only_platforms():
    encoder = _encoder_with_parameter_on(torch.device("cpu"))

    with mock.patch.object(
        minimax_h3_qwen3vl,
        "get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        assert encoder.device == torch.device("cpu")
