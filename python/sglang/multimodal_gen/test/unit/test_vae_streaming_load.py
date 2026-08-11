# SPDX-License-Identifier: Apache-2.0

import torch
from safetensors.torch import save_file
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _stream_safetensors_into_module,
)


def test_stream_safetensors_into_module(tmp_path):
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 2),
        torch.nn.LayerNorm(2),
    )
    expected = {
        name: torch.full_like(tensor, index + 1)
        for index, (name, tensor) in enumerate(module.state_dict().items())
    }
    checkpoint = tmp_path / "model.safetensors"
    save_file(expected, checkpoint)

    _stream_safetensors_into_module(module, [str(checkpoint)])

    for name, tensor in module.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])


def test_stream_safetensors_rejects_missing_keys(tmp_path):
    module = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.zeros_like(module.weight)}, checkpoint)

    try:
        _stream_safetensors_into_module(module, [str(checkpoint)])
    except RuntimeError as error:
        assert "Missing checkpoint keys" in str(error)
    else:
        raise AssertionError("missing checkpoint key was accepted")
