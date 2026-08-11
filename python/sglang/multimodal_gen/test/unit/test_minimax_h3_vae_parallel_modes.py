# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 released VAE decode contract."""

from unittest import mock

import pytest
import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae import (
    PointwiseConv1d,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.conv import (
    BaseConv3d,
)


def _init_kwargs(config: MiniMaxH3VideoVAEConfig):
    with mock.patch.object(
        AutoencoderKLLegacy, "__init__", autospec=True, return_value=None
    ) as init:
        model = MiniMaxH3VideoVAE(config)
    return model, init.call_args.kwargs


def test_pointwise_conv3d_linear_fallback_matches_reference():
    torch.manual_seed(7)
    conv = BaseConv3d(4, 6, kernel_size=1)
    inputs = torch.randn(2, 4, 3, 5, 7)

    expected = torch.nn.functional.conv3d(inputs, conv.weight, conv.bias)
    actual = conv(inputs)

    torch.testing.assert_close(actual, expected)


def test_pointwise_conv1d_linear_fallback_matches_reference():
    torch.manual_seed(11)
    conv = PointwiseConv1d(4, 6, kernel_size=1)
    inputs = torch.randn(2, 4, 17)

    expected = torch.nn.functional.conv1d(inputs, conv.weight, conv.bias)
    actual = conv(inputs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "mode",
    [
        None,
        "auto",
        "tiled",
    ],
)
def test_decode_mode_uses_released_tiled_recipe(mode):
    config = (
        MiniMaxH3VideoVAEConfig()
        if mode is None
        else MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    )
    model, kwargs = _init_kwargs(config)

    assert model.parallel_decode_mode == "tiled"
    assert kwargs["decoder_tiling"] is True
    assert kwargs["parallel_tiling"] is True
    assert kwargs["decoder_parallel"] is False


@pytest.mark.parametrize("mode", ["spatial", "spatial_shard", "patch"])
def test_unvalidated_decode_modes_are_rejected(mode):
    config = MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    with pytest.raises(ValueError, match="use tiled"):
        config.resolved_parallel_decode_mode()
