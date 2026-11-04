"""CPU preprocessing must not accidentally allocate image tensors on CUDA."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors import base_processor
from transformers import BaseImageProcessor


@pytest.mark.parametrize("cpu_only,expected", [(True, "cpu"), (False, "cuda:3")])
def test_preprocess_device_override_preserves_default(monkeypatch, cpu_only, expected):
    args = SimpleNamespace(
        disable_fast_image_processor=False,
        keep_mm_feature_on_device=False,
        rl_on_policy_target=None,
        base_gpu_id=3,
    )
    monkeypatch.setattr(base_processor, "get_global_server_args", lambda: args)
    for name in ["_is_cpu", "_is_xpu", "_is_npu"]:
        monkeypatch.setattr(base_processor, name, False)
    hf = MagicMock()
    hf.image_processor = MagicMock(spec=BaseImageProcessor)
    hf.__call__ = MagicMock(return_value={})
    cls = base_processor.BaseMultimodalProcessor
    with patch.object(cls, "__abstractmethods__", set()):
        proc = object.__new__(cls)
    proc.server_args = args
    proc._processor = hf
    proc.image_config = proc.video_config = proc.audio_config = {}
    proc.FEATURE_NAMES = []
    with envs.SGLANG_MM_PREPROCESS_ON_CPU.override(cpu_only):
        proc.process_mm_data("test", images=[Image.new("RGB", (8, 8))])
    assert hf.__call__.call_args.kwargs["device"] == expected


@pytest.mark.parametrize("cpu_only,gpu_decode", [(True, False), (False, True)])
def test_cpu_override_also_disables_gpu_jpeg_decode(monkeypatch, cpu_only, gpu_decode):
    image = Image.new("RGB", (8, 8))
    loader = MagicMock(return_value=(image, image.size))
    monkeypatch.setattr(base_processor, "load_image", loader)
    with envs.SGLANG_MM_PREPROCESS_ON_CPU.override(cpu_only):
        result = base_processor.BaseMultimodalProcessor._load_single_item(
            b"encoded image", Modality.IMAGE
        )
    assert result is image
    loader.assert_called_once_with(b"encoded image", gpu_decode)
