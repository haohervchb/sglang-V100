# SPDX-License-Identifier: Apache-2.0
"""Numerical boundaries for the one-pass Ref2VA media path."""

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    material_io,
    reference_encoding,
    video_adapter,
)


def test_ffprobe_falls_back_when_stream_side_data_is_unknown(monkeypatch):
    material_io._ffprobe_entries = None
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        entries = command[command.index("-show_entries") + 1]
        if "stream_side_data" in entries:
            raise subprocess.CalledProcessError(
                1,
                command,
                stderr="ffprobe: No match for section 'stream_side_data'",
            )
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [{"codec_type": "audio", "sample_rate": "44100"}],
                    "format": {"format_name": "mp3", "duration": "1.0"},
                }
            )
        )

    monkeypatch.setattr(subprocess, "run", run)
    payload = material_io._ffprobe_media("/input/ref.mp3")

    assert len(calls) == 2
    assert "stream_side_data" in calls[0][calls[0].index("-show_entries") + 1]
    assert "stream_side_data" not in calls[1][calls[1].index("-show_entries") + 1]
    assert payload["format"]["format_name"] == "mp3"
    assert material_io._ffprobe_entries is not None
    assert "stream_side_data" not in material_io._ffprobe_entries


def test_final_output_probe_falls_back_to_pyav(monkeypatch):
    def run(*_args, **_kwargs):
        raise FileNotFoundError("ffprobe")

    duration = 107 / 24
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1344,
                "height": 768,
                "avg_frame_rate": "24",
                "nb_frames": 107,
                "duration": str(duration),
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": 32000,
                "channels": 2,
                "duration": str(duration),
            },
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": str(duration),
        },
    }
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        video_adapter,
        "_probe_minimax_h3_output_with_pyav",
        lambda *_args, **_kwargs: payload,
    )

    fields = video_adapter._probe_minimax_h3_output_fields(
        "/output/result.mp4",
        expected_frame_count=107,
        expected_size=(1344, 768),
    )

    assert fields == {"size": "1344x768", "seconds": "4.458333"}


def test_video_transform_runs_once_and_qwen_samples_shared_rgb(monkeypatch):
    expected = np.arange(25 * 4 * 6 * 3, dtype=np.uint8).reshape(25, 4, 6, 3)
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=expected.tobytes())

    monkeypatch.setattr(subprocess, "run", run)
    frames = reference_encoding.minimax_h3_decode_reference_video_frames(
        "/input/ref.mp4",
        target_width=6,
        target_height=4,
        target_frame_count=25,
        fps=24.0,
        start_time_seconds=2.25,
    )
    sampled = reference_encoding.minimax_h3_sample_reference_video_frames(frames)

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-vf") + 1] == (
        "fps=24,scale=6:4:flags=lanczos,setsar=1"
    )
    assert command[command.index("-frames:v") + 1] == "25"
    assert command[command.index("-ss") + 1] == "2.25"
    assert command.index("-ss") < command.index("-i")
    assert command[-5:] == ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    assert "libx264" not in command
    assert all(np.shares_memory(frame, frames) for frame in sampled["frames"])
    assert [int(frame[0, 0, 0]) for frame in sampled["frames"]] == [
        int(expected[index, 0, 0, 0]) for index in (0, 12, 24)
    ]
    assert sampled["block_timestamps"] == [0.25, 1.0]


def test_audio_decode_is_bounded_float_pcm_without_temp_files(monkeypatch):
    pcm = torch.arange(8, dtype=torch.float32).numpy().tobytes()
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(
                stdout=json.dumps(
                    {"streams": [{"channels": 6, "sample_rate": "44100"}]}
                )
            )
        return SimpleNamespace(stdout=pcm)

    monkeypatch.setattr(subprocess, "run", run)
    waveform, source_rate = reference_encoding._load_waveform(
        "/input/ref.mp4",
        material_chain="video.reference_preserve",
        max_duration_seconds=3.5,
        start_time_seconds=2.25,
    )

    ffmpeg = next(command for command in commands if command[0] == "ffmpeg")
    assert source_rate == 44100
    torch.testing.assert_close(
        waveform,
        torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.float32),
    )
    assert ffmpeg[ffmpeg.index("-t") + 1] == "3.5"
    assert ffmpeg[ffmpeg.index("-ss") + 1] == "2.25"
    assert ffmpeg.index("-ss") < ffmpeg.index("-i")
    assert ffmpeg[-3:] == ["-f", "f32le", "pipe:1"]
