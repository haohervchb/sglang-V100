import json

import pytest

from sglang.srt.model_loader.shard_selection import select_checkpoint_shards


def test_subset_keeps_order_and_complete_shared_shards(tmp_path):
    files = [
        str(tmp_path / name)
        for name in ["b.safetensors", "a.safetensors", "c.safetensors"]
    ]
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.fc.weight": "a.safetensors",
                    "mtp.layer.weight": "b.safetensors",
                    "model.other.weight": "a.safetensors",
                    "model.expert.weight": "c.safetensors",
                }
            }
        )
    )
    assert select_checkpoint_shards(files, tmp_path, ("mtp.",)) == files[:2]
    assert select_checkpoint_shards(files, tmp_path, None) is files
    assert select_checkpoint_shards(files, tmp_path, ("missing.",)) is files


def test_no_index_uses_normal_loader(tmp_path):
    files = ["a.safetensors"]
    assert select_checkpoint_shards(files, tmp_path, ("mtp.",)) is files


def test_missing_required_shard_fails(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.fc.weight": "missing.safetensors",
                }
            }
        )
    )
    with pytest.raises(ValueError, match="Required checkpoint shards"):
        select_checkpoint_shards([str(tmp_path / "a.safetensors")], tmp_path, ("mtp.",))
