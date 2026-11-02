# SPDX-License-Identifier: Apache-2.0
"""Select a model-declared subset without reading unrelated weight tensors."""

import json
from pathlib import Path


def select_checkpoint_shards(files, folder, prefixes):
    """Return shards containing the requested prefixes, or the original list.

    Models must opt in and consume only these prefixes. Missing/unrecognized
    indexes retain the ordinary loader; incomplete indexes must not silently
    omit required weights. Embeddings shared from the target are not loaded by
    the Qwen Flash-Next MTP model.
    """
    if not prefixes:
        return files
    index = Path(folder) / "model.safetensors.index.json"
    if not index.is_file():
        return files
    weight_map = json.loads(index.read_text()).get("weight_map", {})
    required = {
        filename for name, filename in weight_map.items() if name.startswith(prefixes)
    }
    if not required:
        return files
    available = {Path(filename).name for filename in files}
    # The prepared list has already applied the loader's format/index filters.
    # Never invent paths or resurrect files excluded by those checks.
    if not required <= available:
        raise ValueError(
            f"Required checkpoint shards are unavailable: {sorted(required - available)}"
        )
    return [filename for filename in files if Path(filename).name in required]
