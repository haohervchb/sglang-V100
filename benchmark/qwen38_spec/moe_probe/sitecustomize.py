"""Capture a few real TP0 MoE prefill cases for isolated tuning.

Enable only in a dedicated experiment server. Captured checkpoint tensors
remain local and must not be committed with benchmark summaries.
"""

import importlib.abc
import importlib.machinery
import os
import sys


def install(module):
    import functools
    import json
    from pathlib import Path

    import torch

    original = module.fused_marlin_moe
    root = Path(os.environ["SGLANG_MOE_CAPTURE_DIR"])
    selected = {0, 1, 3, 15, 31, 47}
    count = 0

    @functools.wraps(original)
    def capture(*args, **kwargs):
        nonlocal count
        x = kwargs.get("hidden_states")
        if (
            x is not None
            and x.device.index == 0
            and tuple(x.shape) == (2048, 2560)
            and kwargs.get("num_bits") == 4
            and count < 48
        ):
            index = count
            count += 1
            if index in selected:
                root.mkdir(parents=True, exist_ok=True)
                path = root / f"layer_{index:02d}.pt"
                if path.exists():
                    raise FileExistsError(path)
                data = {
                    name: (
                        value.detach().cpu()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for name, value in kwargs.items()
                    if name != "workspace"
                }
                torch.save(data, path)
                histogram = torch.bincount(
                    data["topk_ids"].long().flatten(), minlength=512
                )
                (root / f"layer_{index:02d}.json").write_text(
                    json.dumps(
                        dict(
                            layer=index,
                            shape=list(x.shape),
                            expert_counts=histogram.tolist(),
                            tensors={
                                name: dict(
                                    shape=list(value.shape), dtype=str(value.dtype)
                                )
                                for name, value in data.items()
                                if isinstance(value, torch.Tensor)
                            },
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                print(f"[MoE capture] saved layer {index} to {path}", flush=True)
        return original(*args, **kwargs)

    module.fused_marlin_moe = capture


def install_control(module):
    import functools
    import json
    from pathlib import Path

    path = Path(os.environ["SGLANG_SM70_PREFILL_AB_FILE"])
    previous = None
    original = module.EAGLEWorkerV2.forward_batch_generation

    @functools.wraps(original)
    def controlled(self, batch, *args, **kwargs):
        nonlocal previous
        if batch.forward_mode.is_extend() and batch.reqs:
            identity = tuple(req.rid for req in batch.reqs)
            if identity != previous:
                config = json.loads(path.read_text())
                for key, env in [
                    ("moe", "SGLANG_SM70_MOE_PREFILL_2K"),
                    ("qsa", "SGLANG_SM70_QSA_COMPACT_PREFILL"),
                ]:
                    assert type(config[key]) is bool
                    os.environ[env] = "1" if config[key] else "0"
                grouped = config.get("grouped", False)
                assert type(grouped) is bool
                os.environ["SGLANG_SM70_QSA_GROUPED_PREFILL"] = "1" if grouped else "0"
                previous = identity
        return original(self, batch, *args, **kwargs)

    module.EAGLEWorkerV2.forward_batch_generation = controlled


def install_qsa_sample(module):
    import functools
    from pathlib import Path

    import torch

    root = Path(os.environ["SGLANG_QSA_SAMPLE_DIR"])
    original = module.sm70_cuda_qsa_prefill
    count = 0

    @functools.wraps(original)
    def sample(q, k, v, table, req, indices, lens, scale, **kwargs):
        nonlocal count
        output = original(q, k, v, table, req, indices, lens, scale, **kwargs)
        if (
            count < 4
            and q.device.index == 0
            and q.shape[0] == 2048
            and k.dtype == torch.float16
        ):
            index = count
            count += 1
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"qsa_{index}.pt"
            if not path.exists():
                length = int(lens[0].item())
                request = int(req[0].item())
                slots = table[request, :length].long()
                assert slots.min() >= 0 and slots.max() < k.shape[0]
                torch.save(
                    dict(
                        q=q.detach().cpu(),
                        k=k.index_select(0, slots).cpu(),
                        v=v.index_select(0, slots).cpu(),
                        table=torch.arange(length, dtype=torch.int32)[None],
                        req=torch.zeros(1, dtype=torch.int32),
                        indices=indices.cpu(),
                        lens=lens.cpu(),
                        scale=scale,
                        output=output.cpu(),
                    ),
                    path,
                )
                print(f"[QSA capture] saved {path}", flush=True)
        return output

    module.sm70_cuda_qsa_prefill = sample


if (
    os.environ.get("SGLANG_MOE_CAPTURE_DIR")
    or os.environ.get("SGLANG_SM70_PREFILL_AB_FILE")
    or os.environ.get("SGLANG_QSA_SAMPLE_DIR")
):

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            if module.__name__.endswith("eagle_worker_v2"):
                install_control(module)
            elif module.__name__.endswith("_decode_cuda"):
                install_qsa_sample(module)
            else:
                install(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            capture = (
                fullname == "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe"
                and os.environ.get("SGLANG_MOE_CAPTURE_DIR")
            )
            control = (
                fullname == "sglang.srt.speculative.eagle_worker_v2"
                and os.environ.get("SGLANG_SM70_PREFILL_AB_FILE")
            )
            qsa = (
                fullname == "sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda"
                and os.environ.get("SGLANG_QSA_SAMPLE_DIR")
            )
            if not (capture or control or qsa):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader)
                # Each configured module is imported once by Python.
            return spec

    sys.meta_path.insert(0, Finder())
