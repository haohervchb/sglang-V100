"""Opt-in semantic ranges and prefill events for an isolated profiling server.

No tensors, token values or synchronization are added to unprofiled forwards.
The module import hook attaches only after normal SGLang imports complete.
"""

import importlib.abc
import importlib.machinery
import os
import sys


def install(module):
    import atexit
    import functools
    import json
    import time
    from collections import deque
    from pathlib import Path

    import torch

    def wrap(cls, method, label):
        original = getattr(cls, method)

        @functools.wraps(original)
        def annotated(self, *args, **kwargs):
            if not torch.autograd._profiler_enabled():
                return original(self, *args, **kwargs)
            name = label(self, *args, **kwargs) if callable(label) else label
            with torch.profiler.record_function("core/" + name):
                return original(self, *args, **kwargs)

        setattr(cls, method, annotated)

    name = module.__name__
    if name.endswith("models.qwen4_exp"):
        for cls, method, label in [
            (module.Qwen4ExpPLELayer, "forward", "ple"),
            (module.Qwen4ExpPLELayer, "_short_conv", "ple_short_conv"),
            (module.Qwen4ExpPLELayer, "start_prefetch", "ple_prefetch"),
            (
                module.Qwen4ExpPLELayer,
                "_consume_prefetched_embeddings",
                "ple_gather_reduce",
            ),
            (module.Qwen4ExpLayerExtensionMixin, "_run_qwen4_exp_mlp", "moe_total"),
            (module.Qwen4ExpAttentionDecoderLayer, "self_attention", "qsa_total"),
            (
                module.Qwen4ExpAttentionDecoderLayer,
                "_compute_qsa_topk_indices",
                "qsa_selection",
            ),
        ]:
            wrap(cls, method, label)
    elif name.endswith("models.qwen3_5"):
        wrap(module.Qwen3_5GatedDeltaNet, "forward", "gdn_total")
    elif name.endswith("layers.hyperconnection"):
        wrap(module.GatedResidual, "mix", "hc_mix")
        wrap(module.GatedResidual, "combine", "hc_combine")
    elif name.endswith("attention.qwen_sparse_attn_backend"):
        wrap(module.QwenSparseAttnBackend, "forward_extend", "qsa_attention")
    elif name.endswith("quantization.unquant"):

        def dense_label(self, layer, x, *args, **kwargs):
            return (
                f"dense_m{x.shape[0]}_n{layer.weight.shape[0]}_k{layer.weight.shape[1]}"
            )

        wrap(module.UnquantizedLinearMethod, "apply", dense_label)
    elif name.endswith("quantization.modelopt_quant"):
        wrap(module.ModelOptFp4LinearMethod, "apply", "fp4_dense")
        wrap(module.ModelOptNvFp4FusedMoEMethod, "apply", "moe_experts")
    elif name.endswith("model_executor.model_runner"):
        pending = deque()
        destination = (
            Path(os.environ["SGLANG_CORE_AUDIT_DIR"]) / f"forwards_{os.getpid()}.jsonl"
        )
        sequence = 0

        def drain():
            if not pending or not pending[0][1].query():
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("a") as stream:
                while pending and pending[0][1].query():
                    begin, end, row = pending.popleft()
                    row["gpu_ms"] = begin.elapsed_time(end)
                    stream.write(json.dumps(row) + "\n")

        atexit.register(drain)
        original = module.ModelRunner.forward

        @functools.wraps(original)
        def forward(self, forward_batch, *args, **kwargs):
            nonlocal sequence
            if self.tp_rank == 0:
                drain()
            if not torch.autograd._profiler_enabled():
                return original(self, forward_batch, *args, **kwargs)
            phase = (
                ("draft" if self.is_draft_worker else "target")
                + "_"
                + forward_batch.forward_mode.name.lower()
            )
            row = dict(
                sequence=sequence,
                phase=phase,
                tokens=forward_batch.input_ids.numel(),
                start_unix_s=time.time(),
                rank=self.tp_rank,
            )
            start = time.perf_counter()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.profiler.record_function("core/" + phase):
                output = original(self, forward_batch, *args, **kwargs)
            end.record()
            if self.tp_rank == 0:
                row["cpu_ms"] = 1000 * (time.perf_counter() - start)
                pending.append((begin, end, row))
                sequence += 1
            return output

        module.ModelRunner.forward = forward


def enable():
    names = {
        "sglang.srt.models.qwen4_exp",
        "sglang.srt.models.qwen3_5",
        "sglang.srt.layers.hyperconnection",
        "sglang.srt.layers.attention.qwen_sparse_attn_backend",
        "sglang.srt.layers.quantization.unquant",
        "sglang.srt.layers.quantization.modelopt_quant",
        "sglang.srt.model_executor.model_runner",
    }

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            install(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in names:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
                names.remove(fullname)
                if not names:
                    sys.meta_path.remove(self)
                return spec

    sys.meta_path.insert(0, Finder())
