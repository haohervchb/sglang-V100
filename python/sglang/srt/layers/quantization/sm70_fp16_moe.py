"""V100 FP16 MoE runner backed by GooseLLM's TurboMind s884h kernels."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

logger = logging.getLogger(__name__)

_DEFAULT_OPS_ROOT = Path(__file__).resolve().parents[3] / "jit_kernel"
_OPS_LOADED = False
_LOGGED_CONFIGS: set[tuple[int, int, int, int]] = set()


def _load_sm70_ops() -> bool:
    global _OPS_LOADED
    if _OPS_LOADED:
        return True
    root = Path(os.environ.get("SGLANG_SM70_FP16_MOE_OPS_DIR", _DEFAULT_OPS_ROOT))
    try:
        torch.ops.load_library(str(root / "_sm70_fp16_moe_ops.abi3.so"))
        torch.ops.load_library(str(root / "_sm70_fp16_moe_permute.abi3.so"))
        required = (
            hasattr(torch.ops._goose_C, "sm70_f16_prepare")
            and hasattr(torch.ops._goose_C, "sm70_f16_moe_build_strided_ptrs")
            and hasattr(torch.ops._goose_C, "sm70_f16_moe_gemm_sm70_out")
            and hasattr(torch.ops._goose_moe_C, "moe_permute")
            and hasattr(torch.ops._goose_moe_C, "moe_unpermute")
        )
        if not required:
            raise RuntimeError("one or more required SM70 MoE operators are absent")
    except Exception as exc:
        logger.warning("SM70 FP16 MoE operators unavailable: %s", exc)
        return False
    _OPS_LOADED = True
    return True


def can_use_sm70_fp16_moe(params_dtype: torch.dtype) -> bool:
    # Opt-in until long-prefill model-level correctness is fully validated.
    if os.environ.get("SGLANG_SM70_FP16_MOE", "0") != "1":
        return False
    if params_dtype != torch.float16 or not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 7:
        return False
    return _load_sm70_ops()


class SM70FP16MoEMethod(UnquantizedFusedMoEMethod):
    """Batched FP16 expert GEMMs specialized for V100."""

    def __init__(self):
        if not _load_sm70_ops():
            raise RuntimeError("SM70 FP16 MoE operators are unavailable")
        super().__init__(use_triton_kernels=False)

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts, w13_n, hidden_size = layer.w13_weight.shape
        intermediate_size = w13_n // 2
        dtype = layer.w13_weight.dtype
        device = layer.w13_weight.device

        # TurboMind's gated epilogue expects gate/up rows interleaved.
        gate, up = layer.w13_weight[:, :intermediate_size], layer.w13_weight[:, intermediate_size:]
        w13 = torch.stack((gate, up), dim=2).reshape(-1, hidden_size)
        r13 = torch.ops._goose_C.sm70_f16_prepare(w13)
        w13_k_ld = int(r13[1][0].item())
        w13_tm = r13[0].reshape(num_experts, 2 * intermediate_size, hidden_size)

        w2 = layer.w2_weight.reshape(-1, intermediate_size)
        r2 = torch.ops._goose_C.sm70_f16_prepare(w2)
        w2_k_ld = int(r2[1][0].item())
        w2_tm = r2[0].reshape(num_experts, hidden_size, intermediate_size)

        layer.w13_tm_weight = Parameter(w13_tm, requires_grad=False)
        layer.w2_tm_weight = Parameter(w2_tm, requires_grad=False)
        layer.w13_weight = None
        layer.w2_weight = None

        layer.w13_weight_ptrs = Parameter(
            torch.ops._goose_C.sm70_f16_moe_build_strided_ptrs(
                layer.w13_tm_weight, w13_k_ld, num_experts
            )[0],
            requires_grad=False,
        )
        layer.w2_weight_ptrs = Parameter(
            torch.ops._goose_C.sm70_f16_moe_build_strided_ptrs(
                layer.w2_tm_weight, w2_k_ld, num_experts
            )[0],
            requires_grad=False,
        )

        layer.sm70_num_experts = num_experts
        layer.sm70_hidden_size = hidden_size
        layer.sm70_intermediate_size = intermediate_size
        layer.sm70_top_k = self.moe_runner_config.top_k
        self._allocate_buffers(layer, 32)
        config = (num_experts, hidden_size, intermediate_size, layer.sm70_top_k)
        if config not in _LOGGED_CONFIGS:
            _LOGGED_CONFIGS.add(config)
            logger.info(
                "SM70 FP16 MoE: using GooseLLM TurboMind batched GEMM "
                "(experts=%d hidden=%d intermediate=%d topk=%d)",
                *config,
            )

    @staticmethod
    def _allocate_buffers(layer: torch.nn.Module, max_tokens: int) -> None:
        e, h, i, k = (
            layer.sm70_num_experts,
            layer.sm70_hidden_size,
            layer.sm70_intermediate_size,
            layer.sm70_top_k,
        )
        slots = max_tokens * k
        device = layer.w13_tm_weight.device
        layer.sm70_max_tokens = max_tokens
        layer.sm70_output = torch.empty(max_tokens, h, dtype=torch.float16, device=device)
        layer.sm70_permuted = torch.empty(slots, h, dtype=torch.float16, device=device)
        layer.sm70_gate_up = torch.empty(slots, i, dtype=torch.float16, device=device)
        layer.sm70_sorted_output = torch.empty(slots, h, dtype=torch.float16, device=device)
        layer.sm70_offsets64 = torch.empty(e + 1, dtype=torch.int64, device=device)
        layer.sm70_offsets32 = torch.empty(e + 1, dtype=torch.int32, device=device)
        layer.sm70_inverse = torch.empty(max_tokens, k, dtype=torch.int32, device=device)
        layer.sm70_ids32 = torch.empty(max_tokens, k, dtype=torch.int32, device=device)
        layer.sm70_token_expert = torch.arange(slots, dtype=torch.int32, device=device).view(max_tokens, k)
        layer.sm70_permuted_idx = torch.empty(slots, dtype=torch.int32, device=device)
        layer.sm70_m_indices = torch.empty(slots, dtype=torch.int32, device=device)

    def apply(self, layer, dispatch_output):
        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        tokens = x.shape[0]
        k = layer.sm70_top_k
        slots = tokens * k
        if tokens <= layer.sm70_max_tokens:
            output = layer.sm70_output[:tokens]
            permuted = layer.sm70_permuted[:slots]
            gate_up = layer.sm70_gate_up[:slots]
            sorted_output = layer.sm70_sorted_output[:slots]
            offsets64 = layer.sm70_offsets64
            offsets32 = layer.sm70_offsets32
            inverse = layer.sm70_inverse[:tokens]
            ids32 = layer.sm70_ids32[:tokens]
            token_expert = layer.sm70_token_expert[:tokens]
            permuted_idx = layer.sm70_permuted_idx[:slots]
            m_indices = layer.sm70_m_indices[:slots]
        else:
            # Prefill workspaces must be transient. Retaining one large buffer
            # set per MoE layer exhausts VRAM before the first request.
            device = x.device
            h, i, e = (
                layer.sm70_hidden_size,
                layer.sm70_intermediate_size,
                layer.sm70_num_experts,
            )
            output = torch.empty(tokens, h, dtype=torch.float16, device=device)
            permuted = torch.empty(slots, h, dtype=torch.float16, device=device)
            gate_up = torch.empty(slots, i, dtype=torch.float16, device=device)
            sorted_output = torch.empty(slots, h, dtype=torch.float16, device=device)
            offsets64 = torch.empty(e + 1, dtype=torch.int64, device=device)
            offsets32 = torch.empty(e + 1, dtype=torch.int32, device=device)
            inverse = torch.empty(tokens, k, dtype=torch.int32, device=device)
            ids32 = torch.empty(tokens, k, dtype=torch.int32, device=device)
            token_expert = torch.arange(slots, dtype=torch.int32, device=device).view(tokens, k)
            permuted_idx = torch.empty(slots, dtype=torch.int32, device=device)
            m_indices = torch.empty(slots, dtype=torch.int32, device=device)
        output.zero_()
        if slots == 0:
            return StandardCombineInput(hidden_states=output)

        ids32.copy_(topk.topk_ids)
        torch.ops._goose_moe_C.moe_permute(
            x,
            ids32,
            token_expert,
            None,
            layer.sm70_num_experts,
            layer.sm70_num_experts,
            k,
            None,
            permuted,
            offsets64,
            inverse,
            permuted_idx,
            m_indices,
        )
        offsets32.copy_(offsets64)

        torch.ops._goose_C.sm70_f16_moe_gemm_sm70_out(
            gate_up,
            permuted,
            offsets32,
            layer.w13_weight_ptrs,
            layer.sm70_num_experts,
            layer.sm70_hidden_size,
            2 * layer.sm70_intermediate_size,
            True,
        )
        torch.ops._goose_C.sm70_f16_moe_gemm_sm70_out(
            sorted_output,
            gate_up,
            offsets32,
            layer.w2_weight_ptrs,
            layer.sm70_num_experts,
            layer.sm70_intermediate_size,
            layer.sm70_hidden_size,
            False,
        )
        torch.ops._goose_moe_C.moe_unpermute(
            sorted_output,
            topk.topk_weights,
            inverse,
            offsets64,
            k,
            output,
        )
        return StandardCombineInput(hidden_states=output)
