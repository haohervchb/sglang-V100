import logging

from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils.common import is_blackwell, is_hip, is_musa

logger = logging.getLogger(__name__)


class DraftBackendFactory:
    def __init__(
        self,
        server_args: ServerArgs,
        draft_model_runner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.server_args = server_args
        self.draft_model_runner = draft_model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.draft_attn_backend = server_args.speculative_draft_attention_backend

    def _create_backend(
        self, backend_name: str, backend_map: dict, error_template: str
    ):
        backend_type = (
            self.draft_attn_backend
            if self.draft_attn_backend
            else getattr(self.server_args, backend_name)
        )
        if backend_type is None:
            backend_type = self.server_args.attention_backend

        if backend_type not in backend_map:
            raise ValueError(error_template.format(backend_type=backend_type))

        return backend_map[backend_type]()

    def create_decode_backend(self):
        if self.speculative_num_steps == 1:
            return None

        if self._is_qwen_qsa_draft_model():
            return self._create_qwen_qsa_decode_backend()

        # Returns a per-step CONTAINER, not an AttentionBackend, so
        # attn_backend_wrapper_for_draft_extend cannot give it a conv sidecar.
        _assert_draft_needs_no_conv_sidecar(self.draft_model_runner)

        backend_map = {
            "flashinfer": self._create_flashinfer_decode_backend,
            "triton": self._create_triton_decode_backend,
            # The V100 TileLang backend delegates multi-step built-in MTP
            # draft attention to Triton.
            "tilelang_fa_v100": self._create_triton_decode_backend,
            "flash_attn_v100": self._create_triton_decode_backend,
            "aiter": self._create_aiter_decode_backend,
            "fa3": self._create_fa3_decode_backend,
            "hybrid_linear_attn": (
                self._create_fa3_decode_backend
                if not is_blackwell()
                else self._create_triton_decode_backend
            ),
            "flashmla": self._create_flashmla_decode_backend,
            "trtllm_mha": self._create_trtllm_mha_decode_backend,
            "trtllm_mla": self._create_trtllm_mla_decode_backend,
            "cutedsl_mla": self._create_cutedsl_mla_decode_backend,
            "tokenspeed_mla": self._create_tokenspeed_mla_decode_backend,
            "dsa": self._create_dsa_decode_backend,
            "nsa": self._create_dsa_decode_backend,  # Deprecated alias for "dsa"
            "ascend": self._create_ascend_decode_backend,
            "fa4": self._create_fa4_decode_backend,
            "dsv4": self._create_dsv4_decode_backend,
        }

        return self._create_backend(
            "decode_attention_backend",
            backend_map,
            "EAGLE is not supported in decode attention backend {backend_type}",
        )

    def create_draft_extend_backend(self):
        if self._is_qwen_qsa_draft_model():
            from sglang.srt.layers.attention.qsa.config import (
                QSA_VARIANT_COMPRESSED,
                parse_qsa_profile,
            )

            profile = parse_qsa_profile(
                self.draft_model_runner.model_config.hf_config
            )
            if profile is not None and profile.variant != QSA_VARIANT_COMPRESSED:
                # Tokenwise QSA has no graph-stable indexer metadata; keep
                # the intentional eager draft-extend path and never fall
                # back to a dense backend.
                return None
            # Compressed QSA draft-extend uses the draft model runner's own
            # (QSA-wrapped hybrid) backend.  Its replay path pads the
            # variable accepted-token rows to the captured static width, so
            # the draft-extend CUDA graph expresses the dynamic accept count.
            return self.draft_model_runner.attn_backend

        backend_map = {
            "flashinfer": self._create_flashinfer_prefill_backend,
            "triton": self._create_triton_prefill_backend,
            "tilelang_fa_v100": self._create_triton_prefill_backend,
            "flash_attn_v100": self._create_triton_prefill_backend,
            "aiter": self._create_aiter_prefill_backend,
            "fa3": self._create_fa3_prefill_backend,
            "hybrid_linear_attn": (
                self._create_fa3_prefill_backend
                if not is_blackwell()
                else self._create_triton_prefill_backend
            ),
            "flashmla": self._create_flashmla_prefill_backend,
            "trtllm_mha": self._create_trtllm_mha_prefill_backend,
            "trtllm_mla": self._create_trtllm_mla_prefill_backend,
            # cute-dsl MLA only supports decode; draft-extend falls back to trtllm-gen.
            "cutedsl_mla": self._create_trtllm_mla_prefill_backend,
            "tokenspeed_mla": self._create_tokenspeed_mla_prefill_backend,
            "dsa": self._create_dsa_prefill_backend,
            "nsa": self._create_dsa_prefill_backend,  # Deprecated alias for "dsa"
            "ascend": self._create_ascend_prefill_backend,
            "fa4": self._create_fa4_prefill_backend,
            "dsv4": self._create_dsv4_prefill_backend,
        }
        backend_name = (
            "decode_attention_backend"
            if self.server_args.speculative_attention_mode == "decode"
            else "prefill_attention_backend"
        )
        return self._create_backend(
            backend_name,
            backend_map,
            "EAGLE is not supported in attention backend {backend_type}",
        )

    def _is_qwen_qsa_draft_model(self) -> bool:
        from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

        return is_qwen_qsa(self.draft_model_runner.model_config.hf_config)

    def _create_qwen_qsa_decode_backend(self):
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseMultiStepDraftBackend,
        )

        backend = QwenSparseMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )
        backend.prefill_attention_backend_str = "qsa"
        backend.decode_attention_backend_str = "qsa"
        for child in backend.attn_backends:
            child.prefill_attention_backend_str = "qsa"
            child.decode_attention_backend_str = "qsa"
        return backend

    def _create_dsa_decode_backend(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnMultiStepBackend,
        )

        return DeepseekSparseAttnMultiStepBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_dsa_prefill_backend(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        return DeepseekSparseAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_flashinfer_decode_backend(self):
        if not get_global_server_args().use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            return FlashInferMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            return FlashInferMLAMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )

    def _create_triton_decode_backend(self):
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )

        return TritonMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_aiter_decode_backend(self):
        from sglang.srt.layers.attention.aiter_backend import AiterMultiStepDraftBackend

        return AiterMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_fa_decode_backend(self, fa_impl_ver: int = 3):
        if not is_musa():
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionMultiStepBackend,
            )
        else:
            from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
                MusaFlashAttentionMultiStepBackend as FlashAttentionMultiStepBackend,
            )

        return FlashAttentionMultiStepBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            fa_impl_ver=fa_impl_ver,
        )

    def _create_fa3_decode_backend(self):
        return self._create_fa_decode_backend(fa_impl_ver=3)

    def _create_fa4_decode_backend(self):
        return self._create_fa_decode_backend(fa_impl_ver=4)

    def _create_flashmla_decode_backend(self):
        from sglang.srt.layers.attention.flashmla_backend import (
            FlashMLAMultiStepDraftBackend,
        )

        return FlashMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mha_decode_backend(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            TRTLLMHAAttnMultiStepDraftBackend,
        )

        return TRTLLMHAAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mla_decode_backend(self, backend: str = "trtllm-gen"):
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLAMultiStepDraftBackend,
        )

        return TRTLLMMLAMultiStepDraftBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            backend=backend,
        )

    def _create_cutedsl_mla_decode_backend(self):
        return self._create_trtllm_mla_decode_backend(backend="cute-dsl")

    def _create_tokenspeed_mla_decode_backend(self):
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "tokenspeed_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLAMultiStepDraftBackend,
        )

        return TokenspeedMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_ascend_decode_backend(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnMultiStepDraftBackend,
        )

        return AscendAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_dsv4_decode_backend(self):
        if is_hip():
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4MultiStepBackend,
            )
        else:
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4MultiStepBackend,
            )

        return DeepseekV4MultiStepBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_flashinfer_prefill_backend(self):
        if not get_global_server_args().use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferAttnBackend,
            )

            return FlashInferAttnBackend(self.draft_model_runner, skip_prefill=False)
        else:
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAAttnBackend,
            )

            return FlashInferMLAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_triton_prefill_backend(self):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        return TritonAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_aiter_prefill_backend(self):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        return AiterAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_fa_prefill_backend(self, fa_impl_ver: int = 3):
        if not is_musa():
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )
        else:
            from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
                MusaFlashAttentionBackend as FlashAttentionBackend,
            )
        return FlashAttentionBackend(
            self.draft_model_runner, skip_prefill=False, fa_impl_ver=fa_impl_ver
        )

    def _create_fa3_prefill_backend(self):
        return self._create_fa_prefill_backend(fa_impl_ver=3)

    def _create_fa4_prefill_backend(self):
        return self._create_fa_prefill_backend(fa_impl_ver=4)

    def _create_trtllm_mha_prefill_backend(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        return TRTLLMHAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_trtllm_mla_prefill_backend(self):
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        return TRTLLMMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_tokenspeed_mla_prefill_backend(self):
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "tokenspeed_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )

        return TokenspeedMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_ascend_prefill_backend(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )

        return AscendAttnBackend(self.draft_model_runner)

    def _create_flashmla_prefill_backend(self):
        logger.warning(
            "flashmla prefill backend is not yet supported for draft extend."
        )
        return None

    def _create_dsv4_prefill_backend(self):
        if is_hip():
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4HipRadixBackend,
            )

            return DeepseekV4HipRadixBackend(
                self.draft_model_runner, skip_prefill=False
            )
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        return DeepseekV4AttnBackend(self.draft_model_runner, skip_prefill=False)
