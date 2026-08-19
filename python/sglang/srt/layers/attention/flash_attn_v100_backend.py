"""FlashAttention-2 V100 (SM70) attention backend for sglang.

Prefill (forward_extend): prefers the vendored TileLang FA2 kernel tuned for
SM70 and falls back to ai-bond's ``flash_attn_v100_cuda.paged_fwd``. Both read
the paged KV cache as ``[num_pages, page_size, num_kv_heads, head_dim]``
(block-major, normally page_size=16), giving coalesced block reads on V100.

Native prefix handling is via ``prefix_kv_lens`` — no ragged+paged+merge_state
double-kernel, no FlattenKV, no FlashInfer wrapper ``plan()`` CPU overhead.

Decode (forward_decode): the exact Qwen TP4 H6/Hkv1/D256 FP16 shape uses
1Cat's grouped XQA kernel with its Volta-specific operand movement. Other
shapes retain sglang's GooseLLM-derived Triton SM70 split-K path. The old
ai-bond ``decode_fwd`` kernel is intentionally not used (it produces inf/NaN
exp_sums).

The KV cache layout is shared with the Triton decode path (which views the 4D
cache as flat 3D — see ``triton_backend._flatten_paged_kv_cache``).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# page_size used by this backend (must match server_args SM70 default).
V100_PAGE_SIZE = 16

_paged_forward = None
_paged_forward_loaded = False
_use_tilelang = None  # None = unset, True/False = cached
_xqa_decode = None
_paged_decode = None
_wmma_decode = None
_xqa_decode_loaded = False
_xqa_e4m3_supported = False
_fp8_e5m2_paged_kv_to_fp16 = None

_E5M2_PREFILL_BRIDGE_PAGE_SIZE = 784


def _should_skip_triton_prefill(model_runner: "ModelRunner") -> bool:
    """Keep baseline decode lean while allocating metadata needed by spec verify."""
    uses_sm70_fp8_kv = model_runner.kv_cache_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    return not (model_runner.spec_algorithm.is_speculative() or uses_sm70_fp8_kv)


def _get_native_paged_attention_params(
    layer: "RadixAttention", default_causal: bool
) -> tuple[bool, int]:
    """Resolve the per-layer mask used by the native paged extend kernel."""
    causal = default_causal and not (
        layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY
    )
    sliding_window_size = (
        int(layer.sliding_window_size)
        if causal
        and layer.sliding_window_size is not None
        and layer.sliding_window_size >= 0
        else -1
    )
    return causal, sliding_window_size


def _is_dflash_draft_native_shape_supported(
    layer: "RadixAttention", kv_cache_dtype: torch.dtype = torch.float16
) -> bool:
    """Return whether 1Cat's native paged kernel supports this draft shape."""
    return (
        layer.head_dim == 128
        and layer.tp_k_head_num > 0
        # The grouped TileLang verifier supports any integral GQA ratio.  The
        # earlier DFlash checkpoints use GQA-4; Qwen3.8 DSpark is trained with
        # 40 Q / 8 KV heads and therefore uses GQA-5.
        and layer.tp_q_head_num % layer.tp_k_head_num == 0
        # The E4M3 DFlash integration is validated only for the TP4 layout.
        # The kernel is numerically sound with four KV heads in isolation, but
        # the complete TP2 verify pipeline can otherwise admit corrupt tokens.
        and (kv_cache_dtype != torch.float8_e4m3fn or layer.tp_k_head_num == 2)
    )


def _load_paged_forward():
    """Lazy-load the paged forward kernel. Prefers the vendored tilelang-fa-v100
    on SM70 (from GooseLLM, tuned for V100); falls back to the ai-bond kernel
    (flash_attn_v100_cuda.paged_fwd) when tilelang is unavailable."""
    global _paged_forward, _paged_forward_loaded, _use_tilelang
    if _paged_forward_loaded:
        return _paged_forward
    _paged_forward_loaded = True

    # Try vendored tilelang-fa-v100 first (preferred on SM70).
    try:
        import tilelang  # noqa: F401

        from sglang.srt.layers.attention.tilelang_fa_v100 import (
            paged_forward as _tl_paged,
        )
        from sglang.srt.utils.common import get_device_sm, is_cuda

        if is_cuda() and get_device_sm() == 70:
            _paged_forward = _tl_paged
            _use_tilelang = True
            logger.info("paged prefill: using vendored tilelang-fa-v100 kernel (SM70).")
            return _paged_forward
    except Exception:
        pass

    # Fall back to ai-bond flash_attn_v100_cuda
    _load_ai_bond_paged()
    return _paged_forward


def _load_ai_bond_paged():
    """Lazy-load the ai-bond paged forward kernel via GooseLLM's wrapper."""
    global _paged_forward, _use_tilelang
    if _paged_forward is not None:
        return _paged_forward

    env_root = os.environ.get("FLASH_ATTN_V100_DIR")
    candidate_roots = []
    if env_root:
        candidate_roots.append(Path(env_root).expanduser())
    candidate_roots.extend(
        [
            Path.home() / "GooseLLM" / "csrc" / "flash_attention_v100",
            Path.home() / "flash-attention-v100-ai-bond",
            Path.home() / "flash-attention-v100",
        ]
    )
    for root in candidate_roots:
        root = root.resolve()
        if (root / "flash_attn_v100" / "__init__.py").exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)

    try:
        from flash_attn_v100 import flash_attn_paged_forward  # noqa: F401

        _paged_forward = flash_attn_paged_forward
        _use_tilelang = False
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "flash_attn_v100 backend requires the ai-bond flash_attn_v100_cuda "
            f"kernel + python wrapper. Import failed: {e}. Set FLASH_ATTN_V100_DIR "
            "or install flash-attention-v100."
        ) from e
    return _paged_forward


def _load_xqa_decode():
    """Load 1Cat's SM70 paged decode kernels when they are installed."""
    global _paged_decode, _wmma_decode, _xqa_decode
    global _xqa_decode_loaded, _xqa_e4m3_supported
    global _fp8_e5m2_paged_kv_to_fp16
    if _xqa_decode_loaded:
        return _xqa_decode, _xqa_e4m3_supported, _wmma_decode
    _xqa_decode_loaded = True
    try:
        from flash_attn_v100 import (
            flash_attn_decode_paged,
            flash_attn_decode_paged_wmma,
            flash_attn_decode_paged_xqa,
            flash_attn_decode_paged_xqa_available,
            flash_attn_interface,
        )
        try:
            from flash_attn_v100 import fp8_e5m2_paged_kv_to_fp16
        except ImportError:
            fp8_e5m2_paged_kv_to_fp16 = None
        _fp8_e5m2_paged_kv_to_fp16 = fp8_e5m2_paged_kv_to_fp16

        if not flash_attn_decode_paged_xqa_available():
            return None, False, flash_attn_decode_paged_wmma
        _paged_decode = flash_attn_decode_paged
        _wmma_decode = flash_attn_decode_paged_wmma
        _xqa_decode = flash_attn_decode_paged_xqa
        _xqa_e4m3_supported = bool(
            getattr(
                flash_attn_interface,
                "FLASH_ATTN_V100_XQA_E4M3_SUPPORTED",
                False,
            )
        )
        logger.info(
            "linear verifier: loaded FlashAttention-V100 paged XQA "
            "(E4M3=%s) and strict WMMA decode.",
            _xqa_e4m3_supported,
        )
    except Exception as e:  # noqa: BLE001
        logger.info("linear verifier: enhanced paged decode unavailable (%s).", e)
    return _xqa_decode, _xqa_e4m3_supported, _wmma_decode


def _dflash_target_xqa_requested() -> bool:
    """Select the SM70 DFlash target verifier.

    XQA's independent decode rows reread the same long prefix for every token
    in the speculative block.  The grouped TileLang verifier shares that scan
    for FP16 KV and uses a cached E4M3-to-FP16 lookup for compact KV.
    """
    default = "0"
    value = os.environ.get("SGLANG_V100_DFLASH_TARGET_XQA", default).strip().lower()
    if value not in (
        "0",
        "false",
        "off",
        "no",
        "1",
        "true",
        "on",
        "yes",
    ):
        raise ValueError(
            "SGLANG_V100_DFLASH_TARGET_XQA must be a boolean value, "
            f"got {value!r}."
        )
    return value in ("1", "true", "on", "yes")


def _long_decode_xqa_requested() -> bool:
    """Enable the measured page-16 G6/D256 native decode specialization."""
    value = os.environ.get("SGLANG_V100_LONG_DECODE_XQA", "1").strip().lower()
    if value not in (
        "0",
        "false",
        "off",
        "no",
        "1",
        "true",
        "on",
        "yes",
    ):
        raise ValueError(
            "SGLANG_V100_LONG_DECODE_XQA must be a boolean value, "
            f"got {value!r}."
        )
    return value in ("1", "true", "on", "yes")


@dataclass
class FlashAttnV100ExtendMetadata:
    """Per-forward metadata for the paged prefill (extend) path."""

    page_table: torch.Tensor  # [num_seqs, max_pages] int32 — page indices
    seq_lens: torch.Tensor  # [num_seqs] int32 — total KV length per seq
    query_start_loc: torch.Tensor  # [num_seqs+1] int32 — cumsum of query lens
    prefix_kv_lens: torch.Tensor  # [num_seqs] int32 — cached prefix length
    causal: bool
    # SWA layers use a separate KV pool with a different physical page
    # numbering. Keep a second table for that pool.
    swa_page_table: Optional[torch.Tensor] = None
    # A linear speculative block can be expressed as independent decode rows
    # with monotonically increasing visible sequence lengths. These persistent
    # buffers are shared by every full-attention layer and are graph-safe.
    smallq_page_table: Optional[torch.Tensor] = None
    smallq_swa_page_table: Optional[torch.Tensor] = None
    smallq_seq_lens: Optional[torch.Tensor] = None
    smallq_active_num_partitions: Optional[torch.Tensor] = None
    smallq_max_seq_len: int = 0


class FlashAttnV100Backend(AttentionBackend):
    """V100 attention backend: TileLang/native paged prefill + Triton decode."""

    # Decode metadata is rebuilt by the triton backend from cuda-graph buffers;
    # this backend never reads seq_lens_cpu / seq_lens_sum for decode.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: "ModelRunner",
        skip_prefill: bool = False,
    ):
        super().__init__()
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        self.model_runner = model_runner
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        if self.page_size != V100_PAGE_SIZE:
            logger.warning(
                f"flash_attn_v100 backend expects page_size={V100_PAGE_SIZE}, "
                f"got page_size={self.page_size}. page_size=1 is supported for "
                "Mamba no_buffer scheduling, but page_size=16 gives coalesced "
                "paged-prefill reads."
            )

        self.max_context_len = model_runner.model_config.context_len
        self._max_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.skip_prefill = skip_prefill
        self._uses_sm70_e4m3_kv = (
            model_runner.kv_cache_dtype == torch.float8_e4m3fn
        )
        self._uses_sm70_e5m2_kv = model_runner.kv_cache_dtype == torch.float8_e5m2
        self._uses_sm70_fp8_kv = (
            self._uses_sm70_e4m3_kv or self._uses_sm70_e5m2_kv
        )
        (
            self._xqa_decode,
            self._xqa_e4m3_supported,
            self._wmma_decode,
        ) = _load_xqa_decode()
        self._paged_decode = _paged_decode
        fp8_prefill_scratch = (
            os.environ.get("SGLANG_V100_FP8_PREFILL_SCRATCH", "1").strip().lower()
        )
        if fp8_prefill_scratch not in (
            "0",
            "false",
            "off",
            "no",
            "1",
            "true",
            "on",
            "yes",
        ):
            raise ValueError(
                "SGLANG_V100_FP8_PREFILL_SCRATCH must be a boolean value, "
                f"got {fp8_prefill_scratch!r}."
            )
        self._fp8_prefill_scratch_enabled = (
            self._uses_sm70_fp8_kv
            and fp8_prefill_scratch in ("1", "true", "on", "yes")
            and model_runner.model_config.head_dim
            == model_runner.model_config.v_head_dim
            and (
                self._uses_sm70_e4m3_kv
                or _fp8_e5m2_paged_kv_to_fp16 is not None
            )
        )
        self._long_decode_xqa_enabled = (
            _long_decode_xqa_requested() and self._xqa_decode is not None
        )
        if self._long_decode_xqa_enabled:
            # These are the accepted page-16 settings from an isolated V100
            # sweep. Preserve explicit user overrides for diagnosis/rollback.
            os.environ.setdefault("VLLM_FLASH_V100_XQA_G6_DUAL_CTA", "1")
            os.environ.setdefault("VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT", "2")
            os.environ.setdefault("VLLM_FLASH_V100_XQA_SPLIT_REDUCE", "1")
            logger.info(
                "SM70 decode: enabled native G6/D256 page-16 XQA "
                "(dual CTA, contiguous block layout, split reducer)."
            )
        target_xqa_requested = _dflash_target_xqa_requested()
        self._target_xqa_enabled = (
            target_xqa_requested
            and self._uses_sm70_e4m3_kv
            and self._xqa_decode is not None
            and self._xqa_e4m3_supported
        )
        # 1Cat's optimized quantized-KV contract on Volta is E5M2. Unlike the
        # E4M3 verifier (where grouped TileLang can be faster), E5M2 must use
        # the native byte-decoding XQA path for Qwen3.8's GQA-6/D256 target.
        # The generic Triton fallback below remains available for unsupported
        # shapes and explicit native-linear-verify opt-out.
        self._target_e5m2_xqa_enabled = (
            self._uses_sm70_e5m2_kv
            and self._xqa_decode is not None
            and model_runner.spec_algorithm.is_dflash_family()
            and not model_runner.is_draft_worker
        )
        if (
            target_xqa_requested
            and self._uses_sm70_e4m3_kv
            and model_runner.spec_algorithm.is_dflash_family()
            and not model_runner.is_draft_worker
            and not self._target_xqa_enabled
        ):
            logger.info(
                "DFLASH FP8 target verifier: marked E4M3 XQA is unavailable; "
                "using the FP16-scratch compatibility path."
            )

        # Eagerly validate the kernel where it is required at startup. Ordinary
        # FP8 prefill loads it lazily after materializing active pages as FP16.
        if not self._uses_sm70_fp8_kv or model_runner.spec_algorithm.is_speculative():
            _load_paged_forward()
        if (
            model_runner.spec_algorithm.is_dflash_family()
            and not model_runner.is_draft_worker
        ):
            if self._target_xqa_enabled or self._target_e5m2_xqa_enabled:
                target_verifier = "independent-row native XQA"
            elif _use_tilelang:
                target_verifier = "grouped TileLang block verifier"
            else:
                target_verifier = "FP16-scratch compatibility verifier"
            logger.info("DFLASH target verifier: %s.", target_verifier)

        # Decode is delegated to the Triton backend (GooseLLM SM70 split-K
        # tuning already lives there). Speculative target verification also
        # delegates its extend pass, so it needs Triton's qo/mask indptr
        # buffers; ordinary decoding can retain the lean decode-only setup.
        self._triton = TritonAttnBackend(
            model_runner,
            skip_prefill=_should_skip_triton_prefill(model_runner),
        )

        self.forward_metadata: Optional[FlashAttnV100ExtendMetadata] = None
        # Buffers for piecewise cuda-graph capture of extend.
        self._cg_page_table: Optional[torch.Tensor] = None
        self._cg_swa_page_table: Optional[torch.Tensor] = None
        self._cg_seq_lens: Optional[torch.Tensor] = None
        self._cg_query_start_loc: Optional[torch.Tensor] = None
        self._cg_prefix_kv_lens: Optional[torch.Tensor] = None
        self._cg_strided: Optional[torch.Tensor] = None
        self._cg_smallq_page_table: Optional[torch.Tensor] = None
        self._cg_smallq_swa_page_table: Optional[torch.Tensor] = None
        self._cg_smallq_seq_lens: Optional[torch.Tensor] = None
        self._cg_smallq_active_num_partitions: Optional[torch.Tensor] = None
        self._xqa_decode_page_table = torch.empty(
            1,
            self._max_pages,
            dtype=torch.int32,
            device=self.device,
        )
        self._xqa_decode_active_num_partitions = torch.ones(
            1, dtype=torch.int32, device=self.device
        )
        self._fp8_verify_k_scratch: Optional[torch.Tensor] = None
        self._fp8_verify_v_scratch: Optional[torch.Tensor] = None
        self._fp8_verify_page_table: Optional[torch.Tensor] = None
        self._fp8_verify_kv_indptr: Optional[torch.Tensor] = None
        self._fp8_verify_logical_indices: Optional[torch.Tensor] = None
        self._fp8_prefill_k_scratch: Optional[torch.Tensor] = None
        self._fp8_prefill_v_scratch: Optional[torch.Tensor] = None
        self._fp8_prefill_logical_pages: Optional[torch.Tensor] = None
        self._fp8_prefill_page_capacity = 0
        self._fp8_prefill_head_shape: Optional[tuple[int, int]] = None
        self._fp8_prefill_scratch_logged = False
        self._e5m2_prefill_k_scratch: Optional[torch.Tensor] = None
        self._e5m2_prefill_v_scratch: Optional[torch.Tensor] = None
        self._e5m2_prefill_page_table: Optional[torch.Tensor] = None
        self._e5m2_prefill_page_capacity = 0
        self._e5m2_prefill_head_shape: Optional[tuple[int, int]] = None
        self._e5m2_prefill_scratch_logged = False

        # Older FlashAttention-V100 builds do not provide direct E4M3 XQA.
        # Retain the exact FP16-scratch verifier for those builds and for
        # explicit XQA opt-out. The persistent cache remains E4M3.
        verify_scratch = (
            os.environ.get("SGLANG_V100_DFLASH_FP8_VERIFY_SCRATCH", "1").strip().lower()
        )
        if verify_scratch not in (
            "0",
            "false",
            "off",
            "no",
            "1",
            "true",
            "on",
            "yes",
        ):
            raise ValueError(
                "SGLANG_V100_DFLASH_FP8_VERIFY_SCRATCH must be a boolean "
                f"value, got {verify_scratch!r}."
            )
        verify_scratch_enabled = verify_scratch in ("1", "true", "on", "yes")
        max_running_requests = model_runner.server_args.max_running_requests
        if (
            verify_scratch_enabled
            and not self._target_xqa_enabled
            and not _use_tilelang
            and self._uses_sm70_e4m3_kv
            and model_runner.spec_algorithm.is_dflash_family()
            and not model_runner.is_draft_worker
            and max_running_requests == 1
        ):
            kv_heads = model_runner.model_config.get_num_kv_heads(model_runner.tp_size)
            head_dim = model_runner.model_config.head_dim
            v_head_dim = model_runner.model_config.v_head_dim
            if head_dim == v_head_dim:
                scratch_shape = (
                    self._max_pages,
                    self.page_size,
                    kv_heads,
                    head_dim,
                )
                self._fp8_verify_k_scratch = torch.empty(
                    scratch_shape, dtype=torch.float16, device=self.device
                )
                self._fp8_verify_v_scratch = torch.empty_like(
                    self._fp8_verify_k_scratch
                )
                self._fp8_verify_page_table = torch.arange(
                    self._max_pages,
                    dtype=torch.int32,
                    device=self.device,
                ).unsqueeze(0)
                self._fp8_verify_kv_indptr = torch.zeros(
                    2, dtype=torch.int32, device=self.device
                )
                self._fp8_verify_logical_indices = torch.arange(
                    self.max_context_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                scratch_gib = (
                    2
                    * self._fp8_verify_k_scratch.numel()
                    * self._fp8_verify_k_scratch.element_size()
                    / (1024**3)
                )
                logger.info(
                    "DFLASH FP8 target verifier: allocated %.2f GiB FP16 "
                    "logical-page scratch (persistent KV remains E4M3).",
                    scratch_gib,
                )

    def _uses_fp8_prefill_scratch(self, forward_mode) -> bool:
        return (
            self._fp8_prefill_scratch_enabled
            and forward_mode.is_extend()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend(include_v2=True)
        )

    def _ensure_fp8_prefill_scratch(
        self,
        total_pages: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_shape = (num_kv_heads, head_dim)
        if self._fp8_prefill_head_shape != head_shape:
            self._fp8_prefill_k_scratch = None
            self._fp8_prefill_v_scratch = None
            self._fp8_prefill_logical_pages = None
            self._fp8_prefill_page_capacity = 0
            self._fp8_prefill_head_shape = head_shape

        if self._fp8_prefill_page_capacity < total_pages:
            capacity = 1 << (max(1, total_pages) - 1).bit_length()
            self._fp8_prefill_k_scratch = torch.empty(
                capacity,
                self.page_size,
                num_kv_heads,
                head_dim,
                dtype=torch.float16,
                device=self.device,
            )
            self._fp8_prefill_v_scratch = torch.empty_like(self._fp8_prefill_k_scratch)
            self._fp8_prefill_logical_pages = torch.arange(
                capacity, dtype=torch.int32, device=self.device
            )
            self._fp8_prefill_page_capacity = capacity

        assert self._fp8_prefill_k_scratch is not None
        assert self._fp8_prefill_v_scratch is not None
        assert self._fp8_prefill_logical_pages is not None
        if not self._fp8_prefill_scratch_logged:
            logger.info(
                "SM70 FP8 KV prefill: dequantizing each active prefix page once "
                "into reusable FP16 scratch before native paged attention."
            )
            self._fp8_prefill_scratch_logged = True
        return (
            self._fp8_prefill_k_scratch[:total_pages],
            self._fp8_prefill_v_scratch[:total_pages],
            self._fp8_prefill_logical_pages[:total_pages],
        )

    def _can_use_e5m2_prefill_bridge(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_mode,
    ) -> bool:
        """Bound the 1Cat E5M2 bridge to its measured Qwen3.8 TP4 lane."""
        return (
            self._uses_fp8_prefill_scratch(forward_mode)
            and self._uses_sm70_e5m2_kv
            and _fp8_e5m2_paged_kv_to_fp16 is not None
            and self.page_size == V100_PAGE_SIZE
            # The exact Split-D D256 operator accepts 64-token-aligned Q; the
            # adapter uses 1Cat-style causal-safe padding for short tails.
            # Keep tiny chunks on Triton, but admit both a 4000-token request
            # and 1Cat's much faster 15680-token V100 prefill geometry.
            and q.shape[0] >= 3920
            and q.dtype == torch.float16
            and layer.tp_q_head_num == 6
            and layer.tp_k_head_num == 1
            and layer.qk_head_dim == 256
            and layer.v_head_dim == 256
            and not layer.is_cross_attention
            and layer.attn_type != AttentionType.ENCODER_ONLY
            and (
                layer.sliding_window_size is None
                or layer.sliding_window_size < 0
            )
            and getattr(layer, "logit_cap", None) in (None, 0, 0.0)
            and not torch.cuda.is_current_stream_capturing()
        )

    def _ensure_e5m2_prefill_bridge_scratch(
        self,
        required_pages: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_shape = (num_kv_heads, head_dim)
        if self._e5m2_prefill_head_shape != head_shape:
            self._e5m2_prefill_k_scratch = None
            self._e5m2_prefill_v_scratch = None
            self._e5m2_prefill_page_table = None
            self._e5m2_prefill_page_capacity = 0
            self._e5m2_prefill_head_shape = head_shape

        if self._e5m2_prefill_page_capacity < required_pages:
            capacity = 1 << (max(1, required_pages) - 1).bit_length()
            shape = (
                capacity,
                _E5M2_PREFILL_BRIDGE_PAGE_SIZE,
                num_kv_heads,
                head_dim,
            )
            self._e5m2_prefill_k_scratch = torch.empty(
                shape, dtype=torch.float16, device=self.device
            )
            self._e5m2_prefill_v_scratch = torch.empty_like(
                self._e5m2_prefill_k_scratch
            )
            self._e5m2_prefill_page_table = torch.arange(
                capacity, dtype=torch.int32, device=self.device
            ).unsqueeze(0)
            self._e5m2_prefill_page_capacity = capacity

        assert self._e5m2_prefill_k_scratch is not None
        assert self._e5m2_prefill_v_scratch is not None
        assert self._e5m2_prefill_page_table is not None
        if not self._e5m2_prefill_scratch_logged:
            logger.info(
                "SM70 E5M2 KV prefill: expanding active paged KV once into "
                "a reusable FP16 logical workspace (page=%d).",
                _E5M2_PREFILL_BRIDGE_PAGE_SIZE,
            )
            self._e5m2_prefill_scratch_logged = True
        return (
            self._e5m2_prefill_k_scratch,
            self._e5m2_prefill_v_scratch,
            self._e5m2_prefill_page_table,
        )

    # ------------------------------------------------------------------
    # Metadata construction
    # ------------------------------------------------------------------
    def _uses_native_linear_verify(self, forward_mode) -> bool:
        if not forward_mode.is_target_verify():
            return False
        if self.model_runner.is_draft_worker:
            # The DFlash drafter runs a causal 16-token block through four
            # sliding-attention layers. Triton's SM70 FP8 extend kernel assigns
            # one program per head and serializes the 2K window (~10 ms/layer).
            # 1Cat's paged XQA kernel supports this H128, GQA-4 layout and
            # sliding-window mask, while preserving the drafter's own paged
            # cache. Keep an escape hatch for A/B testing and fall back cleanly
            # when XQA or native E4M3 conversion is unavailable.
            draft_xqa = (
                os.environ.get("SGLANG_V100_DFLASH_DRAFT_XQA", "1").strip().lower()
            )
            if draft_xqa in ("0", "false", "off", "no"):
                return False
            if draft_xqa not in ("1", "true", "on", "yes"):
                raise ValueError(
                    "SGLANG_V100_DFLASH_DRAFT_XQA must be a boolean value, "
                    f"got {draft_xqa!r}."
                )
            return (
                self.model_runner.spec_algorithm.is_dflash_family()
                and self.page_size == V100_PAGE_SIZE
                and self._paged_decode is not None
            )
        native_verify = (
            os.environ.get("SGLANG_V100_NATIVE_LINEAR_VERIFY", "1").strip().lower()
        )
        if native_verify in ("0", "false", "off", "no"):
            return False
        if native_verify not in ("1", "true", "on", "yes"):
            raise ValueError(
                "SGLANG_V100_NATIVE_LINEAR_VERIFY must be a boolean value, "
                f"got {native_verify!r}."
            )
        if self._uses_sm70_e5m2_kv:
            return (
                self._target_e5m2_xqa_enabled
                and self.page_size == V100_PAGE_SIZE
            )
        if self._uses_sm70_e4m3_kv:
            return (
                self.model_runner.spec_algorithm.is_dflash_family()
                and self.page_size == V100_PAGE_SIZE
                and (
                    _use_tilelang
                    or (
                        self._fp8_verify_k_scratch is not None
                        and self._fp8_verify_kv_indptr is not None
                        and self._fp8_verify_logical_indices is not None
                    )
                    or (self._target_xqa_enabled and self._xqa_decode is not None)
                )
            )
        # The ai-bond fallback has no linear-verify or sliding-window API.
        # DFlash drafts contain causal SWA layers, so routing their verify pass
        # through that fallback would either fail on unsupported kwargs or,
        # if the kwargs were dropped, silently use the wrong attention mask.
        # Triton's TARGET_VERIFY path supports both per-layer causal/SWA and
        # final bidirectional attention, so use it unless TileLang is active.
        if not _use_tilelang:
            return False
        spec_algorithm = self.model_runner.spec_algorithm
        return spec_algorithm.is_dflash_family() or (
            spec_algorithm.is_eagle()
            and self.model_runner.server_args.speculative_eagle_topk <= 1
        )

    def _smallq_partition_size(self) -> int:
        """Workspace partition envelope for native linear verification.

        The 1Cat E5M2 GQA-6/D256 XQA graph captures the largest p256 envelope
        and selects p256/p1024 from device-side sequence lengths. Other
        verifier kernels retain their established p1024 decomposition.
        """
        if self._target_e5m2_xqa_enabled and not self.model_runner.is_draft_worker:
            return 256
        return 1024

    def _build_extend_metadata(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        causal: bool,
        page_table_buf: Optional[torch.Tensor] = None,
        swa_page_table_buf: Optional[torch.Tensor] = None,
        build_smallq: bool = False,
    ) -> FlashAttnV100ExtendMetadata:
        """Gather page indices for each sequence and pack into a page table.

        Token slot indices live in ``req_to_token[req, pos]``; the first token
        of each 16-token page determines the page index (``slot // page_size``).
        SWA cache slots are translated through the pool's full-to-SWA mapping.
        """
        # Synthetic prefill warmup uses a decode-oriented seq_len fill value
        # while still submitting a full extend block. Prefix + extend is the
        # lower bound required by both the page table and scratch allocation.
        seq_lens = torch.maximum(
            seq_lens.to(torch.int32),
            extend_prefix_lens.to(torch.int32) + extend_seq_lens.to(torch.int32),
        )
        num_seqs = seq_lens.shape[0]
        max_seq_len = int(seq_lens.max().item()) if num_seqs > 0 else 0
        max_pages = (max_seq_len + self.page_size - 1) // self.page_size

        def build_page_table(
            page_table_buffer: Optional[torch.Tensor],
            translate_to_swa: bool,
        ) -> torch.Tensor:
            # Keep a fixed-width table so the TileLang kernel's max_blocks key
            # remains stable across requests.
            if page_table_buffer is None:
                table = torch.zeros(
                    num_seqs,
                    self._max_pages,
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                table = page_table_buffer[:num_seqs, : self._max_pages]
                table.zero_()

            if max_pages > 0:
                strided = torch.arange(
                    0, max_seq_len, self.page_size, device=self.device
                )  # [max_pages] token positions (page starts)
                token_indices = self.req_to_token[
                    req_pool_indices[:, None], strided[None, :]
                ]  # [num_seqs, max_pages]
                if translate_to_swa:
                    token_indices = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            token_indices
                        )
                    )
                table[:, :max_pages] = (token_indices // self.page_size).to(torch.int32)
            return table

        page_table = build_page_table(page_table_buf, translate_to_swa=False)
        has_swa_mapping = (
            hasattr(self.token_to_kv_pool, "translate_loc_from_full_to_swa")
            and getattr(self.token_to_kv_pool, "full_to_swa_index_mapping", None)
            is not None
        )
        swa_page_table = (
            build_page_table(swa_page_table_buf, translate_to_swa=True)
            if has_swa_mapping
            else None
        )

        query_start_loc = torch.zeros(
            num_seqs + 1, dtype=torch.int32, device=self.device
        )
        query_start_loc[1:] = torch.cumsum(extend_seq_lens.to(torch.int32), dim=0)

        smallq_page_table = None
        smallq_swa_page_table = None
        smallq_seq_lens = None
        smallq_active_num_partitions = None
        if build_smallq:
            total_query_tokens = int(query_start_loc[-1].item())
            query_lens_i32 = extend_seq_lens.to(torch.int32)
            smallq_page_table = torch.repeat_interleave(
                page_table,
                query_lens_i32,
                dim=0,
                output_size=total_query_tokens,
            ).contiguous()
            if swa_page_table is not None:
                smallq_swa_page_table = torch.repeat_interleave(
                    swa_page_table,
                    query_lens_i32,
                    dim=0,
                    output_size=total_query_tokens,
                ).contiguous()
            repeated_prefix_lens = torch.repeat_interleave(
                extend_prefix_lens.to(torch.int32),
                query_lens_i32,
                output_size=total_query_tokens,
            )
            repeated_query_starts = torch.repeat_interleave(
                query_start_loc[:-1],
                query_lens_i32,
                output_size=total_query_tokens,
            )
            token_indices = torch.arange(
                total_query_tokens, dtype=torch.int32, device=self.device
            )
            smallq_seq_lens = (
                repeated_prefix_lens + token_indices - repeated_query_starts + 1
            ).contiguous()
            partition_size = self._smallq_partition_size()
            smallq_active_num_partitions = torch.tensor(
                [max(1, (max_seq_len + partition_size - 1) // partition_size)],
                dtype=torch.int32,
                device=self.device,
            )

        return FlashAttnV100ExtendMetadata(
            page_table=page_table,
            seq_lens=seq_lens.to(torch.int32),
            query_start_loc=query_start_loc,
            prefix_kv_lens=extend_prefix_lens.to(torch.int32),
            causal=causal,
            swa_page_table=swa_page_table,
            smallq_page_table=smallq_page_table,
            smallq_swa_page_table=smallq_swa_page_table,
            smallq_seq_lens=smallq_seq_lens,
            smallq_active_num_partitions=smallq_active_num_partitions,
            smallq_max_seq_len=max_seq_len,
        )

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        mode = forward_batch.forward_mode
        if (
            (
                self._uses_sm70_fp8_kv
                and not self._uses_native_linear_verify(mode)
                and not self._uses_fp8_prefill_scratch(mode)
            )
            or mode.is_decode_or_idle()
            or mode.is_draft_extend(include_v2=True)
            or (mode.is_target_verify() and not self._uses_native_linear_verify(mode))
        ):
            # Decode / spec paths run on the Triton backend.
            self._triton.init_forward_metadata(forward_batch)
            self.forward_metadata = None
            return

        if self._uses_native_linear_verify(mode):
            prefix_lens = forward_batch.seq_lens
            draft_token_num = int(forward_batch.spec_info.draft_token_num)
            extend_seq_lens = torch.full_like(prefix_lens, draft_token_num)
            seq_lens = prefix_lens + draft_token_num
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            extend_seq_lens = forward_batch.extend_seq_lens
            seq_lens = forward_batch.seq_lens

        causal = True
        md = self._build_extend_metadata(
            forward_batch.req_pool_indices,
            seq_lens,
            extend_seq_lens,
            prefix_lens,
            causal=causal,
            build_smallq=self._uses_native_linear_verify(mode),
        )
        self.forward_metadata = md
        if self._uses_sm70_fp8_kv and (
            self._uses_native_linear_verify(mode)
            # Full aligned E5M2 chunks use the native bridge, but the final
            # partial chunk deliberately falls back to Triton. Build both
            # metadata views so a 32K+1 prompt cannot reach Triton with None.
            or self._uses_sm70_e5m2_kv
        ):
            self._triton.init_forward_metadata(forward_batch)

    # ------------------------------------------------------------------
    # CUDA graph state
    # ------------------------------------------------------------------
    def init_cuda_graph_state(
        self, max_bs: int, max_num_tokens: int, kv_indices_buf=None
    ):
        # Decode cuda-graph state lives in the Triton backend.
        self._triton.init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

        self._cg_page_table = torch.zeros(
            max_bs, self._max_pages, dtype=torch.int32, device=self.device
        )
        if (
            hasattr(self.token_to_kv_pool, "translate_loc_from_full_to_swa")
            and getattr(self.token_to_kv_pool, "full_to_swa_index_mapping", None)
            is not None
        ):
            self._cg_swa_page_table = torch.zeros(
                max_bs, self._max_pages, dtype=torch.int32, device=self.device
            )
        self._cg_seq_lens = torch.zeros(max_bs, dtype=torch.int32, device=self.device)
        self._cg_query_start_loc = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self._cg_prefix_kv_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._cg_strided = torch.arange(
            0, self.max_context_len, self.page_size, device=self.device
        )
        self._cg_smallq_page_table = torch.zeros(
            max_num_tokens,
            self._max_pages,
            dtype=torch.int32,
            device=self.device,
        )
        if self._cg_swa_page_table is not None:
            self._cg_smallq_swa_page_table = torch.zeros(
                max_num_tokens,
                self._max_pages,
                dtype=torch.int32,
                device=self.device,
            )
        self._cg_smallq_seq_lens = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._cg_smallq_active_num_partitions = torch.ones(
            1, dtype=torch.int32, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        if (
            (
                self._uses_sm70_fp8_kv
                and not self._uses_native_linear_verify(forward_mode)
            )
            or forward_mode.is_decode_or_idle()
            or forward_mode.is_draft_extend(include_v2=True)
            or (
                forward_mode.is_target_verify()
                and not self._uses_native_linear_verify(forward_mode)
            )
        ):
            return self._triton.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )
        if self._uses_native_linear_verify(forward_mode):
            if self._uses_sm70_fp8_kv:
                self._triton.init_forward_metadata_capture_cuda_graph(
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                )
            self._set_linear_verify_cuda_graph_metadata(
                bs,
                req_pool_indices,
                seq_lens,
                int(spec_info.draft_token_num),
            )
            return
        # Extend capture: metadata buffers are filled at replay time.
        self.forward_metadata = FlashAttnV100ExtendMetadata(
            page_table=self._cg_page_table[:bs],
            seq_lens=self._cg_seq_lens[:bs],
            query_start_loc=self._cg_query_start_loc[: bs + 1],
            prefix_kv_lens=self._cg_prefix_kv_lens[:bs],
            causal=True,
            swa_page_table=(
                self._cg_swa_page_table[:bs]
                if self._cg_swa_page_table is not None
                else None
            ),
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        if (
            (
                self._uses_sm70_fp8_kv
                and not self._uses_native_linear_verify(forward_mode)
            )
            or forward_mode.is_decode_or_idle()
            or forward_mode.is_draft_extend(include_v2=True)
            or (
                forward_mode.is_target_verify()
                and not self._uses_native_linear_verify(forward_mode)
            )
        ):
            return self._triton.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )
        if self._uses_native_linear_verify(forward_mode):
            if self._uses_sm70_fp8_kv:
                self._triton.init_forward_metadata_replay_cuda_graph(
                    bs,
                    req_pool_indices,
                    seq_lens,
                    seq_lens_sum,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                    seq_lens_cpu,
                )
            self._set_linear_verify_cuda_graph_metadata(
                bs,
                req_pool_indices,
                seq_lens,
                int(spec_info.draft_token_num),
            )
            return

        # Extend replay: refresh page table + seq metadata from the new batch.
        seq_lens_b = seq_lens[:bs].to(torch.int32)
        max_len = int(seq_lens_b.max().item()) if bs > 0 else 0
        max_pages = (max_len + self.page_size - 1) // self.page_size
        if max_pages > 0:
            token_indices = self.req_to_token[
                req_pool_indices[:bs, None],
                self._cg_strided[:max_pages][None, :],
            ]
            self._cg_page_table[:bs, :max_pages].copy_(
                (token_indices // self.page_size).to(torch.int32)
            )
            if self._cg_swa_page_table is not None:
                swa_token_indices = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(token_indices)
                )
                self._cg_swa_page_table[:bs, :max_pages].copy_(
                    (swa_token_indices // self.page_size).to(torch.int32)
                )
        if self._cg_swa_page_table is not None:
            self._cg_swa_page_table[:bs, max_pages:].zero_()
        self._cg_seq_lens[:bs].copy_(seq_lens_b)
        # query_start_loc / prefix_kv_lens for extend replay are approximated
        # from seq_lens (pure extend, no prefix) — sufficient for the captured
        # graph shape; mixed-prefix extends fall back to eager metadata.
        self._cg_query_start_loc.zero_()
        self._cg_query_start_loc[1 : bs + 1] = torch.cumsum(seq_lens_b, dim=0)
        self._cg_prefix_kv_lens[:bs].zero_()

    def _set_linear_verify_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        draft_token_num: int,
    ):
        """Refresh fixed buffers for a top-k=1 linear causal verify block."""
        prefix_lens_b = prefix_lens[:bs].to(torch.int32)
        seq_lens_b = prefix_lens_b + draft_token_num
        max_len = int(seq_lens_b.max().item()) if bs > 0 else 0
        max_pages = (max_len + self.page_size - 1) // self.page_size
        if max_pages > 0:
            token_indices = self.req_to_token[
                req_pool_indices[:bs, None],
                self._cg_strided[:max_pages][None, :],
            ]
            self._cg_page_table[:bs, :max_pages].copy_(
                (token_indices // self.page_size).to(torch.int32)
            )
            if self._cg_swa_page_table is not None:
                swa_token_indices = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(token_indices)
                )
                self._cg_swa_page_table[:bs, :max_pages].copy_(
                    (swa_token_indices // self.page_size).to(torch.int32)
                )
        if self._cg_swa_page_table is not None:
            self._cg_swa_page_table[:bs, max_pages:].zero_()
        self._cg_seq_lens[:bs].copy_(seq_lens_b)
        self._cg_query_start_loc[: bs + 1].copy_(
            torch.arange(
                0,
                (bs + 1) * draft_token_num,
                step=draft_token_num,
                dtype=torch.int32,
                device=self.device,
            )
        )
        self._cg_prefix_kv_lens[:bs].copy_(prefix_lens_b)
        num_smallq_rows = bs * draft_token_num
        self._cg_smallq_page_table[:num_smallq_rows].copy_(
            torch.repeat_interleave(
                self._cg_page_table[:bs],
                draft_token_num,
                dim=0,
                output_size=num_smallq_rows,
            )
        )
        if self._cg_smallq_swa_page_table is not None:
            self._cg_smallq_swa_page_table[:num_smallq_rows].copy_(
                torch.repeat_interleave(
                    self._cg_swa_page_table[:bs],
                    draft_token_num,
                    dim=0,
                    output_size=num_smallq_rows,
                )
            )
        repeated_prefix_lens = torch.repeat_interleave(
            prefix_lens_b,
            draft_token_num,
            output_size=num_smallq_rows,
        )
        smallq_offsets = (
            torch.arange(
                draft_token_num,
                dtype=torch.int32,
                device=self.device,
            )
            .add_(1)
            .repeat(bs)
        )
        self._cg_smallq_seq_lens[:num_smallq_rows].copy_(
            repeated_prefix_lens + smallq_offsets
        )
        partition_size = self._smallq_partition_size()
        self._cg_smallq_active_num_partitions.fill_(
            max(1, (max_len + partition_size - 1) // partition_size)
        )
        self.forward_metadata = FlashAttnV100ExtendMetadata(
            page_table=self._cg_page_table[:bs],
            seq_lens=self._cg_seq_lens[:bs],
            query_start_loc=self._cg_query_start_loc[: bs + 1],
            prefix_kv_lens=self._cg_prefix_kv_lens[:bs],
            causal=True,
            swa_page_table=(
                self._cg_swa_page_table[:bs]
                if self._cg_swa_page_table is not None
                else None
            ),
            smallq_page_table=self._cg_smallq_page_table[:num_smallq_rows],
            smallq_swa_page_table=(
                self._cg_smallq_swa_page_table[:num_smallq_rows]
                if self._cg_smallq_swa_page_table is not None
                else None
            ),
            smallq_seq_lens=self._cg_smallq_seq_lens[:num_smallq_rows],
            smallq_active_num_partitions=self._cg_smallq_active_num_partitions,
            smallq_max_seq_len=max_len,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_verify_buffers_to_fill_after_draft(self):
        # Tree verification uses these Triton buffers. Linear verification does
        # not consume the custom mask, but EAGLE's shared tree builder may still
        # fill it while constructing the acceptance metadata.
        return self._triton.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info, cuda_graph_bs: Optional[int]
    ):
        return self._triton.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _can_use_long_decode_xqa(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        sinks,
    ) -> bool:
        if not self._long_decode_xqa_enabled or q.shape[0] != 1:
            return False
        if (
            self.page_size != V100_PAGE_SIZE
            or layer.tp_q_head_num != 6
            or layer.tp_k_head_num != 1
            or layer.qk_head_dim != 256
            or layer.v_head_dim != 256
            or q.dtype != torch.float16
            or sinks is not None
            or layer.xai_temperature_len > 0
            or (
                layer.sliding_window_size is not None
                and layer.sliding_window_size >= 0
            )
            or getattr(layer, "logit_cap", None) not in (None, 0, 0.0)
        ):
            return False

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        return (
            k_cache.dtype in (torch.float16, torch.float8_e5m2)
            and v_cache.dtype == k_cache.dtype
            and k_cache.ndim == 4
            and v_cache.ndim == 4
            and k_cache.shape[1:] == (V100_PAGE_SIZE, 1, 256)
            and v_cache.shape[1:] == (V100_PAGE_SIZE, 1, 256)
            and k_cache.stride(-1) == 1
            and v_cache.stride(-1) == 1
        )

    def _forward_decode_xqa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool,
    ) -> torch.Tensor:
        """Run native page-16 XQA using Triton's already-built KV ordering."""
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        out = torch.empty_like(q)

        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        # For B=1, Triton's logical token list is also an inexpensive source
        # for the native page table: every 16th slot is a logical page start.
        # Eager metadata is runtime-sized; CUDA-graph metadata is capacity-sized,
        # which intentionally fixes the launch grid while seq_lens controls the
        # number of active partitions on device.
        kv_indices = self._triton.forward_metadata.kv_indices
        token_capacity = min(kv_indices.numel(), self.max_context_len)
        page_starts = kv_indices[:token_capacity:V100_PAGE_SIZE]
        page_table = self._xqa_decode_page_table[:, : page_starts.numel()]
        torch.div(
            page_starts,
            V100_PAGE_SIZE,
            rounding_mode="floor",
            out=page_table.view(-1),
        )

        seq_lens = forward_batch.seq_lens[:1]
        torch.add(
            seq_lens,
            255,
            out=self._xqa_decode_active_num_partitions,
        )
        torch.div(
            self._xqa_decode_active_num_partitions,
            256,
            rounding_mode="floor",
            out=self._xqa_decode_active_num_partitions,
        )
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        e5m2_kv = k_cache.dtype == torch.float8_e5m2
        k_scale = layer.k_scale_float if layer.k_scale is not None else 1.0
        v_scale = layer.v_scale_float if layer.v_scale is not None else 1.0
        self._xqa_decode(
            q.view(1, 6, 256),
            k_cache.view(torch.uint8) if e5m2_kv else k_cache,
            v_cache.view(torch.uint8) if e5m2_kv else v_cache,
            page_table,
            seq_lens,
            softmax_scale=layer.scaling,
            out=out.view(1, 6, 256),
            kv_cache_dtype="fp8_e5m2" if e5m2_kv else "auto",
            k_scale=k_scale,
            v_scale=v_scale,
            window_size=(-1, -1),
            workspace_seq_capacity_hint=page_starts.numel() * V100_PAGE_SIZE,
            active_num_partitions=self._xqa_decode_active_num_partitions,
            partition_size_hint=256,
        )
        return out

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        sinks = kwargs.get("sinks")
        if self._can_use_long_decode_xqa(q, layer, sinks):
            return self._forward_decode_xqa(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
            )
        return self._triton.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        use_fp8_prefill_scratch = self._uses_fp8_prefill_scratch(
            forward_batch.forward_mode
        ) and (
            not self._uses_sm70_e5m2_kv
            or self._can_use_e5m2_prefill_bridge(
                q, layer, forward_batch.forward_mode
            )
        )
        if (
            (
                self._uses_sm70_fp8_kv
                and not self._uses_native_linear_verify(forward_batch.forward_mode)
                and not use_fp8_prefill_scratch
            )
            or (
                forward_batch.forward_mode.is_target_verify()
                and not self._uses_native_linear_verify(forward_batch.forward_mode)
            )
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            # SM70 TileLang/ai-bond kernels require FP16 KV. Tree verification
            # additionally needs Triton's custom mask.
            return self._triton.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if save_kv_cache and k is not None:
            preserve_extend_kv = use_fp8_prefill_scratch
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                cache_loc,
                k.clone() if preserve_extend_kv else k,
                v.clone() if preserve_extend_kv else v,
                layer.k_scale,
                layer.v_scale,
            )

        md = self.forward_metadata
        num_tokens = q.shape[0]
        # DFlash's fused QKV projection can return Q as a strided view of the
        # wider projection buffer. TileLang specializes tensor strides, so
        # pack the small extend/verify block before entering the kernel.
        q3 = q.reshape(num_tokens, layer.tp_q_head_num, layer.head_dim).contiguous()

        k_cache, v_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        # paged_fwd requires 4D [num_blocks, block_size, num_kv_heads, head_dim].
        # SGLang's token pool is physically flat for every allocator page size,
        # so expose its contiguous page groups as the 4D layout expected by the
        # TileLang kernel.  Previously this only added a size-1 dimension and
        # made the supported Mamba extra-buffer/page-16 mode fail at runtime.
        if k_cache.ndim == 3:
            k_cache = k_cache.view(
                -1, self.page_size, k_cache.shape[-2], k_cache.shape[-1]
            )
            v_cache = v_cache.view(
                -1, self.page_size, v_cache.shape[-2], v_cache.shape[-1]
            )

        if (
            self.model_runner.is_draft_worker
            and forward_batch.forward_mode.is_target_verify()
            and k_cache.dtype == torch.float8_e4m3fn
            and not _is_dflash_draft_native_shape_supported(layer, k_cache.dtype)
        ):
            raise RuntimeError(
                "V100 DFlash with E4M3 KV is not correctness-validated for "
                "this tensor-parallel draft layout. Use FP16 KV "
                "(--kv-cache-dtype auto) for TP2, or use TP4."
            )

        prefill_page_table = None
        prefill_block_size = self.page_size
        prefill_k_scale = layer.k_scale_float if layer.k_scale is not None else 1.0
        prefill_v_scale = layer.v_scale_float if layer.v_scale is not None else 1.0
        logical_dense_kv = False
        if (
            use_fp8_prefill_scratch
            and k_cache.dtype == torch.float8_e4m3fn
            and k_cache.shape == v_cache.shape
            and k_cache.shape[2:]
            == (
                layer.tp_k_head_num,
                layer.head_dim,
            )
        ):
            from sglang.srt.layers.attention.triton_ops.fp8_sm70 import (
                dequantize_paged_kv_e4m3_sm70,
                store_paged_extend_kv_fp16_sm70,
            )

            batch_size = int(md.prefix_kv_lens.numel())
            # Keep cache and page-table shapes fixed for a given batch size so
            # TileLang compiles once during warmup instead of once per growing
            # prefix chunk. Only active prefix pages are actually dequantized.
            pages_per_sequence = self._max_pages
            total_pages = batch_size * pages_per_sequence
            scratch_k, scratch_v, logical_pages = self._ensure_fp8_prefill_scratch(
                total_pages,
                layer.tp_k_head_num,
                layer.head_dim,
            )
            max_prefix_len = (
                max(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.extend_prefix_lens_cpu is not None
                else int(md.prefix_kv_lens.max().item())
            )
            dequantize_paged_kv_e4m3_sm70(
                k_cache,
                v_cache,
                md.page_table,
                md.prefix_kv_lens,
                scratch_k,
                scratch_v,
                max_prefix_len,
            )
            max_extend_len = (
                max(forward_batch.extend_seq_lens_cpu)
                if forward_batch.extend_seq_lens_cpu is not None
                else int(forward_batch.extend_seq_lens.max().item())
            )
            k_scale = layer.k_scale_float if layer.k_scale is not None else 1.0
            v_scale = layer.v_scale_float if layer.v_scale is not None else 1.0
            store_paged_extend_kv_fp16_sm70(
                k.reshape(num_tokens, layer.tp_k_head_num, layer.head_dim).contiguous(),
                v.reshape(
                    num_tokens, layer.tp_k_head_num, layer.v_head_dim
                ).contiguous(),
                md.query_start_loc,
                md.prefix_kv_lens,
                scratch_k,
                scratch_v,
                max_extend_len,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            k_cache = scratch_k
            v_cache = scratch_v
            prefill_page_table = logical_pages.view(batch_size, pages_per_sequence)
            logical_dense_kv = True
        elif (
            use_fp8_prefill_scratch
            and self._uses_sm70_e5m2_kv
            and k_cache.dtype == torch.float8_e5m2
            and k_cache.shape == v_cache.shape
            and k_cache.shape[2:] == (layer.tp_k_head_num, layer.head_dim)
        ):
            # 1Cat's vectorized bridge resolves the physical page table and
            # converts every active E5M2 value exactly once. The resulting
            # logical FP16 workspace feeds the dense Split-D kernel directly.
            seq_len = md.smallq_max_seq_len
            active_input_pages = (seq_len + self.page_size - 1) // self.page_size
            active_page_table = md.page_table[:, :active_input_pages]
            input_capacity = active_input_pages * self.page_size
            required_pages = (
                input_capacity + _E5M2_PREFILL_BRIDGE_PAGE_SIZE - 1
            ) // _E5M2_PREFILL_BRIDGE_PAGE_SIZE
            scratch_k, scratch_v, logical_pages = (
                self._ensure_e5m2_prefill_bridge_scratch(
                    required_pages,
                    layer.tp_k_head_num,
                    layer.head_dim,
                )
            )
            assert _fp8_e5m2_paged_kv_to_fp16 is not None
            _fp8_e5m2_paged_kv_to_fp16(
                k_cache.view(torch.uint8),
                v_cache.view(torch.uint8),
                active_page_table,
                md.seq_lens,
                scratch_k,
                scratch_v,
                prefill_k_scale,
                prefill_v_scale,
            )
            k_cache = scratch_k
            v_cache = scratch_v
            prefill_page_table = logical_pages
            prefill_block_size = _E5M2_PREFILL_BRIDGE_PAGE_SIZE
            prefill_k_scale = 1.0
            prefill_v_scale = 1.0
            logical_dense_kv = True

        verify_page_table = None
        if (
            self._fp8_verify_k_scratch is not None
            and self._fp8_verify_v_scratch is not None
            and self._fp8_verify_page_table is not None
            and k_cache.dtype == torch.float8_e4m3fn
            and forward_batch.forward_mode.is_target_verify()
            and self._uses_native_linear_verify(forward_batch.forward_mode)
            and (layer.sliding_window_size is None or layer.sliding_window_size < 0)
            and k_cache.shape[2:] == self._fp8_verify_k_scratch.shape[2:]
        ):
            from sglang.srt.layers.attention.triton_ops.fp8_sm70 import (
                dequantize_paged_kv_e4m3_sm70,
                store_linear_verify_kv_fp16_sm70,
            )

            dequantize_paged_kv_e4m3_sm70(
                k_cache,
                v_cache,
                md.page_table,
                md.seq_lens,
                self._fp8_verify_k_scratch,
                self._fp8_verify_v_scratch,
                md.smallq_max_seq_len,
            )
            store_linear_verify_kv_fp16_sm70(
                k.reshape(num_tokens, layer.tp_k_head_num, layer.head_dim).contiguous(),
                v.reshape(
                    num_tokens, layer.tp_k_head_num, layer.v_head_dim
                ).contiguous(),
                md.prefix_kv_lens,
                self._fp8_verify_k_scratch,
                self._fp8_verify_v_scratch,
                k_scale=(layer.k_scale_float if layer.k_scale is not None else 1.0),
                v_scale=(layer.v_scale_float if layer.v_scale is not None else 1.0),
            )
            k_cache = self._fp8_verify_k_scratch
            v_cache = self._fp8_verify_v_scratch
            verify_page_table = self._fp8_verify_page_table

        out = torch.empty(
            num_tokens,
            layer.tp_q_head_num,
            layer.head_dim,
            dtype=q.dtype,
            device=q.device,
        )

        use_fp8_scratch_triton = (
            verify_page_table is not None
            and self._fp8_verify_kv_indptr is not None
            and self._fp8_verify_logical_indices is not None
            and not self.model_runner.is_draft_worker
        )
        if use_fp8_scratch_triton:
            from sglang.srt.layers.attention.triton_backend import logit_capping_mod

            self._fp8_verify_kv_indptr[1].copy_(md.prefix_kv_lens[0])
            triton_md = self._triton.forward_metadata
            causal, _ = _get_native_paged_attention_params(layer, md.causal)
            self._triton.extend_attention_fwd(
                q3,
                k.reshape(num_tokens, layer.tp_k_head_num, layer.head_dim).contiguous(),
                v.reshape(
                    num_tokens, layer.tp_k_head_num, layer.v_head_dim
                ).contiguous(),
                out,
                self._fp8_verify_k_scratch.view(
                    -1, layer.tp_k_head_num, layer.head_dim
                ),
                self._fp8_verify_v_scratch.view(
                    -1, layer.tp_k_head_num, layer.v_head_dim
                ),
                triton_md.qo_indptr,
                self._fp8_verify_kv_indptr,
                self._fp8_verify_logical_indices,
                triton_md.custom_mask,
                causal,
                triton_md.mask_indptr,
                triton_md.max_extend_len,
                layer.k_scale_float if layer.k_scale is not None else 1.0,
                layer.v_scale_float if layer.v_scale is not None else 1.0,
                layer.scaling,
                logit_cap=logit_capping_mod(
                    layer.logit_capping_method, layer.logit_cap
                ),
                sliding_window_size=-1,
                xai_temperature_len=layer.xai_temperature_len,
            )
            return out.reshape(num_tokens, layer.tp_q_head_num * layer.head_dim)

        target_wmma = (
            not self.model_runner.is_draft_worker
            and verify_page_table is not None
            and self._wmma_decode is not None
        )
        use_smallq_decode = (
            (
                self._paged_decode is not None
                if self.model_runner.is_draft_worker
                else target_wmma
                or (
                    (
                        self._target_xqa_enabled
                        or self._target_e5m2_xqa_enabled
                    )
                    and self._xqa_decode is not None
                )
            )
            and self._uses_native_linear_verify(forward_batch.forward_mode)
            # A DFlash2 checkpoint declares is_causal=false: every masked
            # position in its draft block must see the whole block.  The
            # small-Q decode kernel only represents one causally growing row
            # at a time, so route encoder-only draft layers through the native
            # paged forward kernel below.
            and getattr(layer, "attn_type", None) != AttentionType.ENCODER_ONLY
            and md.smallq_page_table is not None
            and md.smallq_seq_lens is not None
            and md.smallq_active_num_partitions is not None
            and (
                (
                    not self.model_runner.is_draft_worker
                    and layer.head_dim == 256
                    and layer.tp_q_head_num == 6 * layer.tp_k_head_num
                    and layer.tp_k_head_num == 1
                    and (
                        layer.sliding_window_size is None
                        or layer.sliding_window_size < 0
                    )
                )
                or (
                    self.model_runner.is_draft_worker
                    and _is_dflash_draft_native_shape_supported(layer, k_cache.dtype)
                )
            )
            and (
                k_cache.dtype == torch.float16
                or (
                    k_cache.dtype == torch.float8_e4m3fn
                    and (self.model_runner.is_draft_worker or self._xqa_e4m3_supported)
                )
                or (
                    k_cache.dtype == torch.float8_e5m2
                    and (
                        self.model_runner.is_draft_worker
                        or self._target_e5m2_xqa_enabled
                    )
                )
            )
        )
        if use_smallq_decode:
            fp8_kv = k_cache.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            )
            if k_cache.dtype == torch.float8_e5m2:
                kv_cache_dtype = "fp8_e5m2"
            elif k_cache.dtype == torch.float8_e4m3fn:
                kv_cache_dtype = "fp8_e4m3"
            else:
                kv_cache_dtype = "auto"
            sliding_window_size = (
                int(layer.sliding_window_size)
                if layer.sliding_window_size is not None
                and layer.sliding_window_size >= 0
                else -1
            )
            smallq_page_table = md.smallq_page_table
            if verify_page_table is not None:
                smallq_page_table = verify_page_table.expand(
                    md.smallq_seq_lens.shape[0], -1
                )
            if sliding_window_size >= 0 and md.smallq_swa_page_table is not None:
                smallq_page_table = md.smallq_swa_page_table
            smallq_decode = (
                self._paged_decode
                if self.model_runner.is_draft_worker
                else self._wmma_decode if target_wmma else self._xqa_decode
            )
            if target_wmma:
                smallq_decode(
                    q3,
                    k_cache,
                    v_cache,
                    smallq_page_table,
                    md.smallq_seq_lens,
                    softmax_scale=layer.scaling,
                    out=out,
                )
            else:
                smallq_decode(
                    q3,
                    k_cache.view(torch.uint8) if fp8_kv else k_cache,
                    v_cache.view(torch.uint8) if fp8_kv else v_cache,
                    smallq_page_table,
                    md.smallq_seq_lens,
                    softmax_scale=layer.scaling,
                    out=out,
                    kv_cache_dtype=kv_cache_dtype,
                    k_scale=(layer.k_scale_float if layer.k_scale is not None else 1.0),
                    v_scale=(layer.v_scale_float if layer.v_scale is not None else 1.0),
                    window_size=(
                        (sliding_window_size, 0)
                        if sliding_window_size >= 0
                        else (-1, -1)
                    ),
                    max_seq_len_hint=max(1, md.smallq_max_seq_len),
                    workspace_seq_capacity_hint=self.max_context_len,
                    active_num_partitions=md.smallq_active_num_partitions,
                    partition_size_hint=self._smallq_partition_size(),
                )
            return out.reshape(num_tokens, layer.tp_q_head_num * layer.head_dim)

        paged_forward = _load_paged_forward()
        causal, sliding_window_size = _get_native_paged_attention_params(
            layer, md.causal
        )
        paged_kwargs = dict(
            out=out,
            block_size=prefill_block_size,
            softmax_scale=layer.scaling,
            causal=causal,
            num_kv_heads=layer.tp_k_head_num,
        )
        if _use_tilelang:
            paged_kwargs.update(
                sliding_window_size=sliding_window_size,
                linear_verify=self._uses_native_linear_verify(
                    forward_batch.forward_mode
                ),
                k_scale=prefill_k_scale,
                v_scale=prefill_v_scale,
                max_seq_len_hint=getattr(md, "smallq_max_seq_len", None),
                logical_dense_kv=logical_dense_kv,
            )
        elif sliding_window_size >= 0:
            # init_forward_metadata normally routes DFlash/SWA verification to
            # Triton before reaching here. Keep this guard so another SWA path
            # cannot silently run full-context attention on the ai-bond kernel.
            raise RuntimeError(
                "flash_attn_v100's ai-bond fallback does not support sliding-window "
                "attention. Install a working TileLang build or use the Triton "
                "attention backend for this request."
            )

        page_table = md.page_table
        if prefill_page_table is not None:
            page_table = prefill_page_table
        if verify_page_table is not None:
            page_table = verify_page_table
        if sliding_window_size >= 0 and md.swa_page_table is not None:
            page_table = md.swa_page_table

        paged_forward(
            q3,
            k_cache,
            v_cache,
            page_table,
            md.seq_lens,
            md.query_start_loc,
            md.prefix_kv_lens,
            **paged_kwargs,
        )
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.head_dim)

    def support_triton(self):
        return True
