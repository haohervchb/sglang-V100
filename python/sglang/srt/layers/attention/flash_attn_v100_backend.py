"""FlashAttention-2 V100 (SM70) attention backend for sglang.

Prefill (forward_extend): prefers the vendored TileLang FA2 kernel tuned for
SM70 and falls back to ai-bond's ``flash_attn_v100_cuda.paged_fwd``. Both read
the paged KV cache as ``[num_pages, page_size, num_kv_heads, head_dim]``
(block-major, normally page_size=16), giving coalesced block reads on V100.

Native prefix handling is via ``prefix_kv_lens`` — no ragged+paged+merge_state
double-kernel, no FlattenKV, no FlashInfer wrapper ``plan()`` CPU overhead.

Decode (forward_decode): delegated to sglang's ``TritonAttnBackend``, which
carries the GooseLLM-derived SM70 split-K tuning. The ai-bond ``decode_fwd``
kernel is intentionally not used (it produces inf/NaN exp_sums); GooseLLM
itself runs Triton/tilelang for decode for the same reason.

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


def _should_skip_triton_prefill(model_runner: "ModelRunner") -> bool:
    """Keep baseline decode lean while allocating metadata needed by spec verify."""
    return not model_runner.spec_algorithm.is_speculative()


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
            logger.info(
                "paged prefill: using vendored tilelang-fa-v100 kernel (SM70)."
            )
            return _paged_forward
    except Exception:
        pass

    # Fall back to ai-bond flash_attn_v100_cuda
    _load_ai_bond_paged()
    return _paged_forward


def _load_ai_bond_paged():
    """Lazy-load the ai-bond paged forward kernel via GooseLLM's wrapper."""
    global _paged_forward
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
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "flash_attn_v100 backend requires the ai-bond flash_attn_v100_cuda "
            f"kernel + python wrapper. Import failed: {e}. Set FLASH_ATTN_V100_DIR "
            "or install flash-attention-v100."
        ) from e
    return _paged_forward


@dataclass
class FlashAttnV100ExtendMetadata:
    """Per-forward metadata for the paged prefill (extend) path."""

    page_table: torch.Tensor  # [num_seqs, max_pages] int32 — page indices
    seq_lens: torch.Tensor  # [num_seqs] int32 — total KV length per seq
    query_start_loc: torch.Tensor  # [num_seqs+1] int32 — cumsum of query lens
    prefix_kv_lens: torch.Tensor  # [num_seqs] int32 — cached prefix length
    causal: bool


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

        # Eagerly validate the kernel is loadable so we fail fast at startup.
        _load_paged_forward()

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
        self._cg_seq_lens: Optional[torch.Tensor] = None
        self._cg_query_start_loc: Optional[torch.Tensor] = None
        self._cg_prefix_kv_lens: Optional[torch.Tensor] = None
        self._cg_strided: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Metadata construction
    # ------------------------------------------------------------------
    def _uses_native_linear_verify(self, forward_mode) -> bool:
        if not forward_mode.is_target_verify():
            return False
        spec_algorithm = self.model_runner.spec_algorithm
        return spec_algorithm.is_dflash() or (
            spec_algorithm.is_eagle()
            and self.model_runner.server_args.speculative_eagle_topk <= 1
        )

    def _build_extend_metadata(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        causal: bool,
        page_table_buf: Optional[torch.Tensor] = None,
    ) -> FlashAttnV100ExtendMetadata:
        """Gather page indices for each sequence and pack into a page table.

        Token slot indices live in ``req_to_token[req, pos]``; the first token
        of each 16-token page determines the page index (``slot // page_size``).
        """
        num_seqs = seq_lens.shape[0]
        max_seq_len = int(seq_lens.max().item()) if num_seqs > 0 else 0
        max_pages = (max_seq_len + self.page_size - 1) // self.page_size

        # Fixed-width page table so the tilelang kernel's max_blocks key never
        # changes: allocate [num_seqs, self._max_pages] and fill only the first
        # max_pages columns.  Unused columns stay 0 (never read by the kernel).
        page_table = torch.zeros(
            num_seqs, self._max_pages, dtype=torch.int32, device=self.device
        )
        if max_pages > 0:
            strided = torch.arange(
                0, max_seq_len, self.page_size, device=self.device
            )  # [max_pages] token positions (page starts)
            token_indices = self.req_to_token[
                req_pool_indices[:, None], strided[None, :]
            ]  # [num_seqs, max_pages]
            page_table[:, :max_pages] = (token_indices // self.page_size).to(torch.int32)

        query_start_loc = torch.zeros(
            num_seqs + 1, dtype=torch.int32, device=self.device
        )
        query_start_loc[1:] = torch.cumsum(
            extend_seq_lens.to(torch.int32), dim=0
        )

        return FlashAttnV100ExtendMetadata(
            page_table=page_table,
            seq_lens=seq_lens.to(torch.int32),
            query_start_loc=query_start_loc,
            prefix_kv_lens=extend_prefix_lens.to(torch.int32),
            causal=causal,
        )

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        mode = forward_batch.forward_mode
        if (
            mode.is_decode_or_idle()
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
        )
        self.forward_metadata = md

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
        self._cg_seq_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._cg_query_start_loc = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self._cg_prefix_kv_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._cg_strided = torch.arange(
            0, self.max_context_len, self.page_size, device=self.device
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
            forward_mode.is_decode_or_idle()
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
            forward_mode.is_decode_or_idle()
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
        self.forward_metadata = FlashAttnV100ExtendMetadata(
            page_table=self._cg_page_table[:bs],
            seq_lens=self._cg_seq_lens[:bs],
            query_start_loc=self._cg_query_start_loc[: bs + 1],
            prefix_kv_lens=self._cg_prefix_kv_lens[:bs],
            causal=True,
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
        if (
            forward_batch.forward_mode.is_target_verify()
            and not self._uses_native_linear_verify(forward_batch.forward_mode)
        ) or forward_batch.forward_mode.is_draft_extend(include_v2=True):
            # Tree verification needs Triton's custom mask. DRAFT_EXTEND may
            # also carry a non-linear mask; linear DFlash and top-k=1 MTP
            # TARGET_VERIFY use the native SM70 path below.
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
            self.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        md = self.forward_metadata
        num_tokens = q.shape[0]
        # DFlash's fused QKV projection can return Q as a strided view of the
        # wider projection buffer. TileLang specializes tensor strides, so
        # pack the small extend/verify block before entering the kernel.
        q3 = q.reshape(
            num_tokens, layer.tp_q_head_num, layer.head_dim
        ).contiguous()

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

        out = torch.empty(
            num_tokens, layer.tp_q_head_num, layer.head_dim,
            dtype=q.dtype, device=q.device,
        )

        paged_forward = _load_paged_forward()
        causal, sliding_window_size = _get_native_paged_attention_params(
            layer, md.causal
        )
        paged_forward(
            q3,
            k_cache,
            v_cache,
            md.page_table,
            md.seq_lens,
            md.query_start_loc,
            md.prefix_kv_lens,
            out=out,
            block_size=self.page_size,
            softmax_scale=layer.scaling,
            causal=causal,
            sliding_window_size=sliding_window_size,
            num_kv_heads=layer.tp_k_head_num,
            linear_verify=(
                self._uses_native_linear_verify(forward_batch.forward_mode)
                and bool(_use_tilelang)
            ),
        )
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.head_dim)

    def support_triton(self):
        return True
