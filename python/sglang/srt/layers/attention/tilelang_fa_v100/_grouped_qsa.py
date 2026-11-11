"""Optional Flash-V100 grouped Page4 prefill for Qwen's structured QSA indices.

Uses the grouped planner/attention API from 1Cat-vLLM's Flash-V100 fork.
The caller must supply the allocator page size and compressed Page4 selections;
arbitrary token permutations or duplicate selections are not this API's input.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_WORKSPACES = {}
# Largest prefill chunk the grouped route serves; rows are padded to a multiple
# of the eight-query group.
_MAX_GROUPED_ROWS = 8192
_GROUP_ROWS = 8


@lru_cache(maxsize=1)
def load_grouped_qsa_ops():
    default = (
        Path(__file__).resolve().parents[4] / "jit_kernel" / "_sm70_grouped_qsa_v100.so"
    )
    path = Path(os.environ.get("SGLANG_SM70_GROUPED_QSA_LIBRARY", str(default)))
    if not path.is_file():
        logger.warning(
            "Grouped QSA library is missing: %s; retaining native QSA.", path
        )
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "_sglang_grouped_qsa.flash_attn_v100_cuda", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not all(
            hasattr(module, name)
            for name in ["grouped_sparse_page4_plan_fwd", "grouped_sparse_page4_fwd"]
        ):
            raise RuntimeError("Flash-V100 library lacks grouped Page4 bindings")
        if not getattr(module, "sglang_masked_sparse_kv", False):
            logger.warning(
                "Grouped QSA library lacks masked KV-load protection; "
                "rebuild with scripts/setup_v100_grouped_qsa.py. "
                "Retaining native QSA."
            )
            return None
        return module
    except Exception:
        logger.exception(
            "Grouped QSA library could not be loaded; retaining native QSA."
        )
        return None


def can_use_grouped_qsa(q, k, v, indices, page_size):
    return (
        os.environ.get("SGLANG_SM70_QSA_GROUPED_PREFILL", "0") == "1"
        and page_size == 16
        and q.is_cuda
        and q.dtype == torch.float16
        and q.ndim == 3
        and tuple(q.shape[1:]) == (6, 256)
        and 0 < q.shape[0] <= _MAX_GROUPED_ROWS
        and tuple(indices.shape) == (q.shape[0], 2051)
        and q.is_contiguous()
        and indices.is_contiguous()
        and indices.dtype == torch.int32
        and k.dtype == v.dtype == torch.float16
        and k.device == v.device == indices.device == q.device
        and k.is_contiguous()
        and v.is_contiguous()
        and k.shape == v.shape
        and tuple(k.shape[1:]) == (1, 256)
        and k.shape[0] % page_size == 0
        and torch.cuda.get_device_capability(q.device) == (7, 0)
    )


@triton.jit
def _prepare_metadata(
    table,
    req,
    lens,
    blocks,
    positions,
    token_to_req,
    STRIDE: tl.constexpr,
    PAGES: tl.constexpr,
    ROWS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(req).to(tl.int64)
    slot = tl.load(
        table + request * STRIDE + offset * PAGE_SIZE, offset < PAGES, other=0
    )
    tl.store(blocks + offset, slot // PAGE_SIZE, offset < PAGES)
    position = tl.load(lens).to(tl.int64) - ROWS + offset
    tl.store(positions + offset, position, offset < ROWS)
    tl.store(token_to_req + offset, 0, offset < ROWS)


@triton.jit
def _mask_future(
    indices, lens, out, ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // WIDTH
    column = offset % WIDTH
    visible = tl.load(lens) - ROWS + row + 1
    token = tl.load(indices + offset, offset < ROWS * WIDTH, other=-1)
    valid = (
        (column < tl.minimum(WIDTH, tl.maximum(visible, 0)))
        & (token >= 0)
        & (token < visible)
    )
    tl.store(out + offset, tl.where(valid, token, -1), offset < ROWS * WIDTH)


def grouped_qsa_prefill(q, k, v, table, req, indices, lens, scale, page_size=16):
    """Read a paged FP16 cache using up to 512 full Page4 groups plus their tail."""
    ops = load_grouped_qsa_ops()
    if ops is None:
        return None
    pad = (-q.shape[0]) % _GROUP_ROWS
    if pad:
        # Rows are the last `rows` positions of the request, so lead with copies
        # of the first row (its position stays valid after masking) and drop
        # their outputs.
        q = torch.cat([q[:1].expand(pad, -1, -1), q]).contiguous()
        indices = torch.cat([indices[:1].expand(pad, -1), indices]).contiguous()
        output = grouped_qsa_prefill(
            q, k, v, table, req, indices, lens, scale, page_size
        )
        return None if output is None else output[pad:]
    rows = q.shape[0]
    assert rows > 0 and rows % 8 == 0 and indices.shape == (rows, 2051)
    assert page_size == 16 and k.shape[0] % page_size == 0
    assert q.is_contiguous() and indices.is_contiguous()
    assert req.numel() == lens.numel() == 1
    assert all(t.device == q.device for t in (k, v, table, req, indices, lens))
    groups = rows // 8
    pages = triton.cdiv(table.shape[1], page_size)
    # One workspace per (device, stream, pages), sized for the largest row
    # count seen so far; ragged chunks use leading slices instead of keeping a
    # separate allocation per distinct row count.
    key = (q.device.index, torch.cuda.current_stream(q.device).cuda_stream, pages)
    capacity = _WORKSPACES.get(key)
    if capacity is None or capacity["rows"] < rows:
        cap = rows if capacity is None else max(rows, capacity["rows"])
        cap_groups = cap // 8
        capacity = dict(
            rows=cap,
            blocks=torch.empty((1, pages), dtype=torch.int32, device=q.device),
            positions=torch.empty(cap, dtype=torch.int64, device=q.device),
            token_to_req=torch.empty(cap, dtype=torch.int32, device=q.device),
            selected=torch.empty((cap, 2051), dtype=torch.int32, device=q.device),
            grouped_pages=torch.empty(
                (cap_groups, 4160), dtype=torch.int32, device=q.device
            ),
            masks=torch.empty(
                (cap_groups, 4160), dtype=torch.uint32, device=q.device
            ),
            lengths=torch.empty(cap_groups, dtype=torch.int32, device=q.device),
            lse=torch.empty((cap, 6), dtype=torch.float32, device=q.device),
        )
        _WORKSPACES[key] = capacity
    workspace = dict(
        blocks=capacity["blocks"],
        positions=capacity["positions"][:rows],
        token_to_req=capacity["token_to_req"][:rows],
        selected=capacity["selected"][:rows],
        grouped_pages=capacity["grouped_pages"][:groups],
        masks=capacity["masks"][:groups],
        lengths=capacity["lengths"][:groups],
        lse=capacity["lse"][:rows],
    )
    _prepare_metadata[(triton.cdiv(max(rows, pages), 256),)](
        table,
        req,
        lens,
        workspace["blocks"],
        workspace["positions"],
        workspace["token_to_req"],
        table.stride(0),
        pages,
        rows,
        page_size,
        256,
    )
    _mask_future[(triton.cdiv(rows * 2051, 256),)](
        indices, lens, workspace["selected"], rows, 2051, 256
    )
    sequence_lengths = lens.to(dtype=torch.int32).contiguous()
    ops.grouped_sparse_page4_plan_fwd(
        workspace["selected"],
        workspace["blocks"],
        workspace["token_to_req"],
        workspace["positions"],
        sequence_lengths,
        workspace["grouped_pages"],
        workspace["masks"],
        workspace["lengths"],
        page_size,
        page_size // 4,
        k.shape[0] // page_size,
    )
    output = torch.empty_like(q)
    ops.grouped_sparse_page4_fwd(
        q,
        k.view(-1, 4, 1, 256),
        v.view(-1, 4, 1, 256),
        output,
        workspace["grouped_pages"],
        workspace["masks"],
        workspace["lengths"],
        workspace["lse"],
        float(scale),
    )
    logger.info_once(
        "SM70 QSA grouped Page4 prefill active: eight queries share selected KV pages."
    )
    return output
