# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QSA attention backed by ``npu_fused_infer_attention_score`` (dense).

When the switch below is on, every real serving step of the QSA main
attention bypasses the GBSA sparse kernel and runs the dense
FlashAttention operator over the same paged QSA main K/V cache, in the
same form as the standard vllm-ascend attention backend
(``attention_v1.py``): per-request TND rows, per-request cumulative
query lengths, per-request block-table rows, and the fixed 2048x2048
int8 triu causal mask with ``sparse_mode=3``.

The per-request form replaces the per-token metadata broadcast of the
first implementation (one TND row per token), which turned every prefill
into an O(T^2) metadata expansion and dominated TTFT.

Switch: ``VLLM_ASCEND_QSA_SHORTSEQ_FIA=1`` (default) run the dense
branch, ``0`` keeps the GBSA sparse kernel.  Startup graph-capture and
profiling runs always take the sparse path: the dense path allocates
per-step buffers and cannot tile on synthetic warmup metadata.
"""

import os
from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.common.qsa_cache import QSAForwardMetadata

SHORTSEQ_FIA_ENV = "VLLM_ASCEND_QSA_SHORTSEQ_FIA"

# The standard backend's causal FIA contract: a fixed 2048x2048 int8
# triu template regardless of sequence length (the operator rejects
# exact-size and larger masks with 561103, while every tested sequence
# length works with this exact template).
FIA_MASK_SIZE = 2048

# The paged FIAS tiling rejects pages larger than this many tokens
# (768/1024-token pages fail with 561103 on 910B3; 512 and below pass).
FIA_MAX_PAGE = 512


_mask_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def qsa_shortseq_fia_enabled() -> bool:
    """Read the dense-attention switch (default: enabled)."""
    return os.environ.get(SHORTSEQ_FIA_ENV, "1") not in ("0", "false", "False")


def _get_causal_mask(device: torch.device) -> torch.Tensor:
    """The fixed triu template shared with the standard backend."""
    key = (device, torch.int8)
    mask = _mask_cache.get(key)
    if mask is None:
        mask = (
            torch.triu(
                torch.ones(FIA_MASK_SIZE, FIA_MASK_SIZE),
                diagonal=1,
            )
            .to(torch.int8)
            .to(device)
        )
        _mask_cache[key] = mask
    return mask


def _as_fia_page_pool(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Normalize the QSA K/V cache to the paged 3-D form FIAS expects.

    Returns ``(key, value, fia_page, storage_page)`` where the first two
    are zero-copy ``[num_blocks, page, nkv*head]`` views (the standard
    backend feeds the operator the same 3-D view shape).  Pages larger
    than ``FIA_MAX_PAGE`` are re-indexed as consecutive sub-pages with a
    pure view; the caller remaps the block table in lockstep.
    """
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("QSA FIA requires 4-D K/V caches")
    if key_cache.shape != value_cache.shape:
        raise ValueError("QSA FIA requires matching K/V cache shapes")
    if key_cache.shape[2] == num_kv_heads and key_cache.shape[1] != num_kv_heads:
        # already [blocks, page, nkv, head]
        pass
    elif key_cache.shape[1] == num_kv_heads and key_cache.shape[2] != num_kv_heads:
        # [blocks, nkv, page, head] -> [blocks, page, nkv, head]
        key_cache = key_cache.transpose(1, 2)
        value_cache = value_cache.transpose(1, 2)
    else:
        raise ValueError(
            "QSA FIA cannot locate the kv-head axis in cache "
            f"shape={tuple(key_cache.shape)}, num_kv_heads={num_kv_heads}"
        )
    num_blocks, page, nkv, head = key_cache.shape
    key = key_cache.view(num_blocks, page, nkv * head)
    value = value_cache.view(num_blocks, page, nkv * head)
    if page > FIA_MAX_PAGE:
        # Split large storage pages into consecutive sub-pages by pure
        # view (GBSA's page-pool approach): one storage block becomes
        # ``sub`` adjacent FIA pages.  The block table is remapped in
        # lockstep: storage block b covers FIA pages [b*sub, (b+1)*sub).
        fia_page = FIA_MAX_PAGE
        while page % fia_page:
            fia_page //= 2
        sub = page // fia_page
        key = key.view(num_blocks * sub, fia_page, nkv * head)
        value = value.view(num_blocks * sub, fia_page, nkv * head)
        return key, value, fia_page, page
    return key, value, page, page


def qsa_fia_shortseq_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: "QSAForwardMetadata",
    token_to_req: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    logical_positions: torch.Tensor,
) -> torch.Tensor:
    """Dense QSA attention via FIAS (standard-backend per-request form).

    ``query``:  ``[num_tokens, num_heads, head_dim]`` bf16.
    ``metadata``: the QSA main-attention metadata (block_table, seq_lens,
        query_start_loc, num_actual_tokens).
    ``token_to_req``: side-cache token->request index (same token count).
    ``logical_positions``: per-token logical position (unused here, kept
        for ABI compatibility with the gate).
    ``output``:  ``[num_tokens, num_heads, head_dim]`` (token-major),
        written in place and returned.
    """
    num_tokens = metadata.num_actual_tokens
    del token_to_req, logical_positions

    key, value, block_size, storage_page = _as_fia_page_pool(
        key_cache, value_cache, num_kv_heads
    )

    # Per-request rows, exactly like the standard backend: the block
    # table already has one row per request.
    block_table = metadata.block_table
    if block_table.dtype not in (torch.int32, torch.int64):
        block_table = block_table.to(torch.int32)
    if storage_page != block_size:
        sub = storage_page // block_size
        block_table = (
            block_table.unsqueeze(-1) * sub
            + torch.arange(sub, dtype=block_table.dtype, device=block_table.device)
        ).reshape(block_table.shape[0], -1)

    # Per-request cumulative query lengths (TND rows are requests).
    query_lens = metadata.query_start_loc[1:] - metadata.query_start_loc[:-1]
    actual_seq_qlen = torch.cumsum(query_lens, dim=0).to(torch.int32)
    actual_seq_kvlen = metadata.seq_lens.to(torch.int32)

    num_heads = query.shape[1]
    logger.info_once(
        "QSA short-sequence attention is using npu_fused_infer_attention_score "
        "(dense, per-request TND)."
    )
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query[:num_tokens].contiguous(),
        key,
        value,
        atten_mask=_get_causal_mask(query.device),
        block_table=block_table,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=actual_seq_qlen,
        actual_seq_lengths_kv=actual_seq_kvlen,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        scale=scale,
        sparse_mode=3,
        pre_tokens=2147483647,
        next_tokens=2147483647,
    )
    dims = (num_heads, output.shape[2])
    output[:num_tokens].copy_(attn_output.view(num_tokens, *dims))
    return output
