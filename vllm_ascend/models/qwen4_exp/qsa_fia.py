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

"""QSA short-sequence attention: dense ``npu_fused_infer_attention_score``.

The short-sequence deployment mode keeps the QSA main cache (already a
standard paged KV cache) and replaces indexer-driven sparse attention by
plain dense attention over the same paged K/V pool:

* K/V are consumed zero-copy as fused-head page pools
  ``[blocks, page, nkv*head]`` — the page-major QSA double-slot views
  squeezed over the singleton head axis; strided (padded block stride)
  pools are accepted directly (verified on 910B3, CANN 9.2.0),
* ``block_table`` carries one row *per query token* (TND rows are
  individual tokens), int32,
* ``actual_seq_lengths`` is ``arange(1, num_tokens + 1)`` (one token per
  row); ``actual_seq_lengths_kv`` is the per-row visible kv length,
* pages up to 512 tokens tile; larger pages are re-packed into sub-pages
  (768/1024-token pages fail with 561103 on 910B3).

Per-row visible kv length:

* gate path (side-cache metadata available): the indexer causal rule
  ``min(logical_position + 1, seq_len)`` — no attention mask materialized;
* stub path: true causal visibility derived from ``query_start_loc`` and
  ``seq_lens`` on device.  A token at batch-local position ``t`` in the
  request covering ``[qsl[r], qsl[r+1])`` sits at global position
  ``seq_lens[r] - (qsl[r+1] - qsl[r]) + (t - qsl[r])``, so it sees
  ``t + 1 - qsl[r+1] + seq_lens[r]`` keys.  This is exact for decode
  (one token per request), whole-prompt and chunked prefill, and
  prefix-cached requests alike.

Switch: ``VLLM_ASCEND_QSA_SHORTSEQ_FIA`` (default) keeps the dense
selection.  The dense operator reads per-row kv lengths as host-visible
scalars, so it cannot run inside a graph capture (107027 during
capture): capture windows and startup profiling runs fall back to the
capture-safe GBSA select-all pattern, which is baked into the replayed
graph, while eager steps (prefill, chunked prefill, and decode outside
captured shapes) run the dense operator for real.
"""

import os
from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.common.qsa_cache import QSAForwardMetadata

SHORTSEQ_FIA_ENV = "VLLM_ASCEND_QSA_SHORTSEQ_FIA"

# Paged FIAS tiling rejects pages larger than this many tokens
# (768/1024-token pages fail with 561103 on 910B3; 512 and below pass).
FIA_MAX_PAGE = 512

# Static all-select pattern width for the profiling-run fallback.  512
# compressed blocks cover 2048 tokens; GBSA's wrapper slices
# ``width - 3`` columns as block indices (the 3-column tail mimics the
# packed LI-v2 buffer contract).
SELECT_BLOCKS = 512
SELECT_RESIDUAL = 3


def qsa_shortseq_fia_enabled() -> bool:
    """Read the dense-attention switch (default: enabled)."""
    return os.environ.get(SHORTSEQ_FIA_ENV, "1") not in ("0", "false", "False")


def _in_capture_window() -> bool:
    """True while the step must not run host-syncing dense operators.

    Mirrors ``QSA AttentionImpl._shortseq_fia_gate``: graph capture runs
    (``_EXTRA_CTX.capturing``), profiling/warmup dummy runs
    (``in_profile_run``), the whole startup capture window
    (``cudagraph_capturing_enabled``), and any non-NONE graph runtime
    mode.
    """
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    try:
        if _EXTRA_CTX.capturing or _EXTRA_CTX.in_profile_run:
            return True
    except Exception:
        # No forward context bound (e.g. unit tests): treat as eager.
        pass
    try:
        from vllm.compilation.monitor import cudagraph_capturing_enabled

        if cudagraph_capturing_enabled:
            return True
    except Exception:
        pass
    from vllm.config.compilation import CUDAGraphMode  # noqa: PLC0415
    from vllm.forward_context import get_forward_context  # noqa: PLC0415

    runtime_mode = getattr(get_forward_context(), "cudagraph_runtime_mode", None)
    return runtime_mode is not CUDAGraphMode.NONE


class _SelectAllState:
    """Persistent per-device buffers for the GBSA select-all fallback.

    ``indices``:      [max_tokens, SELECT_BLOCKS + SELECT_RESIDUAL] int32,
                      row content is arange(SELECT_BLOCKS + RESIDUAL),
                      written once at creation.
    ``token_to_req``: [max_tokens] int64, rewritten every step.
    ``seq_lens``:     [max_num_reqs] int32 mirror of metadata seq_lens.
    """

    def __init__(self, device: torch.device) -> None:
        from vllm.config import get_current_vllm_config

        sched = get_current_vllm_config().scheduler_config
        max_tokens = int(sched.max_num_batched_tokens)
        max_reqs = int(sched.max_num_seqs)
        self.indices = (
            torch.arange(
                SELECT_BLOCKS + SELECT_RESIDUAL,
                dtype=torch.int32,
                device=device,
            )
            .expand(max_tokens, -1)
            .contiguous()
        )
        self.token_to_req = torch.zeros(
            max_tokens, dtype=torch.int64, device=device
        )
        self.seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)


_state: dict[torch.device, _SelectAllState] = {}


def _get_state(device: torch.device) -> _SelectAllState:
    state = _state.get(device)
    if state is None:
        state = _SelectAllState(device)
        _state[device] = state
    return state


def _split_fused_heads(cache: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Normalize a page-major QSA K/V slot view to fused-head 3-D form.

    Accepts ``[blocks, page, nkv, head]`` or ``[blocks, nkv, page, head]``
    (both zero-copy) and returns ``[blocks, page, nkv*head]``.  The page
    axis must end up outermost for FIAS paged tiling.
    """
    if cache.ndim == 3 and cache.shape[2] == num_kv_heads * cache.shape[-1]:
        return cache
    if cache.ndim != 4:
        raise ValueError(f"QSA FIA requires a 3-D/4-D KV cache, got {cache.ndim}-D")
    if cache.shape[2] == num_kv_heads and cache.shape[1] != num_kv_heads:
        # already [blocks, page, nkv, head]
        pass
    elif cache.shape[1] == num_kv_heads and cache.shape[2] != num_kv_heads:
        # [blocks, nkv, page, head] -> [blocks, page, nkv, head]
        cache = cache.transpose(1, 2)
    else:
        raise ValueError(
            "QSA FIA cannot locate the kv-head axis in cache "
            f"shape={tuple(cache.shape)}, num_kv_heads={num_kv_heads}"
        )
    return cache.squeeze(2) if cache.shape[2] == 1 else cache.flatten(2)


def _assemble_paged_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Normalize the QSA K/V cache to the paged form FIAS expects.

    Returns ``(key, value, fia_page, big_page)`` with the page axis
    outermost and the head dims fused into one axis; ``big_page !=
    fia_page`` tells the caller to remap block ids in lockstep.
    """
    if key_cache.shape != value_cache.shape:
        raise ValueError("QSA FIA requires matching K/V cache shapes")
    key = _split_fused_heads(key_cache, num_kv_heads)
    value = _split_fused_heads(value_cache, num_kv_heads)
    page = key.shape[1]
    if page <= FIA_MAX_PAGE:
        return key, value, page, page
    # The paged FIAS tiling only accepts pages up to FIA_MAX_PAGE tokens;
    # re-pack them into sub-pages.  The sub-page size must divide the big
    # page exactly (e.g. 768 splits into 3x256), so walk the supported
    # sizes down from FIA_MAX_PAGE.
    fia_page = FIA_MAX_PAGE
    while page % fia_page:
        fia_page //= 2
    sub = page // fia_page
    key = (
        key.reshape(key.shape[0], sub, fia_page, -1)
        .reshape(key.shape[0] * sub, fia_page, -1)
        .contiguous()
    )
    value = (
        value.reshape(value.shape[0], sub, fia_page, -1)
        .reshape(value.shape[0] * sub, fia_page, -1)
        .contiguous()
    )
    return key, value, fia_page, page


def _token_to_req_from_query_start_loc(
    metadata: "QSAForwardMetadata",
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Rebuild the token→request map on device, static shape.

    ``t2r(t)`` = number of request boundaries <= t.
    """
    token_arange = torch.arange(num_tokens, dtype=torch.int64, device=device)
    boundaries = metadata.query_start_loc[1:].to(torch.int64)
    return (
        token_arange.unsqueeze(1) >= boundaries.unsqueeze(0)
    ).sum(dim=1)


def _selectall_fallback(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: "QSAForwardMetadata",
    output: torch.Tensor,
) -> torch.Tensor:
    """GBSA select-all (capture-safe, value-fresh) for profiling runs.

    Feeds the GBSA sparse kernel a static "all-select" pattern
    (``compact_indices = arange``) so it computes dense attention over
    every compressed block plus the residual tail, while validity masking
    stays on-device.  Coverage boundary: sequences longer than
    ``SELECT_BLOCKS * GBSA_COMPRESS_RATIO + residual`` = 2051 tokens
    attend only over their first 2051 tokens.
    """
    from vllm_ascend.ops.gbsa_qsa import qsa_gbsa_attention

    num_tokens = metadata.num_actual_tokens
    state = _get_state(query.device)

    state.token_to_req[:num_tokens] = _token_to_req_from_query_start_loc(
        metadata, num_tokens, query.device
    )
    # Mirror the device seq_lens into a persistent int32 buffer so the
    # GBSA wrapper indexes stable storage across steps.
    state.seq_lens[: metadata.seq_lens.shape[0]].copy_(metadata.seq_lens)

    logger.info_once(
        "QSA short-sequence attention is using GBSA select-all "
        "(dense-equivalent, capture-safe, value-fresh on replay)."
    )
    return qsa_gbsa_attention(
        query[:num_tokens],
        key_cache,
        value_cache,
        state.indices[:num_tokens],
        metadata.block_table,
        state.token_to_req[:num_tokens],
        state.seq_lens[: metadata.seq_lens.shape[0]],
        output[:num_tokens],
    )


def qsa_fia_shortseq_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: "QSAForwardMetadata",
    token_to_req: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    logical_positions: torch.Tensor | None,
) -> torch.Tensor:
    """Dense QSA attention via FIAS (paged TND, zero-copy).

    ``query``:  ``[num_tokens, num_heads, head_dim]`` bf16.
    ``metadata``: the QSA main-attention metadata (block_table, seq_lens,
        query_start_loc, num_actual_tokens).
    ``token_to_req``: side-cache token→request index when the gate path
        passes it (same token count); the stub path passes ``None`` and
        the map is rebuilt on device from ``query_start_loc``.
    ``logical_positions``: per-token logical position from the QSA
        side-cache metadata (gate path), or ``None`` (stub path) to use
        the true causal visibility derived from ``query_start_loc``.
    ``output``:  ``[num_tokens, num_heads, head_dim]`` (token-major),
        written in place and returned.
    """
    num_tokens = metadata.num_actual_tokens
    if num_tokens == 0:
        return output

    # The dense operator's TND tiling reads per-row kv lengths as
    # host-visible scalars, so it must not run inside a graph capture
    # (107027) nor on the synthetic metadata of startup profiling runs.
    # Capture-safe fallback: GBSA select-all gets baked into the replayed
    # graph (it recomputes validity from on-device seq_lens every step).
    if _in_capture_window():
        return _selectall_fallback(
            query, key_cache, value_cache, metadata, output
        )

    if logical_positions is not None:
        # Gate path: token→request map comes from the side cache.
        row_to_req = token_to_req[:num_tokens].to(torch.int64)
    else:
        # Stub path: no side caches, rebuild the map on device.
        row_to_req = _token_to_req_from_query_start_loc(
            metadata, num_tokens, query.device
        )

    key, value, fia_page, big_page = _assemble_paged_cache(
        key_cache, value_cache, num_kv_heads
    )

    # Paged access: one block-table row per query token (TND rows are
    # individual tokens, so request pages are gathered per token).  The
    # verified operator form takes int32 block ids.
    block_table = metadata.block_table.index_select(0, row_to_req)
    if block_table.dtype is not torch.int32:
        block_table = block_table.to(torch.int32)
    if big_page != fia_page:
        # Remap big-page ids to consecutive FIA_MAX_PAGE sub-page ids:
        # big page b covers sub-pages [b*sub, (b+1)*sub).
        sub = big_page // fia_page
        block_table = (
            block_table.unsqueeze(-1) * sub
            + torch.arange(sub, dtype=block_table.dtype, device=block_table.device)
        ).reshape(block_table.shape[0], -1)

    # One query token per sequence row.  In the TND layout the operator
    # reads the per-row kv length from ``actual_seq_lengths_kv`` (verified
    # on 910B3: the value is the row's own visible length, not a
    # cumulative offset).
    qsl = metadata.query_start_loc.to(torch.int64)
    seq_lens = metadata.seq_lens.to(torch.int64)
    if logical_positions is not None:
        actual_seq_kvlen = torch.minimum(
            logical_positions[:num_tokens].to(torch.int64) + 1,
            seq_lens.index_select(0, row_to_req),
        ).to(torch.int32)
    else:
        # True causal visibility: a token at batch-local position t in the
        # request covering [qsl[r], qsl[r+1]) sees
        # t + 1 - qsl[r+1] + seq_lens[r] keys (exact for decode, whole-
        # prompt and chunked prefill, and prefix-cached requests).
        token_arange = torch.arange(
            num_tokens, dtype=torch.int64, device=query.device
        )
        chunk_end = qsl.index_select(0, row_to_req + 1)
        actual_seq_kvlen = (
            token_arange + 1 - chunk_end + seq_lens.index_select(0, row_to_req)
        ).to(torch.int32)
    actual_seq_qlen = torch.arange(
        1, num_tokens + 1, dtype=torch.int32, device=query.device
    )

    num_heads = query.shape[1]
    logger.info_once(
        "QSA short-sequence attention is using npu_fused_infer_attention_score "
        "(dense, paged TND, zero-copy)."
    )
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query[:num_tokens].contiguous(),
        key,
        value,
        block_table=block_table,
        input_layout="TND",
        block_size=fia_page,
        actual_seq_lengths=actual_seq_qlen,
        actual_seq_lengths_kv=actual_seq_kvlen,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        scale=scale,
        sparse_mode=0,
    )
    dims = (num_heads, output.shape[2])
    output[:num_tokens].copy_(attn_output.view(num_tokens, *dims))
    return output
