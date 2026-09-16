# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Training-free sparse DFlash2 verify: scorer, selector, and consumer.

The sparse route scores 32-token tiles with the main K cache, keeps
sink + recent-window + top-k tiles in a compact virtual page table, and
replays the exact eighty-CTA grouped verifier over the compacted set.
These tests pin the three contracts that make the route safe:

1. when the sink/window/top-k budgets already cover every tile, the
   compact route must reproduce the dense verifier bitwise,
2. the compact table must stay causal (ascending tile bases, last tile
   trimmed to real tokens, draft tokens always covered), and
3. the whole chain must capture into a CUDA graph and replay with new
   seq_lens without host syncs.
"""

from __future__ import annotations

import pytest
import torch

PAGE = 3296  # interleaved K|V page: 1648 K tokens then 1648 V tokens
SPARSE_TILE = 32


def _require_sparse_verify():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("dflash2 sparse verify is SM70-only")
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    sparse_op = getattr(interface, "flash_attn_dflash2_verify_sparse_paged", None)
    if sparse_op is None or not getattr(sparse_op, "available", False):
        pytest.skip("Flash-V100 extension lacks dflash2 sparse verify")
    return interface, sparse_op


def _dense_grouped_verify(interface):
    dense_op = getattr(interface, "flash_attn_grouped_verify_paged", None)
    if dense_op is None:
        pytest.skip("Flash-V100 extension lacks the dense grouped verifier")
    return dense_op


def _make_paged_cache(num_pages: int, seed: int = 11):
    torch.manual_seed(seed)
    # Interleaved layout: each page packs 1648 K tokens then 1648 V tokens,
    # so the K and V halves are strided views over one fp8 storage.
    cache = torch.randn(
        (num_pages, 2 * PAGE, 1, 256), dtype=torch.float16, device="cuda"
    )
    key_cache = cache.to(torch.float8_e5m2).view(torch.uint8)[:, :PAGE]
    value_cache = cache.to(torch.float8_e5m2).view(torch.uint8)[:, PAGE:]
    return key_cache, value_cache


def _block_table_for(seq_len: int) -> torch.Tensor:
    blocks = (seq_len + PAGE - 1) // PAGE
    table = torch.arange(blocks, dtype=torch.int32, device="cuda")
    return table.unsqueeze(0)


def _tile_view(cache: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Dequantized K/V rows for absolute token ids from an interleaved half.

    Token ``t`` of this half lives at page ``t // PAGE``, row ``t % PAGE`` —
    the same resolution the kernel's token-table mode performs per gather.
    """
    rows = cache.view(torch.float8_e5m2).float()
    return rows[token_ids // PAGE, token_ids % PAGE, 0]


def _run_sparse(
    sparse_op,
    query,
    key_cache,
    value_cache,
    block_table,
    seq_lens,
    *,
    topk_tokens=768,
    sink_tokens=256,
    window_tokens=1024,
    out=None,
):
    return sparse_op(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        out=out,
        kv_cache_dtype="fp8_e5m2",
        topk_tokens=topk_tokens,
        sink_tokens=sink_tokens,
        window_tokens=window_tokens,
    )


@pytest.mark.parametrize("seq_len", [1024, 2048])
@torch.inference_mode()
def test_sm70_dflash2_sparse_covered_budget_matches_dense(seq_len: int) -> None:
    """64 tiles are exactly sink 8 + window 32 + top-k 24: sparse == dense."""
    interface, sparse_op = _require_sparse_verify()
    dense_op = _dense_grouped_verify(interface)
    torch.manual_seed(3)
    query = torch.randn((8, 6, 256), dtype=torch.float16, device="cuda") * 0.2
    key_cache, value_cache = _make_paged_cache((seq_len + PAGE - 1) // PAGE)
    block_table = _block_table_for(seq_len)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    sparse_out = _run_sparse(
        sparse_op, query, key_cache, value_cache, block_table, seq_lens
    )
    dense_out = dense_op(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        kv_cache_dtype="fp8_e5m2",
        one_pass=True,
    )
    torch.testing.assert_close(sparse_out, dense_out, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_sm70_dflash2_sparse_permuted_pages_match_dense() -> None:
    """Non-identity block tables: compact entries are PHYSICAL token bases.

    A 64-token page keeps the 2048-token sequence inside the 64-tile
    sink+window+top-k budget while spanning 32 pages, so a compact entry
    that ignored the block table would gather the wrong physical page and
    break the dense match.
    """
    interface, sparse_op = _require_sparse_verify()
    dense_op = _dense_grouped_verify(interface)
    torch.manual_seed(13)
    page = 64
    seq_len = 2048  # 64 tiles == sink 8 + window 32 + top-k 24: full coverage
    num_pages = (seq_len + page - 1) // page
    cache = torch.randn(
        (num_pages, 2 * page, 1, 256), dtype=torch.float16, device="cuda"
    )
    # Scatter pages so logical page l lives at physical page perm[l].
    perm = torch.randperm(num_pages, device="cuda")
    cache = torch.empty_like(cache).index_copy_(0, perm, cache)
    key_cache = cache.to(torch.float8_e5m2).view(torch.uint8)[:, :page]
    value_cache = cache.to(torch.float8_e5m2).view(torch.uint8)[:, page:]
    block_table = perm.to(torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    query = torch.randn((8, 6, 256), dtype=torch.float16, device="cuda") * 0.2

    sparse_out = _run_sparse(
        sparse_op, query, key_cache, value_cache, block_table, seq_lens
    )
    dense_out = dense_op(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        kv_cache_dtype="fp8_e5m2",
        one_pass=True,
    )
    torch.testing.assert_close(sparse_out, dense_out, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_sm70_dflash2_sparse_compact_len_and_causality() -> None:
    """compact_len must trim phantom tail tokens; drafts must stay covered."""
    interface, sparse_op = _require_sparse_verify()
    extension = interface.flash_attn_v100_cuda
    torch.manual_seed(5)
    query = torch.randn((8, 6, 256), dtype=torch.float16, device="cuda") * 0.2
    seq_len = 5000  # 157 tiles: last tile holds 5000 - 156*32 = 8 tokens
    key_cache, value_cache = _make_paged_cache((seq_len + PAGE - 1) // PAGE + 1)
    block_table = _block_table_for(seq_len)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

    scores = torch.empty(8192, dtype=torch.float32, device="cuda")
    pages = torch.empty((1, 72), dtype=torch.int32, device="cuda")
    length = torch.empty(1, dtype=torch.int32, device="cuda")
    extension.dflash2_verify_sparse_topk(
        query,
        key_cache,
        block_table,
        seq_lens,
        scores,
        pages,
        length,
        768,
        256,
        1024,
    )
    compact_len = int(length.item())
    num_selected = (compact_len + SPARSE_TILE - 1) // SPARSE_TILE
    selected = pages[0][:num_selected]
    # Ascending tile bases, the final tile is the real last tile, and the
    # compact length trims its past-seq phantom tokens.
    assert (selected.diff() > 0).all()
    last_tile = int(selected[-1])
    assert last_tile == ((seq_len - 1) // SPARSE_TILE) * SPARSE_TILE
    assert compact_len == (num_selected - 1) * SPARSE_TILE + (
        seq_len - last_tile
    )
    # The 32-token window around the draft tail must always be selected.
    num_tiles = (seq_len + SPARSE_TILE - 1) // SPARSE_TILE
    for tile in range(num_tiles - 32, num_tiles):
        assert ((selected - tile * SPARSE_TILE) == 0).any()

    sparse_out = _run_sparse(
        sparse_op, query, key_cache, value_cache, block_table, seq_lens
    )
    assert sparse_out.shape == (8, 6, 256)
    assert torch.isfinite(sparse_out).all()


@torch.inference_mode()
def test_sm70_dflash2_sparse_matches_fp32_reference() -> None:
    """The compact route must equal a masked FP32 softmax on its own tiles."""
    interface, sparse_op = _require_sparse_verify()
    extension = interface.flash_attn_v100_cuda
    torch.manual_seed(9)
    query = torch.randn((8, 6, 256), dtype=torch.float16, device="cuda") * 0.2
    seq_len = 3000  # sink+window covers 40 of 94 tiles; topk fills the rest
    key_cache, value_cache = _make_paged_cache((seq_len + PAGE - 1) // PAGE + 1)
    block_table = _block_table_for(seq_len)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

    scores = torch.empty(8192, dtype=torch.float32, device="cuda")
    pages = torch.empty((1, 72), dtype=torch.int32, device="cuda")
    length = torch.empty(1, dtype=torch.int32, device="cuda")
    extension.dflash2_verify_sparse_topk(
        query,
        key_cache,
        block_table,
        seq_lens,
        scores,
        pages,
        length,
        768,
        256,
        1024,
    )
    compact_len = int(length.item())
    num_selected = (compact_len + SPARSE_TILE - 1) // SPARSE_TILE
    selected = pages[0][:num_selected]

    # FP32 reference over the very tiles the selector kept: compact entries
    # are absolute token bases, so token j of tile t is base + j.
    token_ids = selected[:, None] + torch.arange(SPARSE_TILE, device="cuda")[
        None, :
    ]
    token_ids = token_ids.reshape(-1)[:compact_len]
    keys = _tile_view(key_cache, token_ids)
    values = _tile_view(value_cache, token_ids)

    prefix = compact_len - query.shape[0]
    scale = query.shape[-1] ** -0.5
    reference = torch.empty((8, 6, 256), dtype=torch.float32, device="cuda")
    for row in range(query.shape[0]):
        visible = prefix + row + 1
        logits = query[row].float() @ keys[:visible].T * scale
        weights = torch.softmax(logits, dim=-1)
        reference[row] = weights @ values[:visible]

    sparse_out = _run_sparse(
        sparse_op, query, key_cache, value_cache, block_table, seq_lens
    )
    # E5M2 storage + fp16 softmax accumulation: the house 3e-2 tolerance.
    torch.testing.assert_close(sparse_out.float(), reference, rtol=3e-2, atol=3e-2)


@torch.inference_mode()
def test_sm70_dflash2_sparse_cuda_graph_replay() -> None:
    """Capture once, then replay with growing seq_lens; outputs stay exact."""
    interface, sparse_op = _require_sparse_verify()
    torch.manual_seed(21)
    query = torch.randn((8, 6, 256), dtype=torch.float16, device="cuda") * 0.2
    seq_len = 40000
    max_seq = seq_len + 4096
    num_pages = (max_seq + PAGE - 1) // PAGE + 1
    key_cache, value_cache = _make_paged_cache(num_pages)
    block_table = _block_table_for(max_seq)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    static_out = torch.empty((8, 6, 256), dtype=torch.float16, device="cuda")

    def _capture_target() -> None:
        _run_sparse(
            sparse_op,
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            out=static_out,
        )

    _capture_target()
    torch.cuda.synchronize()
    eager_reference = static_out.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _capture_target()
    torch.cuda.synchronize()

    for delta in (0, 32, 4096):
        seq_lens.fill_(seq_len + delta)
        graph.replay()
        torch.cuda.synchronize()
        # Same inputs through the captured graph and a fresh eager call must
        # agree bitwise (static workspaces, no host-side branching).
        reference = _run_sparse(
            sparse_op, query, key_cache, value_cache, block_table, seq_lens
        )
        torch.testing.assert_close(static_out, reference, rtol=0.0, atol=0.0)
    seq_lens.fill_(seq_len)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out, eager_reference, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Batched (request-major) sparse verify: the scorer/selector must tile and
# select per block-table row, and the consumer must resolve each row's
# compact table through its own row stride. B1 stays bitwise identical, so
# these tests exercise B > 1 against the batched dense verifier.
# ---------------------------------------------------------------------------


def _batched_block_table(num_seqs: int, pages_per_seq: int) -> torch.Tensor:
    """Row-major identity page tables with a private page range per row."""
    return torch.stack(
        [
            torch.arange(
                r * pages_per_seq,
                (r + 1) * pages_per_seq,
                dtype=torch.int32,
                device="cuda",
            )
            for r in range(num_seqs)
        ]
    )


@pytest.mark.parametrize("num_seqs", [2, 4])
@torch.inference_mode()
def test_sm70_dflash2_sparse_batched_covered_budget_matches_dense(
    num_seqs: int,
) -> None:
    """B2/B4 with per-row covered budgets: batched sparse == batched dense."""
    interface, sparse_op = _require_sparse_verify()
    dense_op = _dense_grouped_verify(interface)
    torch.manual_seed(31)
    seq_lens_list = [1024, 2048] * 2
    seq_lens_list = seq_lens_list[:num_seqs]
    max_seq = max(seq_lens_list)
    pages_per_seq = (max_seq + PAGE - 1) // PAGE
    query = (
        torch.randn((num_seqs * 8, 6, 256), dtype=torch.float16, device="cuda") * 0.2
    )
    key_cache, value_cache = _make_paged_cache(pages_per_seq * num_seqs)
    block_table = _batched_block_table(num_seqs, pages_per_seq)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")

    sparse_out = _run_sparse(
        sparse_op, query, key_cache, value_cache, block_table, seq_lens
    )
    dense_out = dense_op(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        kv_cache_dtype="fp8_e5m2",
        one_pass=True,
    )
    torch.testing.assert_close(sparse_out, dense_out, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_sm70_dflash2_sparse_batched_permuted_pages_match_dense() -> None:
    """Per-row permuted page tables: compact entries stay row-local.

    Each request row permutes its own 32-page slice, so a compact entry that
    ignored the row offset of the batched table would gather another row's
    physical pages and break the dense match.
    """
    interface, sparse_op = _require_sparse_verify()
    dense_op = _dense_grouped_verify(interface)
    torch.manual_seed(37)
    page = 64
    seq_len = 2048  # 64 tiles == sink 8 + window 32 + top-k 24: full coverage
    num_seqs = 2
    pages_per_seq = seq_len // page
    num_pages = pages_per_seq * num_seqs
    cache = torch.randn(
        (num_pages, 2 * page, 1, 256), dtype=torch.float16, device="cuda"
    )
    perm_rows = []
    for r in range(num_seqs):
        # Permute only this row's slice, reading from a clone: logical page l
        # of row r lives at physical page perm_rows[r][l].
        local = torch.randperm(pages_per_seq, device="cuda")
        perm = local + r * pages_per_seq
        src = cache[r * pages_per_seq : (r + 1) * pages_per_seq].clone()
        cache.index_copy_(0, perm, src)
        perm_rows.append(perm)
    key_cache = cache.to(torch.float8_e5m2).view(torch.uint8)[:, :page]
    value_cache = cache.to(torch.float8_e5m2).view(torch.uint8)[:, page:]
    block_table = torch.stack(perm_rows).to(torch.int32)
    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device="cuda")
    query = (
        torch.randn((num_seqs * 8, 6, 256), dtype=torch.float16, device="cuda") * 0.2
    )

    sparse_out = _run_sparse(
        sparse_op, query, key_cache, value_cache, block_table, seq_lens
    )
    dense_out = dense_op(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        kv_cache_dtype="fp8_e5m2",
        one_pass=True,
    )
    torch.testing.assert_close(sparse_out, dense_out, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_sm70_dflash2_sparse_batched_mixed_lengths() -> None:
    """Rows of different lengths select independently; both hit the reference."""
    interface, sparse_op = _require_sparse_verify()
    extension = interface.flash_attn_v100_cuda
    torch.manual_seed(41)
    num_seqs = 2
    seq_lens_list = [4096, 3000]  # 128 and 94 tiles: both trigger real top-k
    pages_per_seq = 2
    query = (
        torch.randn((num_seqs * 8, 6, 256), dtype=torch.float16, device="cuda") * 0.2
    )
    key_cache, value_cache = _make_paged_cache(pages_per_seq * num_seqs + 1)
    block_table = _batched_block_table(num_seqs, pages_per_seq)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")

    scores = torch.empty(num_seqs * 8192, dtype=torch.float32, device="cuda")
    pages = torch.empty((num_seqs, 72), dtype=torch.int32, device="cuda")
    length = torch.empty(num_seqs, dtype=torch.int32, device="cuda")
    extension.dflash2_verify_sparse_topk(
        query,
        key_cache,
        block_table,
        seq_lens,
        scores,
        pages,
        length,
        768,
        256,
        1024,
    )
    compact_lens = []
    for r, seq_len in enumerate(seq_lens_list):
        compact_len = int(length[r].item())
        num_selected = (compact_len + SPARSE_TILE - 1) // SPARSE_TILE
        selected = pages[r][:num_selected]
        # Ascending tile bases ending at the row's real last tile, with the
        # compact length trimmed to it — per row, not just row 0. Entries are
        # PHYSICAL bases: row r's pages start at physical page r*pages_per_seq,
        # so subtract that row offset before comparing with logical tiles.
        row_page_offset = r * pages_per_seq * PAGE
        assert (selected.diff() > 0).all()
        last_logical = int(selected[-1]) - row_page_offset
        assert last_logical == ((seq_len - 1) // SPARSE_TILE) * SPARSE_TILE
        assert compact_len == (num_selected - 1) * SPARSE_TILE + (
            seq_len - last_logical
        )
        compact_lens.append((compact_len, selected, row_page_offset))

    sparse_out = _run_sparse(
        sparse_op, query, key_cache, value_cache, block_table, seq_lens
    )
    assert sparse_out.shape == (num_seqs * 8, 6, 256)
    scale = query.shape[-1] ** -0.5
    for r, (compact_len, selected, _offset) in enumerate(compact_lens):
        token_ids = selected[:, None] + torch.arange(SPARSE_TILE, device="cuda")[
            None, :
        ]
        token_ids = token_ids.reshape(-1)[:compact_len]
        keys = _tile_view(key_cache, token_ids)
        values = _tile_view(value_cache, token_ids)
        q_rows = query[r * 8 : (r + 1) * 8]
        prefix = compact_len - q_rows.shape[0]
        reference = torch.empty((8, 6, 256), dtype=torch.float32, device="cuda")
        for row in range(q_rows.shape[0]):
            visible = prefix + row + 1
            logits = q_rows[row].float() @ keys[:visible].T * scale
            weights = torch.softmax(logits, dim=-1)
            reference[row] = weights @ values[:visible]
        torch.testing.assert_close(
            sparse_out[r * 8 : (r + 1) * 8].float(), reference, rtol=3e-2, atol=3e-2
        )


@torch.inference_mode()
def test_sm70_dflash2_sparse_batched_cuda_graph_replay() -> None:
    """B2 capture once, replay with growing seq_lens; outputs stay exact."""
    interface, sparse_op = _require_sparse_verify()
    torch.manual_seed(43)
    num_seqs = 2
    query = (
        torch.randn((num_seqs * 8, 6, 256), dtype=torch.float16, device="cuda") * 0.2
    )
    seq_len = 40000
    max_seq = seq_len + 4096
    pages_per_seq = (max_seq + PAGE - 1) // PAGE + 1
    key_cache, value_cache = _make_paged_cache(pages_per_seq * num_seqs)
    block_table = _batched_block_table(num_seqs, pages_per_seq)
    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device="cuda")
    static_out = torch.empty(
        (num_seqs * 8, 6, 256), dtype=torch.float16, device="cuda"
    )

    def _capture_target() -> None:
        _run_sparse(
            sparse_op,
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            out=static_out,
        )

    _capture_target()
    torch.cuda.synchronize()
    eager_reference = static_out.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _capture_target()
    torch.cuda.synchronize()

    for delta in (0, 32, 4096):
        seq_lens.fill_(seq_len + delta)
        graph.replay()
        torch.cuda.synchronize()
        # Same inputs through the captured graph and a fresh eager call must
        # agree bitwise (static workspaces, no host-side branching).
        reference = _run_sparse(
            sparse_op, query, key_cache, value_cache, block_table, seq_lens
        )
        torch.testing.assert_close(static_out, reference, rtol=0.0, atol=0.0)
    seq_lens.fill_(seq_len)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out, eager_reference, rtol=0.0, atol=0.0)
