# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch


def _require_grouped_page4():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("grouped QSA page4 is SM70-only")
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    extension = interface.flash_attn_v100_cuda
    if not hasattr(extension, "grouped_sparse_page4_fwd"):
        pytest.skip("Flash-V100 extension lacks grouped sparse page4")
    return extension


def _grouped_page4_abi_version(extension) -> int:
    capability = getattr(extension, "grouped_sparse_page4_abi_version", None)
    if callable(capability):
        return int(capability())
    doc = getattr(extension.grouped_sparse_page4_fwd, "__doc__", "") or ""
    return 2 if "arg11:" in doc else 1 if "arg8:" in doc else 0


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3", "fp8_e5m2"])
@torch.inference_mode()
def test_sm70_qsa_grouped_page4_calibrated_kv(kv_cache_dtype: str) -> None:
    extension = _require_grouped_page4()
    abi_version = _grouped_page4_abi_version(extension)
    if kv_cache_dtype == "fp8_e4m3" and abi_version < 2:
        pytest.skip("installed grouped page4 ABI does not support quantized K/V")
    if kv_cache_dtype == "fp8_e5m2" and abi_version < 3:
        pytest.skip("installed grouped page4 ABI does not support E5M2 K/V")
    torch.manual_seed(7)
    query = torch.randn((8, 6, 256), dtype=torch.float16, device="cuda") * 0.2
    key = torch.randn((1, 4, 1, 256), dtype=torch.float16, device="cuda") * 0.35
    value = torch.randn_like(key) * 0.3
    k_scale = float(key.abs().max().item()) / 448.0
    v_scale = float(value.abs().max().item()) / 448.0
    if kv_cache_dtype == "fp8_e4m3":
        key_cache = (key / k_scale).to(torch.float8_e4m3fn).view(torch.uint8)
        value_cache = (value / v_scale).to(torch.float8_e4m3fn).view(torch.uint8)
        reference_key = key_cache.view(torch.float8_e4m3fn).float() * k_scale
        reference_value = value_cache.view(torch.float8_e4m3fn).float() * v_scale
    elif kv_cache_dtype == "fp8_e5m2":
        # E5M2 carries no calibrated scale in this route (same semantics as
        # the dense grouped verifier): the storage round-trip is the only
        # quantization, so the reference dequantizes at unit scale.
        key_cache = key.to(torch.float8_e5m2).view(torch.uint8)
        value_cache = value.to(torch.float8_e5m2).view(torch.uint8)
        reference_key = key_cache.view(torch.float8_e5m2).float()
        reference_value = value_cache.view(torch.float8_e5m2).float()
        k_scale = v_scale = 1.0
    else:
        key_cache = key
        value_cache = value
        reference_key = key.float()
        reference_value = value.float()
        k_scale = v_scale = 1.0

    block_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    token_masks = torch.full((1, 1), 0xFFFFFFFF, dtype=torch.uint32, device="cuda")
    seq_lens = torch.tensor([4], dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    lse = torch.empty((8, 6), dtype=torch.float32, device="cuda")
    args = (
        query,
        key_cache,
        value_cache,
        output,
        block_table,
        token_masks,
        seq_lens,
        lse,
        256**-0.5,
    )
    if abi_version >= 2:
        extension.grouped_sparse_page4_fwd(
            *args,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    else:
        extension.grouped_sparse_page4_fwd(*args)

    reference_key = reference_key.view(4, 256)
    reference_value = reference_value.view(4, 256)
    scores = torch.matmul(query.float(), reference_key.transpose(0, 1)) / math.sqrt(256)
    probabilities = torch.softmax(scores, dim=-1)
    reference = torch.matmul(probabilities, reference_value)
    torch.testing.assert_close(output.float(), reference, atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_sm70_qsa_grouped_page4_e5m2_masked_reference() -> None:
    """E5M2 planner-contract round trip against a torch reference.

    Builds the exact artifacts the grouped sparse planner emits (packed
    page4 lists plus per-row token-mask nibbles, several groups, a tail of
    unused entries) and checks the consumer against masked softmax.
    """
    extension = _require_grouped_page4()
    if _grouped_page4_abi_version(extension) < 3:
        pytest.skip("installed grouped page4 ABI does not support E5M2 K/V")
    torch.manual_seed(11)
    groups, used_entries = 3, 13
    list_width = 16  # tail entries stay masked off and must be ignored
    pool = groups * list_width
    kv = torch.randn((pool, 4, 1, 256), dtype=torch.float16,
                     device="cuda") * 0.3
    key_cache = kv.to(torch.float8_e5m2).view(torch.uint8)
    value_cache = kv.mul(0.8).to(torch.float8_e5m2).view(torch.uint8)
    key_ref = key_cache.view(torch.float8_e5m2).float().view(pool, 4, 256)
    value_ref = value_cache.view(torch.float8_e5m2).float().view(pool, 4, 256)

    query = torch.randn((groups * 8, 6, 256), dtype=torch.float16,
                        device="cuda") * 0.2
    # Disjoint page ranges per group keep the reference aliasing-free.
    block_table = torch.arange(pool, dtype=torch.int32,
                               device="cuda").view(groups, list_width)
    token_masks = torch.zeros((groups, list_width), dtype=torch.uint32,
                              device="cuda")
    seq_lens = torch.full((groups,), used_entries * 4, dtype=torch.int32,
                          device="cuda")
    random_masks = torch.randint(0, 1 << 32, (groups, used_entries),
                                 device="cuda", dtype=torch.int64)
    token_masks[:, 0] = 0xFFFFFFFF  # keep every row's visible set non-empty
    token_masks[:, 1:used_entries] = random_masks[:, 1:].to(torch.uint32)

    output = torch.empty_like(query)
    lse = torch.empty((groups * 8, 6), dtype=torch.float32, device="cuda")
    extension.grouped_sparse_page4_fwd(
        query,
        key_cache,
        value_cache,
        output,
        block_table,
        token_masks,
        seq_lens,
        lse,
        256**-0.5,
        "fp8_e5m2",
        1.0,
        1.0,
    )

    reference = torch.zeros((groups * 8, 6, 256), dtype=torch.float32,
                            device="cuda")
    masks = token_masks.cpu()
    pages = block_table.cpu()
    for group in range(groups):
        entries = pages[group, :used_entries].tolist()
        group_masks = masks[group, :used_entries].tolist()
        for row in range(8):
            positions = [
                (entry, sub) for entry, mask in zip(entries, group_masks)
                for sub in range(4) if (mask >> (row * 4 + sub)) & 1
            ]
            page_idx = torch.tensor([p for p, _ in positions],
                                    device="cuda")
            subs = torch.tensor([s for _, s in positions], device="cuda")
            keys = key_ref[page_idx, subs]  # [N, 256]
            values = value_ref[page_idx, subs]
            q_row = query[group * 8 + row].float()  # [6, 256]
            scores = q_row @ keys.T * 256**-0.5  # [6, N]
            probabilities = torch.softmax(scores, dim=-1)
            reference[group * 8 + row] = probabilities @ values
    torch.testing.assert_close(
        output.float(), reference, atol=3e-2, rtol=3e-2)
