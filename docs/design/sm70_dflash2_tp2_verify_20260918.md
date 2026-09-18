# SM70 DFlash2 verify on TP2: adaptation plan

Sister document to `sm70_dflash2_concurrent_20260917.md` (concurrency,
TP4-validated) and `sm70_dflash2_verify_kv_pipeline_20260916.md`
(single-request sparse verify). This plan extends the DFlash2 long-context
verify routes (sparse + request-major batched) to **TP2**, where the
per-rank topology is H12/Hkv2/D256 instead of the H6/Hkv1 the current
contract pins.

## 1. Why TP2 is currently locked out

Qwen3.8-27B has 24 query heads and 4 KV heads. TP4 gives 6/1 per rank
(the measured topology); TP2 gives **12/2**. Three layers pin 6/1:

| Layer | Location | Pin |
|---|---|---|
| Python gate | `_dflash2_grouped_verify_allowed` (`flash_attn_v100.py`) | `query.shape == (N, 6, 256)`, `key_cache.shape[2:] == (1, 256)` |
| C++ binding | `flash_attention_dflash2_verify_sparse_topk` / `flash_attention_grouped_verify_paged` TORCH_CHECK | `q [B*8, 6, 256]`, `k_cache [P, page, 1, 256]`, workspace `numel == batch * cap` |
| Kernel constants | `kGroupedVerifyHeads = 6`; Traits `kHeadsPerCta/kHeadGroups`; combine grid | compile-time 6 heads, 1 KV group |

Gate rejection is silent by design: TP2 today falls back to
`_flash_v100_small_query_prefill_as_decode` (correct results, the ~20 tok/s
aggregate serialization wall the concurrency doc describes).

## 2. Findings that make the adaptation small

Reading the current kernel overturns the "hard-coded" first impression:

1. **The partial kernel already has a head-group grid axis.**
   `flash_attention_grouped_verify_e5m2_partial_kernel` computes
   `head_group = blockIdx.x`, `head_start = head_group *
   Traits::kHeadsPerCta`, guards loads with `head_idx < kGroupedVerifyHeads`,
   and the epilogue composes `head_idx = head_start + local_head`. The q16
   single-request route already runs **2 head groups** (`kHeadsPerCta = 3`,
   `kHeadGroups = 2`) through this exact machinery. Multi-KV-group grids are
   a supported shape; only the *count* is pinned.

2. **The KV panel loader already takes `kv_head_idx`.**
   Both `load_xqa_tc_kv_vector` and the FP8 pair-load branch of
   `load_xqa_tc_kv_panel` compute
   `physical_offset = block*block_stride + off*token_stride +
   kv_head_idx*head_stride + panel_offset` in the runtime-stride path. The
   verify kernel passes `0` today. Passing `head_group` requires **no loader
   changes**. (`CONTIGUOUS_HKV1_LAYOUT` bakes hkv=1 into its address math,
   but it is selected by stride checks — `k_cache.stride(1) ==
   kGroupedVerifyHeadDim` — that an `(P, page, 2, 256)` TP2 cache fails
   naturally, and the env `VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED` gates
   it. The runtime-stride path is already exercised in production by the
   1728/3456 LABD pages.)

3. **Host grid dims are already parameterized.**
   `partial_grid = dim3(head_groups, grouped_splits, batch_size)`,
   `combine_grid = (query_len, kGroupedVerifyHeads, batch_size)`.

So the whole task reduces to: make the per-rank head count
`6 * kv_head_groups` instead of a constant 6, thread the KV-group index into
the KV loader address, and scale the per-request workspaces by the KV-group
count.

## 3. Design

### 3.1 Core decision: per-KV-head compact tables (Option A)

The sparse scorer ranks 32-token tiles per request against the K cache and
keeps sink+window+top-k tiles. With two KV heads per rank there are two
defensible shapes:

- **Option A (chosen): each KV head gets its own compact tile table.**
  Strictly preserves TP4 semantics — each KV group's verifier runs over the
  tiles *its own* heads scored best. Scorer K traffic is identical to any
  alternative (both heads' K must be read either way), and the verifier
  reads compact K/V per group regardless. Costs: tile_scores/compact
  workspaces scale by hkv (B×72×4 B×2 per request — negligible), and
  per-row rebasing in three kernels.
- **Option B (rejected, kept as fallback): one shared table scored by
  max-merging both KV heads.** Smaller diff (workspace shapes unchanged,
  verifier reads the same compact row for both groups) but dilutes each
  group's tile budget with the other group's targets. If validation shows
  no acceptance-rate difference, Option B can replace A later as a
  simplification — never the reverse.

### 3.2 Kernel changes (`flash_decode_paged.cu`)

**KV-group template axis.** Extend the traits and both kernels:

```cpp
template <int MAX_QUERY_TOKENS, int KV_GROUPS = 1>
struct GroupedVerifyTraits {
  static constexpr int kHeadsPerCta = kGroupedVerifyRows / MAX_QUERY_TOKENS;
  static constexpr int kHeadGroups = KV_GROUPS;   // was 6 / kHeadsPerCta
  static constexpr int kRankHeads  = KV_GROUPS * kGroupedVerifyHeads;
  static_assert(kGroupedVerifyRows ==
                    MAX_QUERY_TOKENS * kHeadsPerCta, "...");
};
```

The row budget invariant `kGroupedVerifyRows == MAX_QUERY_TOKENS *
kHeadsPerCta` is unchanged; the CTA maps
`head_idx = head_group * kHeadsPerCta + local_head` into `[0, kRankHeads)`.
For q8 (`kHeadsPerCta = 6`), `KV_GROUPS = 2` means each CTA holds one KV
group's 6 heads — 2 groups × 6 heads = the 12 per-rank heads. For q16
(`kHeadsPerCta = 3`) the existing 2 groups split *within* one KV head, so
q16 does not generalize this way — it is out of scope (§3.5).

- `flash_attention_grouped_verify_e5m2_partial_kernel`: add `KV_GROUPS`,
  guard `head_idx < Traits::kRankHeads`, pass `kv_head_idx = head_group`
  into `load_xqa_tc_kv_panel`, rebase the compact table per
  `(group_idx, head_group)`:
  `row_compact_table = compact_table + (group_idx * KV_GROUPS +
  head_group) * compact_table_row_stride`, and offset `partial_out` /
  `partial_lse` base strides by `kRankHeads` instead of
  `kGroupedVerifyHeads`.
- `flash_attention_grouped_verify_e5m2_combine_kernel`: add `KV_GROUPS`;
  `blockIdx.y` ranges over `kRankHeads`; partial indexing uses `kRankHeads`.
  Launch grid already parameterized.
- `dflash2_verify_sparse_score_kernel`: build the head-mean query per KV
  group (heads `[kvh*6, kvh*6+6)` of the 8 draft rows), load the K panel
  with `kv_head_idx = kvh`, and write
  `tile_scores[(group_idx * KV_GROUPS + kvh) * kDflash2ScoreCapTiles +
  tile]`. Pass `KV_GROUPS` (or derive from `q.size(1) / 6`) and rebase
  `block_table_row`/`seq_lens` by `row = blockIdx.z / KV_GROUPS`. The
  per-tile score stays "max over the group's 6 heads".
- `dflash2_verify_sparse_select_kernel`: grid and row indexing
  `batch * KV_GROUPS`; unchanged selection math (budgets are per table and
  stay `sink 8 + window 32 + top-k 24 <= 72` tiles).

**Composite seq_lens contract (token-table mode).** As implemented, the
token-table (sparse) verifier keys `seq_lens` per *composite*
`(request, KV head)` row — `seq_lens[group_idx * KV_GROUPS + kv_head]` in
the partial kernel and `seq_lens[request * KV_GROUPS + kv_head_of(head)]`
in the combine — because each per-rank KV head compacts its own tile set,
so the covered token count (and therefore `active_splits`) can differ
across the KV heads of one request. Dense (block-table) runs keep the
per-request `seq_lens` both KV heads share. The host check is
`seq_lens.size(0) == batch_size * kv_head_groups` when `token_table`,
`batch_size` otherwise; the sparse wrapper already passes its
`[batch * kv_heads]` `compact_len` tensor, so the Python side needs no
change. The combine learns the mode through a new trailing
`composite_seq_rows` argument (`kv_head_groups` when `token_table &&
!wide_query`, else 0); the E4M3 FP32 combine call site relies on the
default and is untouched.

**Dispatch shape (as implemented).** Template args must be compile-time
constants, so `DISPATCH_GROUPED_VERIFY_PARTIAL` splits into two literal
ladders — `..._HKV1` (the original eight branches) and `..._HKV2`
(`token_table` plus runtime/`1648`/`3296` paths, all with literal
`KV_GROUPS=2`, staged-page-id variants excluded) — selected by a runtime
`if (kv_head_groups == 2)`.

**Instantiation budget.** The `DISPATCH_GROUPED_VERIFY_PARTIAL` macro
already multiplies variants; because the shared macro body routes on
`kv_head_groups`, the Q16 dispatch sites also expand the KV2 ladder, so
the binary carries 18 `KV_GROUPS=2` partial instantiations
(`{Q8×{two_pass∈{t,f}}×{single_query∈{t,f}} + Q16×{two_pass}} ×
PAGE∈{0,1648,3296}`; Q16×KV2 is runtime-rejected but its templates still
compile, and staged-page-id variants stay KV1-only). Measured: 47 KV1 +
18 KV2 partial symbols. Compile-time and binary impact is bounded and
measurable.

**Host dispatcher (`flash_attention_grouped_verify_paged`).**
- TORCH_CHECK: `hkv = k_cache.size(2)`; accept `hkv ∈ {1, 2}` with
  `q.size(1) == kGroupedVerifyHeads * hkv` (replaces `size(2) == 1` and the
  literal `6` checks). Same for the sparse scorer binding.
- `head_groups = kGroupedVerifyHeads / heads_per_cta` becomes
  `hkv * (kGroupedVerifyHeads / heads_per_cta)` for q8.
- Dispatch: `hkv == 2` routes to the `KV_GROUPS=2` instantiations with
  `CONTIGUOUS_LAYOUT` forced false (stride checks already reject it; make it
  explicit so the invariant is load-bearing, not incidental).
- Workspaces: `partial_out` numel checks scale the heads dim by `kRankHeads`;
  scorer binding checks `tile_scores.numel() == batch * hkv *
  kDflash2ScoreCapTiles`, `compact_pages.numel() == batch * hkv *
  kDflash2CompactMaxPages`.
- **New ABI getter** `flash_attention_grouped_verify_kv_heads_abi_version()`
  returning 2, so a stale extension on TP2 degrades the same way the sparse
  entry's `.available` flag does today: gate off, logged, smallq fallback —
  never a crash.

### 3.3 Python changes

- **Gate** `_dflash2_grouped_verify_allowed`: replace
  `tuple(query.shape) == (num_query_tokens, 6, 256)` with
  `(num_query_tokens, H, 256)` where `H = 6 * key_cache.shape[2]`, and
  `key_cache.shape[2:] == (1, 256)` with `key_cache.shape[2] in (1, 2)`.
  Everything else (E5M2-only, page list, request-major q8 shapes, strides)
  unchanged. `_dflash2_sparse_verify_allowed` needs no shape edits (it
  delegates), only the ABI getter check when `hkv == 2`.
- **Guarded rollout**: new env
  `VLLM_FLASH_V100_DFLASH2_VERIFY_MAX_KV_HEADS`, default `"1"` (today's
  behavior). `"2"` admits the TP2 routes. This is the kill switch and the
  staged-rollout lever; flip the default to `"2"` only after §5 validation,
  mirroring how `VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY` graduated.
- **Workspace** `_get_grouped_verify_workspace`
  (`flash_attn_interface.py`): the `6` in `partial_out`/`partial_lse`
  shapes becomes `q.shape[1]` (`(B, 80, 8, H, 256)` / `(B, 80, 8, H)`);
  cache key gains `H`. TP1 shapes are bit-identical.
- **Sparse wrapper** `flash_attn_dflash2_verify_sparse_paged`:
  `_get_dflash2_sparse_workspace(q, (q.shape[0] // 8) * hkv)` with
  `hkv = k_cache.shape[2]`; E5M2-only check unchanged.
- **Log line**: the one-shot activation log prints
  `request-major B%d/q8/H%d/D256` with the real head count, so TP2
  admission is visible in capture logs (`H12`).

### 3.4 What does not change

- The loader internals (both FP8 pair-load and vector paths already honor
  `kv_head_idx`), page-layout specializations, and the compact
  `block_size < 0` token-table resolution.
- E5M2-only KV contract; top-k/sink/window budgets; `num_reqs ∈ {1,2,4,8}`
  request-major shapes; CUDA-graph capture stability (all capture-time
  values are shape-derived, never live `seq_lens`).
- `b4a48a464` (drafter YaRN) and `b72f21b84` (AWQ warmup) — already
  topology-agnostic; no changes.
- Upstream main's E4M3 XQA family and `is_dflash_selector_target` handling.

### 3.5 Explicit non-goals

- **q16 dense verify at TP2.** q16's `kHeadsPerCta = 3` splits *within* one
  KV head's 6 heads; a TP2 q16 route needs 96-row CTAs (smem budget blown)
  or a restructured tiling. Conc-1 long-context perf rides the sparse q8
  route (`num_reqs=1`, `num_query_tokens=8`), so dense q16 stays TP4-only.
- E4M3 KV for the DFlash2 verify family.
- TP3/TP6 (4 KV heads do not divide); TP8 (q_per_kv stays 6 but the
  instantiation axis generalizes to `KV_GROUPS ∈ {1,2,4}` later if asked).

## 4. Capacity reality check (TP2)

Per rank, per token, per full-attention layer: 2 KV heads × 256 × 1 B ×
2(K+V) = **1 KB** (vs 512 B at TP4). Weights per rank double. The TP4
858 K-token pool therefore lands at roughly **350–430 K tokens on 2×32 GB**,
and the per-step KV re-read cost per row doubles (16 layers × 1 KB). The
sparse route remains the only path that avoids the concurrency
serialization wall, but the scheduler ceiling moves:

| Concurrent 200 K reqs | Tokens | 858 K (TP4) | ~400 K (TP2 est.) |
|---|---:|---:|---:|
| 1 | 200 K | fits | fits |
| 2 | 400 K | fits | **marginal** |
| 4 | 800 K | fits | no |

At 32 K context, B8 (256 K tokens) still fits. Validation must measure the
actual pool before promising a concurrency target (§5).

Drafter placement: `draft_tensor_parallel_size`
(`vllm/config/speculative.py`) must divide the target's TP; default (follow
target) gives TP2 — confirm drafter head divisibility at TP2 in validation.

## 5. Test and validation plan

1. **Unit numerics** (`tests/kernels/attention/test_sm70_dflash2_sparse_verify.py`):
   parametrize existing cases over `(H, hkv) ∈ {(6,1), (12,2)}` ×
   `B ∈ {1,2,4}`; compare against a dense PyTorch reference on synthetic
   E5M2 caches (existing tolerance); keep the short-sequence degenerate case
   (every tile marked ⇒ equals dense); add a case where the two KV heads'
   top tiles differ, asserting per-group tables diverge (Option A invariant).
2. **Gate policy** (`tests/v1/attention/test_sm70_flash_v100_policy.py`):
   H12/Hkv2 accepted with env `=2`, rejected (falls through to smallq, no
   exception) with default `=1`; stale-extension path (ABI getter absent)
   rejected; TP1 shapes unaffected by the refactor.
3. **Graph capture**: capture/replay the B2/B4 verify step under
   `torch.cuda.CUDAGraph` with H12 shapes; assert no host syncs and stable
   route (route-summary counters).
4. **Strides probe** (first TP2 run): log `k_cache.stride()` /
   `v_cache.stride()` / page size at TP2, confirm the fixed-interleaved
   layout rejects and the runtime-stride path engages (expected, since the
   LABD pages already run it).
5. **Benchmark matrix** (mirrors the concurrency doc): conc {1,2,4} ×
   {32K,128K,200K}, sparse vs `VLLM_FLASH_V100_DFLASH2_SPARSE_TOPK=0`
   (dense-verify-if-eligible) vs env-off (smallq baseline). Acceptance-rate
   A/B for the sparse route at TP2 (per the YaRN doc's quality protocol).
   Record in a new section of the concurrency control log.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Occupancy: partial CTA count doubles (grid `(2,80,B)`), ~53 KB smem ⇒ 1 CTA/SM ⇒ 2 waves vs 1 at TP4 | Measured in §5.5; follow-up (not this change): make `kGroupedVerifyQ8Splits` runtime to re-tune splits per head-group count — requires the Python workspace splits dim on both sides, so deliberately deferred |
| TP2 allocator produces unexpected strides/pages | Runtime-stride path is layout-general and already load-bearing (LABD pages); §5.4 probe; gate falls back cleanly |
| Acceptance-rate drift | Option A preserves per-head semantics; A/B in §5.5; Option B exists as simplification if parity is trivially met |
| Stale extension without KV-group support | ABI getter; gate off + one-shot log, today's fallback behavior |
| Compile time / binary size | 6 bounded instantiations; measure in CI |
| KV pool smaller than expected kills concurrency targets | §4 table is an estimate; measure pool first, set `--max-num-seqs` accordingly |

## 7. Staging

| Stage | Content | Exit criterion |
|---|---|---|
| S1 | Kernel: `KV_GROUPS` template axis, scorer/select per-kvh rows, host TORCH_CHECK + dispatch + ABI getter | Existing TP4 tests green, bit-identical (`KV_GROUPS=1` is the same math) |
| S2 | Python: gate relaxation + `VLLM_FLASH_V100_DFLASH2_VERIFY_MAX_KV_HEADS`, workspace heads param, sparse wrapper hkv, log line | Policy tests green; TP1 default behavior unchanged |
| S3 | Tests: §5.1–5.3 | Green on TP4 hardware (H12 shapes are synthetic; SM70-only kernel checks unchanged) |
| S4 | TP2 validation run: §5.4–5.5, capacity measurement, flip env default if clean | Route admission logged at capture, perf table + acceptance A/B recorded in the control log |

Rollback at any stage: `VLLM_FLASH_V100_DFLASH2_VERIFY_MAX_KV_HEADS=1`
(then remove the env entirely if S4 is aborted).

## 8. S4 validation results (2026-09-18, image v1.5.0-dgkfa-p3)

Hardware: 2×V100 32GB on GPUs 2+3 (both PCIe x16, NUMA1, one PHB leg;
GPU0 negotiates x8 and was excluded), `cpuset 14-27,42-55`,
`NCCL_P2P_DISABLE=1`. Config: production compose minus topology —
TP2, `--max-model-len 204800` (user-directed after the 512K pool check),
`--max-num-seqs 2`, YaRN 512K hf-overrides kept for rope parity,
fp8_e5m2 KV, DFlash2 probabilistic.

**Measured KV pool: 209,391 tokens** (vs 858 K at TP4). 200 K conc1 fits
with 1.02× headroom; conc2 tops out at ~96 K/req; conc4×32 K (128 K) fits.

**Route admission** (both ranks, capture time):

- sparse: `DFlash2 sparse verifier active (request-major B2/q8/H12/D256,
  fp8_e5m2 KV, topk=768 window=1024 sink=256)`
- dense (`SPARSE_TOPK=0`): `DFlash2 exact grouped verifier active
  (request-major B2/q8/H12/Hkv2/D256, fp8_e5m2 KV, one-pass)`
- baseline (env removed): `DFlash2 grouped verifier gate rejected …
  q=(16,12,256) k=(228,3296,2,256)` on both ranks → smallq fallback, no
  crash — the kill-switch contract holds.

**Same-prompt trio** (conc1, greedy `temperature=0`, ctx 194,993 tok,
~140 output chunks, salt-3 prompt; prefix cache empty after each recreate
so TTFT is comparable; decode rate = chunks/(wall−TTFT)):

| Leg | Route | TTFT | Decode | Acceptance |
|---|---|---:|---:|---|
| A3 sparse | hkv2 compact topk=768 | 248.4 s | **22.8 tok/s** | 35.7% (365/1022) |
| B dense | hkv2 exact one-pass | 245.7 s | 15.3 tok/s | 38.0% (372/980) |
| C baseline | gate off → smallq | 250.7 s | 4.9 tok/s | 39.1% (375/959) |

(A2, sparse on a different salt-3-free prompt: TTFT 255.3 s, 33.4% —
consistent.)

Reading: sparse is **4.6× baseline decode** (dense 3.1×) at 200 K; the
smallq serialization wall (KV re-read per draft row) reproduces at TP2
exactly as the TP4 concurrency doc describes. Acceptance ordering
baseline > dense > sparse is the expected exact-vs-approximation split;
the sparse cost is 2.3 pp absolute (~6% relative) against dense for +49%
decode. TTFT spread 245–251 s confirms prefill is route-independent.

**Not run** (production downtime window closed the session): conc2×96 K
and conc4×32 K legs, and the conc-matrix stagger fix (backlog P2) that
would make concurrent decode numbers trustworthy. TP2 at 200 K is a
latency play, not a throughput play — the pool backs one long request.
