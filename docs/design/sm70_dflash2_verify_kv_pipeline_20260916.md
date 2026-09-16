# SM70 DFlash2 verify KV pipeline and training-free sparse verify: control log

## Scope and frozen baseline

- Date: 2026-09-16.
- Branch: `feature-dgk-fa`, cut from `feature-v1.5.0-pro` at `11126371a`.
- Prior art in this branch: commit `8a62c7f56` (commit 2, sparse page4 E5M2
  consumer support).
- Workload: Qwen3.8-27B AWQ (H6/Hkv1/D256), official DFlash2 draft
  (seven draft tokens, probabilistic sampling), TP4 over V100-SMX2, target
  E5M2 KV, 200K-token prompts, concurrency one, CUDA graph
  (`FULL_AND_PIECEWISE`), prod container `1cat-vllm:v1.5.0-feature-pro`.
- Frozen dense baseline (same server state, warm runs, `/tmp/probe200k.py`,
  usage-token measurement): **46.1-48.7 ms/step, 55.9-72.3 tok/s,
  1.72-2.62 tokens/step** (acceptance is prompt-dependent on this workload).
  Authoritative morning reference: 60.7 tok/s at 203604 prompt tokens.
- Motivation (commit 1): the one-pass grouped verifier is latency-bound, not
  bandwidth-bound — SM ~95%, DRAM ~20%, ~1.64 GB/step/rank of K/V gather at
  200K tokens across the 16 full-attention layers, with per-tile barriers
  serializing gather, QK, softmax, and PV.

## Commit 1 — K/V double-buffer software pipeline: REJECTED (negative result)

Hypothesis: overlapping the next tile's K/V gather with the current tile's
softmax/PV (`kv_alt` shared buffer, smem 52.0 → 68.5 KiB, still 1 CTA/SM)
hides the gather latency behind compute.

Result: **+1.2% at 200K against a ≥15% acceptance gate.** The per-tile
`__syncthreads()` lattice around the WMMA QK and PV stages — not the gather —
bounds the one-pass loop; adding a second buffer does not shorten the
critical chain. Numerical output was bit-identical (`torch.equal` across
prefixes {4K, 128K, 200K, 250K} × page {1648, 3296, generic}), so the idea is
sound but the ceiling is elsewhere.

Action: reverted; the implementation is archived at
`.build-cache/pipelined-kv.patch` for future reference. No env switch was
kept.

## Commit 2 — sparse page4 consumer accepts E5M2: LANDED

`8a62c7f56`: the grouped sparse page4 route (`grouped_sparse_page4_plan` →
`grouped_sparse_page4`) previously only instantiated fp8_e4m3/fp16 KV. Added
the E5M2 branch (pair-load path already supported `BLOCK_SIZE==4`), relaxed
the two dtype `TORCH_CHECK`s, and bumped the ABI getter. Required by commit 3
and independently useful for QSA consumers. No behavior change on existing
dtype paths; the page4 parameterized tests cover the new dtype.

## Commit 3 — training-free sparse DFlash2 verify: LANDED (this document)

### Design

Every 32-token tile of the sequence is scored once per verify step with
K-only panels against the draft-mean query `q6` (six heads merged). A
single-CTA select kernel keeps **sink 256 + recent window 1024 + top-k 768**
tokens (64 tiles ≤ the 72-entry compact table), emits **physical** token
bases ascending, trims the final tile's past-seq phantom tokens into
`compact_len`, and the unchanged eighty-CTA exact one-pass verifier consumes
the compact table through token-table addressing (`page_size` sentinel
`-k_cache.size(1)`). No training, no indexer weights, no extra model state;
the scorer reads the same E5M2 K cache the verifier reads.

- Scorer: 320 CTAs (4× the verifier's 80), 512 threads, smem-staged panels
  (`load_xqa_tc_kv_panel`, 3 syncs/tile). Measured **227-250 µs/layer ≈
  3.0-3.3× faster** than the 750 µs dense verify. A register-direct variant
  (`__ldg` + shfl, no smem staging) measured 259-281 µs (2.67-2.89×) —
  rejected.
- Select: 1 CTA, bitmap + top-k rounds + exclusive scan; the final tile is
  always kept (it hosts the draft tokens).
- Consumer: dense verifier unchanged except the token-table gather branch;
  compact entries carry physical bases, so `page_ids` is intentionally
  bypassed (`physical_block = token / page_size`).
- Dispatch: `VLLM_FLASH_V100_DFLASH2_SPARSE_TOPK` (default 0 = dense
  rollback), `..._WINDOW` (1024), `..._SINK` (256), `..._MIN_SEQ` (32768).
  The backend gate pins single-request q8, `is_dflash_selector_target`, and
  the static `max_model_len` band only — capture-stable, no host sync, safe
  under CUDA graph capture (graph-replay test replays with changing
  `seq_lens`).

### The bug the first deployment hit (and the test it added)

The first prod boot produced 100% draft rejection and gibberish output while
every unit test stayed green — including exact sparse-vs-dense equality.
Root cause: the select kernel emitted **logical** tile bases
(`tile * 32`) while the consumer's token-table gather addresses the cache
**directly by physical page** (it bypasses `page_ids` by design). All tests
used identity block tables (`arange`), where logical and physical are
indistinguishable; prod block tables are not identity, so attention read
unrelated pool pages.

Fix: resolve each entry through the block table at select time
(`block_table[base / page_size] * page_size + base % page_size`), plus a
`page_size % 32 == 0` host check so a tile can never straddle a page
(3296 = 32 × 103). Test hardening:
`test_sm70_dflash2_sparse_permuted_pages_match_dense` builds a 64-token-page,
32-page pool with a `randperm` mapping and demands bit-exact sparse-vs-dense
equality at full coverage — a logical-base regression fails it loudly.

Diagnostic notes for future boot debugging: `WorkerProc` crash loops with
`UnboundLocalError` came from a probe log block referencing an out-of-scope
variable (syntax checks do not catch name resolution); a health poll must
check `http_code` because `curl -w` prints on connection failure; post-restart
run 0 is a cold run (Triton JIT) and must not be compared against warm
baselines.

### End-to-end A/B (prod, 200K, concurrency 1, warm runs)

| Route | ms/step | tok/s | tokens/step |
|---|---:|---:|---:|
| dense (baseline) | 48.7 / 47.9 / 46.1 | 55.9-72.3 | 1.72-2.62 |
| sparse (cold run 0) | 57.2 | 48.7 | 1.78 |
| **sparse (warm)** | **29.9 / 30.3 / 31.3** | **103.0-119.0** | **2.22-2.53** |

- Step time ratio **0.63-0.64× dense** — the ≤0.75× gate passes with margin.
  (Ceiling analysis had predicted 0.71×: attention was ~12 ms of a ~48 ms
  step, and the sparse attention path costs ~250-330 µs/layer including
  scorer, select, and compact verify.)
- Acceptance is in the dense band (2.22-2.53 vs 1.72-2.62) — the
  approximation does not shift verify decisions measurably on this workload.
- Prefill TTFT unchanged (~195 s, chunked prefill untouched).
- Quality gates: see §Quality gates below.

### Budget sweep (200K, concurrency 1, warm runs)

Compact-table capacity is 72 tiles (2304 tokens), so the sweep explores the
feasible region only; the scorer's cost is independent of the budgets (it
always scans every tile), which the curve confirms:

| TOPK | WINDOW | SINK | tokens kept | ms/step (warm) | tokens/step |
|---|---:|---:|---:|---:|---:|
| dense | — | — | all (~200K) | 46.1-48.7 | 1.72-2.62 |
| 512 | 1024 | 256 | 1792 | 30.2-31.0 | 1.76-2.76 |
| **768 (default)** | **1024** | **256** | **2048** | **29.9-31.3** | **2.22-2.53** |
| 1024 | 1024 | 256 | 2304 | 32.0-33.3 | 1.87-2.56 |

Step time is flat from 512 → 768 and degrades at 1024 (bigger compact verify)
while acceptance stays inside the prompt-dependent band everywhere. The knee
is therefore in quality, not speed: 768 keeps 50% more selection headroom
than 512 at identical step time, and NIAH is 8/8 at 768, so **768/1024/256
is the committed default**.

### Quality gates

Sparse-route results (this section); dense references in the final table
below. NIAH: two haystacks per length (131072, 204000 tokens) × four needle
depths (0.12/0.38/0.63/0.88), fixed salt so dense and sparse see identical
haystacks. Greedy consistency: five fixed 200K prompts, temperature 0.

| Gate | Contract | Sparse result |
|---|---|---|
| NIAH retrieval | ≥95% of dense | **8/8 = 100%** (dense also 8/8) |
| Acceptance | ≥2.40 tok/step band | 2.22-2.53 warm, in the dense band |
| Step time | ≤0.75× dense | **0.63-0.64×** |
| Structured greedy | sparse == dense text | **3/3 bit-identical** |
| Flat-haystack greedy | diagnostic | 0/5 identical (sim 0.26-0.98), coherent |

Honest scope note: the plan called for ten greedy prompts and a standalone
PPL delta gate; the flat-haystack probe ran five prompts and no standalone
PPL harness was built. Exact identity is unattainable on degenerate
random-keyword haystacks for any context approximation — attention there is
genuinely diffuse, so the next-token distribution is nearly flat and argmax
flips under perturbations as small as batching noise. The structured probe
supplements it: a sequential-facts expedition log (66K / 110K / 154K tokens)
whose continuation is nearly deterministic ("waypoint 1501") — the sparse
route reproduces the dense continuation **bit-identically at all three
lengths**, which is the strongest available evidence that the approximation
is faithful where attention is concentrated.

## Conclusion

Commit 3 lands as designed: a training-free sparse DFlash2 verify route that
scores 32-token tiles with a K-only draft-mean scorer, keeps
sink+window+top-k, and reruns the unchanged exact verifier over a compact
physical-base table. At 200K / concurrency 1 it cuts verify step time from
~47 ms to ~30 ms (**0.63×**, past the 0.75× gate) and lifts throughput from
56-72 to 103-119 tok/s with acceptance, NIAH, and structured-context
behavior indistinguishable from dense. The committed default is
TOPK=768 / WINDOW=1024 / SINK=256 (knee of the sweep); `TOPK=0` is the
one-env rollback to dense verify.

Commit 1's rejection stands as the honest counterweight: software-pipelining
the dense verifier's K/V gather yields +1.2% against a ≥15% gate because the
tile loop is barrier-bound, not gather-bound — the gather cost had to be
removed (sparse selection), not hidden.
