# SM70 DFlash2 concurrency optimization: control log

Standing control log for decode/prefill optimization at concurrency >= 2 on
the DFlash2 target-model verify path (SM70, TP4). Sister document to
`sm70_dflash2_verify_kv_pipeline_20260916.md`, which covers the
single-request (concurrency 1) sparse verify route. Every concurrency-related
change lands here with its measurement, gate, and rollback.

## Scope and frozen reference

- Dates: analysis and P0 measured 2026-09-16/17.
- Branch: `feature-dgk-fa` (image `1cat-vllm:v1.5.0-feature-dgk-fa`,
  commits `8a62c7f56` + `55fe8995c`). **P0 is env-only; no code change.**
- Workload: Qwen3.8-27B AWQ + official DFlash2 draft (7 tokens,
  probabilistic sampling), TP4 over V100-SMX2, E5M2 KV,
  `--max-model-len 262144`, `--max-num-seqs 4`,
  `--max-num-batched-tokens 32768`, chunked prefill, prefix caching.
- Reference benchmark: `/home/bso/benchmark_vllm.py`
  (`BENCH_CONC`/`BENCH_PP`/`BENCH_OUT` subsets), usage-token measurement,
  per-request decode window = e2e - TTFT.
- Baseline (main image `1cat/vllm:sm70-v1.5.0-cuda12.8`, 2026-09-16 morning,
  `1Cat-vLLM.xlsx` == `1Cat-vLLM-tp4-main.xlsx`): conc1 200K decode
  60.7 tok/s; per-request decode tok/s:

| Concurrency | 32K | 128K | 200K |
|---|---:|---:|---:|
| 2 | 32.4* | 16.4 | 10.4 |
| 4 | 25.1* | 10.7 | 5.0 |

  \* the 32K-conc2 numbers are polluted by a benchmark artifact (see
  Measurement pitfalls); 128K/200K are clean.
- The phenomenon to explain: **aggregate decode throughput pins at
  ~20 tok/s for any concurrency >= 2 at long prompt lengths**
  (conc1 60.7, conc2 2 x 10.4 = 20.8, conc4 4 x 5.0 = 20.0) — decode is
  fully serialized by something that concurrency should have amortized.

## Path analysis — why concurrency >= 2 was slow

The verify call site (`_flash_v100_small_query_prefill_as_decode` in
`vllm/v1/attention/backends/flash_attn_v100.py`) evaluates three fast-route
gates before falling back:

| Route | Gate | conc2/4 without P0 |
|---|---|---|
| sparse verify (commit 3) | `num_reqs == 1 and num_query_tokens == 8` | rejected |
| batched dense grouped verify | `VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY` (default **False**) | rejected |
| single dense grouped verify | `num_reqs == 1` | rejected |

Every verify step at concurrency >= 2 therefore landed in the **smallq
fallback**, whose design treats *each draft row as an independent decode row
re-reading the full paged KV* — 16 rows at conc2, 32 rows at conc4. Cost
model (16 full-attention layers x 512 B/token, effective bandwidth
~110-130 GB/s):

```
t_step ≈ T_fixed(~22 ms) + rows × seq × 512 B × 16 / BW_eff

conc2 128K: 16 rows × 1.07 GB × 16 = 17.2 GB → ~156 ms  (measured ~154 ms)
conc2 200K: 16 rows × 1.67 GB × 16 = 26.7 GB → ~243 ms  (measured ~237 ms)
conc4 200K: 32 rows × 1.67 GB × 16 = 53.4 GB → ~500 ms  (baseline ~516 ms)
```

The model matches measurement, which pins the bottleneck: KV re-read
amplification proportional to `num_reqs × 8` rows. Prefill is NOT affected:
aggregate prefill throughput is flat across concurrency (conc1 ~955,
conc2 ~896, conc4 ~994 tok/s); TTFT growth with concurrency is chunk
serialization (queueing), not waste.

## P0 — request-major batched grouped verify (env-only): LANDED (2026-09-17)

### Lever

`VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY=1` (compose env; rollback =
remove the line and recreate). The whole chain already existed in code:

- Native side: request-major kernel contract is `query_len == 8` per request
  for any batch (`flash_decode_paged.cu:4814-4816`); B2 (q16) and B4 (q32)
  satisfy it; workspace is batch-aware
  (`_get_grouped_verify_workspace(q, batch)`).
- Python gate: `batched_request_shape` in `_dflash2_grouped_verify_allowed`
  (`num_reqs in (2, 4, 8)`, `max_query_len == 8`,
  `num_query_tokens == num_reqs * 8`, request-major ABI >= 1 — the extension
  getter returns 1).
- Call site: passes full `block_table[:num_reqs]` / `seq_lens[:num_reqs]`.
- KV reads drop from `num_reqs × 8` row-scans to **one** scan per step.

Concurrency 1 is untouched: `single_request_shape` matches first and the
sparse route (commit 3, 103-119 tok/s at 200K) is unaffected.

### Routing evidence (and two log pitfalls)

- Capture-time one-shot log confirms the batched kernel is captured:
  `DFlash2 exact grouped verifier active (request-major B4/q8/H6/Hkv1/D256,
  fp8_e5m2 KV, one-pass)`.
- Pitfall 1: the one-shot `gate rejected` log burned at capture time on a
  **B3 dummy shape** (`num_reqs in (2,4,8)` excludes 3). A `reqs=3`
  rejection in the log is a capture artifact, not a runtime diagnosis; and
  **concurrency 3 genuinely falls back to smallq** (acceptable while
  max_num_seqs = 4; see backlog P3).
- Pitfall 2: all one-shot activation logs print during graph capture, seconds
  after boot — a `docker logs --since <window>` grep misses them. Grep the
  full log.

### End-to-end A/B (warm, usage-token measurement)

"Before" for conc2 was re-measured on this branch's image (matches the
main-image baseline — the sparse commit does not engage at conc >= 2);
conc4 "before" stands on the main-image baseline (the branch conc4 run was
interrupted; sparse is gate-off at conc >= 2 so the images are equivalent
there).

| Case | Before (smallq) | **After (batched)** | Gain |
|---|---:|---:|---:|
| conc2 128K | 16.8 | **34.1** | 2.03x |
| conc2 200K | 10.9 | **29.9** | 2.74x |
| conc4 128K | 10.7 | **29.0** | 2.71x |
| conc4 200K | 5.0 | **21.8** | 4.36x |

(prefill speed tok/s, per request, before → after: 555.9 → 555.9-class at
every point; 128K conc2 549.8 → 555.9, 200K conc2 505.6 → 510.5, 128K conc4
272.3 → 271.9, 200K conc4 248.5 → 249.6 — prefill untouched.)

Raw: `/tmp/bench_batched_ab.log`, copy at `/home/bso/1Cat-vLLM-batched-p0.xlsx`.

The gain grows with concurrency and length, as the mechanism predicts:
smallq cost scales with `num_reqs × 8` row-scans while the batched kernel
reads KV once.

### Quality gates

- Semantic: two structured sequential-facts probes (64K + 108K tokens) fired
  **concurrently** at temperature 0 both continue with the correct next
  waypoint under the batched route.
- Acceptance: per-position accepted-token curve decays smoothly
  (949/643/412/268/183/120/80); aggregate 2.045 tok/step over a mixed
  short-probe + long-bench boot is not directly comparable to the
  prompt-dependent 1.72-2.62 dense band and shows no pathology. The verify
  is exact dense math (request-major packing changes scheduling, not the
  per-row computation), so no approximation gate is required.
- Gap vs the theoretical ceiling (est. ~37 ms steps at conc2 200K vs
  ~86 ms implied): the fixed pipeline (draft propose x N, sampling,
  scheduling) scales with batch, and decode windows still contain some
  TTFT-stagger pollution. The measured 2.0-4.4x is the honest end-to-end
  number.

## Measurement pitfalls (standing notes for this log)

1. **32K-conc2 artifact**: a 32K prompt fits in exactly one
   `--max-num-batched-tokens` chunk, so the two requests' prefills serialize
   into two ~30 s steps; the first request's decode window (e2e - TTFT_mean)
   swallows the second request's entire prefill. 32K-conc2 decode numbers
   are meaningless until fixed (backlog P2). 128K+ dilutes the stagger and
   is comparable.
2. **One-shot logs burn at capture**: see P0 routing evidence.
3. **KV capacity ceiling**: pool = 878,233 tokens → conc4 x 256K
   (1.05 M) is physically infeasible; conc2 x 256K (512 K) fits.

## P1 — multi-request (batched) sparse verify: LANDED (2026-09-17)

Generalizes the single-request sparse verifier (commit 3, B1-only) to the
request-major batched route: every request row now scores, selects, and
verifies against its own compact table inside one B2/B4 step, so the
conc >= 2 decode path reads `sink+window+topk` tiles per request instead of
the full paged KV (P0) — and instead of `num_reqs x 8` full row-scans
(smallq).

### What changed (single commit on `feature-dgk-fa`)

- `flash-attention-v100/kernel/flash_decode_paged.cu`
  - Score kernel: `blockIdx.z` selects the request row; q is rebased per row
    (`[batch*8, 6, 256]`), the block table and the `[batch*8192]` score
    buffer are row-strided; grid becomes `(1, 320, batch)`.
  - Select kernel: one CTA per request row; per-row `seq_lens`, score
    window, `[batch, 72]` compact table, and `[batch]` compact length;
    launch `<<<batch, 256>>>`.
  - Consumer verify kernel: signature gains a trailing
    `compact_table_row_stride`; the token-table loader receives
    `row_compact_table = compact_table + group_idx * stride` at all three
    panel-loader call sites; the launch macro and both direct launch sites
    thread the stride (`compact_table.stride(0)`); token-table host check
    now requires `compact_table [batch, max_compact]`.
  - `flash_attention_dflash2_verify_sparse_topk`: `batch = block_table.size(0)`;
    shape checks scale with batch. **Op signature unchanged — no ABI bump.**
- `flash_attn_v100/flash_attn_interface.py`: sparse workspace is keyed by
  `(device, stream, batch)` and sized `[batch*8192] / [batch, 72] / [batch]`;
  wrapper derives `batch = q.shape[0] // 8`. B1 allocates identical shapes,
  so the single-request path is bitwise unchanged.
- `vllm/v1/attention/backends/flash_attn_v100.py`:
  `_dflash2_sparse_verify_allowed` accepts `num_reqs in (1, 2, 4, 8)` with
  `num_query_tokens == num_reqs * 8` (capture-stable as before — the dense
  gate it delegates to still pins `max_query_len == 8` for batches); the
  call site passes the full gate-pinned `block_table`/`seq_lens`; the
  one-shot log now carries the batch (`request-major B4/q8/...`).
- `tests/kernels/attention/test_sm70_dflash2_sparse_verify.py`: four new
  batched test functions (five cases: B2/B4 covered-budget bitwise vs
  batched dense, per-row permuted page tables bitwise, mixed per-row
  lengths with per-row compact_len + fp32 reference, B2 graph replay). The
  six B1 tests are untouched and pass unchanged — B1 is bitwise identical
  by construction (row-0 offsets are zero).

### End-to-end A/B (warm, usage-token measurement, same protocol as P0)

| Case | P0 dense batched | **P1 batched sparse** | Gain | vs main baseline |
|---|---:|---:|---:|---:|
| conc2 128K | 34.1 | **50.3** | 1.47x | 2.1x (16.4) |
| conc2 200K | 29.9 | **41.0** | 1.37x | 3.9x (10.4) |
| conc4 128K | 29.0 | **38.6** | 1.33x | 3.6x (10.7) |
| conc4 200K | 21.8 | **43.0** | 1.97x | 8.6x (5.0) |

(decode tok/s per request; prefill unchanged at every point: 556/511/273/252
vs P0's 556/511/272/250.) All four cases clear the >=15% adoption bar; the
stretch targets (conc2 200K >= 45, conc4 200K >= 35) are 1/2 met — conc2
200K landed at 41.0 (+37%), conc4 200K overshot to 43.0 (+97%). Per-request
decode is now nearly flat from conc2 to conc4 at 200K (41.0 vs 43.0), which
is what the mechanism predicts: the scorer reads the K cache once per
32-token tile and the verifier reads only the selected ~2048 tokens per
request, so batch growth no longer multiplies KV traffic.

Raw: `/tmp/bench_p1_ab.log`, copy at `/home/bso/1Cat-vLLM-batched-p1.xlsx`.

### B1 regression gate

Fresh-boot 200K single-request probe: 30.7 ms/step, 122.7 tok/s (band:
~30 ms, 103-119+ tok/s), smooth acceptance decay. Sparse B1 is bitwise
identical by construction and the unit tests pin it.

### Quality gates (concurrency, all passed)

- Concurrent NIAH (B2): one 108K + one 168K request in flight together,
  4 needles each at depths 0.12/0.38/0.63/0.88 — sparse 8/8 (100%),
  dense (TOPK=0, P0 route) 8/8; identical per-needle hits.
- Concurrent structured greedy (B2): ~66K + ~110K sequential-fact logs fired
  together, temperature 0 — both continue at exactly waypoint 1501/2501 on
  sparse AND dense; identical walls (93.8/105.1 s sparse vs 93.6/105.1 s
  dense).
- Acceptance (mixed boot: B1 probes + B2 bench + concurrent gates):
  890/632/412/263/182/130/82 per position, aggregate 2.19 tok/step — smooth
  monotone decay, no pathology.

### Rollout and rollback

- Image: `1cat-vllm:v1.5.0-dgkfa-p1` (commit of the verified container
  state; the P0 tag `1cat-vllm:v1.5.0-feature-dgk-fa` is preserved).
  Compose `docker-compose-v150-pro-dflash.yaml` points at the P1 tag.
- Runtime rollback: `VLLM_FLASH_V100_DFLASH2_SPARSE_TOPK=0` disables sparse
  for B1 AND batches (falls back to P0 dense batched, quality-neutral per
  the gates above). Image rollback: switch the compose tag back.
- The B3 (concurrency 3) capture-dummy still rejects and conc3 still falls
  back to smallq (backlog P3 unchanged).

### New pitfalls recorded

1. **First decode request after boot**: the first probe on a fresh recreate
   showed 221 ms/step; the second on the same boot showed 30.7 ms. Always
   discard the first decode probe after boot (or warm with a short request)
   before measuring.
2. **Serving containers carry TWO copies of the extension**: the dist-packages
   root AND `flash_attn_v100/` package dir (the package dir is what
   `from . import flash_attn_v100_cuda` loads). Copying only the root copy
   boots fine but crashes workers with the OLD kernel's checks — update both,
   or use the committed image which has them identical (md5-verified).
3. **Incremental builds after a failed compile can silently skip the
   extension** (distutils `newer_group` / ninja regeneration races): the
   relinked .so was 8 s newer than the touched source, so `build_ext`
   skipped the whole flash extension. `--force` (or `rm -rf build`) plus a
   `strings <so> | grep <new symbol>` check after every build.

## Backlog

### P2 — benchmark stagger fix

Fix the decode-window artifact for short-prompt concurrent cases: stagger
request launches, or derive decode speed from server-side step logs.
Required before trusting any 32K-class concurrent number.

### P3 — batched gate shape coverage

`num_reqs in (2, 4, 8)` excludes 3 (and 5-7): conc3 falls back to smallq.
Either extend the gate to contiguous `2..8` after verifying the kernel
contract (`query_len == 8` per request holds for any B) and re-capturing
graphs, or document conc3 as degraded.

### P4 — batched scorer cost ceiling

P1's conc2 200K (41.0) still trails single-request sparse (122.7): the
fixed pipeline (propose x N, sampling, scheduling) and the scorer's
320-CTA-per-row K sweep scale with batch. Options: fewer scorer CTAs per
row at B >= 2, topk re-selection every N steps, or splitting the compact
verify into per-row splits like the dense two-pass path.
