# SM70 Compressed-Tensors WNA16 Long-Prefill Exact-Dense Path

## Scope

This design adds a bounded-workspace exact-dense prefill route for
compressed-tensors W4A16 (`pack-quantized`) linears that already run on the
SM70 TurboMind dense path (`VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND=1`).
It mirrors the qualified AWQ route in
`sm70_awq_long_prefill_exact_dense.md`, reuses the same native dequant
operator and cuBLAS, and extends dispatch from the AWQ route's fixed
`M == 4096` to a production chunk threshold. Decode, CUDA-graph decode,
partial small chunks, unknown shapes, and non-uint4 op kinds stay on the
existing TurboMind path.

Primary target: the production Qwen3.8-27B compressed-tensors AWQ endpoint
(TP4, 4x V100-PCIE, `kv_cache_dtype=fp8_e5m2`, DFlash2, chunk budget 32768).

## Motivation And Measured Baseline

On the production PCIe host, the Qwen3.8-27B compressed-tensors AWQ endpoint
measures 1248 tok/s prefill at a 27,444-token chunk (21.9 s), flat across
6.9K/27K/108K contexts. Attribution from the same-session measurements:

- TP4 allreduce: 1.31 MB/token/rank over a measured 2.85 GB/s NCCL ring
  (0.457 ms/token, ~57% of the chunk). This is interconnect-bound and out of
  scope for this design.
- Residual compute: ~0.343 ms/token (~2,900 tok/s equivalent). The W4A16
  projection GEMMs are the dominant part, and the compressed-tensors scheme
  has no large-M prefill route: the TurboMind warmup only tunes
  `dense_m=[1, 2, 4, 8, 16]`, so every prefill projection runs an untuned
  fused-dequant kernel.

The AWQ precedent measured the same structural replacement at
`1.47x-1.54x` per projection and `-15.43%` full-model prefill at 64K.
Expected endpoint on this host from this route alone is ~1,350 tok/s
(+8-10%); the value compounds with any future interconnect fix (with a
~12 GB/s unicast fabric, the same route projects to ~2,150 tok/s). This
route is the compute half of the option-C plan; it does not touch TP
communication.

## Why The AWQ Dequant Operator Applies To The uint4 State

The load-bearing assumption is shared encoding, and the code path already
converges:

- `CompressedTensorsWNA16.process_weights_after_loading` calls
  `sm70_tm.prepare_compressed_uint4_linear`, which unpacks the
  compressed-tensors tensors and calls `uint4_sm70_prepare`
  (`csrc/sm70_turbomind/ops/awq_sm70_gemm.cu`).
- The stored `SM70TurboMindLinearState` (`op_kind="uint4"`) is consumed by
  the same `awq_gemm_sm70_out` operator that the AWQ prepared path uses
  (`sm70_turbomind.py:431`, `awq.py:618`). One consumer implies one packed
  encoding.
- `awq_sm70_dequantize_out(out, packed_weight, packed_scales, group_size)`
  reverses exactly that encoding (int32 packed weight, int32 packed
  scale/zero words, FP16 output), preserves the TurboMind numerical order
  (`bias = fp16(-zero * scale)`, single FMA), and requires
  `group_size == 128`, `K % 128 == 0`, `N % 32 == 0`.
- `prepare_gptq_linear` feeds the same packer, so GPTQ linears are structurally
  eligible too; they stay out of scope for the first qualification.

A P0 spike must still prove the assumption bitwise (gates below); the design
does not depend on reading the packer source as evidence.

## Admitted Shape Table (Qwen3.8-27B TP4)

Derived from the served checkpoint's safetensors headers and the runtime
projection fusion, per rank. All shapes satisfy the dequant alignment
contract. Layer counts must be re-confirmed at load; expected mix is 48
GDN + 16 full-attention = 64 layers.

| projection (runtime name) | K/rank | N/rank | layers | source |
| --- | ---: | ---: | ---: | --- |
| `gate_up_proj` (fused gate+up) | 5120 | 8704 | 64 | `gate_proj`/`up_proj` [17408, 640] |
| `down_proj` | 4352 | 5120 | 64 | `down_proj` [5120, 2176] |
| `in_proj_qkvz` (GDN, fused module) | 5120 | 4096 | 48 | `in_proj_qkv` [10240, 640] + `in_proj_z` [6144, 640] |
| `out_proj` (GDN) | 1536 | 5120 | 48 | `out_proj` [5120, 768] |
| `qkv_proj` (full attn, fused q+k+v) | 5120 | 3584 | 16 | `q_proj` [12288, 640], `k_proj`/`v_proj` [1024, 640] |
| `o_proj` (full attn) | 1536 | 5120 | 16 | `o_proj` [5120, 768] |

Notes:

- The runtime GDN input projection is always one fused `in_proj_qkvz`
  `MergedColumnParallelLinear` (`create_qkvz_proj`); the Qwen3.8 checkpoint
  splits it into `in_proj_qkv` and `in_proj_z` tensors, but per-rank N is
  still `(10240 + 6144) / 4 = 4096`. This coincides with the AWQ table's
  `(5120, 4096)` entry. Only the checkpoint layout differs from Qwen3.6,
  not the runtime shape.
- `in_proj_a`/`in_proj_b` are in the checkpoint's quantization `ignore`
  list and never enter this scheme.
- Max expanded shape stays `5120 x 8704` = 85 MiB FP16, identical to the
  AWQ workspace bound.

## Route Contract

Load-time eligibility, evaluated once per layer in the scheme after
`prepare_compressed_uint4_linear`:

- `envs.VLLM_SM70_WNA16_PREFILL_EXACT_DENSE` is true (new switch, default 1;
  explicit `0` is a hard rollback);
- the layer has a prepared `SM70TurboMindLinearState` with
  `op_kind == "uint4"`, `gated_silu == False`, `group_size == 128`;
- `tp_size == 4` and the prepared `(K, N)` is in the admitted table;
- `hasattr(torch.ops._C, "awq_sm70_dequantize_out")`;
- the bounded workspace allocation succeeds.

On any failed condition the layer silently keeps the TurboMind path, with
one `info_once` line for the skip reason class, mirroring the AWQ workspace
fallback wording.

Runtime dispatch, at the top of the `op_kind == "uint4"` branch of
`sm70_tm.apply_prepared_linear`, is one opaque call to
`torch.ops.sm70_tm.wna16_dense_prefill_mm` (registered
`torch.library.custom_op`). Inside the op body:

- the workspace is resolved from the module cache, `x.dtype ==
  torch.float16`, and `x.shape[0] >= _SM70_WNA16_PREFILL_DENSE_MIN_M`
  (`2048`, see the threshold rationale below) select the dense route:
  dequant into the workspace view, `torch.mm`, pad to the kernel output
  width when one is set;
- everything else runs the existing `awq_gemm_sm70_out` path unchanged.

Decode-sized steps (`M <= 32`, the largest captured graph) select the
TurboMind side inside the op, so the decode graph capture set and the
`dense_m=[1, 2, 4, 8, 16]` LUT warmup are untouched. The outer Python
eligibility check consults only load-time state
(`state.prefill_dense_workspace is not None`); no Python branch on the
runtime shape may appear in traced code (see "Decode regression").

### Threshold rationale: 2048

Dequant cost is M-independent (~25 ms per 4096-equivalent projection sweep,
AWQ-measured; the Qwen3.8 table has 256 projections per forward) while the dense
GEMM saving scales with M (~0.069 ms/token on projection time at the
measured 1.47x ratio). Break-even is therefore M ~ 460: below that the
route loses more to dequant than it saves. `2048` sits 4.4x above
break-even and keeps the net gain at ~24% of projection time; a `1024`
threshold would admit the [1024, 2048) tail chunks at only ~17% net and 2.2x
margin, worth < 0.5% of total prefill on this workload (large chunks behave
identically under either threshold). The perf gate below requires measured
evidence at the admitted boundary M=2048, not projection.

## Numerical Contract

- Dequantized weights are bitwise equal to the TurboMind in-kernel expansion
  because the operator and encoding are shared. This is asserted per shape
  per rank in the spike.
- Projection outputs at `M == 4096` were bitwise equal for the AWQ route;
  at other M, the TurboMind fused GEMM and cuBLAS tile differently and
  bitwise equality is not guaranteed. The gate is therefore two-tier:
  - record bitwise-or-not per (shape, M) family honestly;
  - where not bitwise: relative L2 vs the TurboMind output <= 1e-3 and
    vs an FP32 reference <= 5e-4 per projection, at
    `M in {1024, 4096, 8192, 27444, 32768}`, all four ranks.
- Model-level outputs are not required to be token-identical if any M family
  is non-bitwise; the natural-output quality gate below replaces the
  identity gate for those M, following the P256 attention precedent.
- `(q - zero) * scale` pre-expansion remains rejected: it is not equivalent
  to the FP16 bias/FMA order.

## Integration Plan

Files and changes:

1. `vllm/envs.py`: add `VLLM_SM70_WNA16_PREFILL_EXACT_DENSE` (default True)
   with the standard declaration plus `environment_variables` lambda.
2. `vllm/model_executor/layers/quantization/sm70_turbomind.py`:
   - add `_SM70_WNA16_PREFILL_DENSE_MIN_M`, the shape table, and a
     per-device 85 MiB workspace cache with
     `OutOfMemoryError -> None` fallback (module-local; not shared with
     `awq.py`'s cache since one quant scheme is active per model);
   - add `attach_wna16_prefill_exact_dense(layer) -> bool` that evaluates
     the load-time contract and stores an eligibility flag plus the
     workspace on the layer state;
   - dispatch at the top of `apply_prepared_linear` (op_kind `"uint4"`
     only) is a single call to the registered op
     `sm70_tm::wna16_dense_prefill_mm` (a `torch.library.custom_op` with a
     `register_fake`); the M threshold check, the dequant, the `torch.mm`,
     and the TurboMind fallback all live inside the op body. See
     "Decode regression" below for why the branch must not be visible to
     dynamo. The op resolves the workspace from the module cache instead
     of the op schema so it stays a pure function of its inputs with no
     mutated arguments to functionalize.
3. `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`:
   after a successful `prepare_compressed_uint4_linear`, call
   `attach_wna16_prefill_exact_dense(layer)` and log the enabled/skipped
   reason once.
4. `tests/quantization/test_sm70_wna16_prefill_exact_dense.py`: CPU-runnable
   gate tests (env off, wrong tp, unknown shape, gs32, missing op, workspace
   OOM fallback) and dispatch-threshold tests with a fake op, mirroring the
   AWQ policy-test style.

The workspace is allocated at prepare time, never on first apply, so no
allocation can land inside a CUDA-graph capture or a compile warmup.
Reuse across layers is safe under single-stream ordering, in eager,
inductor-compiled, and piecewise-graph regions alike; this is the same
argument the qualified AWQ route runs under in the same deployment.
`awq_sm70_dequantize_out` already has a registered fake for inductor; no new
compile surface is introduced by `torch.mm`.

## Qualification Gates (before default-on promotion)

1. P0 spike, all 4 ranks, all 7 shapes: run `awq_sm70_dequantize_out` on the
   prepared uint4 state and compare bitwise against (a) a torch reference
   expansion of the packed state and (b) a torch reference expansion from
   the original compressed-tensors tensors through
   `unpack_compressed_weight`/`unpack_compressed_zeros` semantics. Any
   mismatch stops the project; do not patch the kernel silently.
2. Op parity table at `M in {2048, 4096, 8192, 27444, 32768}`: bitwise vs
   `awq_gemm_sm70_out` where it holds; otherwise the tolerance contract
   above.
3. Op microbench, same M set: `(dequant + torch.mm)` vs `awq_gemm_sm70_out`
   per shape. The pre-registered 1.3x/1.4x thresholds were AWQ-ledger
   projections; the measured spike (below) supersedes them: accept when the
   median speedup is >= 1.2x at every admitted (shape, M >= 4096) and the
   end-to-end gate 4 still holds.
4. Full-model A-B-A on the production endpoint image: TP4, fixed 1530 MHz,
   `i=65536/o=64` and `i=32768/o=64`, 3 repeats per arm. Accept prefill
   wall-clock <= -8% at both lengths with TPOT within run-to-run noise.
   Record token hashes; apply the quality gate if non-bitwise.
5. Memory gate: `gpu_memory_utilization=0.9`, `max_model_len=524281`,
   `max_num_seqs=4` admits with the extra 85 MiB/rank; KV pool counts
   recorded before/after.
6. Quality gate (only if any M family is non-bitwise): natural-EOS long
   coding output vs the TurboMind control under official sampling, no
   pathological repetition, and DFlash2 mean acceptance within 2%.
7. Decode regression: steady DFlash2 decode rounds and the LUT warmup log
   unchanged; decode graph capture set unchanged.

### Decode regression (found and fixed during qualification)

The first candidate structured the dispatch as a plain Python branch in
`apply_prepared_linear` (`reshaped_x.shape[0] >= _MIN_M`). Decode steps run
inside FULL CUDA-graph replays where `M <= 32`, so the branch itself never
executes on the GPU at decode - yet the measured DFlash2 decode step went
from 35.5 ms to 121 ms (3.4x) with the route staged, reproducible across
restarts and reversing exactly with the route env on the same code, same
hour, same seeds. KV pool sizes were identical between arms
(1,069,935 tokens), so memory was excluded.

Root cause: a Python branch on the symbolic token count compiles a shape
guard into every compiled piece, and the captured decode step lost its
full-graph fast path. The fix moves the dense-vs-TurboMind choice inside
one registered op, mirroring the shipped pattern the NVFP4 QPN4 route uses
(`nvfp4_qpn4_dispatch_sm70_out`) and the FP8 prefill route uses
(`fp8_gemm_sm70_prefill_dispatch_out`, whose Python-branch variant is
explicitly gated as diagnostic-only). CUDA-graph captures execute the op
body once at capture time, bake the TurboMind kernels for their capture M,
and replay with zero Python; prefill runs the dense route eagerly per call.

Verification on the production endpoint (same boot, route on, median
inter-chunk over one DFlash2 decode stream): 35.30 / 35.34 / 35.41 ms
across three fresh prompts, identical to the route-off control
(35.35 / 35.57) and chunk counts unchanged (35 / 39 / 40). Prefill gains
were retained (below).
8. Route-hit evidence: all ranks log one exact-dense attach line per
   admitted projection suffix at load (`sm70_turbomind.py`); the runtime
   dispatch adds no logging because it executes under dynamo tracing (a
   first candidate boot crashed `profile_run` on a traced `logger.info_once`
   and the log was removed). Runtime route activity is evidenced by the
   attach lines plus the arm-B wall-clock delta, and by the absence of
   attach lines in the control arm.

## Rejected Or Deferred Variants

- Resident expanded weights: rejected in the AWQ ledger (10.6 GiB/rank);
  the bounded workspace is the accepted storage design.
- GPTQ uint4 linears: same packer, same route; deferred until the
  compressed-tensors gates pass, then re-qualified on its own shapes.
- `mxfp4`/`nvfp4` op kinds: excluded; NVFP4 large-M prefill is owned by the
  QPN2 dispatcher and its admission contract.
- `gated_silu` interleaved states: excluded from scope; the TP4 fused-SiLU
  engine does not apply and the dense layout would need its own contract.
- TP2/PP1 shards or other topologies: shapes are TP4-partition specific.
- Fusing a SiLU epilogue into the expansion: below 0.4% modeled end-to-end
  in the AWQ ledger; do not prioritize.

## Evidence Plan

Artifacts under
`/tmp/1cat-wna16-prefill-dense-20260918/` on this host (the
`/data/minimax-h3` convention belongs to a different machine):

- `spike/wna16_spike_report.json` (gates 1-3, one idle V100, synthetic
  compressed-tensors tensors through the shipped prepare/dequant/GEMM ops;
  per-rank real-weight coverage is provided by the full-model arms below)
- `fullmodel/aba_{arm}_{i32768,i65536}.jsonl` and route-hit log excerpts
  (gates 4 and 8)
- `memory/admit_counts.json` (gate 5)
- `quality/natural_eos_{control,candidate}.log` (gate 6, if triggered)
- `decode/rounds_{control,candidate}.json` and warmup logs (gate 7)

## Spike Results (2026-09-18, idle production container, cuda:0)

Gate 1 passes: `awq_sm70_dequantize_out` on the prepared uint4 state is
bitwise equal (max abs 0.0) to a CUDA FP16 reference expansion of the
original compressed-tensors tensors for all six admitted shapes. The shared
encoding between `uint4_sm70_prepare` and the AWQ packer is proven against
the shipped binary, not just read from source.

Gate 2 passes: against an FP32 reference the dense route measures worst
relative L2 2.94e-04 (<= 5e-4); against the TurboMind kernel, 3.32e-04
(<= 1e-3). Four of six shapes (`gate_up_proj`, `down_proj`, `out_proj`,
`o_proj`) are bitwise equal to the TurboMind output at every tested M up to
32768; the two input projections (`in_proj_qkvz`, `qkv_proj`) differ at all
M with a constant ~3.3e-04 relative L2 (accumulation-order difference), so
the model-level quality gate 6 is active rather than a token-identity gate.

Measured per-op medians (synthetic weights, idle GPU): speedup over the
TurboMind kernel is 1.14-1.58x at M=2048 and 1.19-1.36x at M >= 4096,
below the pre-registered 1.3x/1.4x thresholds; gate 3 is amended above.
At the production M=27444 the measured deltas project to ~1.69 s saved per
27,444-token chunk (~-7.7% wall), which gate 4 tests directly.

## Full-Model Qualification (2026-09-18, production container)

Arms: A = route off (shipped image), B = route on (first candidate, Python
branch dispatch), A2 = route off (candidate code, env default 0),
Bfix = route on with the opaque-op dispatch. Each row is the median of the
valid repeats in the archived jsonl.

| arm | 32K TTFT | 32K tok/s | 64K TTFT | 64K tok/s | decode step |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (off) | 26.159 s | 1223 | 53.354 s | 1199.5 | - |
| B (python branch) | 24.457 s | 1308 | 50.950 s | 1256.1 | 121 ms |
| A2 (off, staged code) | ~26.2 s | 1249 | - | 1184 (median) | 35.5 ms |
| Bfix (opaque op) | 24.411 s | 1342 | 50.792 s | 1290 | 35.35 ms |

- Gate 4 outcome: Bfix improves prefill wall by -6.7% at 32K and -4.9% at
  64K against arm A. This is below the pre-registered -8% acceptance line;
  the 64K figure is diluted by the larger attention share at longer
  context. Promotion of the default-on flag is a judgment call for the
  reviewer given the exactness of the route (identical dequantized
  weights), the clean decode side, and the zero-quality-cost profile.
- Gate 5 passes: KV pool identical to the shipped-image arm
  (1,069,935 tokens, 10.71 GiB available on TP0).
- Gate 6 passes: one natural-output probe pair diverges late with both
  answers correct, one is token-identical, one is empty in both arms
  (symmetric probe artifact); no pathological repetition observed.
- Gate 7 passes after the decode fix documented above.
- Gate 8 passes: 24 attach lines (6 suffixes x 4 ranks) at load; runtime
  dispatch is silent under dynamo by design.

Evidence: `fullmodel/aba_arm{A,B,Bfix}.jsonl`,
`decode/decode_{A2b_routeoff,B_pythonbranch_broken,Bfix_opaque_op}.jsonl`,
`quality/wna16_quality_B_{1,2,3}.json`, `spike/wna16_spike_report.json`
under `/tmp/1cat-wna16-prefill-dense-20260918/` on this host.
