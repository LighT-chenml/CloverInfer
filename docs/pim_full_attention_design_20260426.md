# PIM-Based Attention Boundary and Full-PIM Roadmap

## 1. Core design principle

The reason to use PIM for attention is not "remove the host from the
computation graph entirely".

The real reason is:

- attention is dominated by KV-cache traffic
- historical KV is large
- each KV element is typically consumed once per decode step
- arithmetic intensity is low relative to the amount of data touched

So the correct design target is:

- keep the host as the control plane
- move the KV-dominated data plane as close to PIM as possible

In practical terms, the most important rule is:

- large historical KV should not keep moving back to the host
- the host should exchange only small per-step objects when possible

The desired dataflow is therefore:

- host / GPU sends small per-token inputs: `q`, `k_new`, `v_new`
- PIM keeps historical KV resident
- PIM performs the KV-scan-heavy attention path
- host / GPU receives only the small output `context`

This is the most relevant notion of "PIM-based attention" for this project.

## 2. Recommended system boundary

### Host / GPU responsibilities

These should remain outside PIM:

- request lifecycle and scheduling
- batching and wavefront formation
- QKV projection on the GPU
- output projection on the GPU
- FFN / LM head on the GPU

Rationale:

- these parts are control-heavy or compute-dense
- they are not the KV-bandwidth bottleneck
- moving them to PIM would not target the main problem

### PIM responsibilities

These are the right target for PIM:

- resident KV allocation / free
- KV append for each decode step
- historical K scan
- historical V scan
- QK score computation
- softmax-related reductions
- AV accumulation

Rationale:

- this is the bandwidth-intensive part
- it repeatedly touches the resident KV state
- it benefits most from keeping data local to the memory-side engine

## 3. Current implementation boundary

The current codebase should be described as:

- PD-disaggregated
- AF-disaggregated
- resident-KV PIM-assisted attention

What already runs on / through PIM:

- KV resident storage in `UpmemKVSlotStore`
- KV append / free through the kvslot helper
- batched QK slot scoring support
- resident AV support

Key implementation anchors:

- scheduler and control plane:
  - `src/core/scheduler.py`
- attention node and backend:
  - `src/core/nodes.py`
  - `src/core/attention_backend.py`
- resident KV store:
  - `src/core/resident_kv_store.py`
- UPMEM helper and DPU kernel:
  - `src/pim/upmem_kvslot/host_kvslot.c`
  - `src/pim/upmem_kvslot/dpu_kvslot.c`

### What is still host-resident in the attention datapath

Two important attention steps are still on the host side today:

1. a full CPU score path still exists in the main decode record preparation
   path
2. softmax is still performed on the host, and the resulting weights are then
   sent into resident AV

Concrete code points:

- full host-side score construction:
  - `src/core/attention_backend.py`
  - `scores = torch.einsum("hd,lhd->hl", q_fp32, keys.float()) * score_scale`
- host-side softmax before AV:
  - `weights = torch.softmax(record["scores"], dim=-1)`

So the current state is approximately:

- KV append: PIM
- KV storage: PIM
- QK: full-PIM primary path is now implemented behind a flag, but it is still
  under validation
- softmax: host
- AV: PIM
- context return: host

This is not yet full-PIM attention.

## 4. What "full-PIM attention" should mean here

For this project, the recommended operational definition is:

- the host remains the control plane
- the host still performs QKV projection and downstream dense compute
- but the attention datapath over historical KV is fully closed inside the PIM
  side

The target per-step interface should be:

- input from host / GPU:
  - `q`
  - `k_new`
  - `v_new`
  - lightweight request / layer metadata
- work performed by PIM:
  - append `k_new/v_new`
  - compute `QK`
  - perform softmax-related reductions / normalization
  - compute `AV`
- output to host / GPU:
  - `context`

In short:

- context-only return
- no full score matrix returned to the host
- no full weight matrix sent back from the host to PIM

This is the correct full-PIM target boundary for our system.

## 5. Current evidence and lessons

We now have a reasonably clear picture from the experiments so far:

- simple scheduler batching works and is enough as a baseline
- stronger scheduler-cohort persistence did not help
- helper-side AV rank-subset batching produced real latency gains
- coarse AV DPU-kernel repartition attempts regressed

The most important lesson is:

- current bottlenecks are still more about helper/runtime/transport overhead
  and data-path structure than about missing scheduler sophistication

This supports a roadmap that pushes the attention datapath boundary inward,
instead of spending more time on scheduler complexity right now.

## 6. Recommended roadmap to full-PIM attention

The roadmap should be staged so that each step shrinks host involvement in the
attention datapath while preserving a clean correctness story.

### Phase 0: Stable baseline

Before pushing deeper into full-PIM, keep a stable experimental baseline:

- simple decode-step sync and attention batching
- no decode-wave persistence by default
- resident KV enabled
- helper-side AV rank-subset batching enabled
- current stable DPU AV kernel kept unchanged

Purpose:

- all future full-PIM work must compare against a fixed, credible baseline

### Phase 1: Full-QK offload while keeping host softmax

Goal:

- remove the host-side full score path as the main computation path
- make PIM the primary producer of attention scores

Target state:

- KV append: PIM
- QK: full PIM
- softmax: host
- AV: PIM

Required changes:

1. introduce a full-QK resident path that produces the full score tensor for a
   layer/request batch
2. stop relying on the host-side full `torch.einsum` score path as the normal
   path
3. keep the host-side path only as a debug / validation fallback
4. preserve existing correctness checks while the new path is being validated

Why this step first:

- it removes the biggest remaining host-side KV scan
- it still avoids the additional complexity of PIM-side softmax

### Phase 1 current status

This phase is now partially complete in implementation:

- `pim_qk_full_enabled` now routes decode-time QK through the resident-slot
  batch path in `src/core/attention_backend.py`
- the old host `torch.einsum` score path is retained as a shadow/reference path
- cluster-level correctness for a small single-request decode check passes with
  `qk_full_shadow_max_abs_diff = 0.0`

We also found and fixed one helper/runtime issue while bringing this up:

- full-QK on real traces exposed an 8-byte MRAM readback alignment bug in
  `src/pim/upmem_kvslot/host_kvslot.c`
- the helper now rounds QK-score readback size up to 8-byte alignment before
  copying the real payload back to the host buffer

However, Phase 1 is not numerically closed yet on realistic concurrent traces.

On:

- `OPT-125M`
- `dataset/humaneval.jsonl`
- `concurrency = 8`
- `max_new_tokens = 4`
- `256 DPU`

artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync0ms_fullqk_retry1.jsonl`

observed initially:

- the run completed successfully
- `avg_latency ~= 21.54 s`
- `max_latency ~= 21.71 s`
- `qk_full_batch_calls` grew as expected
- but `qk_full_shadow_max_abs_diff = 16.8175`

Interpretation:

- the full-QK path was functionally integrated and stable enough to execute
  real traces
- but there was still a correctness gap in the `qk_slot` data path
- so at that point this phase was only "running but not fully validated"

Immediate follow-up for Phase 1:

- keep `qk_full_shadow_check` enabled
- locate whether the remaining diff comes from:
  - slot-query ordering / regrouping
  - helper batch packing / unpacking
  - DPU-side slot-QK semantics on longer windows
  - accumulation / dtype behavior in the helper return path

### Phase 1 follow-up result

That follow-up has now been completed.

Root cause:

- odd `window` in `qk_slot` score writeback used packed 64-bit stores
- but the per-head MRAM row stride still used the unpadded `window`
- so the last packed score pair of one head overlapped the first score of the
  next head

Fix:

- switch internal `qk_slot` score layout to a padded stride:
  - `score_stride = round_up_even(window)`
- keep the external Python-visible output shape unchanged

Files:

- `src/pim/upmem_kvslot/dpu_kvslot.c`
- `src/pim/upmem_kvslot/host_kvslot.c`

Validated on:

- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync0ms_fullqk_retry2.jsonl`

Observed after the fix:

- `avg_latency ~= 15.61 s`
- `max_latency ~= 15.65 s`
- `qk_full_shadow_max_abs_diff = 7.629e-06`

Updated interpretation:

- Phase 1 full-QK offload is now numerically validated enough to serve as the
  next baseline
- the next blocker is no longer QK correctness
- the next major boundary shift should move toward fused `softmax + AV`

### Phase 2: Fuse softmax + AV into the helper-side attention path

Goal:

- remove score/weight round-trips through the host

Target state:

- host provides `q, k_new, v_new`
- PIM returns `context`
- no full scores exposed to the host in the normal path

Required changes:

1. add a fused helper command / API for:
   - append KV
   - QK
   - softmax reductions
   - AV
2. define the numerical-stability strategy for softmax:
   - row max
   - exponentiation / normalization
   - reduction order
3. keep an optional host-side shadow check path during bring-up

Why this step matters most:

- this is the real boundary shift from PIM-assisted to near-full-PIM attention
- it eliminates the largest remaining host-side attention dataflow

### Phase 3: Tighten the PIM-side runtime

Goal:

- after the datapath boundary is correct, make it fast

Focus areas:

- helper protocol packing and header overhead
- fused command granularity
- DPU launch/transfer amortization
- selective metadata elision
- only then, carefully revisit DPU-kernel specialization

Important constraint:

- avoid coarse repartition ideas that destroy tasklet utilization
- any future kernel redesign must preserve enough intra-head or intra-row
  parallelism

## 7. Near-term implementation priorities

If the objective is to reach full-PIM attention quickly, the recommended
priority order is:

1. make QK full-offload the primary path
2. design the fused `append + QK + softmax + AV -> context` helper contract
3. implement a first correct fused full-PIM path
4. only then optimize its runtime

What should *not* be the short-term priority:

- more scheduler redesign
- more coarse AV-kernel repartition experiments
- moving GPU-dense work into PIM

## 8. Concrete engineering implications

To keep future work organized, each code change should be tagged mentally as
belonging to one of these layers:

- control plane:
  - scheduler, request lifecycle, actor orchestration
- PIM data plane:
  - resident KV store
  - helper protocol
  - QK / softmax / AV kernels
- dense compute plane:
  - projection / FFN / output projection on GPU

The architectural rule should be:

- reduce host involvement in the PIM data plane
- do not blur dense compute and PIM data-plane responsibilities

## 9. Experimental framing

For future experiments, comparisons should be organized around these system
states:

1. disaggregated CPU attention baseline
2. resident-KV + partial-PIM attention baseline
3. resident-KV + full-QK PIM
4. fused full-PIM attention (`QK + softmax + AV`)

This gives a clean experimental story:

- first move KV locality
- then move QK
- then close the attention loop
- then optimize

## 10. Summary

The recommended full-PIM direction is not "remove the host entirely".

It is:

- host = control plane + dense compute plane
- PIM = KV-resident attention data plane

The shortest correct path forward is:

1. full-QK offload
2. fused softmax + AV in the helper
3. context-only return
4. performance optimization on that final boundary

That boundary best matches the real reason PIM is attractive for attention:

- high KV bandwidth demand
- low arithmetic intensity
- one-pass consumption of historical KV per decode step

## 11. Current implemented workflow after helper-boundary fusion

As of 2026-04-26, the running decode-time attention workflow is:

1. host/GPU computes `q`, `k_new`, `v_new`
2. attention node appends `k_new/v_new` into resident KV slots on `.7`
3. resident-slot full-QK path computes per-group score rows through
   `qk_slot_scores_batch`
4. Python backend regroups per-slot score rows back into the full head order
5. if `pim_softmax_av_fused_enabled = True`, the backend sends per-slot score
   matrices to the kvslot helper through a fused `softmax+AV` command
6. helper performs row-wise softmax on CPU, launches the existing resident-AV
   path, and returns only per-group `context`
7. backend concatenates group contexts and returns the final layer `context`
   to the dense/GPU side

This means the current normal-path attention boundary is now:

- host/GPU sends:
  - `q`
  - `k_new`
  - `v_new`
- PIM side owns:
  - resident KV
  - full-QK over resident K
  - softmax at the helper boundary
  - AV over resident V
- host/GPU receives:
  - `context`

Important nuance:

- softmax is fused at the helper boundary, not yet inside the DPU kernel
- so this is not the final full-PIM endpoint yet
- but it already removes the previous host-side `scores -> softmax -> weights`
  main-path round trip

## 12. Phase 2 bring-up status: helper-boundary fused softmax + AV

Phase 2 has now reached a first correct end-to-end baseline.

What changed:

- `src/core/attention_backend.py`
  - added `softmax_av_fused_enabled`
  - added `softmax_av_shadow_check`
  - decode path now uses `ResidentKVStore.softmax_weighted_value_sum_batch(...)`
    when resident AV is enabled
- `src/core/resident_kv_store.py`
  - added a fused `softmax_weighted_value_sum_batch(...)` interface
  - host store provides a CPU reference implementation
  - UPMEM store forwards the fused call into the kvslot helper
- `src/pim/upmem_kvslot/common.h`
  - added `KVSLOT_CMD_SOFTMAX_AV_BATCH`
- `src/pim/upmem_kvslot/host_kvslot.c`
  - added helper handling for fused softmax + AV
  - helper computes row-wise stable softmax on CPU:
    - row max
    - `exp`
    - normalize
  - helper then reuses the existing resident-AV launch path

This is intentionally a staged implementation:

- QK is already on the resident path
- softmax is now fused into the helper boundary
- AV is still the existing DPU resident-AV path

So the current system should be described as:

- context-only return at the Python/backend boundary
- helper-side fused softmax + AV
- not yet DPU-kernel-fused softmax + AV

## 13. Validation results for the current fused boundary

### Helper-level fused-path check on `.7`

Direct helper/store check:

- `UpmemKVSlotStore.softmax_weighted_value_sum_batch(...)`
- compared against host reference

Observed:

- `max_diff = 5.96e-08`

Interpretation:

- helper protocol and fused numerical path are correct for a direct store-level
  call

### Three-machine cluster correctness check

Command shape:

- `pim_qk_full_enabled = True`
- `pim_qk_mixed_enabled = False`
- `pim_softmax_av_fused_enabled = True`
- `64 DPU`
- `OPT-125M`
- `max_new_tokens = 2`

Observed:

- three-machine placement check passed
- end-to-end generation succeeded
- `qk_full_shadow_max_abs_diff = 0.0`
- `softmax_av_fused_shadow_max_abs_diff = 4.768e-07`

Interpretation:

- the new main path is now correct end-to-end for a small real decode
- the backend no longer needs host-side softmax as the normal path

### Small concurrent trace

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_tok2_dpu64_fp32_fullqk_fused.jsonl`

Workload:

- `dataset/humaneval.jsonl`
- `OPT-125M`
- `concurrency = 4`
- `max_new_tokens = 2`
- `64 DPU`

Observed:

- `avg_latency ~= 3.18 s`
- `max_latency ~= 3.20 s`
- `max qk_full_shadow_max_abs_diff = 5.722e-06`
- `max softmax_av_fused_shadow_max_abs_diff = 6.104e-05`
- no fallback allocations
- no DPU allocation failures

Interpretation:

- the helper-boundary fused path is stable enough to serve as the next baseline
- the main remaining gap to the full-PIM target is no longer correctness at
  the backend/helper boundary
- the next step should move more of the softmax/AV path into the true PIM
  data plane

## 14. What still prevents this from being "full-PIM"

Even after the new fused path, the following still remain outside the true DPU
kernel:

- row-wise softmax is computed inside the helper CPU process
- AV launch orchestration is still helper-managed
- score matrices are still materialized at the helper boundary before softmax

So the current design is best understood as:

- host control plane
- helper-side fused attention boundary
- DPU-resident KV data path

not yet:

- single DPU-kernel `QK + softmax + AV` closure

## 15. Recommended next technical step

The next implementation step should be:

1. preserve the current helper-boundary fused path as the correctness baseline
2. push the fused path inward so the helper no longer has to materialize full
   score rows as the stable long-term boundary
3. move toward a DPU-oriented softmax/AV workflow with minimal host/helper
   data movement

Concretely, future work should focus on:

- softmax reduction ownership:
  - what stays on helper CPU
  - what can move to DPU
- score/weight materialization lifetime:
  - avoid unnecessary helper-side copies
- DPU-local KV and context accumulation pipeline:
  - keep the `context-only` return boundary stable
