# PIM Allocator Trace Round 3

Date: 2026-04-25

## Goal

Use the new resident-footprint accounting to explain why:

- `concurrency = 3` remains stable on the full-DPU path
- `concurrency = 4` begins triggering host fallback

The target here is not a partial-residency workaround. The mainline design is
still:

- keep resident KV on the DPU/PIM side whenever capacity allows
- treat host fallback as overflow protection and a comparison baseline

## Code Changes Used In This Round

Updated files:

- `src/core/attention_backend.py`
- `tests/trace_pim_allocator.py`

Added debug visibility for:

- per-group resident `live_elems`
- per-group reserved `capacity_elems`
- per-layer resident footprint
- per-request resident footprint
- per-request per-DPU resident footprint

This makes allocator pressure interpretable in terms of actual request state,
instead of only allocator-side snapshots.

## Validation Setup

Cluster:

- `192.168.123.3`: prefill GPU node
- `192.168.123.4`: decode-dense GPU node and Ray entry point
- `192.168.123.7`: attention CPU/PIM node

Model/workload:

- `OPT-125M`
- dataset: `dataset/humaneval.jsonl`
- `max_new_tokens = 2`
- resident backend: `upmem_kvslot`
- `pim_num_dpus = 4`

Commands:

```bash
python tests/trace_pim_allocator.py \
  --address 192.168.123.4:26379 \
  --data dataset/humaneval.jsonl \
  --model /home/cml/CloverInfer/model/opt-125m \
  --model-name opt-125m \
  --limit 6 \
  --max-new-tokens 2 \
  --concurrency 3 \
  --pim-num-dpus 4 \
  --pim-length 8 \
  --pim-resident-store-backend upmem_kvslot \
  --output artifacts/pim_allocator_trace_opt125m_humaneval6_conc3_footprint.jsonl
```

```bash
python tests/trace_pim_allocator.py \
  --address 192.168.123.4:26379 \
  --data dataset/humaneval.jsonl \
  --model /home/cml/CloverInfer/model/opt-125m \
  --model-name opt-125m \
  --limit 4 \
  --max-new-tokens 2 \
  --concurrency 4 \
  --pim-num-dpus 4 \
  --pim-length 8 \
  --pim-resident-store-backend upmem_kvslot \
  --output artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_footprint.jsonl
```

## Observed Summary

### Concurrency 3

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval6_conc3_footprint.jsonl`

Summary:

- `num_requests = 6`
- `concurrency = 3`
- `max_usage_ratio = 0.863525390625`
- `max_live_slot_count = 36`
- `max_dpu_capacity_fallbacks = 0`
- `max_fallback_allocations = 0`

Average request footprint:

- request `capacity_elems ~= 1,205,760`
- request `live_elems ~= 1,196,544`
- per-DPU request `capacity_elems ~= 301,440`

Interpretation:

- three overlapping requests fit cleanly:
  - `3 * 301,440 = 904,320` elems per DPU
- DPU pool capacity is `1,048,576` elems per DPU
- so the full-DPU path still has about `144k` elems of headroom per DPU

### Concurrency 4

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_footprint.jsonl`

Summary:

- `num_requests = 4`
- `concurrency = 4`
- `max_usage_ratio = 0.99169921875`
- `max_live_slot_count = 41`
- `max_dpu_capacity_fallbacks = 28`
- `max_fallback_allocations = 28`
- `failed = false`

Average request footprint:

- request `capacity_elems ~= 1,313,280`
- request `live_elems ~= 1,304,064`
- per-DPU request `capacity_elems ~= 328,320`

Representative per-request reserved capacity:

- request 1:
  - `per_dpu_capacity_elems = 331,776`
- request 2:
  - `per_dpu_capacity_elems = 331,776`
- request 3:
  - `per_dpu_capacity_elems = 327,168`
- request 4:
  - `per_dpu_capacity_elems = 322,560`

Interpretation:

- four overlapping requests would need about:
  - `4 * 328,320 = 1,313,280` elems per DPU on average
- that exceeds the pool limit:
  - `1,313,280 > 1,048,576`
- fallback therefore starts because reserved resident capacity no longer fits,
  not because of allocator corruption or poor DPU placement

## Important Finding

This round makes the root cause much clearer:

- `concurrency = 4` overflow is primarily a reserved-capacity problem
- not a slot-count problem
- not a cross-DPU imbalance problem
- not a helper-crash problem

The evidence chain is:

1. per-request footprint is balanced across DPUs
2. `concurrency = 3` fits below the per-DPU MRAM pool limit
3. `concurrency = 4` exceeds that limit even before considering fragmentation
4. the guarded allocator now degrades via fallback instead of crashing

So the next optimization target should be reducing reserved resident footprint
per request, not changing the architectural principle of keeping KV on DPU.

## What This Means For Next Steps

Most promising full-DPU optimization directions:

1. reduce over-reservation
   - current resident capacity uses `seq_len + decode_reserve_tokens`
   - this is correctness-safe, but expensive under overlap

2. reduce resident granularity cost
   - a request currently reserves capacity independently for each
     `layer x group`
   - smaller reserve policy or tiered reserve may save substantial MRAM

3. later, reduce element footprint itself
   - quantized or compressed KV is a true next-stage optimization
   - but it should come after the current reserve-accounting challenge is
     fully characterized

## Suggested Immediate Next Step

Stay on the full-DPU mainline and evaluate a reserve-aware optimization such as:

- smaller decode reserve
- dynamic growth for resident slots
- two-tier reserve policy

Then rerun the same `concurrency = 3/4` traces and compare:

- fallback count
- peak usage ratio
- fragmentation
- latency impact

## Additional DPU-Scaling Check

After the capacity-threshold analysis above, I also tested whether increasing
the available DPU count can recover the full-DPU path without changing the
reserve policy.

Command:

```bash
python tests/trace_pim_allocator.py \
  --address 192.168.123.4:26379 \
  --data dataset/humaneval.jsonl \
  --model /home/cml/CloverInfer/model/opt-125m \
  --model-name opt-125m \
  --limit 4 \
  --max-new-tokens 2 \
  --concurrency 4 \
  --pim-num-dpus 8 \
  --pim-length 8 \
  --pim-resident-store-backend upmem_kvslot \
  --output artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu8.jsonl
```

Observed summary:

- `concurrency = 4`
- `max_usage_ratio = 0.5855712890625`
- `max_dpu_capacity_fallbacks = 0`
- `max_fallback_allocations = 0`
- `failed = false`

Important detail:

- `OPT-125M` has `12` heads
- with `num_dpus = 8`, current grouping uses:
  - `num_groups = min(8, 12) = 8`
  - `heads_per_group = ceil(12 / 8) = 2`
- because the last two groups would be empty, only `6` DPUs actually carry
  resident KV in this configuration

Observed per-request reserved capacity:

- about `221,184` elems on each active DPU
- active DPUs: `6`
- idle DPUs: `2`

Interpretation:

- even without using all 8 DPUs, the wider grouping reduces per-DPU KV pressure
  enough for `concurrency = 4` to remain fully DPU-resident
- this strongly supports the conclusion that the current bottleneck is per-DPU
  resident footprint, not a fundamental flaw in the disaggregated design

This also suggests a useful next question:

- should head-group mapping be redesigned to use all available DPUs more evenly
  when `num_dpus > num_heads / 2`?

## Placement Follow-Up

After the DPU-scaling check, I tried to improve `8 DPU` placement so that all
available DPUs participate more evenly.

### Important implementation bug found

At first, I changed only the head-group metadata in `attention_backend.py`.
That looked like a placement experiment, but it did **not** change the real DPU
placement.

Root cause:

- actual physical DPU selection in `UpmemKVSlotStore` was implicitly determined
  by:
  - `slot_id % num_dpus`
- the `HeadGroupState.dpu_id` chosen by the attention backend was only metadata
  and did not propagate to the allocator

This was a real implementation gap in the placement path.

### Fix

Updated:

- `src/core/attention_backend.py`
- `src/core/resident_kv_store.py`

Added:

- `preferred_dpu` propagation from the attention backend to the resident store
- per-DPU slot-id assignment in `UpmemKVSlotStore`
- actual placement control instead of metadata-only placement labels

### Three 8-DPU variants

1. Original 8-DPU grouping
   - artifact:
     - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu8.jsonl`
   - request-1 per-DPU capacity:
     - `[221184, 221184, 221184, 221184, 221184, 221184, 0, 0]`
   - summary:
     - `max_usage_ratio = 0.5855712890625`
     - `max_live_slot_count = 36`
     - `fallback = 0`

2. Balanced groups, but before real placement fix
   - artifact:
     - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu8_balanced.jsonl`
   - request-1 per-DPU capacity:
     - `[221184, 221184, 221184, 221184, 110592, 110592, 110592, 110592]`
   - summary:
     - `max_usage_ratio = 0.78076171875`
     - `max_live_slot_count = 48`
     - `fallback = 0`
   - interpretation:
     - this version increased active groups, but heavy `2-head` groups stayed
       aligned on the same DPUs across requests

3. Balanced groups with real rotated placement
   - artifact:
     - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu8_rotated_real.jsonl`
   - request-1 per-DPU capacity:
     - `[165888, 175104, 184320, 175104, 165888, 156672, 147456, 156672]`
   - summary:
     - `max_usage_ratio = 0.6026611328125`
     - `max_live_slot_count = 48`
     - `fallback = 0`

### Interpretation

This follow-up reveals two separate effects:

1. using more DPU groups increases parallel resident placement opportunity
2. but without real placement control, heavier groups can create persistent DPU
   hotspots

Once the real placement path was fixed, rotated placement reduced peak usage
substantially:

- from `0.7808` down to `0.6027`

while preserving:

- `0` fallback allocations
- `0` DPU allocation failures

This is a strong signal that placement policy is itself a meaningful
optimization dimension, separate from:

- raw DPU count
- raw resident footprint
- reserve policy

## Policy Matrix Baseline

To make the grouping and placement experiments reproducible, I exposed them as
independent config/CLI policies:

- `pim_head_grouping_policy`
  - `legacy`
  - `balanced`
- `pim_dpu_placement_policy`
  - `identity`
  - `rotated`

This enables clean comparisons between:

- old behavior
- improved grouping only
- improved grouping plus real placement

### 8-DPU concurrency-4 comparison

Command family:

```bash
python tests/trace_pim_allocator.py \
  --address 192.168.123.4:26379 \
  --data dataset/humaneval.jsonl \
  --model /home/cml/CloverInfer/model/opt-125m \
  --model-name opt-125m \
  --limit 4 \
  --max-new-tokens 2 \
  --concurrency 4 \
  --pim-num-dpus 8 \
  --pim-length 8 \
  --pim-resident-store-backend upmem_kvslot \
  --pim-head-grouping-policy <policy> \
  --pim-dpu-placement-policy <policy>
```

Observed results:

1. `legacy + identity`
   - artifact:
     - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu8_legacy_identity.jsonl`
   - `max_usage_ratio = 0.78076171875`
   - `max_live_slot_count = 48`
   - `fallback = 0`

2. `balanced + rotated`
   - artifact:
     - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu8_balanced_rotated.jsonl`
   - `max_usage_ratio = 0.6114501953125`
   - `max_live_slot_count = 48`
   - `fallback = 0`

Interpretation:

- compared with the legacy policy pair, the explicit balanced-grouping +
  rotated-placement baseline reduces peak allocator pressure materially
- the resident system remains fully DPU-resident in both cases, so this is a
  real placement/pressure improvement rather than a fallback artifact

Current recommended default for 8-DPU experiments:

- `pim_head_grouping_policy = balanced`
- `pim_dpu_placement_policy = rotated`

## 4-DPU Control Check

I also ran the same policy comparison back on the tighter `4 DPU` setting to
see whether placement alone can rescue the overflow case.

Artifacts:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu4_legacy_identity.jsonl`
- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu4_balanced_rotated.jsonl`

Observed summary for both:

- `max_usage_ratio = 0.99169921875`
- `max_dpu_capacity_fallbacks = 28`
- `max_fallback_allocations = 28`

Interpretation:

- with `12 heads` on `4 DPUs`, the grouping is already effectively fixed at:
  - `3 heads / DPU`
- so both policy settings place an equal amount of resident KV on every DPU
- in this regime, rotating placement does not create new headroom

This is an important negative result:

- placement policy helps when there is placement asymmetry to exploit
- but it does **not** solve the core per-DPU capacity limit in the fully dense
  `4 DPU` case

That means the next meaningful optimization for `4 DPU` should go back to:

- reserve policy
- KV footprint reduction
- or a larger PIM resource budget

## FP16 Resident-KV Baseline

After the negative `4 DPU` placement result, I added a new resident-KV storage
mode:

- `pim_resident_kv_dtype = fp16`

This changes the DPU-resident KV representation from the original `fp32`
bitwise-storage path to a packed `fp16` path while still materializing back to
`float32` for the host-side attention computation.

Updated components:

- `src/core/config.py`
- `src/core/scheduler.py`
- `src/core/attention_backend.py`
- `src/core/resident_kv_store.py`
- `src/pim/upmem_kvslot/common.h`
- `src/pim/upmem_kvslot/host_kvslot.c`
- `tests/trace_pim_allocator.py`

### Smoke check

Artifact:

- `artifacts/pim_allocator_trace_opt125m_smoke_fp16.jsonl`

Observed:

- `concurrency = 1`
- `max_usage_ratio = 0.158203125`
- `fallback = 0`

Compared with the earlier `fp32` smoke:

- `fp32` peak was about `0.316`
- `fp16` peak is about `0.158`

This confirms the expected near-2x resident-capacity reduction.

### 4-DPU concurrency-4 check

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu4_fp16.jsonl`

Configuration:

- `pim_num_dpus = 4`
- `concurrency = 4`
- `pim_head_grouping_policy = balanced`
- `pim_dpu_placement_policy = rotated`
- `pim_resident_kv_dtype = fp16`

Observed summary:

- `max_usage_ratio = 0.5855712890625`
- `max_dpu_capacity_fallbacks = 0`
- `max_fallback_allocations = 0`
- `failed = false`

### Interpretation

This is the first optimization that actually fixes the `4 DPU` overflow case.

Compared with the `4 DPU fp32` baseline:

- `fp32`
  - `max_usage_ratio = 0.99169921875`
  - `fallback_allocations = 28`
- `fp16`
  - `max_usage_ratio = 0.5855712890625`
  - `fallback_allocations = 0`

So for the current workload, resident-KV compression is much more effective
than placement policy in the constrained `4 DPU` regime.

## Current Best Baselines

For the current naive resident-KV PIM stack, the most useful baselines are now:

1. `4 DPU + fp32`
   - shows the raw capacity bottleneck and overflow behavior

2. `4 DPU + fp16`
   - shows that resident-KV compression can recover the full-DPU path

3. `8 DPU + fp32 + balanced + rotated`
   - shows how added PIM resources plus better placement reduce pressure
