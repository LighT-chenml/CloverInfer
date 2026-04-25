# Resident-KV PIM Implementation Round 8

Date: 2026-04-24

## Goal

Add allocator observability for the DPU-backed resident-KV store so that later
dataset-driven experiments can distinguish:

- capacity exhaustion
- fragmentation
- low utilization
- healthy reuse

from the backend debug output directly.

## Code Changes

Updated files:

- `src/pim/upmem_kvslot/common.h`
- `src/pim/upmem_kvslot/host_kvslot.c`
- `src/core/resident_kv_store.py`
- `tests/verify_cluster_placement.py`

### New helper command

Added:

- `KVSLOT_CMD_GET_STATS`

The stdio helper now supports a read-only allocator stats query. It returns one
record per DPU with:

- `next_free_elem`
- `free_range_count`
- `free_elems_total`
- `largest_free_range`
- `live_slot_count`
- `live_elems_total`

### Python-side derived metrics

`UpmemKVSlotStore.get_debug_info()` now derives and exposes per-DPU:

- `tail_free_elems`
- `total_free_elems`
- `pool_capacity_elems`
- `used_elems_estimate`
- `usage_ratio`

This keeps the hot path unchanged while making allocator state visible through
the normal backend debug interface.

### Verification script update

`tests/verify_cluster_placement.py` now validates that:

- `allocator_stats` exists
- it has one record per configured DPU
- pool accounting is internally consistent

## Validation

### Direct stats smoke on `.7`

Observed after a small request was initialized and decoded:

- each DPU had `live_slot_count = 2`
- each DPU had `live_elems_total = 128`
- each DPU had `next_free_elem = 128`
- each DPU had `usage_ratio = 0.0001220703125`

Observed after freeing the request:

- each DPU returned to:
  - `live_slot_count = 0`
  - `live_elems_total = 0`
  - `next_free_elem = 0`
  - `usage_ratio = 0.0`

Interpretation:

- allocator accounting reflects both allocation and full reclamation correctly

### Fragmentation smoke on `.7`

Constructed a 1-DPU test with uneven capacities:

1. allocate A(8), B(16), C(8)
2. free middle block B
3. allocate D(12)
4. free everything

Observed:

- after freeing B:
  - `free_range_count = 1`
  - `largest_free_range = 512`
- after allocating D:
  - `free_range_count = 1`
  - `largest_free_range = 128`
- after freeing all:
  - `next_free_elem = 0`
  - `free_range_count = 0`

Interpretation:

- the new stats can directly expose fragmentation
- allocator reuse is visible, not inferred indirectly from fallback behavior

### Three-machine end-to-end smoke

Validated command:

```bash
/home/cml/anaconda3/envs/clover_infer/bin/python tests/verify_cluster_placement.py \
  --address 192.168.123.4:26379 \
  --model /home/cml/CloverInfer/model/opt-125m \
  --attention-backend pim_naive \
  --pim-resident-store-backend upmem_kvslot \
  --no-pim-qk-mixed-enabled \
  --pim-num-dpus 4 \
  --pim-length 8 \
  --max-new-tokens 2
```

Observed result:

- `Placement verification passed`
- `dpu_allocations = 48`
- `fallback_allocations = 0`
- `dpu_allocate_failures = 0`
- `resident_shadow_max_abs_diff = 0.0`
- `allocator_stats` present for all 4 DPUs

## What This Enables

The system can now separate several future failure modes cleanly:

- if `fallback_allocations` rises and `largest_free_range` is small, that points
  to fragmentation
- if `usage_ratio` is low but performance is bad, the challenge is not raw MRAM
  pressure
- if `next_free_elem` grows but `tail_free_elems` does not recover, we can see
  poor reclamation behavior directly

This is the right instrumentation layer to support the next stage of challenge
analysis.

## Suggested Next Step

Run longer dataset-driven workloads and log allocator stats over time, then
summarize:

1. max usage ratio per DPU
2. max free-range count per DPU
3. largest-free-range decay over time
4. any point where fallback begins despite substantial total free space
