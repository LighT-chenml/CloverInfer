# Resident-KV PIM Implementation Round 6

Date: 2026-04-24

## Goal

Extend the first DPU-backed resident-KV path from:

- "a few groups can land on DPU"

to:

- "the normal OPT-125M end-to-end smoke can place all resident groups on DPU"

This round focused on finishing the multi-slot-per-DPU helper path and wiring
real lifecycle cleanup through the Python store.

## Code Changes

Updated files:

- `src/pim/upmem_kvslot/common.h`
- `src/pim/upmem_kvslot/host_kvslot.c`
- `src/core/resident_kv_store.py`

### Helper/runtime changes

The UPMEM kvslot helper now supports multiple logical slots per DPU:

- route logical slot ids with:
  - `physical_dpu = slot_id % num_dpus`
  - `local_slot = slot_id / num_dpus`
- maintain a slot table sized as:
  - `num_dpus * KVSLOT_MAX_SLOTS_PER_DPU`
- assign each slot a per-DPU MRAM element offset
- read/write resident KV at offset-based MRAM addresses instead of always using
  offset 0

Added helper command:

- `KVSLOT_CMD_FREE`

This allows the host-side store to explicitly release a logical DPU slot when a
request is freed.

### Python store changes

`UpmemKVSlotStore` now:

- treats the DPU slot space as `num_dpus * 32` logical slots instead of only
  `num_dpus`
- reuses freed logical slot ids
- sends an explicit `FREE` command for DPU-backed groups
- records `dpu_free_ops`, `free_slot_ids`, and `next_slot_id` in debug output
- falls back to the host resident store if DPU allocation fails

## Validation

### Direct backend smoke on `.7`

Command shape:

- instantiate `PimNaiveAttentionBackend(..., resident_store_backend="upmem_kvslot")`
- init one request with 2 layers
- decode both layers once
- free the request

Observed result:

```python
{
  "shadow_diff": 0.0,
  "store_backend": "upmem_kvslot_store",
  "dpu_allocations": 8,
  "fallback_allocations": 0,
  "dpu_free_ops": 8,
  "live_slots_end": 0,
  "free_slot_ids": 8,
  "next_slot_id": 8,
}
```

Interpretation:

- all resident groups in this smoke case now land on DPU
- CPU shadow remains exactly aligned
- freed requests return their logical slot ids to the store

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

Observed resident-store debug:

- `resident_store_debug.backend = upmem_kvslot_store`
- `resident_store_debug.dpu_allocations = 48`
- `resident_store_debug.dpu_free_ops = 48`
- `resident_store_debug.fallback_allocations = 0`
- `resident_store_debug.helper_restarts = 1`
- `resident_shadow_max_abs_diff = 0.0`

Interpretation:

- the real distributed OPT-125M smoke now places all resident groups on the
  DPU-backed store
- the previous host fallback in this configuration is no longer needed
- the resident-KV path remains numerically aligned with the CPU oracle

### Sequential request reuse smoke

Additional direct stress test on `.7`:

- reuse a single backend/helper instance
- run 24 requests sequentially
- each request allocates, decodes, and frees 12 resident groups

Observed behavior:

- requests 1-24 all stayed at `fallback_allocations = 0`
- `free_slot_ids` stayed at 48 after each request
- `next_slot_id` stayed at 48 after the first request, which confirms logical
  slot-id reuse

Interpretation:

- explicit `FREE` plus slot-id reuse works for repeated request lifecycles
- the current helper is stable enough for repeated functional experiments

## Current Remaining Limitation

The helper now reuses logical slot ids, but it does not yet reclaim or compact
the per-DPU MRAM pool:

- `elem_offset` allocation is still monotonic inside a helper lifetime
- `FREE` clears logical slot metadata, but not the MRAM pool allocator state

This did not block the current smoke tests, but it is the next real systems
limit for longer-running workloads.

## Environment Note

The three-machine run still emits a Ray warning about Python patch-version
mismatch across nodes. The smoke passes, but the cluster environment should
eventually be normalized to reduce debugging noise.

## What This Round Achieved

This is the first point where the naive PIM resident-KV path is not merely
"hybrid-capable".

For the current OPT-125M functional setup, it now provides:

- fully DPU-backed resident-group placement
- repeated request lifecycle support
- explicit DPU slot cleanup
- end-to-end three-machine validation

## Suggested Next Step

The next implementation step should target real DPU-resident longevity rather
than only placement:

1. add reusable/free-list management for the per-DPU MRAM region allocator
2. expose per-DPU pool usage in debug output
3. run longer multi-request or dataset-driven experiments to find the true
   steady-state limit
