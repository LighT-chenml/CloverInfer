# Resident-KV PIM Implementation Round 5

Date: 2026-04-24

## Goal

Bring the first DPU-backed resident-KV store into the real backend path.

This round does not yet move all resident KV onto DPUs. Instead, it introduces
the first mixed store:

- use DPU-backed slots where the current helper can support them
- fall back to the existing host-backed slot store otherwise

That is enough to prove the backend/store abstraction can drive a real
DPU-backed resident path.

## Code Changes

### New/Updated Store Path

Updated file:

- `src/core/resident_kv_store.py`

Added:

- `_KVSlotHelperClient`
- `UpmemKVSlotStore`

Behavior:

- `UpmemKVSlotStore` talks to `src/pim/upmem_kvslot/build/host_kvslot`
- DPU-backed groups are stored via the stdio helper
- unsupported groups fall back to `HostResidentKVStore`

Current allocation policy:

- assign a unique logical slot id per request/layer/group
- use DPU-backed slots while `slot_id < num_dpus`
- route later slots to host fallback

### Backend Wiring

Updated files:

- `src/core/attention_backend.py`
- `src/core/config.py`
- `src/core/scheduler.py`

Added config/backend option:

- `resident_store_backend = "host" | "upmem_kvslot"`

### UPMEM KV Slot Helper

Updated files:

- `src/pim/upmem_kvslot/host_kvslot.c`
- `src/pim/upmem_kvslot/dpu_kvslot.c`
- `src/pim/upmem_kvslot/common.h`

The helper now:

- manages slot state through a persistent stdio process
- writes payloads into DPU MRAM symbols
- reads resident KV back from DPU MRAM

## Important Constraint

The current helper still supports only a very small DPU slot model:

- effectively one active DPU-backed slot per DPU

That is why the current store is hybrid rather than fully DPU-backed.

This is acceptable for the current round because the implementation goal is:

- prove that the resident store abstraction can already drive a real DPU-backed
  KV path

not yet:

- move the entire model's resident KV footprint to DPU storage

## Validation

### Direct Backend Smoke on `.7`

Constructed:

- `PimNaiveAttentionBackend(..., resident_store_backend="upmem_kvslot")`

Observed result:

```python
{
  'shadow_diff': 0.0,
  'store_backend': 'upmem_kvslot_store',
  'dpu_allocations': 4,
  'fallback_allocations': 4,
  'live_slots_end': 0
}
```

Interpretation:

- the backend can use the DPU-backed store path directly
- some resident groups are truly routed through the DPU helper
- CPU shadow correctness is preserved exactly on the smoke case

### Three-Machine End-to-End Smoke

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

Observed store debug:

- `resident_store_debug.backend = upmem_kvslot_store`
- `resident_store_debug.dpu_allocations = 4`
- `resident_store_debug.fallback_allocations = 44`
- `resident_store_debug.helper_restarts = 1`
- `resident_shadow_max_abs_diff = 0.0`

Interpretation:

- the real three-machine decode flow is now using a DPU-backed resident-KV
  store for a subset of groups
- the remaining groups still go through the host fallback store
- the hybrid path remains numerically aligned with the CPU oracle

## What This Means

This is the first point where the system is no longer only "preparing for KV on
DPUs".

It now has:

- a real DPU-backed resident store implementation
- a backend-level switch to enable it
- a distributed end-to-end smoke demonstrating that it participates in the
  decode flow

That is a meaningful architectural transition.

## Next Step

The next bottleneck is no longer "can we route any resident KV through a DPU
store at all?"

It is:

- how to extend the helper/runtime from the current tiny slot model toward a
  multi-slot-per-DPU resident layout

That work should focus on:

1. multi-slot addressing inside the helper/runtime
2. a clearer DPU-region layout for multiple resident groups
3. reducing the current heavy dependence on host fallback
