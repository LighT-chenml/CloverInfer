# Resident-KV PIM Implementation Round 4

Date: 2026-04-24

## Goal

Introduce an explicit storage/runtime boundary for resident KV, so the
attention backend no longer directly owns the resident tensor storage logic.

This is the key abstraction step before a future DPU-backed KV store.

## Code Changes

New file:

- `src/core/resident_kv_store.py`

New interface:

- `ResidentKVStore`

First implementation:

- `HostResidentKVStore`

Provided operations:

- `allocate_group`
- `append_group`
- `materialize_group`
- `slot_debug`
- `free_group`
- `get_debug_info`

Backend integration:

- `src/core/attention_backend.py` now uses `HostResidentKVStore`
- `HeadGroupState` keeps slot identifiers and metadata, but not the backing
  tensors directly
- request free now reclaims resident store groups explicitly
- backend debug now exposes:
  - `resident_store_debug`

## Why This Matters

Before this round, the resident-KV path was already slot-based, but the storage
logic still lived inside the attention backend.

After this round:

- the attention backend owns request lifecycle and attention logic
- the resident store owns slot allocation, append, materialization, and free

This is exactly the boundary we need before introducing a DPU-oriented store.

## Validation

### `.7` Direct Store Smoke

Observed result:

```python
{
  'shadow_diff': 0.0,
  'slot_backend': 'host_slot_store',
  'live_slots_mid': 8,
  'append_ops_store': 8,
  'materialize_ops_store': 8,
  'peak_bytes': 4096,
  'slot_shape': [8, 1, 8],
  'live_slots_end': 0
}
```

Interpretation:

- the backend is using the store abstraction
- slot allocation/append/materialization/free all pass through the store
- live slot accounting behaves correctly
- resident results still match the CPU oracle exactly

### Three-Machine End-to-End Smoke

Regression script:

- `tests/verify_cluster_placement.py`

Observed store-related debug in the real three-machine path:

- `resident_store_debug.backend = host_slot_store`
- `resident_store_debug.total_allocations = 48`
- `resident_store_debug.append_ops = 48`
- `resident_store_debug.materialize_ops = 48`
- `resident_store_debug.peak_allocated_bytes = 294912`
- `resident_shadow_max_abs_diff = 0.0`

Interpretation:

- the storage/runtime boundary is active inside the real distributed decode path
- store-level accounting is now visible in benchmark/debug outputs
- request cleanup returns `live_slots` to zero

## Current Status

The resident-KV work is now at a stronger abstraction point:

1. metadata lifecycle
2. resident host-side data path
3. slot-based resident host-side storage
4. explicit storage/runtime interface

This means the next step can focus on replacing the store backend rather than
rewriting attention backend control flow.

## Next Step

The most meaningful next move is to add a second store implementation that is
more DPU-oriented.

A practical first version could be:

- `SimulatedDPUResidentKVStore`

Properties:

- keeps the same `ResidentKVStore` interface
- models DPU slot IDs and region ownership explicitly
- optionally routes group operations through a UPMEM host helper instead of
  directly touching host tensors

That would let us evolve from:

- "host store with DPU-like semantics"

to:

- "store API that actually behaves like a DPU-managed runtime boundary"
