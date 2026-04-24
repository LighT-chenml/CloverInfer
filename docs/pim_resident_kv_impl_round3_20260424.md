# Resident-KV PIM Implementation Round 3

Date: 2026-04-24

## Goal

Move the host-side resident-KV path one step closer to a real DPU-managed
storage model:

- replace dynamic `torch.cat` growth with preallocated slots
- append using logical `seq_len` pointers
- keep the materialization and CPU shadow path unchanged for validation

This round makes the host-side resident layout behave more like a future
fixed-capacity DPU KV region.

## Code Changes

Updated file:

- `src/core/attention_backend.py`

Main changes:

- resident head-group partitions are now allocated as fixed-capacity tensors
  with shape:
  - `[capacity, group_heads, head_dim]`
- initial prefill KV is copied into the front of the slot
- decode append writes into:
  - `group.k_cache[group.seq_len:expected_seq_len]`
  - `group.v_cache[group.seq_len:expected_seq_len]`
- `_grow_group_capacity` was added for capacity expansion when the requested
  sequence length exceeds current capacity
- materialization now slices only the live prefix:
  - `group.k_cache[:group.seq_len]`
  - `group.v_cache[:group.seq_len]`

## Why This Matters

The previous round already had resident host-side partitions, but their
behavior still resembled dynamic host tensors.

This round changes the storage semantics to be closer to the intended PIM-side
model:

- storage has explicit capacity
- writes are append-by-offset
- logical live length is separate from allocated size

That is much closer to how a DPU-managed KV region will need to behave.

## Validation

### `.7` Direct Backend Smoke

Observed result:

```python
{
  'seq_len': 3,
  'append_ops': 8,
  'materialize_ops': 2,
  'shadow_diff': 0.0,
  'resident_shape': [8, 1, 8],
  'resident_capacity': 8,
  'resident_seq_len': 4
}
```

Interpretation:

- the slot is preallocated to capacity `8`
- the logical sequence length advances only to `4`
- resident storage now has explicit capacity semantics
- the resident path still matches the CPU shadow exactly

### Three-Machine End-to-End Smoke

Regression script:

- `tests/verify_cluster_placement.py`

Validated command:

```bash
/home/cml/anaconda3/envs/clover_infer/bin/python tests/verify_cluster_placement.py \
  --address 192.168.123.4:26379 \
  --model /home/cml/CloverInfer/model/opt-125m \
  --attention-backend pim_naive \
  --no-pim-qk-mixed-enabled \
  --pim-num-dpus 4 \
  --pim-length 8 \
  --max-new-tokens 2
```

Observed resident-path signals:

- `resident_append_ops = 48`
- `resident_materialize_ops = 12`
- `resident_shadow_max_abs_diff = 0.0`

Interpretation:

- the slot-based resident path is active in the real three-machine decode flow
- resident append/materialize behavior remains correct after the storage-model
  change
- the shadow oracle still matches exactly

## Current Status

At this point, the resident-KV work has progressed through three stages:

1. metadata lifecycle only
2. resident host-side data path
3. slot-based resident host-side storage model

This means the software model now has:

- request-local placement state
- head-group partitions
- append-by-offset semantics
- explicit capacity tracking

These are the right prerequisites for mapping the same model onto DPU-managed
regions.

## Next Step

The next meaningful implementation step is to introduce a lower-level storage
boundary that no longer treats resident slots as ordinary host tensors.

That can begin with one of the following:

1. a simulated DPU-region abstraction on the host side
2. a UPMEM host/runtime extension that initializes per-group slots once and
   updates them incrementally

Either way, the key transition is:

- from "host tensors that imitate DPU slots"
- to "slots that are managed through a DPU-oriented storage/runtime interface"
