# Resident-KV PIM Implementation Round 2

Date: 2026-04-24

## Goal

Convert the resident-KV work from "metadata only" to "metadata drives the host
data path":

- maintain persistent head-group KV partitions on the attention backend
- append incremental decode KV into those partitions
- materialize the current layer KV from resident partitions
- keep the existing CPU cache as the shadow oracle

This round is still host-side resident KV, not yet DPU-resident KV.

## Code Changes

Updated file:

- `src/core/attention_backend.py`

Main additions:

- `HeadGroupState` now stores:
  - `k_cache`
  - `v_cache`
- backend debug/state fields:
  - `resident_compute_enabled`
  - `resident_append_ops`
  - `resident_materialize_ops`
  - `resident_shadow_max_abs_diff`

New logic:

- `_build_head_groups` now creates persistent per-group host KV partitions
- `_append_resident_kv` appends incremental `k_new/v_new` into those partitions
- `_materialize_layer_kv` reconstructs full-layer KV from resident partitions
- `_update_resident_shadow_diff` compares resident materialization against the
  existing CPU oracle cache

Decode-path change:

- `decode_layer` now:
  1. appends to resident partitions
  2. updates the CPU shadow cache
  3. materializes KV from resident partitions
  4. computes attention from the resident-materialized KV

## Design Boundary

This round still does **not**:

- keep KV inside DPU memory
- run DPU-side resident append
- query DPU-resident KV directly

But it does create a real resident-KV execution model on the host side, which
is the intended staging area for later DPU residency.

## Validation

### Local Syntax Check

Command:

```bash
/home/cml/anaconda3/envs/clover_infer/bin/python -m py_compile src/core/attention_backend.py
```

Result:

- passed

### `.7` Resident-Path Smoke

The updated backend was synced to `192.168.123.7` and tested directly with the
real `clover_infer` environment.

Observed result:

```python
{
  'seq_len': 3,
  'mid_count': 1,
  'mid_context_len': 4,
  'append_ops': 8,
  'materialize_ops': 2,
  'shadow_diff': 0.0,
  'end_count': 0,
  'last_freed': 'req0'
}
```

Interpretation:

- resident head-group partitions are receiving decode appends
- decode uses resident-materialized KV rather than only the CPU shadow cache
- resident-materialized KV matches the CPU oracle exactly on the smoke case

### Three-Machine End-to-End Smoke

Regression script:

- `tests/verify_cluster_placement.py`

This script was updated to assert resident-path fields for `pim_naive` after a
real decode request:

- `resident_metadata_enabled`
- `resident_compute_enabled`
- `resident_append_ops > 0`
- `resident_materialize_ops > 0`
- `resident_shadow_max_abs_diff == 0.0`
- `resident_last_freed_request_id` is populated
- `resident_request_count == 0` after request cleanup

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

Observed end-to-end resident debug signals:

- `resident_append_ops = 48`
- `resident_materialize_ops = 12`
- `resident_shadow_max_abs_diff = 0.0`

Interpretation:

- the resident host-side KV path is active inside the real three-machine decode
  flow
- request lifecycle cleanup remains correct after generation
- the resident materialization still matches the CPU oracle exactly in the
  end-to-end path

## Why This Matters

This is the first point where the resident-KV design affects real computation,
not just debug state.

Before this round:

- resident state existed only as lifecycle metadata

After this round:

- the backend has a persistent resident-style KV representation that drives the
  actual layer computation path

That makes the next transition much cleaner:

- swap host-resident partitions for DPU-resident partitions
- keep the same request/layer/head-group state model

## Next Step

The next implementation step should validate this resident host-side path in a
more realistic end-to-end flow, then begin introducing a lower-level storage
boundary that can map onto DPU-managed KV regions.
