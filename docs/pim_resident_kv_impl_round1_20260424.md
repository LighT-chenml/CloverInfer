# Resident-KV PIM Implementation Round 1

Date: 2026-04-24

## Goal

Land the first code step for the resident-KV PIM baseline:

- add persistent request/layer/head-group metadata on the attention backend
- keep the current helper-based naive PIM execution path intact
- prepare the backend for later resident-KV append and placement logic

This round is intentionally a state-model round, not a kernel round.

## Code Changes

Updated file:

- `src/core/attention_backend.py`

Added:

- `HeadGroupState`
- `LayerState`
- `RequestState`

Added backend state:

- `resident_metadata_enabled`
- `request_states`
- `last_freed_request_id`

Added helper methods:

- `_build_head_groups`
- `_build_request_state`
- `_append_layer_metadata`
- `_summarize_request_state`

Lifecycle changes:

- `init_request` now builds persistent request metadata
- `decode_layer` now updates resident metadata append state before the current
  CPU/reference attention path runs
- `free_request` now reclaims resident metadata separately from the DPU runtime
- `get_debug_info` now exposes resident-metadata debug fields

## Design Boundary

This round does **not** yet:

- keep KV physically resident inside DPU memory
- change the current helper-based QK execution contract
- implement DPU-side append
- implement resident-KV query routing

It only makes the request-state model explicit inside the backend.

## Validation

### Local Syntax Check

Command:

```bash
/home/cml/anaconda3/envs/clover_infer/bin/python -m py_compile src/core/attention_backend.py
```

Result:

- passed

### `.7` Metadata Lifecycle Smoke

The updated backend file was synced to `192.168.123.7` and a small direct
backend smoke was run with the real `clover_infer` environment on that node.

Validated sequence:

1. construct `PimNaiveAttentionBackend`
2. `init_request`
3. run two `decode_layer` calls for a two-layer toy request
4. inspect debug info
5. `free_request`
6. inspect debug info again

Observed result:

```python
{
  'seq_len': 3,
  'mid_count': 1,
  'mid_context_len': 4,
  'mid_layer0_seq': 4,
  'end_count': 0,
  'last_freed': 'req0'
}
```

Interpretation:

- request metadata is created correctly
- per-layer append metadata advances with decode
- request metadata is reclaimed correctly on free

## Why This Matters

This round gives the resident-KV design a concrete place to live in code.

Before this round:

- the backend had no explicit persistent placement model

After this round:

- the backend can track request-local placement and append state independently
  from the existing CPU oracle cache

That separation is necessary before:

- resident DPU storage
- incremental DPU append
- head-group routing

can be implemented safely.

## Next Step

The next implementation step should focus on host-side resident-KV behavior:

1. add explicit host-side per-group KV partitions derived from the request
   metadata
2. stop treating the request state as only a debug structure
3. prepare the decode path for:
   - append incremental `k_new/v_new`
   - query selected head-groups
   - avoid rebuilding ad hoc slices from the full CPU cache
