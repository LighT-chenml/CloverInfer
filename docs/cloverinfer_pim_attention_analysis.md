# CloverInfer PIM Attention Analysis

## Scope

This document summarizes the performance exploration done while building
`CloverInfer` on top of `disagg_pim_naive`.

Important constraint:

- `disagg_pim_naive` remains the comparison baseline.
- CloverInfer logic stays isolated in `src/core/clover_attention_backend.py`.
- Baseline `PimNaiveAttentionBackend` semantics are not changed.

## Implementation Separation

- Baseline path:
  - `src/core/attention_backend.py`
- CloverInfer path:
  - `src/core/clover_attention_backend.py`
- Resident KV runtime:
  - `src/core/resident_kv_store.py`

## Main Findings

### 1. Initial long-context `pim_full` was fake-PIM

Early long-context runs appeared to use PIM, but profiling showed most slots
fell back to host memory because a single DPU slot only supports
`capacity <= 256`.

Relevant limits:

- `src/core/resident_kv_store.py`
- `src/pim/upmem_kvslot/common.h`

Key evidence:

- Long-context profiling showed `dpu_allocations = 0` and large
  `fallback_allocations`.
- Increasing DPU count did not help because long contexts were not actually
  resident on DPU.

### 2. Segmented resident KV fixed correctness, but exposed the real bottleneck

We added segmented logical KV slots so one logical group can be split into
multiple physical `<=256` DPU slots.

This made long-context exact PIM execution possible, but it was initially very
slow because one logical attention group exploded into many small helper calls.

### 3. The first real bottleneck was segment fanout and helper orchestration

Before batching fixes, segmented decode caused repeated recursive calls into:

- `qk_slot_scores_batch`
- `weighted_value_sum_batch`

This produced far too many helper launches per decode layer.

### 4. Flattened segmented batching fixed the call explosion

We changed CloverInfer segmented execution to flatten all segment-level work
across the batch before invoking the helper.

Result:

- DPU calls per decode layer dropped close to the theoretical lower bound.
- This made the segmented exact path substantially cleaner and more scalable.

Representative artifact:

- `artifacts/clover_pim_full_d8_ctx1024_gen8_segmented_batchfix.jsonl`

For `ctx=1024`, theory is:

- `5` segments per logical group
- `8` groups per layer
- about `40` segment items per decode layer

Observed after batching fix:

- `qk_items_per_layer ~= 37.9`
- `av_items_per_layer ~= 35.0`

This means the recursive orchestration overhead was mostly removed.

### 5. Experimental segment-local fused context is slower than segmented QK+AV

We then implemented an experimental partial-fused segmented path:

- each segment performs `QK + local softmax + local context` on DPU
- host merges segment outputs with exact log-sum-exp composition

Relevant artifacts:

- `artifacts/clover_pim_full_d8_ctx1024_gen8_partialfused_retry2.jsonl`
- `artifacts/clover_pim_full_d8_ctx1024_gen8_partialfused_ctxbatch.jsonl`

This path works functionally, but is slower than the segmented batched QK+AV
path.

Observed on `prompt_token_length=1024`, `max_new_tokens=8`, `pim_num_dpus=8`:

- segmented batch path latency: about `81.5s`
- partial-fused path latency: about `142.0s`

Even after batching context fetches on the host side, latency stayed nearly
unchanged.

## Current Diagnosis

The current first-order bottleneck is no longer:

- host fallback
- insufficient DPU count
- segment call explosion
- per-item context copy on the host side

The current first-order bottleneck is:

- the `context_fused` DPU kernel / helper round itself

Empirically:

- segmented batched `QK + AV` costs about `12.2 ms / DPU item`
- experimental partial-fused costs about `44.9 ms / DPU item`

So the current fused implementation is about `3.7x` slower per item.

## Practical Conclusion

For CloverInfer today, the best PIM path is:

- resident segmented long-context KV
- flattened batched segmented QK
- flattened batched segmented AV
- exact behavior preserved

The experimental fused path should not be the default fast path.

## Default Policy In Repo

`clover_pim_attention_enabled=True` still enables CloverInfer's independent
PIM-first path, but the experimental context-fused route is gated behind:

- `clover_pim_context_fused_experimental_enabled`

Default:

- `False`

So current `pim_full` runs use the faster segmented batch implementation by
default.

## Recommended Next Steps

### Near-term

- Keep the segmented batch path as CloverInfer's default PIM implementation.
- Focus performance work on:
  - decode batching
  - cross-request parallelism
  - group/head layout
  - reducing residual host fallback

### Longer-term

Only revisit fused segmented attention if we are ready to redesign the DPU
kernel itself, especially:

- context accumulation memory access pattern
- local softmax implementation cost
- fused kernel launch/transfer structure

Incremental helper-side tweaks are unlikely to close the current gap.

## Key Artifacts

- Fake-PIM / fallback diagnosis:
  - `artifacts/clover_pim_full_d8_long_ctx1024_gen32_segmented_smoke.jsonl`
- Segmented batch fixed path:
  - `artifacts/clover_pim_full_d8_ctx512_gen8_segmented_batchfix.jsonl`
  - `artifacts/clover_pim_full_d8_ctx1024_gen8_segmented_batchfix.jsonl`
- Experimental partial-fused path:
  - `artifacts/clover_pim_full_d8_ctx1024_gen8_partialfused_retry2.jsonl`
  - `artifacts/clover_pim_full_d8_ctx1024_gen8_partialfused_ctxbatch.jsonl`
