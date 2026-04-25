# PIM Baseline Findings

Date: 2026-04-25

## Scope

This note summarizes the most important findings from the current naive
resident-KV PIM baselines for `OPT-125M` on the three-machine cluster.

Workload:

- dataset: `dataset/humaneval.jsonl`
- `max_new_tokens = 2`
- `concurrency = 4`

## Main Comparison

### 1. 4 DPU + fp32 resident KV

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu4_legacy_identity.jsonl`

Observed:

- `avg_latency = 2.7851 s`
- `max_usage_ratio = 0.9917`
- `max_fallback_allocations = 28`

Interpretation:

- this is the constrained-capacity failure baseline
- the full-DPU path cannot hold the concurrent workload

### 2. 4 DPU + fp16 resident KV

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu4_fp16.jsonl`

Observed:

- `avg_latency = 2.8382 s`
- `max_usage_ratio = 0.5856`
- `max_fallback_allocations = 0`

Interpretation:

- compressing resident KV to `fp16` removes the overflow entirely
- this is the strongest optimization so far for the constrained `4 DPU` regime

### 3. 8 DPU + fp32 + balanced/rotated

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu8_balanced_rotated.jsonl`

Observed:

- `avg_latency = 3.1028 s`
- `max_usage_ratio = 0.6115`
- `max_fallback_allocations = 0`

Interpretation:

- increasing PIM resources plus better placement also recovers the full-DPU path
- but it is slower than the `4 DPU + fp16` path on this small workload

## Correctness Check

Comparing `4 DPU + fp32` and `4 DPU + fp16` on the same four requests:

- all `task_id` values match
- all sampled completions match
- `resident_shadow_max_abs_diff = 0.0`

For this workload, the `fp16` resident-KV path preserves the observed output
behavior of the original `fp32` path.

## Key Findings

1. The dominant bottleneck in the `4 DPU` setting is resident-KV capacity, not
   placement policy.

2. `fp16` resident-KV compression is enough to recover the full-DPU path in the
   constrained setting:
   - from `28` fallback allocations
   - down to `0`

3. `8 DPU + fp32` also recovers the full-DPU path, but at higher end-to-end
   latency on this small benchmark.

4. For the current setup, the most informative comparison set is:
   - `4 DPU + fp32`
   - `4 DPU + fp16`
   - `8 DPU + fp32 + balanced + rotated`

## Current Takeaway

There are now two distinct recovery routes for the disaggregated PIM-style
design:

- increase available PIM capacity
- reduce resident-KV footprint

In the current cluster and workload, reducing resident-KV footprint with
`fp16` is the better tradeoff.

## Real-Model Heavy Decode Check

We also ran a heavier real-model trace to test whether the earlier small-model
conclusions still hold.

Workload:

- dataset: `dataset/humaneval.jsonl`
- model: `Qwen-1_8B`
- first `4` samples
- `max_new_tokens = 8`
- `concurrency = 2`

### 4. 4 DPU + fp16 resident KV

Artifact:

- `artifacts/pim_allocator_trace_qwen1.8b_humaneval4_conc2_tok8_dpu4_fp16.jsonl`

Observed:

- `avg_latency = 22.4315 s`
- `max_usage_ratio = 1.0`
- `max_fallback_allocations = 128`

Interpretation:

- the earlier `4 DPU + fp16` success does not generalize to this heavier
  real-model setting
- resident-KV compression alone is no longer enough at this scale

### 5. 8 DPU + fp32 + balanced/rotated

Artifact:

- `artifacts/pim_allocator_trace_qwen1.8b_humaneval4_conc2_tok8_dpu8_fp32_balanced_rotated.jsonl`

Observed:

- `avg_latency = 24.0875 s`
- `max_usage_ratio = 1.0`
- `max_fallback_allocations = 256`

Interpretation:

- scaling to `8 DPU` without reducing resident-KV precision is still
  insufficient on this workload
- more PIM capacity by itself does not recover the full-resident path here

### 6. 8 DPU + fp16 + balanced/rotated

Artifact:

- `artifacts/pim_allocator_trace_qwen1.8b_humaneval4_conc2_tok8_dpu8_fp16_balanced_rotated.jsonl`

Observed:

- `avg_latency = 25.7025 s`
- `max_usage_ratio = 0.75`
- `max_fallback_allocations = 0`

Interpretation:

- combining scale-out with resident-KV compression recovers the full-resident
  path again
- for this heavier real-model setting, `8 DPU + fp16` is the first stable
  resident-KV baseline among the three tested options

## Updated Takeaway

The earlier lightweight result still matters, but the heavier Qwen trace
changes the practical baseline story:

1. capacity pressure grows much faster on a real model with longer decode than
   it did on `OPT-125M`

2. `4 DPU + fp16` is not a robust real-model baseline by itself

3. `8 DPU + fp32` is also not enough in this setting

4. the most reliable recovery route so far is:
   - more DPU capacity
   - plus smaller resident-KV representation

The new most informative real-model comparison set is:

- `4 DPU + fp16`
- `8 DPU + fp32 + balanced + rotated`
- `8 DPU + fp16 + balanced + rotated`

## Shared-Owner Scheme 2 Status

We then switched from the earlier dual-helper design to a shared-owner layout:

- resident KV and QK-mixed traffic now go through the same persistent
  `upmem_kvslot` helper
- `upmem_dot` initialization smoke is skipped in this mode, so the attention
  actor does not allocate a second full DPU set during startup

This removes the original structural conflict that made `512 DPU` impossible
when both `host_kvslot` and `host_qk` tried to reserve large DPU sets at the
same time.

### Transient allocation issue on `.7`

We briefly observed a misleading state where:

- `446 DPU` succeeded
- `447 DPU` failed

This turned out not to be a hardware or software design limit.

Root cause:

- a stale `host_qk --stdio --num-dpus 512` helper from an earlier experiment
  was still alive on `.7`
- that leftover process was occupying a large DPU set and made later
  `host_kvslot` allocations fail

After killing the stale helper:

- direct `host_kvslot` allocation at `512 DPU` succeeded again
- direct `host_kvslot` allocation at `1020 DPU` also succeeded

So the earlier `446` ceiling was a transient runtime-leak artifact, not the
real allocatable capacity of `.7`.

### 7. 446 DPU + fp16 + balanced/rotated + shared-owner

Artifact:

- `artifacts/pim_allocator_trace_qwen1.8b_humaneval4_conc2_tok8_dpu446_fp16_balanced_rotated_shared.jsonl`

Observed:

- `avg_latency = 25.3456 s`
- `max_latency = 27.1023 s`
- `max_usage_ratio = 0.25`
- `max_fallback_allocations = 0`
- `max_dpu_allocate_failures = 0`

Interpretation:

- the shared-owner Scheme 2 path is now functionally stable near the current
  large-DPU regime
- the run no longer fails from duplicate DPU reservation during actor startup
- it also avoids fallback allocations completely on this workload

### 8. 512 DPU + fp16 + balanced/rotated + shared-owner

Artifact:

- `artifacts/pim_allocator_trace_qwen1.8b_humaneval4_conc2_tok8_dpu512_fp16_balanced_rotated_shared.jsonl`

Observed:

- `avg_latency = 25.9202 s`
- `max_latency = 27.6668 s`
- `max_usage_ratio = 0.25`
- `max_fallback_allocations = 0`
- `max_dpu_allocate_failures = 0`

Interpretation:

- after removing the stale `host_qk` process, Scheme 2 also runs cleanly at
  the true `512 DPU` target
- this confirms the remaining blocker was runtime leakage, not the shared-owner
  refactor itself

### 9. 1020 DPU + fp16 + balanced/rotated + shared-owner

Artifact:

- `artifacts/pim_allocator_trace_qwen1.8b_humaneval4_conc2_tok8_dpu1020_fp16_balanced_rotated_shared.jsonl`

Observed:

- `avg_latency = 26.4178 s`
- `max_latency = 28.0796 s`
- `max_usage_ratio = 0.25`
- `max_fallback_allocations = 0`
- `max_dpu_allocate_failures = 0`

Interpretation:

- the shared-owner Scheme 2 path also runs correctly at the full `1020 DPU`
  scale on `.7`
- once stale helper leakage is removed, large-DPU scale-up no longer introduces
  a new functional failure on this workload

### Runtime leak fix

We also fixed the main software cause of the stale-helper issue:

- when the last DPU-backed resident slot is freed, `UpmemKVSlotStore` now
  closes its persistent `host_kvslot` helper automatically

This means later experiments should not inherit an orphaned helper that keeps a
large DPU set pinned after the request lifecycle has already ended.

### DPU-side QK milestone

We then replaced the temporary CPU-side `QK_BATCH` implementation inside
`host_kvslot` with a real DPU launch path:

- host partitions each query's key window across the allocated DPU set
- the kvslot DPU program computes dot products on DPU
- host collects per-DPU score shards and reassembles the final score matrix

Standalone correctness check:

- local `UpmemKVSlotStore.qk_scores_batch(...)` matched the host reference with
  `max_abs_diff = 0`

### 10. 512 DPU + fp16 + balanced/rotated + shared-owner + DPU-QK

Artifact:

- `artifacts/pim_allocator_trace_qwen1.8b_humaneval4_conc2_tok8_dpu512_fp16_balanced_rotated_shared_dpuqk.jsonl`

Observed:

- `avg_latency = 27.7712 s`
- `max_latency = 29.4477 s`
- `max_usage_ratio = 0.2422`
- `max_fallback_allocations = 0`

Interpretation:

- the first real DPU-side QK implementation preserves end-to-end correctness
  and stability on the real-model shared-owner path
- latency is slightly worse than the temporary CPU-inside-helper version, which
  is expected for a first functional DPU-QK path without optimization

### 11. 1020 DPU + fp16 + balanced/rotated + shared-owner + DPU-QK

Artifact:

- `artifacts/pim_allocator_trace_qwen1.8b_humaneval4_conc2_tok8_dpu1020_fp16_balanced_rotated_shared_dpuqk.jsonl`

Observed:

- `avg_latency = 29.3233 s`
- `max_latency = 31.2214 s`
- `max_usage_ratio = 0.125`
- `max_fallback_allocations = 0`

Interpretation:

- the real DPU-side QK path also remains stable at `1020 DPU`
- this confirms the new QK implementation scales functionally to the full
  machine-wide DPU count

## AV Batching Results

We then tested the first `AV` batching optimization on the small
resident-`AV` path.

What changed:

- `attention_backend` now issues one `weighted_value_sum_batch(...)` call per
  layer instead of one `weighted_value_sum(...)` call per head-group
- `host_kvslot` gained a new `KVSLOT_CMD_AV_BATCH` stdio command
- this first version only batches the Python-to-helper protocol
- inside the helper it still executes each `AV` item separately

### 12. 4 DPU + fp32 + balanced/rotated + resident AV

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_upmemav.jsonl`

Observed:

- `avg_latency = 1.8385 s`
- `avg_tpot = 1.6575 s`
- `avg_resident_av_ops = 30`
- `avg_resident_materialize_ops = 0`
- `max_resident_av_shadow_max_abs_diff = 0.000244140625`

### 13. 4 DPU + fp32 + balanced/rotated + resident AV + protocol batch

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_upmemav_batch1.jsonl`

Observed:

- `avg_latency = 1.8391 s`
- `avg_tpot = 1.6603 s`
- `avg_resident_av_ops = 30`
- `avg_resident_materialize_ops = 0`
- `max_resident_av_shadow_max_abs_diff = 0.000244140625`

Interpretation:

- correctness is unchanged
- protocol-level batching alone gives essentially no speedup on this workload
- the dominant cost is still inside the helper:
  - per-item `dpu_copy_to`
  - per-item `dpu_launch`
  - per-item `dpu_copy_from`

Current takeaway:

- this is still a useful negative result
- it shows the next optimization must happen inside `host_kvslot`, not only at
  the Python stdio boundary

### 14. 4 DPU + fp32 + balanced/rotated + resident AV + helper-side async batch

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_upmemav_batch2.jsonl`

Observed:

- `avg_latency = 1.8437 s`
- `avg_tpot = 1.6626 s`
- `avg_resident_av_ops = 30`
- `avg_resident_materialize_ops = 0`
- `max_resident_av_shadow_max_abs_diff = 0.000244140625`

What changed:

- `host_kvslot` now groups one `AV_BATCH` round by physical DPU
- each round launches at most one resident `AV` item per DPU asynchronously
- outputs are still written back in the original request order

Interpretation:

- correctness is still unchanged
- helper-internal per-DPU async launch also does not improve the small
  end-to-end trace
- on this workload, launch parallelism alone is still not enough to overcome:
  - per-item weight transfer
  - per-item context readback
  - helper-side packing / scheduling overhead

Updated takeaway:

- both batching attempts are now recorded as negative results on the current
  small trace
- the next promising direction is no longer finer launch orchestration
- the next promising direction is reducing data movement more fundamentally:
  - especially host-side resident-key assembly on mixed `QK`
  - and, if needed later, a deeper multi-slot DPU-kernel refactor for `AV`

## Resident-Slot Mixed-QK Baseline

We then changed the mixed-`QK` path itself.

What changed:

- mixed-`QK` no longer slices `keys[-window:]` from the CPU cache and repacks
  them into a generic `qk_scores_batch(...)` payload
- instead, the backend now maps each mixed head to its resident slot and asks
  `upmem_kvslot` to compute scores directly from the resident `k_cache`
- this is the first baseline where mixed-`QK` actually reads resident `K`
  instead of host-materialized key slices

### 15. Minimal three-machine correctness check with resident-slot mixed-QK

Observed:

- placement verification passed
- `qk_mixed_count = 24`
- `qk_mixed_last_max_abs_diff = 0.0`
- `resident_materialize_ops = 0`
- `resident_av_ops = 12`

Interpretation:

- the new resident-slot mixed-`QK` path is functionally active
- the current implementation preserves the observed mixed-`QK` result exactly
  on the minimal cluster check

### 16. 4 DPU + fp32 + balanced/rotated + resident-slot mixed-QK + resident AV

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_upmemav.jsonl`

Observed:

- `avg_latency = 1.9812 s`
- `max_latency = 2.4791 s`
- `max_usage_ratio = 0.31640625`
- `max_fallback_allocations = 0`

Comparison against the earlier resident-AV baseline:

- previous resident-AV path:
  - `avg_latency = 1.8385 s`
- resident-slot mixed-`QK` baseline:
  - `avg_latency = 1.9812 s`

Interpretation:

- removing host-side mixed-`QK` key repacking is not enough by itself to
  improve small-trace latency
- the current resident-slot mixed-`QK` implementation is still naive and likely
  paying too much per-head/per-slot kernel overhead
- this is still an important milestone because it moves mixed-`QK` onto the
  real resident-`K` path and gives us a truthful baseline for the next round

### 17. 4 DPU + fp32 + balanced/rotated + grouped resident-slot mixed-QK + resident AV

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_upmemav.jsonl`

Observed:

- `avg_latency = 1.9548 s`
- `avg_tpot = 1.7792 s`

Comparison:

- ungrouped resident-slot mixed-`QK` baseline:
  - `avg_latency = 1.9812 s`
  - `avg_tpot = 1.7980 s`
- grouped resident-slot mixed-`QK`:
  - `avg_latency = 1.9548 s`
  - `avg_tpot = 1.7792 s`
- original resident-AV baseline without resident-slot mixed-`QK`:
  - `avg_latency = 1.8385 s`
  - `avg_tpot = 1.6575 s`

Interpretation:

- grouping multiple mixed heads from the same slot does recover a small part of
  the regression introduced by the first resident-slot mixed-`QK` baseline
- but the current grouped version is still slower than the earlier
  resident-AV-only path
- this suggests the direction is right, but the remaining cost is now deeper
  than Python/helper framing:
  - helper still launches one kernel per head inside each grouped slot item
  - cross-slot execution is still serialized

Updated takeaway:

- the grouped resident-slot mixed-`QK` path is now the stronger baseline than
  the earlier ungrouped resident-slot version
- the next likely win requires either:
  - true multi-head `QK` inside one slot/kernel
  - or better overlap across multiple slot groups / DPUs

### DPU-QK active-set follow-up

We also tried a host-side launch optimization that capped each QK batch to at
most `16` active DPUs instead of launching the whole persistent set.

Artifacts:

- `artifacts/pim_allocator_trace_qwen1.8b_humaneval4_conc2_tok8_dpu512_fp16_balanced_rotated_shared_dpuqk_active16.jsonl`
- `artifacts/pim_allocator_trace_qwen1.8b_humaneval4_conc2_tok8_dpu1020_fp16_balanced_rotated_shared_dpuqk_active16.jsonl`

Observed:

- `512 DPU`: `avg_latency = 27.8377 s`
- `1020 DPU`: `avg_latency = 29.3965 s`

Interpretation:

- this did not improve over the earlier DPU-QK baselines
- the main DPU-QK overhead is not simply caused by launching too many DPUs per
  batch
- this should be recorded as a negative optimization result, not a mainline
  direction

### DPU-side AV milestone

The shared-owner `upmem_kvslot` path now also has a first real DPU-side `AV`
implementation over resident `V`.

What changed:

- a new `KVSLOT_CMD_AV` protocol path was added to `host_kvslot`
- the kvslot DPU program now supports weighted value aggregation on resident
  `V`
- the attention backend now calls resident `AV` directly on the
  `upmem_kvslot` path instead of materializing `V` back to host

Standalone correctness check on `.7`:

- `fp32`: `max_abs_diff = 0.0`
- `fp16`: `max_abs_diff = 0.0`

Interpretation:

- resident `AV` is now functionally correct for both currently supported KV
  formats
- the implementation is still a naive kernel aimed at correctness first
- architecturally, the system has now crossed the key milestone where both
  `QK` and `AV` have real DPU-side execution paths

### Small end-to-end trace after resident-AV integration

We also ran a small cluster trace to compare the new `upmem_kvslot + resident
AV` path against the existing host-store path on the same workload.

Workload:

- model: `OPT-125M`
- dataset: first `4` samples from `dataset/humaneval.jsonl`
- `max_new_tokens = 2`
- `concurrency = 1`
- `4 DPU`, `fp32`, balanced/rotated grouping

Artifacts:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_upmemav.jsonl`
- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_hoststore.jsonl`

Observed:

- `upmem_kvslot + resident AV`
  - `avg_latency = 1.8385 s`
  - `avg_tpot = 1.6575 s`
  - `avg resident_materialize_ops = 0`
  - `avg resident_av_ops = 30`
  - `max resident_av_shadow_max_abs_diff = 0.000244140625`
- `host store`
  - `avg_latency = 0.8360 s`
  - `avg_tpot = 0.6555 s`
  - `avg resident_materialize_ops = 30`
  - `avg resident_av_ops = 0`

Interpretation:

- the new resident-AV path is definitely active in real decode execution:
  - materialization dropped to `0`
  - resident `AV` calls replaced those host-side materializations
- correctness remains intact within the current naive path
- performance is still much worse than the host-store path on this small
  workload, so the current bottleneck is now clearly in the naive DPU-side
  `AV` path and its associated host/DPU traffic, not in allocator fallback

## Updated Practical Takeaway

There are now two separate conclusions:

1. Scheme 2 fixes the software ownership problem for large-DPU shared
   resident-KV/QK execution.

2. Large-DPU failures after the refactor should first be checked for stale
   helper processes before being interpreted as architectural limits.

3. With the stale-helper leak fixed, the current real-model shared-owner path
   is now verified at both:
   - `512 DPU`
   - `1020 DPU`

4. QK is no longer just "logically shared-owner"; it now executes on DPU in
   the kvslot path, giving us a true PIM-side attention-compute baseline to
   optimize next.

### 18. 4 DPU + fp32 + balanced/rotated + grouped resident-slot mixed-QK + true multi-head kernel + resident AV

We then completed the next mixed-`QK` step:

- one grouped slot item now executes as one real multi-head `QK` kernel on DPU
- the helper no longer launches one slot-`QK` kernel per head inside the
  grouped item
- minimal three-machine verification still passed with
  `qk_mixed_last_max_abs_diff = 0.0`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_upmemav.jsonl`

Observed at the time:

- `avg_latency = 1.3796 s`
- `avg_tpot = 1.1931 s`

Comparison on the same workload:

- original resident-AV baseline:
  - `avg_latency = 1.8385 s`
  - `avg_tpot = 1.6575 s`
- grouped resident-slot mixed-`QK` before true multi-head kernel:
  - `avg_latency = 1.9548 s`
  - `avg_tpot = 1.7792 s`
- grouped resident-slot mixed-`QK` with true multi-head kernel:
  - `avg_latency = 1.3796 s`
  - `avg_tpot = 1.1931 s`

Important correction:

- this result was recorded before we exposed a grouped multi-head
  `QK_SLOT_BATCH` score-layout bug
- later debugging showed the DPU wrote grouped scores with a per-head stride
  while the helper read them back as one compact contiguous matrix
- because of that mismatch, later head rows in real decode could return
  garbage / `NaN`, so this earlier fast result should not be treated as a
  trustworthy baseline

Updated interpretation:

- the direction was still correct: helper/kernel execution granularity really
  matters
- but the absolute performance number here is now considered invalid for
  comparison purposes

Updated takeaway:

- keep the architectural conclusion
- discard this exact numeric result in later comparisons

### 19. 4 DPU + fp32 + balanced/rotated + grouped resident-slot mixed-QK + true multi-head kernel + helper-side cross-DPU async overlap + resident AV

We then tried one more helper-side scheduling step on top of the new
multi-head slot kernel:

- `QK_SLOT_BATCH` now preloads all grouped slot items
- it launches at most one grouped item per physical DPU asynchronously in each
  round
- responses are still written back in original item order

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_async_upmemav.jsonl`

Observed:

- `avg_latency = 1.3810 s`

Comparison against the immediately previous version:

- grouped resident-slot mixed-`QK` + true multi-head kernel:
  - `avg_latency = 1.3796 s`
- plus helper-side cross-DPU async overlap:
  - `avg_latency = 1.3810 s`

Interpretation:

- this is effectively no improvement on the current small trace
- after the true multi-head kernel fix, the next bottleneck is no longer
  dominated by simple slot-level host/helper serialization
- this should be recorded as another useful negative result rather than pushed
  further right now

### 20. 4 DPU + fp32 + balanced/rotated + grouped resident-slot mixed-QK + true multi-head kernel + per-row query WRAM staging + resident AV

We then moved one level deeper into the `QK` kernel itself.

What changed:

- each grouped slot `QK` head row now stages its query row from MRAM into WRAM
  once before scanning the decode window
- the token loop no longer re-reads the same query elements from MRAM for
  every token
- the slot-`QK` reduction also now stays in float space instead of converting
  through the earlier scaled-int accumulation path

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_rowstage_upmemav.jsonl`

Observed at the time:

- `avg_latency = 1.1986 s`
- `avg_tpot = 1.0118 s`

Comparison on the same workload:

- grouped resident-slot mixed-`QK` + true multi-head kernel:
  - `avg_latency = 1.3796 s`
  - `avg_tpot = 1.1931 s`
- plus helper-side cross-DPU async overlap:
  - `avg_latency = 1.3810 s`
  - `avg_tpot = 1.1994 s`
- plus per-row query WRAM staging inside the `QK` kernel:
  - `avg_latency = 1.1986 s`
  - `avg_tpot = 1.0118 s`

Important correction:

- this result was also produced before the grouped multi-head score-layout bug
  was root-caused
- after fixing the DPU-write / helper-readback mismatch, the first correct
  row-staging reruns moved back to:
  - `avg_latency = 1.9139 s`
  - `avg_tpot = 1.7329 s`
  - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_rowstage_fixedreadback_upmemav.jsonl`
- a second compact-readback rerun was similar:
  - `avg_latency = 1.9378 s`
  - `avg_tpot = 1.7552 s`
  - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_rowstage_compactreadback_upmemav.jsonl`

Updated interpretation:

- the earlier `~1.20 s` row-stage result was not trustworthy
- the real post-fix row-stage baseline is slightly worse than the original
  resident-AV baseline, so more kernel work was still needed

Updated takeaway:

- query-row staging is still a reasonable kernel optimization direction
- but the corrected baseline after the readback fix is no longer a clear win
- any later faster result must be compared only against the corrected
  post-fix numbers above

### 21. Grouped multi-head readback bug fix + token-parallel compact-score kernel

We then root-caused the real correctness failure behind the suspect earlier
numbers.

Root cause:

- grouped multi-head slot `QK` on DPU wrote `qk_slot_scores_bits` with a
  per-head stride
- the host helper read those bytes back as one compact contiguous score
  matrix
- in end-to-end decode this could make later head rows become all `NaN` even
  when resident materialized keys exactly matched CPU keys

How we verified it:

- direct small `qk_slot_scores_batch()` tests were still fine
- real decode mixed-`QK` checks showed `layer0/head1` DPU scores turning into
  all `NaN`
- dumped failing cases showed:
  - finite CPU scores
  - `NaN` DPU scores
  - `resident_window_key_diff_max = 0.0`
- this isolated the bug to grouped multi-head score output / readback layout

Fix:

- DPU grouped slot kernel now writes scores in compact layout
- host helper now reads back the same compact layout consistently
- mixed-`QK` verification returned to exact agreement:
  - `qk_mixed_last_max_abs_diff = 0.0`
  - `qk_mixed_last_head_diffs = [0.0, 0.0]`

We then replaced the slower correct row-stage path with a safer token-parallel
kernel:

- each tasklet handles different `token_offset`s
- each tasklet computes the full dot product for its assigned tokens
- the kernel keeps the compact grouped score layout
- the per-token barrier / partial-reduction path is removed

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_tokenparallel_compactreadback_upmemav.jsonl`

Observed:

- `avg_latency = 1.8203 s`
- `avg_tpot = 1.6384 s`

Comparison against trustworthy baselines:

- original resident-AV baseline:
  - `avg_latency = 1.8385 s`
  - `avg_tpot = 1.6575 s`
- corrected row-stage baseline:
  - `avg_latency = 1.9139 s`
  - `avg_tpot = 1.7329 s`
- token-parallel compact-score grouped mixed-`QK`:
  - `avg_latency = 1.8203 s`
  - `avg_tpot = 1.6384 s`

Interpretation:

- this is the first trustworthy post-fix mixed-`QK` result that recovers the
  row-stage regression
- it slightly beats the original resident-AV-only baseline on this small
  trace while keeping exact mixed-`QK` correctness
- the useful optimization here came from changing tasklet work partitioning
  inside the kernel, not from helper-side overlap

Current trustworthy takeaway:

- discard the earlier `1.3796 s` and `1.1986 s` results as suspect
- keep the compact-layout fix
- treat the token-parallel compact-score kernel as the current correct
  small-trace mixed-`QK` baseline
