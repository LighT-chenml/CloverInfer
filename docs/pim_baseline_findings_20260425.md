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

## Full-QK Phase 1 Bring-Up

Date: 2026-04-26

We pushed the decode-time QK path further inward so that resident-slot batched
QK is now the primary path behind a config flag:

- `pim_qk_full_enabled = True`
- `pim_qk_mixed_enabled = False`

This is still not full-PIM attention because:

- softmax remains on the host
- AV still returns through the existing resident-AV path

but it is the first implementation step where the main decode QK path no
longer depends on host-side full score construction.

### 9. Single-request cluster correctness

Workload:

- cluster: `.3` prefill GPU, `.4` dense GPU, `.7` attention CPU+UPMEM
- model: `OPT-125M`
- `max_new_tokens = 2`
- `256 DPU`

Observed:

- 3-machine placement check passed
- generation succeeded
- `qk_full_shadow_max_abs_diff = 0.0`
- `resident_av_shadow_max_abs_diff = 0.0`

Interpretation:

- the new full-QK path is correctly wired end-to-end for a small decode check

### 10. Real trace with full-QK primary path

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync0ms_fullqk_retry1.jsonl`

Workload:

- dataset: `dataset/humaneval.jsonl`
- model: `OPT-125M`
- `concurrency = 8`
- `max_new_tokens = 4`
- `256 DPU`
- `fp32` resident KV
- balanced head grouping
- rotated placement

Observed:

- `avg_latency ~= 21.54 s`
- `max_latency ~= 21.71 s`
- the run completed successfully after the helper alignment fix
- `qk_full_batch_calls` and `resident_av_batch_calls` both grow normally
- `qk_full_shadow_max_abs_diff = 16.8175`

Interpretation:

- full-QK is now runnable on a real concurrent trace
- but the path is not yet numerically trustworthy under realistic batching
- so this should be treated as a Phase 1 bring-up baseline, not a final
  correctness baseline

### Helper issue discovered during bring-up

While enabling full-window QK score readback, we hit a helper/runtime bug:

- `invalid mram access (size and offset need to be 8-byte aligned)`

Root cause:

- QK slot-score readback length was not always 8-byte aligned
- the helper read back `num_heads * window * 4` bytes directly from MRAM

Fix:

- `src/pim/upmem_kvslot/host_kvslot.c` now rounds readback size up to 8-byte
  alignment and then copies the real payload into the output buffer

Effect:

- the earlier full-QK trace crash is gone
- the trace now finishes, exposing the remaining numerical-diff issue as the
  next real blocker

### Root cause of the remaining full-QK diff

We then isolated the correctness gap with a direct helper-level repro on `.7`:

- resident KV append was correct
- helper batched-vs-single QK was correct before append
- but after append, the newest token score in `qk_slot` became wrong

Root cause:

- the DPU-side `qk_slot_scores_bits` writeback packed two scores into one
  64-bit word
- but each head row still used `window` as its MRAM stride
- for odd `window`, the final packed write of one head overlapped the start of
  the next head row

This produced the observed pattern:

- earlier tokens correct
- newest / last token wrong
- both single-item and batched helper calls wrong in the same way

### Fix

We fixed `qk_slot` score writeback to use a padded per-head stride:

- `score_stride = round_up_even(window)`

Implementation:

- DPU side:
  - `src/pim/upmem_kvslot/dpu_kvslot.c`
- helper side:
  - `src/pim/upmem_kvslot/host_kvslot.c`

The helper now:

- allocates the padded score buffer
- reads back the padded layout from MRAM
- serializes only the real `window` scores back to Python

### 11. Full-QK trace after the stride fix

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync0ms_fullqk_retry2.jsonl`

Observed:

- `avg_latency ~= 15.61 s`
- `max_latency ~= 15.65 s`
- `qk_full_shadow_max_abs_diff = 7.629e-06`
- `resident_av_shadow_max_abs_diff = 2.441e-04`
- no failures

Interpretation:

- the full-QK correctness gap is now effectively closed for this trace
- Phase 1 has moved from "runnable but not numerically validated" to a usable
  correctness-checked baseline

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

### 22. First K-read optimization pass inside the token-parallel grouped mixed-QK kernel

We then started the next planned kernel-side optimization round by reducing
`K` read overhead without changing the host/helper protocol.

#### 22a. fp32 pair-read fast path

What changed:

- grouped mixed-`QK` keeps the same token-parallel structure
- for `fp32` rows that are pair-aligned, the kernel now reads two adjacent
  `K` values with one `mram_read`
- non-aligned or unsupported cases still fall back to the older scalar path

Correctness:

- minimal three-machine verification still passed
- `qk_mixed_last_max_abs_diff = 0.0`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_tokenparallel_compactreadback_pairkfastpath_upmemav.jsonl`

Observed:

- `avg_latency = 1.8189 s`
- `avg_tpot = 1.6370 s`

Comparison:

- token-parallel compact-score baseline:
  - `avg_latency = 1.8203 s`
  - `avg_tpot = 1.6384 s`
- plus fp32 pair-read fast path:
  - `avg_latency = 1.8189 s`
  - `avg_tpot = 1.6370 s`

Interpretation:

- the gain is small but real
- this supports the current hypothesis that `K`-side MRAM access is still part
  of the remaining bottleneck
- it is worth keeping this fast path because it improves performance without
  complicating correctness or protocol behavior

#### 22b. Larger four-float read attempt

We then tried pushing the same idea one step further with a larger aligned
`K` read block.

What changed:

- for stricter alignment conditions, the kernel attempted to read four
  adjacent `fp32` values in one larger chunk
- the earlier pair-read fast path remained available as fallback

Correctness:

- minimal three-machine verification still passed
- `qk_mixed_last_max_abs_diff = 0.0`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_tokenparallel_compactreadback_quadkfastpath_upmemav.jsonl`

Observed:

- `avg_latency = 1.8280 s`
- `avg_tpot = 1.6459 s`

Comparison:

- pair-read fast path:
  - `avg_latency = 1.8189 s`
  - `avg_tpot = 1.6370 s`
- larger four-float read attempt:
  - `avg_latency = 1.8280 s`
  - `avg_tpot = 1.6459 s`

Interpretation:

- larger block reads were not automatically better on this kernel
- the four-float variant regressed relative to both:
  - the pair-read version
  - the original token-parallel compact-score baseline
- this suggests the next useful direction is not simply "make the read block
  larger"

Updated takeaway:

- keep the fp32 pair-read fast path
- discard the four-float read attempt as a negative result
- the next kernel round should focus on better `K` tiling / staging strategy,
  not blindly larger contiguous reads

### 23. Small K-tile staging by reading fp32 directly into a float WRAM tile

We then tried the next more structured `K`-side optimization step.

What changed:

- the grouped mixed-`QK` kernel still keeps the token-parallel work split
- for aligned `fp32` rows, the kernel now reads a small `K` tile directly into
  a WRAM `float` buffer
- the dot product then reuses those `float` values directly instead of
  converting each element through repeated `u32_bits_to_float(...)`
- the earlier pair-read path still covers the remaining aligned tail
- non-aligned or unsupported cases still fall back to the generic path

Correctness:

- minimal three-machine verification still passed
- `qk_mixed_last_max_abs_diff = 0.0`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_tokenparallel_compactreadback_ktile8float_upmemav.jsonl`

Observed:

- `avg_latency = 1.8019 s`
- `avg_tpot = 1.6224 s`
- `avg_ttft = 0.1795 s`

Comparison:

- token-parallel compact-score baseline:
  - `avg_latency = 1.8203 s`
  - `avg_tpot = 1.6384 s`
  - `avg_ttft = 0.1819 s`
- fp32 pair-read fast path:
  - `avg_latency = 1.8189 s`
  - `avg_tpot = 1.6370 s`
  - `avg_ttft = 0.1819 s`
- small `K` tile read directly into `float` WRAM:
  - `avg_latency = 1.8019 s`
  - `avg_tpot = 1.6224 s`
  - `avg_ttft = 0.1795 s`

Interpretation:

- this is a clearer improvement than the earlier pair-read micro-optimization
- the gain suggests that reducing per-element unpack / conversion overhead is
  worthwhile in addition to reducing raw MRAM read count
- a small structured tile is more effective than simply making the direct read
  block larger

Updated takeaway:

- the current strongest trustworthy grouped mixed-`QK` kernel is now the
  token-parallel compact-score version with small `fp32` `K`-tile staging
- the next likely direction is to refine tile shape / staging order further,
  not to go back to larger unstructured direct reads

### 24. Inner-loop base-hoisting / branch-hoisting attempt on top of small K-tile staging

We then tried one more engineering-oriented cleanup pass on top of the
`ktile8float` kernel.

What changed:

- token-row base arithmetic was partially hoisted / rewritten into a
  recurrence-style inner loop
- some fast-path condition checks were moved outside the hottest inner loop
- the small `fp32` `K`-tile staging structure itself stayed the same

Correctness:

- minimal three-machine verification still passed
- `qk_mixed_last_max_abs_diff = 0.0`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_tokenparallel_compactreadback_ktile8float_hoistedbase_upmemav.jsonl`

Observed:

- `avg_latency = 1.8228 s`
- `avg_tpot = 1.6354 s`
- `avg_ttft = 0.1874 s`

Comparison:

- small `K` tile read directly into `float` WRAM:
  - `avg_latency = 1.8019 s`
  - `avg_tpot = 1.6224 s`
  - `avg_ttft = 0.1795 s`
- plus base-hoisting / branch-hoisting cleanup:
  - `avg_latency = 1.8228 s`
  - `avg_tpot = 1.6354 s`
  - `avg_ttft = 0.1874 s`

Interpretation:

- this engineering cleanup did not help the current kernel
- the regression suggests the current bottleneck is not simply index arithmetic
  inside the inner loop
- the more valuable gains are still coming from memory-access structure, not
  from algebraic loop rewrites alone

Updated takeaway:

- keep the simpler `ktile8float` version as the active kernel baseline
- record this as a negative result
- deprioritize further pure index-hoisting / branch-hoisting tweaks unless a
  profiler later points back to them directly

### 25. K-tile size sweep: 8-float vs 16-float direct fp32 staging

We then checked whether the current small-tile win would continue if the tile
size grew further.

What changed:

- the grouped mixed-`QK` kernel kept the same structure as the current
  `ktile8float` baseline
- only the direct `fp32` `K`-tile size changed from `8` to `16`
- host/helper protocol remained unchanged

Correctness:

- minimal three-machine verification still passed
- `qk_mixed_last_max_abs_diff = 0.0`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_tokenparallel_compactreadback_ktile16float_upmemav.jsonl`

Observed:

- `avg_latency = 1.8181 s`
- `avg_tpot = 1.6341 s`
- `avg_ttft = 0.1839 s`

Comparison:

- token-parallel compact-score baseline:
  - `avg_latency = 1.8203 s`
  - `avg_tpot = 1.6384 s`
  - `avg_ttft = 0.1819 s`
- pair-read fast path:
  - `avg_latency = 1.8189 s`
  - `avg_tpot = 1.6370 s`
  - `avg_ttft = 0.1819 s`
- small `8-float` `K` tile:
  - `avg_latency = 1.8019 s`
  - `avg_tpot = 1.6224 s`
  - `avg_ttft = 0.1795 s`
- larger `16-float` `K` tile:
  - `avg_latency = 1.8181 s`
  - `avg_tpot = 1.6341 s`
  - `avg_ttft = 0.1839 s`

Interpretation:

- increasing the tile from `8` to `16` did not preserve the earlier win
- `16-float` staging is still slightly better than the original baseline, but
  clearly worse than the `8-float` tile
- this is another sign that the current local optimum is a small structured
  tile, not an arbitrarily larger one

Updated takeaway:

- keep `ktile8float` as the active kernel baseline
- record `ktile16float` as a negative tile-size result
- future kernel work should pivot away from simply enlarging the staging tile

### 26. Bottleneck attribution by disabling mixed-QK

Before continuing to optimize grouped mixed-`QK`, we ran a quick attribution
experiment to see how much of the remaining latency still came from that path
versus resident `AV`.

What changed:

- keep resident `AV` enabled
- disable mixed-`QK` replacement entirely
- leave the rest of the resident-KV path unchanged

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_nomixedqk_upmemav.jsonl`

Observed:

- `avg_latency = 1.6745 s`
- `avg_tpot = 1.4918 s`
- `avg_ttft = 0.1827 s`

Comparison:

- current best mixed-`QK` + resident `AV` kernel before this attribution check:
  - `avg_latency = 1.8019 s`
  - `avg_tpot = 1.6224 s`
- same path with mixed-`QK` disabled:
  - `avg_latency = 1.6745 s`
  - `avg_tpot = 1.4918 s`

Interpretation:

- grouped mixed-`QK` is still contributing a meaningful latency increment
- but the resident `AV` path itself remains a large absolute part of the total
  decode cost
- this was enough evidence to justify switching the next exploration round to
  the `AV` kernel

### 27. First resident-AV kernel optimization: same-head adjacent-dim fast path

We then tried the first direct `AV` kernel optimization.

What changed:

- when one output pair corresponds to two adjacent dims from the same head:
  - reuse the same attention weight for both dims
  - on aligned `fp32` rows, read the two `V` values together
- all other `AV` cases fall back to the older path

Correctness:

- minimal three-machine verification still passed
- `resident_av_shadow_max_abs_diff` stayed within the current tolerance
- mixed-`QK` correctness stayed exact

Artifact with mixed-`QK` enabled:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_tokenparallel_compactreadback_ktile8float_avpairdim_upmemav.jsonl`

Observed with mixed-`QK` enabled:

- `avg_latency = 1.7883 s`
- `avg_tpot = 1.6092 s`
- `avg_ttft = 0.1790 s`

Comparison against the best prior mixed-`QK` + resident `AV` baseline:

- `ktile8float` mixed-`QK` baseline:
  - `avg_latency = 1.8019 s`
  - `avg_tpot = 1.6224 s`
  - `avg_ttft = 0.1795 s`
- plus `AV` same-head adjacent-dim fast path:
  - `avg_latency = 1.7883 s`
  - `avg_tpot = 1.6092 s`
  - `avg_ttft = 0.1790 s`

Artifact with mixed-`QK` disabled:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_nomixedqk_avpairdim_upmemav.jsonl`

Observed with mixed-`QK` disabled:

- `avg_latency = 1.6420 s`
- `avg_tpot = 1.4632 s`
- `avg_ttft = 0.1788 s`

Comparison against the earlier no-mixed-`QK` resident-`AV` path:

- no mixed-`QK`, old `AV` path:
  - `avg_latency = 1.6745 s`
  - `avg_tpot = 1.4918 s`
  - `avg_ttft = 0.1827 s`
- no mixed-`QK`, optimized `AV` path:
  - `avg_latency = 1.6420 s`
  - `avg_tpot = 1.4632 s`
  - `avg_ttft = 0.1788 s`

Interpretation:

- this `AV` optimization gives a real win both:
  - with grouped mixed-`QK` enabled
  - and with grouped mixed-`QK` disabled
- that means the gain is genuinely from the `AV` kernel, not from an accidental
  interaction with the mixed-`QK` path

Updated takeaway:

- the best current end-to-end resident attention baseline is now:
  - grouped mixed-`QK` with `ktile8float`
  - plus the `AV` same-head adjacent-dim fast path
- switching some optimization attention toward resident `AV` was the right
  move after the earlier attribution check

### 28. Resident-AV weight-tile checkpoint

We then tried a deeper `AV` optimization on top of the same-head adjacent-dim
fast path.

What changed:

- for same-head adjacent-dim output pairs, the kernel stages `8` consecutive
  attention weights at once
- the goal is to reduce repeated scalar weight reads while keeping the earlier
  paired-`V` fast path

#### First attempt was invalid

Our first `weight-tile[8]` implementation looked faster, but it was not a safe
result.

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_tokenparallel_compactreadback_ktile8float_avweighttile8_upmemav.jsonl`
- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_nomixedqk_avweighttile8_upmemav.jsonl`

Observed:

- mixed-`QK`: `avg_latency = 1.7573 s`
- no mixed-`QK`: `avg_latency = 1.6240 s`
- but `resident_av_shadow_max_abs_diff` jumped to about `2.03`

Root cause:

- the tiled weight path bypassed the earlier aligned packed-read logic
- this introduced a correctness bug in the staged `AV` weight reads

Interpretation:

- those faster numbers must be treated as invalid
- they should not be used as a baseline or reported as a real optimization

#### Corrected aligned version

We then fixed the tiled weight path by reading packed aligned `uint64` weight
pairs and unpacking them in WRAM before reuse.

Artifact with mixed-`QK` enabled:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_tokenparallel_compactreadback_ktile8float_avweighttile8_aligned_upmemav.jsonl`

Observed with mixed-`QK` enabled:

- `avg_latency = 1.7764 s`
- `avg_tpot = 1.5986 s`
- `avg_ttft = 0.1778 s`
- `max_resident_av_shadow_max_abs_diff = 0.0001220703125`

Artifact with mixed-`QK` disabled:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_nomixedqk_avweighttile8_aligned_upmemav.jsonl`

Observed with mixed-`QK` disabled:

- `avg_latency = 1.6168 s`
- `avg_tpot = 1.4386 s`
- `avg_ttft = 0.1782 s`
- `max_resident_av_shadow_max_abs_diff = 0.000030517578125`

Comparison against the current pair-dim `AV` baseline:

- mixed-`QK` + `avpairdim`:
  - `avg_latency = 1.7883 s`
  - `avg_tpot = 1.6092 s`
- mixed-`QK` + aligned `avweighttile8`:
  - `avg_latency = 1.7764 s`
  - `avg_tpot = 1.5986 s`
- no mixed-`QK` + `avpairdim`:
  - `avg_latency = 1.6420 s`
  - `avg_tpot = 1.4632 s`
- no mixed-`QK` + aligned `avweighttile8`:
  - `avg_latency = 1.6168 s`
  - `avg_tpot = 1.4386 s`

Interpretation:

- after fixing the alignment bug, the `AV` weight-tiling direction still gives
  a real positive result
- the gain is smaller than the invalid first attempt suggested, but it is
  still consistent in both:
  - mixed-`QK`
  - no mixed-`QK`
- the current trustworthy small-trace resident-attention baseline is now:
  - grouped mixed-`QK` with `ktile8float`
  - `AV` same-head adjacent-dim fast path
  - aligned `AV` weight-tile staging

### 29. Helper-side AV batch push-xfer checkpoint

We then moved the next optimization step out of the DPU kernel and into
`host_kvslot`.

What changed:

- in `AV_BATCH`, if one launch round has:
  - one item per physical DPU
  - matching padded weight/context sizes
  - and `nr_dpus <= 16`
- then the helper now uses:
  - one batched `runtime_slot_args` push-xfer
  - one batched `av_weights_bits` push-xfer
  - one whole-set launch
  - one batched `av_context_bits` pull-xfer
- instead of per-item:
  - `dpu_copy_to`
  - `dpu_launch`
  - `dpu_copy_from`

This is intentionally gated to the small-DPU regime so it does not accidentally
turn a large-DPU run into a "launch the whole machine for a tiny round" path.

Artifact with mixed-`QK` enabled:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_avbatchpushxfer_upmemav.jsonl`

Observed with mixed-`QK` enabled:

- `avg_latency = 1.1833 s`
- `max_latency = 1.6468 s`

Artifact with mixed-`QK` disabled:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_nomixedqk_avbatchpushxfer_upmemav.jsonl`

Observed with mixed-`QK` disabled:

- `avg_latency = 1.0241 s`
- `max_latency = 1.4638 s`

Comparison against the previous trustworthy resident-`AV` baseline:

- mixed-`QK` + aligned `avweighttile8`:
  - `avg_latency = 1.7764 s`
- mixed-`QK` + helper-side AV batch push-xfer:
  - `avg_latency = 1.1833 s`
- no mixed-`QK` + aligned `avweighttile8`:
  - `avg_latency = 1.6168 s`
- no mixed-`QK` + helper-side AV batch push-xfer:
  - `avg_latency = 1.0241 s`

Interpretation:

- this is a much larger gain than the earlier kernel-only `AV` micro-steps
- the dominant small-trace resident-`AV` bottleneck was not just DPU-side
  arithmetic or unpacking
- a large fraction of the cost was still in helper-side transfer/launch
  granularity

Updated takeaway:

- for the current `4 DPU` small-trace regime, helper-side launch/xfer
  amortization is the strongest `AV` optimization so far
- this is the new trustworthy small-DPU resident-attention baseline

### 30. Helper-side mixed-QK batch push-xfer checkpoint

We then tried the same helper-side batching idea on `QK_SLOT_BATCH`.

What changed:

- keep the current trusted resident-attention baseline:
  - grouped resident-slot mixed-`QK`
  - `ktile8float`
  - helper-side batched resident `AV`
- in `QK_SLOT_BATCH`, if one launch round has:
  - one item per physical DPU
  - matching `num_heads`
  - matching `window`
  - matching `head_dim`
  - and `nr_dpus <= 16`
- then the helper now uses:
  - one batched `runtime_slot_args` push-xfer
  - one batched `qk_slot_args` push-xfer
  - one batched `qk_slot_head_indices` push-xfer
  - one batched `qk_query` push-xfer
  - one whole-set launch
  - one batched `qk_slot_scores_bits` pull-xfer
- otherwise it falls back to the previous per-item path

Correctness:

- placement verification still passed
- `resident_av_shadow_max_abs_diff = 0.0`
- `qk_mixed_last_max_abs_diff = 0.0`

Artifacts from the clean serial attribution rerun:

- mixed-`QK` enabled:
  - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_qkbatch_serial_upmemav.jsonl`
- mixed-`QK` disabled:
  - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_qkbatch_serial_nomixedqk_upmemav.jsonl`

Observed with mixed-`QK` enabled:

- `avg_latency = 1.2104 s`
- `avg_tpot = 0.9860 s`
- `avg_ttft = 0.2244 s`

Observed with mixed-`QK` disabled:

- `avg_latency = 1.0835 s`
- `avg_tpot = 0.8562 s`
- `avg_ttft = 0.2273 s`

Comparison against the previous trustworthy AV-batched baseline:

- previous mixed-`QK` baseline:
  - `avg_latency = 1.1653 s`
  - `avg_tpot = 0.9790 s`
  - `avg_ttft = 0.1862 s`
- helper-side mixed-`QK` batching:
  - `avg_latency = 1.2104 s`
  - `avg_tpot = 0.9860 s`
  - `avg_ttft = 0.2244 s`
- previous no-mixed-`QK` attribution:
  - `avg_latency = 1.0260 s`
  - `avg_tpot = 0.8425 s`
  - `avg_ttft = 0.1834 s`
- helper-side no-mixed-`QK` attribution:
  - `avg_latency = 1.0835 s`
  - `avg_tpot = 0.8562 s`
  - `avg_ttft = 0.2273 s`

Interpretation:

- unlike the helper-side resident-`AV` batching step, the analogous
  helper-side mixed-`QK` batching does not help on the current `4 DPU`
  small-trace regime
- correctness is fine, but latency regresses slightly in both:
  - mixed-`QK`
  - no mixed-`QK`
- this suggests the remaining mixed-`QK` bottleneck is not dominated by the
  same helper launch/xfer granularity issue that previously dominated resident
  `AV`

Updated takeaway:

- keep the code path only as a correctness-preserving experiment checkpoint
- do not treat helper-side whole-set mixed-`QK` batching as the next active
  optimization direction
- the next mixed-`QK` steps should return to:
  - kernel/dataflow structure
  - active-set launch structure for larger DPU counts
  - or reducing host-side packing/copy work before launch

### 31. Python-side helper packing / unpacking checkpoint

We then tried a smaller host-language cleanup around the helper client.

What changed:

- keep the current helper-side mixed-`QK` batching code in place
- reduce Python-side serialization / deserialization overhead in
  `resident_kv_store.py`:
  - use streamed helper writes instead of `b\"\".join(...)`
  - use `numpy.frombuffer(...).copy()` for score/context readback instead of
    large `struct.unpack(...)` tuples
  - fast-path already-CPU, already-`fp32`, already-contiguous tensors
- reduce mixed-`QK` query packing overhead in `attention_backend.py`:
  - convert `q` to `fp32` once
  - reuse row slices from that tensor
  - avoid building a list of per-head tensors and then `torch.stack(...)`

Correctness:

- placement verification still passed
- `resident_av_shadow_max_abs_diff = 0.0`
- `qk_mixed_last_max_abs_diff = 0.0`

Artifacts:

- mixed-`QK` enabled:
  - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_packopt_serial_upmemav.jsonl`
- mixed-`QK` disabled:
  - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_packopt_serial_nomixedqk_upmemav.jsonl`

Observed with mixed-`QK` enabled:

- `avg_latency = 1.2095 s`
- `avg_tpot = 0.9763 s`
- `avg_ttft = 0.2332 s`

Observed with mixed-`QK` disabled:

- `avg_latency = 1.0605 s`
- `avg_tpot = 0.8355 s`
- `avg_ttft = 0.2251 s`

Comparison against the immediately previous helper-batched checkpoint:

- previous mixed-`QK` path:
  - `avg_latency = 1.2104 s`
  - `avg_tpot = 0.9860 s`
  - `avg_ttft = 0.2244 s`
- Python-side packing cleanup:
  - `avg_latency = 1.2095 s`
  - `avg_tpot = 0.9763 s`
  - `avg_ttft = 0.2332 s`
- previous no-mixed-`QK` path:
  - `avg_latency = 1.0835 s`
  - `avg_tpot = 0.8562 s`
  - `avg_ttft = 0.2273 s`
- Python-side packing cleanup:
  - `avg_latency = 1.0605 s`
  - `avg_tpot = 0.8355 s`
  - `avg_ttft = 0.2251 s`

Interpretation:

- this cleanup does not materially improve mixed-`QK`
- it does recover a small amount of latency on the no-mixed path, which is
  consistent with a modest helper-client serialization benefit
- the dominant mixed-`QK` bottleneck still does not appear to be Python-side
  packing / unpacking alone

Updated takeaway:

- keep the lighter helper-client serialization path because it is cleaner and
  gives a small general benefit
- do not expect further Python-only packing cleanup to unlock the main
  mixed-`QK` gap
- the next meaningful steps should move back toward:
  - mixed-`QK` kernel/dataflow structure
  - active-set launch design
  - or larger-granularity batching choices

### 32. Active-rank QK launch checkpoint

We then implemented the first active-set-style launch step inside
`host_kvslot`.

What changed:

- keep the current mixed-`QK` helper batching path
- cache DPU topology in the helper:
  - physical DPU handles
  - rank handles
  - physical-DPU to rank mapping
- for batched mixed-`QK` rounds, build a temporary `dpu_set_t` containing only
  the active ranks touched by that round
- run the batched `runtime_slot_args / qk_slot_args / qk_slot_head_indices /
  qk_query / qk_slot_scores_bits` transfers and launch against that temporary
  rank subset instead of unconditionally using the whole runner set

Important scope note:

- this is an active-rank subset, not yet a perfect arbitrary active-DPU subset
- within an active rank, inactive DPUs still receive dummy buffers for the
  batched push/pull operations
- the main purpose of this checkpoint is to remove the hard dependency on
  whole-runner launch before scaling to larger multi-rank DPU counts

Correctness:

- placement verification still passed
- `resident_av_shadow_max_abs_diff = 0.0`
- `qk_mixed_last_max_abs_diff = 0.0`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_activerank_serial_upmemav.jsonl`

Observed with mixed-`QK` enabled:

- `avg_latency = 1.2120 s`
- `max_latency = 1.5669 s`

Comparison:

- previous mixed-`QK` helper-side checkpoint:
  - `avg_latency = 1.2104 s`
- Python-side packing cleanup checkpoint:
  - `avg_latency = 1.2095 s`
- active-rank launch checkpoint:
  - `avg_latency = 1.2120 s`

Interpretation:

- no measurable gain appears on the current `4 DPU` small-trace setup
- this is expected for a small regime where the active round likely touches the
  same rank anyway
- the value of this step is structural:
  - the mixed-`QK` batched path is no longer forced to use the full runner set
  - the codebase is now ready for future multi-rank experiments where
    active-rank reduction can matter

Updated takeaway:

- keep the active-rank path as the new structural baseline for mixed-`QK`
  batching
- do not expect this step alone to improve single-rank small traces
- the next useful measurement for this direction is not another `4 DPU` rerun,
  but a larger multi-rank DPU regime

### 33. Active-rank large-DPU follow-up

After the active-rank path was verified on the small `4 DPU` setup, we moved
to larger multi-rank DPU counts to test the intended use case directly.

Artifacts:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu256_fp32_balanced_rotated_activerank_upmemav.jsonl`
- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu512_fp32_balanced_rotated_activerank_upmemav.jsonl`

Observed:

- `256 DPU`:
  - `avg_latency = 1.9515 s`
  - `max_latency = 2.3289 s`
  - `avg_usage_ratio ~= 0.00494`
- `512 DPU`:
  - `avg_latency = 1.9673 s`
  - `max_latency = 2.3734 s`
  - `avg_usage_ratio ~= 0.00244`

Comparison against the small active-rank trace:

- `4 DPU`:
  - `avg_latency = 1.2120 s`
- `256 DPU`:
  - `avg_latency = 1.9515 s`
- `512 DPU`:
  - `avg_latency = 1.9673 s`

Interpretation:

- active-rank launch is not enough to make this light mixed-`QK` workload
  scale well to large DPU counts
- the end-to-end latency gets worse as the persistent DPU pool grows, even
  though only a tiny fraction of total capacity is actually active
- the utilization numbers make the root cause visible:
  - `256 DPU`: only about `0.5%` of total per-DPU pool capacity is active on
    average
  - `512 DPU`: only about `0.24%` is active on average
- this points to a structural mismatch between:
  - very small mixed-`QK` working sets
  - and very large resident DPU pools

Updated takeaway:

- the current grouped mixed-`QK` path is still over-distributed for this light
  workload
- rank-level active-set reduction removes one architectural blocker, but it
  does not solve the deeper issue that too little work is being spread over too
  many DPUs
- the next optimization phase should focus on reducing distribution granularity
  or increasing useful work per launch, rather than simply allocating more
  DPUs

### 34. Light-workload group compaction checkpoint

We then directly targeted the over-distribution problem in the resident
head-group construction path.

What changed:

- add an internal `_effective_head_group_count(...)` heuristic in
  `src/core/attention_backend.py`
- keep the external experiment interface unchanged
- for light per-head KV workloads, reduce the number of resident groups by:
  - requiring more heads per group
  - and requiring a minimum amount of live KV work per group
- use this effective group count inside `_build_head_groups(...)` instead of
  blindly using `min(num_dpus, num_heads)`

Design intent:

- this is not a DPU-count optimization by itself
- it is a granularity optimization:
  - keep fewer active groups for light requests
  - reduce slot-level `QK` / `AV` helper call fanout
  - avoid spreading tiny per-layer workloads over too many resident units

Correctness:

- placement verification still passed at `256 DPU`
- `resident_av_shadow_max_abs_diff = 0.0`
- `qk_mixed_last_max_abs_diff = 0.0`

Important observed behavior:

- on the verification request, each OPT-125M layer compacted to
  `group_count = 1`
- that means the layer was no longer fragmented across multiple tiny resident
  groups for this light decode regime

Artifacts:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu256_fp32_balanced_rotated_compactgroups_upmemav.jsonl`
- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu512_fp32_balanced_rotated_compactgroups_upmemav.jsonl`

Observed:

- `256 DPU`:
  - `avg_latency = 1.8611 s`
  - previous active-rank baseline: `1.9515 s`
  - improvement: about `4.6%`
- `512 DPU`:
  - `avg_latency = 1.9031 s`
  - previous active-rank baseline: `1.9673 s`
  - improvement: about `3.3%`

Interpretation:

- this is the first larger-DPU result that clearly improves the
  over-distributed mixed-`QK`/resident-`AV` baseline
- the result supports the working diagnosis:
  - the previous large-DPU slowdown was not only about launch topology
  - it was also about splitting too little work into too many resident groups
- the utilization ratio is still extremely low because the persistent pool is
  still large, but reducing group fanout already cuts real end-to-end latency

Updated takeaway:

- light-workload group compaction is a valid next baseline direction
- "active-rank launch + compact resident groups" is better than
  "active-rank launch alone" in the current light-trace regime
- the next useful follow-up is to push granularity reduction further at a
  higher level:
  - compact more requests together
  - compact more layer work per launch
  - or make the grouping heuristic request-aware instead of purely
    layer-shape-based

### 35. Attention micro-batching checkpoint

We then implemented the first request-level batching path at the attention
actor boundary.

What changed:

- add `decode_layer_batch(...)` to the attention backends
- for the PIM-backed backend, batch multiple requests together across one
  attention round:
  - append resident KV per request as before
  - flatten mixed-`QK` slot queries across requests into one
    `qk_slot_scores_batch(...)`
  - flatten resident `AV` slot weights across requests into one
    `weighted_value_sum_batch(...)`
- add a short-window micro-batcher inside `AttentionNode` so concurrent
  `decode_layer` RPCs can be collected and flushed together
- surface batching counters in debug info and trace artifacts

Correctness:

- placement verification still passed
- `resident_av_shadow_max_abs_diff = 0.0`
- `qk_mixed_last_max_abs_diff = 0.0`

Important validation result:

- in a direct two-request concurrent scheduler check, the attention actor did
  batch requests together:
  - `decode_batching.max_observed_size = 2`
  - `decode_batch_calls = 23`
  - `decode_batch_items = 24`
  - `resident_av_batch_calls = 23`
- this proves the new path is functionally live:
  - some attention rounds were merged across requests
  - helper round count became smaller than total request-layer item count

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc2_tok2_dpu256_fp32_balanced_rotated_compactgroups_microbatchstats_upmemav.jsonl`

Observed on the current trace driver:

- trace correctness stayed clean
- but per-row trace stats showed:
  - `attention_decode_batching.max_observed_size = 1`
  - `decode_batch_calls = decode_batch_items`
- end-to-end latency was therefore not improved on this trace:
  - `avg_latency = 2.9912 s` at `concurrency = 2`

Interpretation:

- the attention-side batching mechanism itself works
- however, the current scheduler / trace submission rhythm does not reliably
  create overlap at the `decode_layer` RPC boundary
- in other words:
  - request-level batching is now implementable
  - but the present end-to-end driver does not naturally feed it enough
    same-time attention work to produce stable gains

Updated takeaway:

- keep the micro-batched attention path as a working capability
- do not treat the current trace result as evidence that request-level batching
  is useless
- the next issue is now above the attention actor:
  - orchestrate decode requests so they reach the attention boundary together
  - or explicitly batch layer work in the scheduler instead of hoping
    opportunistic overlap is enough

### 36. Scheduler-side explicit attention batching checkpoint

We then moved one level upward and added explicit scheduler-managed attention
batching keyed by `(decode_step, layer_idx)`.

What changed:

- the scheduler no longer depends only on opportunistic overlap at the actor
  boundary
- concurrent requests reaching the same decode step and layer are now queued
  into one scheduler-side wavefront bucket
- the scheduler can flush that bucket either:
  - immediately when it reaches the current target batch size
  - or after a short wait window
- the attention node receives one direct `decode_layer_batch(...)` RPC for that
  wavefront

Correctness:

- three-machine placement verification still passed with the new config path
- batched decode results remained numerically clean

Important validation result:

- with `attention_rpc_batch_window_s = 0.01` and actor-side batching disabled,
  direct concurrent validation showed the scheduler really was merging work:
  - `scheduler_attention_batching.flushes = 12`
  - `scheduler_attention_batching.total_items = 24`
  - `scheduler_attention_batching.max_observed_size = 2`
  - backend `decode_batch_calls = 12`
  - backend `decode_batch_items = 24`

Observed on the end-to-end light trace:

- batching now happens reliably at the scheduler level
- but end-to-end latency still became worse:
  - direct two-request check: about `1.48 s` per request
  - saved four-sample trace: `avg_latency = 2.975 s`

Artifacts:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc2_tok2_dpu256_fp32_balanced_rotated_schedbatch10ms_upmemav.jsonl`
- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc2_tok2_dpu256_fp32_balanced_rotated_schedbatch5ms_upmemav.jsonl`

Interpretation:

- missing batch machinery is no longer the main problem
- scheduler-managed batching does reduce attention round count
- however, under this light workload the waiting cost needed to create the
  batch is larger than the saved backend work

### 37. Wavefront batching refinement

We then refined the scheduler queue from a generic time-window batch into an
explicit wavefront map keyed by `(decode_step, layer_idx)`.

Why this matters:

- this is structurally closer to the real decode dependency graph
- it prevents unrelated layer work from being mixed together
- it gives us a cleaner base for later decode-step coordination

Observed result:

- the structural refactor is correct
- but it does not remove the arrival-skew problem by itself
- on the current workload:
  - a `10 ms` wait is still enough to get `max_observed_size = 2`
  - a `5 ms` wait is still usually too short and falls back to
    `max_observed_size = 1`

Updated takeaway:

- we now have working batching at three levels:
  - backend batched decode
  - attention-actor micro-batching
  - scheduler-side wavefront batching
- the new bottleneck is request synchronization cost, not missing batch APIs
- the next step should therefore coordinate requests earlier in the decode
  pipeline so same-step work reaches the attention wavefront together

### 38. Decode-step sync bug fix and first real scheduler batching

We then fixed a scheduler bug in the new decode-step synchronization path.

Root cause:

- the step-sync target size was derived from "requests already inside decode"
- the first request reaching a step therefore often saw target size `1`
- it was released immediately and never really waited for peer requests

Fix:

- derive the step-sync target from the current inflight request count instead

Immediate result on the earlier `concurrency = 2`, `max_new_tokens = 2`
light trace:

- scheduler wavefront batching became truly active
- `scheduler_attention_batching.max_observed_size = 2`
- backend `decode_batch_calls / decode_batch_items = 36 / 48`
- `avg_latency = 2.7789 s`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc2_tok2_dpu256_fp32_balanced_rotated_stepsync5ms_fix_schedbatch5ms_upmemav.jsonl`

Interpretation:

- the previous batching ceiling was partly an implementation bug, not just
  workload shape
- after the fix, scheduler-coordinated attention merging is now genuinely
  working end to end

### 39. Concurrency scaling checkpoint

We then increased decode concurrency to see whether the same scheduler path
could scale the wavefront batch size upward.

#### `concurrency = 4`, `max_new_tokens = 2`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc4_tok2_dpu256_fp32_balanced_rotated_stepsync5ms_fix_schedbatch5ms_upmemav.jsonl`

Observed:

- `scheduler_attention_batching.max_observed_size = 3`
- `scheduler_decode_step_sync.max_observed_size = 2`
- backend `decode_batch_calls / decode_batch_items = 59 / 96`
- `avg_latency = 4.7446 s`

Interpretation:

- the wavefront batch size does grow with concurrency
- but short decode still limits how often all requests align on the same layer

#### `concurrency = 4`, longer decode

Artifacts:

- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc4_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_fix_schedbatch5ms_upmemav.jsonl`
- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc4_tok8_dpu256_fp32_balanced_rotated_stepsync5ms_fix_schedbatch5ms_upmemav.jsonl`

Observed:

- `tok4`:
  - `scheduler_attention_batching.max_observed_size = 3`
  - `scheduler_decode_step_sync.max_observed_size = 3`
  - `avg_latency = 11.7315 s`
- `tok8`:
  - `scheduler_attention_batching.max_observed_size = 3`
  - `scheduler_decode_step_sync.max_observed_size = 3`
  - `avg_latency = 27.2719 s`

Interpretation:

- longer decode makes the batching path more active and more stable
- however, with `concurrency = 4` the practical wavefront ceiling on this
  setup remains `3`
- the next limiter is no longer decode lifetime alone, but residual
  per-layer arrival skew

### 40. `concurrency = 8` and batch-target sensitivity

We then pushed decode concurrency to `8` and compared two scheduler target
policies.

Workload:

- `concurrency = 8`
- `max_new_tokens = 4`
- step-sync window `5 ms`

Artifacts:

- target `8`:
  - `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_fix_batchtarget8_upmemav.jsonl`
- target `4`:
  - `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_fix_batchtarget4_upmemav.jsonl`

Observed:

- target `8`:
  - `scheduler_attention_batching.max_observed_size = 4`
  - `scheduler_decode_step_sync.max_observed_size = 5`
  - backend `decode_batch_calls / decode_batch_items = 123 / 288`
  - `avg_latency = 23.2235 s`
- target `4`:
  - `scheduler_attention_batching.max_observed_size = 4`
  - `scheduler_decode_step_sync.max_observed_size = 4`
  - backend `decode_batch_calls / decode_batch_items = 146 / 288`
  - `avg_latency = 23.3370 s`

Interpretation:

- the batching ceiling continues to rise with concurrency:
  - we now reliably reach attention wavefront batch size `4`
- reducing the target size from `8` to `4` does not increase the achieved
  attention batch size
- it mostly increases flush count, i.e. it is more eager but not more
  productive on this workload

### 41. Attention-window sensitivity above the step-sync path

The next question was:

- if step sync can briefly align more than `4` requests,
  why does attention wavefront batching still stop at `4`?

We tested a larger scheduler attention window:

- step-sync window `5 ms`
- attention wavefront window `10 ms`
- `concurrency = 8`
- `max_new_tokens = 4`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_fix_schedbatch10ms_target8_upmemav.jsonl`

Observed:

- `scheduler_attention_batching.max_observed_size = 5`
- backend `decode_batch_calls / decode_batch_items = 113 / 288`
- `avg_latency = 23.7549 s`

Interpretation:

- the residual bottleneck is indeed per-layer arrival skew, not an absolute
  structural limit of the current scheduler batching path
- a larger attention-side wait can recover a larger wavefront batch
- but it does so by paying additional waiting latency

Updated takeaway:

- current scheduler coordination can now reach:
  - batch size `2` at `concurrency = 2`
  - batch size `3` at `concurrency = 4`
  - batch size `4` at `concurrency = 8` with a short attention window
  - batch size `5` at `concurrency = 8` with a longer attention window
- the next system question is no longer whether batching is possible
- the next question is how to achieve the larger wavefronts with less
  explicit waiting

### 42. Layer-barrier scheduler checkpoint

We then implemented a stronger scheduler-side coordination path:

- add an explicit layer barrier keyed by `(decode_step, layer_idx)`
- release a bounded group of requests together at the layer boundary
- let the attention wavefront inherit that expected group size

This is structurally stronger than a pure attention-side time window because:

- grouping happens before the attention RPC boundary
- the batch has an explicit expected size instead of only "wait and hope"

#### Barrier-only result

Workload:

- `concurrency = 8`
- `max_new_tokens = 4`
- step sync `5 ms`
- attention wavefront window `0 ms`

Artifacts:

- barrier `5 ms`:
  - `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_layerbarrier5ms_target8_attn0_upmemav.jsonl`
- barrier `10 ms`:
  - `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_layerbarrier10ms_target8_attn0_upmemav.jsonl`

Observed:

- barrier `5 ms`:
  - `scheduler_attention_layer_barrier.max_observed_size = 4`
  - `scheduler_attention_batching.max_observed_size = 4`
  - backend `decode_batch_calls / decode_batch_items = 150 / 288`
  - `avg_latency = 23.1171 s`
- barrier `10 ms`:
  - `scheduler_attention_layer_barrier.max_observed_size = 4`
  - `scheduler_attention_batching.max_observed_size = 4`
  - backend `decode_batch_calls / decode_batch_items = 112 / 288`
  - `avg_latency = 23.1939 s`

Interpretation:

- the explicit layer barrier is functionally correct
- it is slightly more latency-friendly than the earlier `10 ms` pure
  attention-window scheme
- however, in this current form it still does not recover batch size `5`
  by itself

#### Barrier + short residual attention window

We also tested a hybrid path:

- layer barrier `5 ms`
- plus a short attention wavefront window `5 ms`

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_layerbarrier5ms_attn5ms_target8_upmemav.jsonl`

Observed:

- `scheduler_attention_layer_barrier.max_observed_size = 4`
- `scheduler_attention_batching.max_observed_size = 4`
- backend `decode_batch_calls / decode_batch_items = 145 / 288`
- `avg_latency = 23.3052 s`

Interpretation:

- a short residual attention window does not raise the achieved batch ceiling
  above the barrier-only result here
- so the remaining gap versus the earlier batch-size-`5` run is not solved by
  merely stacking a small extra window on top of the current barrier

### 43. Updated scheduler takeaway

At this point the evidence is:

- pure short wavefront window:
  - best latency among the simple schemes
  - but only reaches batch size `4`
- longer wavefront window:
  - reaches batch size `5`
  - but pays more waiting latency
- current explicit layer barrier:
  - improves structure and keeps latency low
  - but still tops out at batch size `4`

Updated interpretation:

- the remaining limitation is not just "requests did not meet at the
  attention boundary"
- the remaining limitation is that the current scheduler still does not keep a
  decode wave tightly coupled through the whole per-layer pipeline

The next stronger scheduler idea should therefore be:

- keep an explicit active decode wave together across the layer loop
- instead of independently waiting at each layer key

### 44. Decode-wave persistence checkpoint

We then tried the stronger scheduler variant sketched above:

- decode-step sync now returns a persistent cohort id
- attention batching is keyed by `(decode_step, layer_idx, cohort_id)`
- the intended effect was to keep the same decode wave coupled across the full
  layer loop of one decode step

Trace setup:

- `concurrency = 8`
- `max_new_tokens = 4`
- `attention_rpc_batch_window_s = 0.0`
- `attention_actor_batch_window_s = 0.0`
- `decode_step_sync_window_s = 5 ms`
- compare `layer barrier = 5 ms` and `10 ms`

Artifacts:

- wavepersist + barrier `5 ms`:
  - `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_wavepersist_stepsync5ms_layerbarrier5ms_target8_attn0_upmemav.jsonl`
- wavepersist + barrier `10 ms`:
  - `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_wavepersist_stepsync5ms_layerbarrier10ms_target8_attn0_upmemav.jsonl`

Observed results:

- wavepersist + barrier `5 ms`:
  - `avg_latency = 27.0421 s`
  - `scheduler_attention_batching.max_observed_size = 3`
  - `scheduler_attention_batching.flushes = 203`
  - `scheduler_attention_layer_barrier.max_observed_size = 3`
  - `decode_batch_calls = 203`
- wavepersist + barrier `10 ms`:
  - `avg_latency = 23.2994 s`
  - `scheduler_attention_batching.max_observed_size = 3`
  - `scheduler_attention_batching.flushes = 168`
  - `scheduler_attention_layer_barrier.max_observed_size = 3`
  - `decode_batch_calls = 168`

Comparison versus earlier non-wavepersist runs on the same workload:

- pure short wavefront `5 ms`:
  - `avg_latency = 23.2235 s`
  - `max_observed_size = 4`
  - `flushes = 123`
- pure long wavefront `10 ms`:
  - `avg_latency = 23.7549 s`
  - `max_observed_size = 5`
  - `flushes = 113`
- layer barrier `5 ms`:
  - `avg_latency = 23.1171 s`
  - `max_observed_size = 4`
  - `flushes = 150`
- layer barrier `10 ms`:
  - `avg_latency = 23.1939 s`
  - `max_observed_size = 4`
  - `flushes = 112`

Updated interpretation:

- this decode-wave persistence implementation does not help
- in fact it regresses both batching quality and latency:
  - batch size falls from `4` or `5` down to `3`
  - helper / backend rounds increase sharply
- the current cohort-id mechanism is therefore not preserving a useful active
  wave in practice
- the likely failure mode is over-fragmentation:
  - once requests diverge across cohorts, later layers inherit that split
  - the scheduler loses cross-request merge opportunities that the simpler
    time-window schemes still exploited

Conclusion from this checkpoint:

- we should not keep pushing this particular scheduler-cohort design as the
  main path forward
- the scheduler evidence is now good enough:
  - simple windows can buy larger batches, but only with more waiting
  - the first structured barrier is latency-competitive, but capped at `4`
  - the first cohort-persistence design is strictly worse than both

### 45. AV rank-subset batched round checkpoint

We then returned to the compute / transport side and fixed a concrete missing
fast path in the UPMEM kvslot helper:

- the AV batched round had been effectively disabled for larger DPU counts
- specifically, `can_use_batched_av_round(...)` returned false when
  `runner->nr_dpus > 16`
- on our real setup with `256` DPUs, this meant AV never used the intended
  batched multi-DPU launch path

What changed:

- keep the same AV batched protocol
- but launch only on the active rank subset, mirroring the batched QK path
- this removes the artificial large-DPU disable while avoiding a full-system
  dummy launch across all provisioned DPUs

Important engineering note:

- the harmful decode-wave persistence scheduler path is now gated behind an
  explicit config flag and is disabled by default again
- this restores a clean baseline for performance comparison

Comparison workload:

- `concurrency = 8`
- `max_new_tokens = 4`
- `decode_step_sync_window_s = 5 ms`
- compare pure attention window `5 ms` and `10 ms`

Artifacts:

- pure wavefront `5 ms`, old:
  - `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_fix_batchtarget8_upmemav.jsonl`
- pure wavefront `5 ms`, AV rank-subset batched:
  - `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_fix_batchtarget8_upmemav_avrankbatch.jsonl`
- pure wavefront `10 ms`, old:
  - `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_fix_schedbatch10ms_target8_upmemav.jsonl`
- pure wavefront `10 ms`, AV rank-subset batched:
  - `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_fix_schedbatch10ms_target8_upmemav_avrankbatch.jsonl`

Observed results:

- pure wavefront `5 ms`:
  - old: `avg_latency = 23.2235 s`, `max_observed_size = 4`, `flushes = 123`
  - new: `avg_latency = 21.0997 s`, `max_observed_size = 4`, `flushes = 141`
  - improvement: `-2.1237 s` (`-9.14%`)
- pure wavefront `10 ms`:
  - old: `avg_latency = 23.7549 s`, `max_observed_size = 5`, `flushes = 113`
  - new: `avg_latency = 20.1384 s`, `max_observed_size = 5`, `flushes = 114`
  - improvement: `-3.6165 s` (`-15.22%`)

Interpretation:

- batching quality did not improve materially:
  - the `5 ms` run still peaks at batch size `4`
  - the `10 ms` run still peaks at batch size `5`
- yet end-to-end latency improved substantially in both cases
- this strongly suggests the gain came from cheaper AV execution per batched
  round on large-DPU configurations, not from scheduler effects
- this is exactly the kind of optimization signal we wanted:
  - compute / helper overhead is still a large, real bottleneck
  - reducing it can buy double-digit latency improvement without needing more
    aggressive waiting or more complex scheduling

Updated takeaway:

- the current best direction is again clearly compute-side optimization
- scheduler restructuring remains secondary unless future kernel/runtime
  improvements make additional batching significantly more valuable

### 46. AV kernel repartition attempts that regressed

After the helper-side AV rank-subset fix landed, we tried to push further on
the DPU kernel itself by changing the AV tasklet work decomposition.

Attempt A: head-centric fp32 AV kernel

- idea:
  - assign one tasklet a full attention head
  - reuse one weight row across all head dimensions inside the tasklet
- expectation:
  - eliminate most duplicated weight-row MRAM reads

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_fix_schedbatch10ms_target8_upmemav_avrankbatch_avheadkernel.jsonl`

Result:

- correctness still passed
- but performance collapsed badly:
  - `avg_latency = 58.3023 s`
  - versus `20.1384 s` for the previous AV-rank-batched baseline

Interpretation:

- this mapping destroyed effective tasklet utilization
- with only `12` heads in OPT-125m, the kernel no longer exposed enough
  parallel work to keep the DPU execution efficient

Attempt B: fp32 AV kernel with `8-dim` output blocks

- idea:
  - keep more parallel blocks than the head-centric version
  - still reduce repeated weight-row reads relative to the original
    `2-dim pair` mapping

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval8_conc8_tok4_dpu256_fp32_balanced_rotated_stepsync5ms_fix_schedbatch10ms_target8_upmemav_avrankbatch_avblock8.jsonl`

Result:

- correctness still passed
- but performance was still worse than the previous baseline:
  - `avg_latency = 23.5841 s`
  - versus `20.1384 s` for the previous AV-rank-batched baseline

Interpretation:

- reducing duplicated weight reads alone is not sufficient here
- the original per-pair mapping keeps enough parallelism to offset that extra
  duplication
- our attempted repartitions traded away too much concurrency / balance for
  the amount of MRAM-read reduction they achieved

Action taken:

- both experimental AV-kernel repartitions were discarded
- the repository and `.7` runtime were restored to the previous stable kernel
  that pairs well with the AV rank-subset helper optimization

Updated lesson:

- for this workload, AV kernel optimization is constrained by a careful
  balance between:
  - duplicated weight-row traffic
  - available tasklet parallelism
  - per-tasklet workload balance
- future DPU-kernel work should therefore avoid coarse head-level ownership
  unless it also introduces a stronger mechanism for within-head parallelism

## Helper-Boundary Fused Softmax+AV Baseline

Date: 2026-04-26

We then moved the attention boundary one step further inward:

- full-QK remains on the resident-slot path
- host-side softmax is removed from the normal decode path
- per-slot score matrices are sent to the kvslot helper
- the helper performs row-wise softmax on CPU and reuses the resident-AV path
- the backend receives only per-group `context`

This gives us the first stable `context-only return` baseline at the
Python/backend boundary, even though softmax is still helper-CPU-side rather
than DPU-kernel-side.

### 47. Three-machine cluster correctness with fused softmax+AV

Workload:

- cluster: `.3` prefill GPU, `.4` dense GPU, `.7` attention CPU+UPMEM
- model: `OPT-125M`
- `max_new_tokens = 2`
- `64 DPU`
- `pim_qk_full_enabled = True`
- `pim_qk_mixed_enabled = False`
- `pim_softmax_av_fused_enabled = True`

Observed:

- placement verification passed
- end-to-end generation succeeded
- `qk_full_shadow_max_abs_diff = 0.0`
- `softmax_av_fused_shadow_max_abs_diff = 4.768e-07`
- `softmax_av_fused_ops = 12`
- `softmax_av_fused_batch_calls = 12`

Interpretation:

- the new fused path is correctly wired through the real three-machine cluster
- from the backend perspective, the host no longer performs softmax in the
  normal path

### 48. Small concurrent fused trace

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_tok2_dpu64_fp32_fullqk_fused.jsonl`

Workload:

- dataset: `dataset/humaneval.jsonl`
- model: `OPT-125M`
- first `4` samples
- `concurrency = 4`
- `max_new_tokens = 2`
- `64 DPU`
- `fp32` resident KV
- full-QK enabled
- helper-boundary fused softmax+AV enabled

Observed:

- `avg_latency ~= 3.18 s`
- `max_latency ~= 3.20 s`
- `max_usage_ratio = 0.2095`
- `max_live_slot_count = 8`
- `max_dpu_allocate_failures = 0`
- `max_fallback_allocations = 0`
- `max qk_full_shadow_max_abs_diff = 5.722e-06`
- `max softmax_av_fused_shadow_max_abs_diff = 6.104e-05`

Interpretation:

- the fused boundary is now stable on a small concurrent real trace
- the current path is numerically trustworthy enough to use as the next
  experimental baseline
- this is a better baseline for future full-PIM work than the earlier
  `host softmax + resident AV` split path

## Updated Baseline Reading

After this step, the attention baselines are best ordered as:

1. disaggregated CPU attention
2. resident-KV + host-softmax + resident-AV
3. resident-KV + full-QK + helper-boundary fused softmax+AV

The new third baseline matters because it isolates the next remaining gap more
cleanly:

- not backend correctness
- not host-side softmax structure
- but the fact that softmax is still helper-CPU-side rather than truly inside
  the PIM-side kernel/data plane

## Deeper Fused `QK+softmax+AV->context` Bring-Up

Date: 2026-04-27

We then removed one more host/backend-side boundary:

- the earlier fused path still exposed full score matrices to Python
- the new path sends slot-group queries directly to the kvslot helper
- the helper runs:
  - resident-slot QK
  - helper-side softmax
  - resident AV
- the backend receives only `context`

This is still not final full-PIM attention, but it is a more faithful
`context-only` baseline than the earlier helper-boundary softmax+AV design.

### 49. `.7` local single-RPC fused check

Direct test:

- `UpmemKVSlotStore.qk_softmax_weighted_value_sum_batch(...)`

Observed:

- `max_diff = 5.96e-08`

Interpretation:

- the new single-helper-RPC fused path is numerically correct at the store
  boundary

### 50. Three-machine correctness with deeper fused path

Workload:

- cluster: `.3` prefill GPU, `.4` dense GPU, `.7` attention CPU+UPMEM
- model: `OPT-125M`
- `64 DPU`
- `max_new_tokens = 2`
- `pim_qk_full_enabled = True`
- `pim_qk_mixed_enabled = False`
- `pim_softmax_av_fused_enabled = True`

Observed:

- placement verification passed
- generation succeeded
- `qk_full_shadow_max_abs_diff = 0.0`
- `softmax_av_fused_shadow_max_abs_diff = 4.768e-07`

Interpretation:

- the deeper fused path is correctly wired end-to-end on the real cluster
- Python/backend no longer needs the full score tensor in the normal path

### 51. Small concurrent deeper-fused trace

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_tok2_dpu64_fp32_fullqk_qksavfused.jsonl`

Observed:

- `avg_latency ~= 4.10 s`
- `max_latency ~= 4.12 s`
- `max_usage_ratio = 0.2095`
- `max_dpu_allocate_failures = 0`
- `max_fallback_allocations = 0`
- `max qk_full_shadow_max_abs_diff = 5.722e-06`
- `max softmax_av_fused_shadow_max_abs_diff = 6.104e-05`

Important reading:

- this run still keeps correctness shadows enabled
- so it is not a pure performance measurement of the new boundary
- the extra shadow QK path still appears in the trace counters

Interpretation:

- this should be treated as a correctness/stability baseline
- performance conclusions should wait until we can compare:
  - shadow-on correctness mode
  - shadow-off steady-state mode

## Updated Baseline Ladder

The current functional baselines are now:

1. disaggregated CPU attention
2. resident-KV + host-softmax + resident-AV
3. resident-KV + full-QK + helper-boundary softmax+AV
4. resident-KV + single-helper-RPC `QK+softmax+AV->context`

The new fourth baseline is the right starting point for the next round of
full-PIM-oriented optimization, because the remaining gap is now much more
specific:

- helper CPU still owns softmax
- helper still orchestrates AV
- shadow validation still costs extra work during correctness runs

## 52. DPU-side softmax baseline

We then advanced the fused path again so that softmax normalization now runs on
the DPU side instead of the helper CPU side.

Current fused boundary:

1. resident KV stays on DPU
2. DPU computes QK
3. DPU computes normalized attention weights
4. helper forwards those weights into AV
5. DPU computes context

### 52.1 `.7` local fused-store validation

Observed:

- `max_diff = 1.1921e-07`

Interpretation:

- DPU-side softmax is numerically consistent at the direct store boundary

### 52.2 Three-machine correctness after DPU-side softmax

Workload:

- cluster: `.3` prefill GPU, `.4` dense GPU, `.7` attention CPU+UPMEM
- model: `OPT-125M`
- `64 DPU`
- `max_new_tokens = 2`
- `pim_qk_full_enabled = True`
- `pim_qk_mixed_enabled = False`
- `pim_softmax_av_fused_enabled = True`

Observed:

- placement verification passed
- generation succeeded
- `qk_full_shadow_max_abs_diff = 0.0`
- `softmax_av_fused_shadow_max_abs_diff = 4.7684e-07`

Intermediate finding during bring-up:

- a lower-order DPU exponential approximation initially produced
  `softmax_av_fused_shadow_max_abs_diff = 1.2207e-04`
- after upgrading the approximation, verify-level correctness was restored

Interpretation:

- this gives us a stronger full-PIM-oriented baseline than the previous
  helper-softmax variant
- the main remaining inefficiency is now the intermediate normalized-weight
  bounce:
  - DPU -> helper
  - helper -> DPU AV path

## 53. Removing the normalized-weight bounce

We then removed the intermediate normalized-weight bounce from the fused path.

New fused dataflow:

1. resident KV stays on DPU
2. DPU computes QK
3. DPU computes normalized weights
4. DPU writes normalized weights directly into resident `av_weights_bits`
5. AV reuses the already-resident weights in place
6. helper returns only context

### 53.1 Bring-up issue

The first direct-write version was incorrect for odd window lengths.

Observed on the `.7` local fused-store test:

- `max_diff = 2.57`
- then `max_diff = 0.70` after a partial fix

Root cause:

- AV reads weights using a globally compact `num_heads * seq_len` layout
- writing normalized weights row-by-row caused a packing mismatch when
  `window` was odd

Final fix:

- write normalized weights using the same globally compact packed layout that
  AV consumes

### 53.2 Final validation

Observed:

- `.7` local fused-store test:
  - `max_diff = 1.1921e-07`
- three-machine placement verify:
  - passed
  - `qk_full_shadow_max_abs_diff = 0.0`
  - `softmax_av_fused_shadow_max_abs_diff = 4.7684e-07`

Interpretation:

- the fused path now removes both:
  - Python-side score materialization
  - normalized-weight host bounce
- this is the strongest current functional full-PIM-style baseline in the
  codebase
