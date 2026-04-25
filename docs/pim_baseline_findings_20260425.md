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

## Updated Practical Takeaway

There are now two separate conclusions:

1. Scheme 2 fixes the software ownership problem for large-DPU shared
   resident-KV/QK execution.

2. Large-DPU failures after the refactor should first be checked for stale
   helper processes before being interpreted as architectural limits.
