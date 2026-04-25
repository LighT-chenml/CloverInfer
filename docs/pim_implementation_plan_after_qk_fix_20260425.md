# PIM Implementation Plan After QK Correctness Fix

Date: 2026-04-25

## Purpose

This document records the implementation roadmap after the grouped
multi-head resident-slot `QK` correctness bug was fixed.

The goal here is practical execution:

- keep future optimization work anchored to trustworthy baselines
- avoid repeating older conclusions drawn from invalid measurements
- define the next implementation steps in a stable order

## Current Trustworthy Baseline

Current state:

- three-machine `PD + AF` separation is running end-to-end
- resident KV is active on the UPMEM path
- grouped mixed-`QK` correctness has been restored
- resident `AV` is active on the PIM node

Current trustworthy small-trace baseline:

- grouped resident-slot mixed-`QK`
- compact grouped score layout
- token-parallel `QK` kernel
- resident `AV`

Current small-trace result:

- `avg_latency = 1.8203 s`
- `avg_tpot = 1.6384 s`
- artifact:
  - `artifacts/pim_allocator_trace_opt125m_humaneval4_conc1_tok2_dpu4_fp32_balanced_rotated_residentqk_slotqk_grouped_kernelmulti_tokenparallel_compactreadback_upmemav.jsonl`

Important rule:

- all future optimization comparisons should use this post-fix baseline
- older fast grouped multi-head numbers from before the readback-layout fix
  should be treated as suspect and not used for decision-making

## Working Principles

### 1. Correctness before speed

Every optimization step must preserve:

- end-to-end three-machine execution
- exact mixed-`QK` agreement on the current debug path
- stable resident-KV lifecycle behavior

### 2. Optimize the real bottleneck

Recent evidence says:

- helper-side batching / overlap is not the main bottleneck right now
- kernel-internal work partitioning and data movement matter more

So the mainline should stay inside the grouped mixed-`QK` kernel before
returning to helper-protocol work.

### 3. Keep each step measurable

Each implementation round should produce:

- code changes
- one minimal correctness result
- one benchmark artifact
- one short written conclusion

## Short-Term Plan

### Stage 1. Optimize grouped mixed-QK K-side access

This is the highest-priority next step.

Target:

- reduce the cost of reading resident `K` inside the grouped mixed-`QK`
  kernel

Why this is first:

- token-parallel work partitioning already improved performance
- the remaining kernel cost is still dominated by repeated `K` access and
  scalar unpack behavior
- this path is more promising than further helper overlap

Candidate implementation directions:

- reduce repeated `read_k_value()` calls
- replace scalar-style reads with small packed / tiled reads
- improve sequential MRAM access locality
- stage small `K` tiles into WRAM where possible

Recommended order inside Stage 1:

1. keep the token-parallel structure
2. optimize `K` reads without changing host/helper protocol
3. re-measure
4. only then consider a larger kernel restructuring

Acceptance criteria:

- `qk_mixed_last_max_abs_diff = 0.0`
- no new `NaN` / `Inf` grouped mixed-`QK` failures
- small-trace latency improves or at least does not regress materially

### Stage 2. Revisit tasklet partitioning after K-read cleanup

Only do this after Stage 1 is measured.

Goal:

- find a better tasklet mapping once `K` access is less wasteful

Candidate directions:

- keep token-parallel but use token tiles
- assign each tasklet a contiguous token block
- try small `(head, token)` tile mappings if WRAM and synchronization allow

What not to do first:

- do not immediately add more helper-side scheduling complexity
- do not mix several kernel changes into one unmeasurable patch

Acceptance criteria:

- exact mixed-`QK` correctness still holds
- end-to-end small-trace latency is better than the current trustworthy
  baseline

### Stage 3. Re-check whether resident AV becomes the next limiter

After mixed-`QK` improves further, re-evaluate the bottleneck split.

Questions to answer:

- is mixed-`QK` still the dominant attention-node cost?
- has resident `AV` become the next main optimization target?

Only after this check should we decide whether the next engineering round is:

- deeper mixed-`QK` kernel work
- or resident-`AV` kernel / data-movement optimization

## Mid-Term Plan

### Stage 4. Scale DPU count and workload

The current small trace is still lightweight.

We should next test whether the current conclusions survive under:

- more DPUs:
  - `32 / 64 / 128 / 256 / 512`
- longer contexts
- more decode steps
- higher concurrency

Goal:

- identify which optimizations are real architectural wins
- identify which gains disappear once workload shape changes

Metrics to track:

- latency
- TPOT
- allocator usage ratio
- fallback count
- live-slot distribution across DPUs

### Stage 5. Move to more realistic workloads

After small-trace kernel work stabilizes, run broader traces with:

- real datasets from `dataset/`
- longer decode
- real model configurations already supported in the repo

Purpose:

- validate that the current dense causal-attention baseline still works
- expose heavier communication and memory behavior
- see whether host/helper overhead reappears at larger scale

## Longer-Term Plan

### Stage 6. Refine the KV-in-DPU workflow

This is not the immediate optimization target, but it should stay on the
roadmap.

Topics to develop later:

- long-lived DPU allocation strategy
- request-to-DPU routing
- append behavior under multi-request load
- MRAM space reuse and fragmentation control
- per-request / per-layer / per-group balancing

This becomes more important once current kernel performance is stable enough
that storage-management overhead starts to matter.

### Stage 7. Explore sparse attention and GQA as evaluation branches

These should be treated as later branches, not immediate blockers.

#### Sparse attention

Worth evaluating because PIM may benefit when the sparsity structure remains
regular.

More reasonable first candidates:

- sliding-window attention
- block-sparse attention

Less suitable first candidates:

- highly irregular query-dependent sparse schemes

#### Grouped-Query Attention

GQA should be included as a realistic evaluation setting because it changes:

- KV footprint
- bandwidth pressure
- head/group mapping

But it should not replace the current dense baseline as the primary
architecture target until the dense path is better understood.

## Implementation Order

Recommended execution order from this point:

1. optimize grouped mixed-`QK` `K`-side reads
2. run minimal correctness and small-trace measurement
3. if needed, refine tasklet partitioning
4. re-check whether mixed-`QK` or resident `AV` is now the main limiter
5. run DPU-count and workload sweeps
6. only then shift attention to deeper resident-`AV` work or fuller KV
   workflow redesign

## Validation Checklist Per Round

Each round should include:

1. code implementation
2. syntax / build check
3. minimal three-machine verification
4. one benchmark artifact
5. one short result summary added to `docs/`

Required checks:

- grouped mixed-`QK` diff stays exact on the current verification path
- no stale helper / allocator regression
- resident request lifecycle returns to empty after request free
- no silent fallback unless explicitly intended

## Bottom Line

The current mainline should be:

- continue optimizing grouped mixed-`QK` inside the kernel
- prioritize `K`-side read efficiency over more helper scheduling work
- keep every step tied to the corrected post-fix baseline
- expand to larger DPU counts and heavier workloads only after the next kernel
  round is measured cleanly
