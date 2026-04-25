# PIM Optimization Next Steps

Date: 2026-04-25

## Purpose

This document records the near-term optimization direction after the current
resident-KV baselines.

The goal here is not paper writing. It is to keep the implementation work
focused on the most promising technical paths.

## Current State

We now have three useful baselines:

1. `4 DPU + fp32 resident KV`
   - capacity bottleneck baseline
   - overflows under `concurrency = 4`

2. `4 DPU + fp16 resident KV`
   - compression baseline
   - recovers the full-DPU path

3. `8 DPU + fp32 + balanced + rotated`
   - scale-out / placement baseline
   - also recovers the full-DPU path

Key observation:

- in the constrained `4 DPU` setting, reducing resident-KV footprint is more
  effective than placement tuning

## Mainline Priorities

### 1. Stress the current best baselines harder

Before adding more mechanisms, test whether the present conclusions still hold
under heavier workloads:

- longer prompts
- larger `max_new_tokens`
- higher concurrency
- real models, starting with `Qwen-1_8B`

This should answer:

- does `4 DPU + fp16` remain stable under longer decode?
- when is `8 DPU + fp32` still insufficient?
- when do we need both scale-out and resident-KV compression?

Current answer from the first heavier real-model trace:

- `Qwen-1_8B`, `max_new_tokens = 8`, `concurrency = 2`
- `4 DPU + fp16` still overflows
- `8 DPU + fp32` still overflows
- `8 DPU + fp16` recovers the full-resident path

### 2. Push real PIM-side attention computation

The most important architectural gap is still:

- KV is resident on the PIM side
- but attention computation is not yet fully staying there

Recommended order:

1. move `QK` computation further onto the PIM side
2. then implement `AV`
3. reduce host-side materialization as much as possible

The real target is not only resident KV, but also reducing return traffic from
the attention node.

### 3. Keep resident-KV compression as a first-class optimization axis

The successful `fp16` result means resident representation should remain a core
optimization dimension.

Likely next sub-directions:

- validate `fp16` under heavier workloads
- measure numerical stability on larger models
- later consider more aggressive compression:
  - `int8 KV`
  - mixed-precision KV
  - asymmetric K/V formats

## Secondary Priorities

### 4. Revisit allocator / reserve policy later

At the current lightweight workload, reserve policy is not the main bottleneck.
It matters more once:

- generation length grows
- per-request lifetime becomes longer
- append-heavy workloads dominate

This should stay on the list, but behind resident-KV footprint reduction and
actual PIM-side compute.

### 5. Improve batching / pipeline overlap

After the core backend path is more mature, worthwhile system optimizations
include:

- continuous batching
- overlapping dense and attention stages better
- reducing RPC overhead

These are important, but they are not the next bottleneck to solve first.

## Sparse Attention

Sparse attention is worth considering for a PIM-based design, but not all forms
of sparsity are equally suitable.

### More promising for PIM

- sliding-window attention
- block-sparse attention
- local + global structured sparsity

Why:

- memory access is more regular
- block mapping is easier
- metadata overhead is lower
- placement and capacity accounting stay manageable

### Less suitable as the first sparse target

- top-k token sparse patterns
- query-dependent irregular sparse attention
- runtime-dynamic sparse connectivity

Why:

- index management becomes heavy
- memory access becomes irregular
- DPU balance becomes harder
- control overhead can erase theoretical savings

### Recommendation

If sparse attention is explored, start with:

- block-sparse
- or sliding-window

Do not start with highly irregular sparse schemes.

## Grouped-Query Attention

GQA should absolutely be considered, but it changes the PIM story.

### Why it matters

GQA reduces:

- KV-head count
- KV-cache footprint
- bandwidth pressure

### What that means for PIM

Positive:

- PIM attention can still help with disaggregated KV ownership
- long context and concurrency can still create attention-node pressure

Negative:

- one of PIM's clearest advantages is large KV capacity / bandwidth pressure
- GQA directly weakens that advantage
- relative benefit versus strong GPU baselines may shrink

### Practical implication

GQA is best treated as:

- a more realistic and more difficult evaluation setting

not as the first architecture-driving assumption.

### Recommendation

- include GQA models in evaluation soon
- but do not redesign the whole system around GQA first
- first establish the dense-attention and resident-KV optimization story

## Suggested Execution Order

Recommended next sequence:

1. run the current best baselines on heavier workloads
   - especially `8 DPU + fp16 + balanced + rotated`
   - and compare against `4 DPU + fp16`
   - and `8 DPU + fp32 + balanced + rotated`

2. switch to a more realistic model
   - use `Qwen-1_8B` first on the current `V100-16GB` cluster
   - keep `Qwen-7B` as a later target after reducing dense-side memory pressure

3. continue moving `QK` onto the PIM side

4. move `AV` onto the PIM side

5. only after the dense path is mature:
   - add structured sparse-attention exploration

6. keep GQA as a realistic evaluation branch throughout

## Bottom Line

The current mainline should remain:

- resident-KV footprint reduction
- real PIM-side attention compute
- heavier and more realistic workloads

Sparse attention and GQA are both important, but they should be treated as
controlled next-stage branches rather than replacing the current mainline.

## Scheme 2 Follow-Up

The shared-owner refactor has now reached a useful checkpoint:

- resident KV and QK-mixed traffic share the same `upmem_kvslot` helper
- startup no longer runs a separate `upmem_dot` allocation smoke in this mode
- the current real-model trace is stable at `446 DPU`

The next actions should separate system issues from algorithm issues.

### Immediate system follow-up

- always check for stale helper processes before concluding that `.7` cannot
  allocate a requested DPU count
- especially look for leftover `host_qk --stdio` or `host_kvslot --stdio`
  processes from previous experiments
- after cleaning stale helpers, retry `512 DPU` and then `1020 DPU`

### Immediate backend follow-up

- keep Scheme 2 as the mainline
- do not reintroduce dual-owner `host_qk` allocation for large-DPU runs
- add clearer diagnostics when helper startup fails because a previous helper is
  still occupying DPUs

### After the allocatable-budget issue is clarified

- move QK from CPU-inside-kvslot-helper to true DPU-side compute
- then start moving AV onto the same shared-owner path
