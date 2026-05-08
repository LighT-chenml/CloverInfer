# CloverInfer Prompt Alignment (2026-05-08)

## Goal

This note maps the current CloverInfer repository against the five proposed
design prompts, calls out what is already present, and records the adjustments
needed to make the design fit the existing codebase instead of competing with
it.

## Current Repository Reality

### What already exists

- `src/core/attention_backend.py`
  - request-level `preferred_dpu_stripe`
  - `rankset_plan`
  - `head_grouping_policy`
  - `load_aware` placement
  - dynamic stripe expansion
- `src/core/scheduler.py`
  - scheduler-owned decode queue
  - predictive batch selection based on observed dense/attention timings
  - rankset overlap planning and task-graph scaffolding
- `src/core/nodes.py`
  - batched attention decode
  - simulated rankset transfer / partial completion path
- `src/core/resident_kv_store.py`
  - resident KV abstraction
  - UPMEM helper client
  - allocator telemetry already surfaced to Python
- `src/pim/upmem_kvslot`
  - host-side slot allocator and helper runtime
  - DPU-side KV/QK/AV kernels

### What is not yet cleanly implemented

- There is no standalone sharding planner contract.
- Predictive scheduling is tied to the Ray runtime rather than exposed as a
  reusable algorithm component.
- The rankset overlap path is still mostly a scaffold on top of serial compute.
- There is no isolated host reduction engine API for Prompt 3.
- There is no standalone DPU MRAM allocator module matching Prompt 2.
- There is no minimal AFD simulator that can compare fixed batching against
  alignment-aware batching outside the full Ray stack.

## Prompt-By-Prompt Adjustment

### Prompt 1: Sharding Planner

Adjustment:

- Do not replace `preferred_dpu_stripe` / `rankset_plan`.
- Introduce a planner module that produces a stable plan object first.
- Later, let `AttentionNode.init_request()` and `ResidentKVStore` consume that
  plan.

Reason:

- Current head grouping and stripe selection are scattered across backend
  heuristics.
- A stable planner contract is the missing layer that can unify those
  heuristics without rewriting the runtime first.

Implemented first step:

- `src/core/clover_planner.py`
  - `plan_sharding(requests, D, H)`
  - `update_sharding(existing_plan, new_requests)`
  - per-DPU token load summary
  - load standard deviation

Important design choice:

- Token ranges stay contiguous per DPU.
- That is less theoretically flexible than arbitrary token scatter, but it
  matches the repository's blocked resident-KV layout and is much easier to
  materialize on UPMEM.

### Prompt 2: DPU MRAM Allocator

Adjustment:

- Do not fold this directly into `upmem_kvslot` yet.
- Land it as an isolated DPU-side allocator scaffold first.

Reason:

- `upmem_kvslot` already has a host-managed slot allocator.
- Replacing it in-place would create too much risk before the allocator API is
  validated.

Implemented first step:

- `src/pim/upmem_allocator/dpu_allocator.c`
  - WRAM bitmap
  - next-fit allocation
  - free
  - compaction with remap table

Important design choice:

- `mem_compact()` is documented as stop-the-world.
- That matches the practical requirement that the host must refresh all moved
  pointers anyway.

### Prompt 3: Host Reduction Engine

Adjustment:

- Treat reduction as an independent host library rather than embedding it into
  the current Python scheduler first.

Reason:

- Current CloverInfer overlap work is mostly at the request/rankset orchestration
  layer.
- Prompt 3 needs a clean math kernel and buffer contract before it can be
  connected to real UPMEM result transfers.

Implemented first step:

- `src/host/reduction/attention_reducer.h`
- `src/host/reduction/attention_reducer.cc`

Current scope:

- global max / global sum reduction
- numerically correct output rescaling
- double-buffer slot bookkeeping scaffold

Not yet wired:

- actual `dpus_copy_from` / asynchronous host-DPU I/O
- integration with `AttentionNode`

### Prompt 4: Capacity-Aware Micro-Batch Scheduler

Adjustment:

- Keep the repository's existing runtime predictor in place.
- Add a standalone scheduler component with injected planner/predictor/capacity
  checker interfaces.

Reason:

- The main scheduler already contains predictive selection logic, but it is hard
  to test in isolation.
- Prompt 4 is best implemented as a reusable algorithm layer first.

Implemented first step:

- `src/core/clover_scheduler_components.py`
  - greedy batching
  - optional lookahead reordering
  - injected PIM/host predictors
  - injected capacity checker

### Prompt 5: End-to-End AFD Main Loop

Adjustment:

- Start with a deterministic simulator, not the full Ray runtime.

Reason:

- The repository already has a correctness-first distributed runtime.
- What is missing for design iteration is a compact throughput simulator that
  lets us compare policies quickly.

Implemented first step:

- `src/core/clover_pipeline_sim.py`
  - transfer / DPU / reduce / FC stage overlap model
  - fixed-batch vs optimized-batch comparison support
  - stage utilization reporting

## Recommended Integration Order

1. Use `clover_planner.py` to replace ad hoc stripe-selection heuristics.
2. Connect `clover_scheduler_components.py` to `GlobalScheduler._take_decode_batch`.
3. Replace simulated rankset transfer timing in `nodes.py` with real reducer and
   host-copy stages.
4. Evaluate whether the new DPU allocator should replace or augment the current
   `upmem_kvslot` host-managed slot allocator.

## Design Calls That Should Change From The Original Prompt

### 1. Non-uniform token splitting should remain contiguous

- Original prompt allowed arbitrary non-uniform token cuts.
- In this repository, contiguous ranges are the right default because resident
  KV is already blocked and layered around contiguous logical growth.

### 2. Prompt 2 allocator should not immediately own all resident-KV placement

- The repository already has working host-side placement and telemetry.
- A clean allocator module should be validated before deeper replacement.

### 3. Prompt 3 should reduce per-request partials, not only raw DPU outputs

- The existing CloverInfer path already reasons in ranksets and grouped head
  slices.
- The reducer contract should stay compatible with that granularity.

### 4. Prompt 4 should not duplicate runtime prediction logic

- The new scheduler component should be the testable algorithm surface.
- The existing scheduler should eventually call into it, not maintain a second
  unrelated predictive policy.

### 5. Prompt 5 should begin as a policy simulator

- The repository already has an end-to-end runtime.
- The missing tool is a cheap environment to validate overlap and alignment
  policy choices before pushing them into Ray + UPMEM.
