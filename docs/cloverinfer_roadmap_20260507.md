# CloverInfer Roadmap (2026-05-07)

## Purpose

This document records the current CloverInfer optimization roadmap relative to
the `Naive PIM` baseline.

It serves three roles:

- preserve the paper-facing optimization story
- map each optimization target to the current codebase
- define a staged implementation and validation plan

Important scope rule:

- `PimNaiveAttentionBackend` remains the baseline
- CloverInfer-specific optimization logic stays isolated as much as possible
- scheduler and runtime changes may be shared, but baseline semantics should
  remain intact unless explicitly intended

## Baseline vs CloverInfer

### Naive PIM baseline today

Primary implementation:

- `src/core/attention_backend.py`
- `src/core/resident_kv_store.py`

Current baseline semantics:

- DPU-first resident KV path
- `QK + AV` main path on DPU for benchmark entrypoints
- default baseline policy now prefers:
  - `dpu_placement_policy = load_aware`
  - `head_grouping_policy = coarse`

Baseline intent:

- serve as a correctness-preserving but relatively unoptimized PIM comparison
- tolerate worse performance if it better reflects a simpler offload design

### CloverInfer today

Primary implementation:

- `src/core/clover_attention_backend.py`
- `src/core/scheduler.py`
- `src/core/resident_kv_store.py`

Current CloverInfer-specific advantages over `Naive PIM`:

1. CPU shadow KV is separated from the fast path
2. shadow checks are sampled and configurable
3. hot-path sub-operations are individually profiled
4. batched QK / AV paths are richer than the baseline path
5. an experimental `QK + softmax + AV` fused context path already exists
6. request packing hints already flow from attention backend to scheduler
7. scheduler already contains:
   - decode continuous batching
   - decode wave persistence
   - attention layer barrier
   - attention wavefront batching
   - cross-key batch merge

Practical interpretation:

- CloverInfer already has the right skeleton
- but the three target optimizations are not yet fully connected into a
  complete end-to-end policy

## Target Optimizations

This roadmap focuses on the following three target optimizations.

### 1. Bubble-free pipeline and mixed-granularity relayout

Goal:

- eliminate layout mismatch and cross-chip / cross-subarray traffic inside the
  DIMM-PIM path
- keep the PIM-side execution pipeline busy
- avoid paying extra data movement cost caused by a mismatch between logical
  attention grouping and physical PIM layout

Desired paper story:

- use coarse-grained placement when it improves resident locality and transfer
  amortization
- use finer-grained layout only where it reduces load imbalance or unlocks
  better overlap
- combine these without creating extra pipeline bubbles

### 2. Rankset-granularity communication-computation overlap

Goal:

- overlap GPU-side dense compute with PIM-side transfer / execution
- make GPU-PIM communication asynchronous at the finest independent granularity
- hide transfer cost by slicing attention-side work into independently
  schedulable rankset units

Desired paper story:

- communication is no longer performed as a monolithic attention-stage fence
- each rankset becomes the atomic unit for overlap
- end-to-end decode time becomes closer to the max of dense-side and PIM-side
  critical paths rather than their sum

### 3. Alignment-aware predictive scheduling

Goal:

- choose request sub-batches using a latency model for GPU and PIM devices
- minimize cross-device waiting time
- reduce bubbles caused by mismatched dense-side and attention-side service time

Desired paper story:

- sub-batch formation is no longer heuristic-only
- the scheduler predicts per-request latency on both devices
- selected requests should align better in their parallel execution timeline

## Current Code Mapping

### CloverInfer backend hot path

Relevant files:

- `src/core/clover_attention_backend.py`
- `src/core/attention_backend.py`
- `src/core/resident_kv_store.py`

Relevant current mechanisms:

- `_apply_qk_context_fused_batch`
- `_apply_qk_full_batch`
- `_apply_qk_mixed_batch`
- `_finalize_decode_records`
- request resident metadata and `preferred_dpu_stripe`
- resident blocked-slot KV placement

What already exists:

- fused context experimental path
- rank-spread allocation experimental flag
- fine-head-grouping experimental flag
- target-heads-per-group experimental knob

What is still missing:

- a single mixed-granularity relayout policy
- an explicit rankset abstraction in the backend/runtime
- explicit pipeline-stage orchestration tied to rankset completion

### Scheduler side

Relevant file:

- `src/core/scheduler.py`

Relevant current mechanisms:

- decode continuous batching
- attention layer barrier
- attention wavefront batching
- cross-key merge
- decode wave persistence
- request packing hints:
  - `rank_index`
  - `stripe_width`
  - `preferred_dpu_stripe`

What already exists:

- batching and synchronization skeletons
- limited locality-aware reordering

What is still missing:

- explicit latency prediction model
- explicit communication / compute overlap state per rankset
- a cost function that optimizes predicted cross-device bubble time

### Attention node batching

Relevant file:

- `src/core/nodes.py`

What already exists:

- backend-side `decode_layer_batch`
- actor-side decode batching window

What is still missing:

- rankset-aware partial completion path
- asynchronous streaming of intermediate PIM-side results back into dense-side
  progress

## Gaps Relative To The Three Targets

### Target 1 gap: mixed-granularity relayout is only experimental and local

Current status:

- layout knobs exist
- placement policy exists
- request stripe metadata exists

Missing pieces:

- no global policy that decides when to use coarse vs fine grouping
- no explicit physical relayout objective tied to cross-chip traffic avoidance
- no pipeline orchestration that guarantees bubble-free progression after
  relayout

Conclusion:

- target 1 is partially scaffolded but not implemented as a complete strategy

### Target 2 gap: overlap skeleton exists, but rankset is not yet the atomic unit

Current status:

- scheduler can batch, gate, merge, and persist decode waves
- attention RPC batching can merge multiple keys

Missing pieces:

- no per-rankset task graph
- no independent readiness / transfer / compute tracking per rankset
- no streaming completion path from attention-side partial work to dense-side
  progress

Conclusion:

- target 2 is architecturally feasible now, but requires a real rankset
  execution abstraction

### Target 3 gap: scheduler is still heuristic-driven

Current status:

- scheduler uses packing hints and locality-aware heuristics
- CloverInfer already records the timings needed for a first latency model

Missing pieces:

- no device execution model
- no sub-batch score based on predicted idle time
- no online parameter adaptation using observed dense/PIM stage timings

Conclusion:

- target 3 is the easiest near-term implementation and should be the first
  production optimization round

## Recommended Implementation Order

Recommended order:

1. alignment-aware predictive scheduling
2. rankset-granularity communication-computation overlap
3. bubble-free mixed-granularity relayout

Reasoning:

- target 3 has the lowest implementation risk
- target 2 can reuse and extend scheduler batching skeletons
- target 1 is the most invasive and benefits from first knowing the scheduler
  and overlap behavior we want to optimize for

## Phase Plan

### Phase A: Roadmap baseline and instrumentation lock-in

Goal:

- freeze the measurement surface before deeper optimization

Tasks:

- keep CloverInfer profiling enabled on optimization runs
- keep stage timing and backend timing fields stable
- preserve request packing hint output in attention init

Primary files:

- `src/core/clover_attention_backend.py`
- `src/core/scheduler.py`
- `src/core/attention_backend.py`

Deliverable:

- reproducible metrics for scheduler, dense-side, attention-side, and resident
  store timing

### Phase B: Alignment-aware predictive scheduling

Goal:

- replace heuristic-only decode batch formation with a latency-aware selector

Primary idea:

- build an online latency model for:
  - dense prepare/start/finish
  - attention decode
  - optional resident fused path
- score candidate sub-batches by predicted cross-device bubble time

Likely implementation points:

- `src/core/scheduler.py`

Suggested design:

- add an online EMA model per backend and per coarse context bucket
- estimate:
  - `dense_cost(request)`
  - `attention_cost(request)`
- choose batch members that minimize:
  - `max(pred_dense_batch, pred_attention_batch)` slack
  - plus locality penalties if needed

Deliverable:

- a scheduler mode that can be enabled without changing CloverInfer backend
  semantics

Expected risks:

- noisy measurements on short outputs
- overfitting to one model or one dataset

Validation:

- reduced scheduler-side idle/bubble indicators
- lower `attention_decode_rpc_s` variance across batched requests
- improved TPOT under concurrency > 1

### Phase C: Rankset-granularity overlap

Goal:

- turn attention-side work into independently overlappable rankset units

Primary idea:

- split a decode-layer attention task into rankset sub-operations
- allow GPU-side progress and PIM-side transfer/compute to overlap
- avoid all-or-nothing attention stage synchronization

Likely implementation points:

- `src/core/scheduler.py`
- `src/core/nodes.py`
- `src/core/clover_attention_backend.py`
- possibly `src/core/resident_kv_store.py`

Suggested design:

- introduce rankset descriptors into payload preparation
- track subtask lifecycle:
  - ready
  - in transfer
  - computing
  - completed
- return partial attention results at rankset granularity

Deliverable:

- a rankset-aware asynchronous attention execution path

Expected risks:

- higher scheduling complexity
- partial-result reassembly cost
- correctness drift if partial synchronization is mishandled

Validation:

- visible overlap between dense-side compute and attention-side transfer/compute
- reduced exposed communication cost in stage timing
- improved TPOT at medium/high concurrency

Current implementation progress as of 2026-05-08:

- request packing hints now include `rankset_plan` and per-layer
  `layer_group_map`
- scheduler task graphs no longer stop at one work item per rankset; when
  `--clover-rankset-overlap-transfer-granularity rankset` is enabled they can
  split a single request into multiple group-slice work items inside the same
  rankset
- `AttentionNode` supports:
  - async transfer staging
  - serial compute consumption
  - per-work-item event timelines
- CloverInfer backend now exposes a partial path:
  - prepare decode records once
  - compute group-slice partial contexts per work item
  - assemble the final per-request context after all partials complete

Practical interpretation:

- the codebase now has a real single-request multi-work-item execution path
- overlap is still conservative:
  - transfer is async
  - compute is still consumed serially inside the attention actor
- this is enough to validate task-graph granularity, partial readiness, and
  reassembly correctness before attempting deeper async execution

### Phase D: Mixed-granularity relayout and bubble-free pipeline

Goal:

- jointly optimize grouping, placement, and pipeline flow

Primary idea:

- choose grouping granularity using both:
  - resident locality / capacity needs
  - rankset overlap opportunities
- relayout requests so physical execution avoids avoidable cross-chip movement

Likely implementation points:

- `src/core/attention_backend.py`
- `src/core/clover_attention_backend.py`
- `src/core/resident_kv_store.py`

Suggested design:

- define a relayout policy over:
  - head grouping
  - stripe width
  - rankset allocation
  - block growth behavior
- integrate relayout decisions with scheduler-side overlap and prediction

Deliverable:

- a coherent CloverInfer pipeline policy instead of isolated experimental flags

Expected risks:

- large surface area
- difficult regression analysis
- interaction with model-specific head/group geometry

Validation:

- lower internal transfer pressure
- fewer exposed pipeline bubbles
- end-to-end gains over both `Naive PIM` and current CloverInfer

## Immediate Next Step

The next implementation step should be:

- Phase B: alignment-aware predictive scheduling

Why:

- it requires the least invasive changes
- it already has the required instrumentation support
- it can improve both current CloverInfer and future overlap-aware paths

## Predictive Scheduling Status (2026-05-07)

Current implementation status:

- online EMA-based predictive models for dense-side and attention-side latency
  have been added in `src/core/scheduler.py`
- predictive scheduling is CloverInfer-only and opt-in
- benchmark entrypoints now expose:
  - `--decode-continuous-batch-window-s`
  - `--decode-continuous-batch-window-ms`
  - `--decode-continuous-batch-max-size`
  - `--clover-predictive-scheduling-*`

Important validation finding:

- predictive scheduling only has a chance to make decisions when:
  - `concurrency > decode_continuous_batch_max_size`
  - and a positive decode batching window allows queue formation
- if `concurrency <= max_size`, the scheduler often flushes the whole queue and
  there is no sub-batch selection opportunity

Validated trigger configuration:

- dataset: `dataset/longbench/qasper.jsonl`
- model: `opt-125m`
- prompt length: `256`
- output length: `16`
- decode continuous batch window: `2 ms`
- decode continuous batch max size: `3`
- concurrency: `6`

Observed A/B result on the validated trigger configuration:

- predictive disabled:
  - `TTFT = 0.441 s`
  - `TPOT = 3.013 s`
  - `Throughput = 2.094 tok/s`
- predictive enabled:
  - `TTFT = 0.482 s`
  - `TPOT = 2.978 s`
  - `Throughput = 2.110 tok/s`

Observed scheduler evidence:

- `predictive_batch_decisions = 57`
- `predictive_model_updates = 180`
- `reordered_flushes = 1`

Interpretation:

- predictive scheduling is now functionally exercised end-to-end
- the current gain is modest but real on TPOT / throughput
- TTFT regression is small and should be treated as an optimization target
- future comparisons must use workloads with genuine candidate-selection
  pressure, not just batched flushing

Recommended benchmarking rule for predictive scheduling:

- always keep `concurrency > decode_continuous_batch_max_size`
- use a small positive decode batching window such as `1-2 ms`
- prefer medium or long outputs so online models have enough warm-up samples

## Metrics To Track

For all CloverInfer optimization rounds, report:

- TTFT
- TPOT
- throughput
- scheduler stage timing
- attention backend timing
- resident store timing
- number of DPU items vs host fallback items
- batching statistics:
  - decode continuous batching
  - attention wavefront batching
  - cross-key merge frequency
- request packing statistics:
  - stripe width
  - rank hint usage
  - reorder frequency

For predictive scheduling specifically, add:

- predicted dense cost
- predicted attention cost
- predicted batch slack / bubble cost
- observed batch slack / bubble cost

## Success Criteria

### For predictive scheduling

- lower average waiting time between dense-side and attention-side stages
- improved TPOT at concurrency > 1 without hurting correctness

### For rankset overlap

- measurable overlap between communication and compute
- reduced exposed communication time per decode step

### For mixed-granularity relayout

- lower end-to-end latency than current CloverInfer under long-context decode
- lower imbalance and lower transfer-induced stalls

## Non-Goals For This Roadmap

The following are explicitly not the first priority of this roadmap:

- making `Naive PIM` fast
- optimizing monolithic GPU baselines
- changing benchmark naming/alias behavior
- redesigning the DPU kernel before scheduler-side gains are understood

## Related Files

- `src/core/clover_attention_backend.py`
- `src/core/attention_backend.py`
- `src/core/resident_kv_store.py`
- `src/core/scheduler.py`
- `src/core/nodes.py`
- `docs/cloverinfer_optimization_status_20260428.md`
- `docs/cloverinfer_pim_attention_analysis.md`
- `docs/pim_optimization_roadmap_20260424.md`
