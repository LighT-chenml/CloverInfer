# PIM Optimization Roadmap

Date: 2026-04-24

## Purpose

This document summarizes the current challenge landscape of the prototype and
proposes optimization directions. The challenges are split into:

- PIM-specific challenges
- non-PIM-specific system challenges

The goal is to preserve a paper-usable roadmap from the current prototype state
 to later optimized implementations.

## PIM-Specific Challenges

### 1. Per-head / per-layer PIM invocation granularity is too fine

Observed issue:

- The current `pim_naive` path scales roughly with `mixed_heads`.
- Attention-side decode time grows almost linearly as more heads are routed
  through the PIM-backed QK path.

Why this matters:

- The current implementation does not amortize host-side setup and launch cost.
- Too much runtime cost is paid per head instead of per useful batch of work.

Optimization directions:

- batch multiple heads into a single QK invocation
- move from per-head invocation toward per-layer invocation
- later consider batching across tokens or across requests when correctness is
  stable

### 2. Host-side orchestration dominates the naive PIM path

Observed issue:

- The current `.7` path is dominated by orchestration overhead rather than just
  raw arithmetic.
- Existing evidence points to subprocess-style and marshaling-heavy execution as
  a major bottleneck source.

Why this matters:

- Even correct PIM kernels cannot help much if the host spends too much time
  preparing, launching, and collecting tiny jobs.

Optimization directions:

- replace subprocess / file-oriented execution with a persistent host runtime
- reuse allocated buffers and DPU state across decode steps
- reduce repeated input packing and output unpacking
- keep frequently used control/data structures resident on the attention node

### 3. Too little of the attention path is currently amortized inside the PIM-side workflow

Observed issue:

- The present naive path mainly demonstrates a QK-oriented offload path.
- The full attention flow is still fragmented between host CPU and PIM-related
  work.

Why this matters:

- Excessive host/PIM handoff can erase any bandwidth-side benefit.

Optimization directions:

- first make the QK path coarse-grained and efficient
- then evaluate whether adjacent stages should be grouped into fewer host-side
  transitions
- revisit whether additional stages such as `AV` should move only after QK
  batching is improved

### 4. PIM advantage has not yet emerged in the tested context-length regime

Observed issue:

- In the current context sweep, `pim_naive` cost changes only mildly with
  context length, while CPU attention cost grows more naturally with context.

Why this matters:

- This strongly suggests the current implementation is dominated by fixed
  overhead and has not reached the region where PIM-side bandwidth benefits can
  dominate.

Optimization directions:

- reduce fixed invocation overhead first
- then rerun longer-context sweeps
- extend experiments to larger context lengths only after batching/orchestration
  is improved

### 5. KV cache placement and update policy on the attention node is still naive

Observed issue:

- The current design is correctness-first and does not yet optimize KV layout or
  update behavior for the PIM-side path.

Why this matters:

- KV movement, append behavior, and layout decisions can dominate remote
  attention efficiency.

Optimization directions:

- use more stable preallocated KV storage on the attention node
- reduce reformatting and repeated host-side tensor reshaping
- explore layer/head-aware layout choices aligned with the eventual PIM kernel
- evaluate windowed or staged KV policies only after the baseline path is
  stable

### 6. PIM correctness validation is still limited to lightweight checks

Observed issue:

- Current correctness signals are good, but still relatively narrow in scope.

Why this matters:

- Later optimization work will be difficult to trust without a stable reference.

Optimization directions:

- keep `CpuAttentionBackend` as the correctness oracle
- add automated backend-level comparisons on selected requests/layers
- record numerical drift metrics and failing configurations as part of the
  benchmark output

## Non-PIM-Specific Challenges

### 1. Remote attention RPC cadence is already expensive even without PIM

Observed issue:

- `disagg_cpu` is much slower than `split_gpu_full_decode`.
- The remote-attention baseline already pays substantial cost before naive PIM
  overhead is added.

Why this matters:

- Not all slowdown should be attributed to PIM.
- The system already has a remote-attention structural cost.

Optimization directions:

- reduce the number of `.4 <-> .7` round trips
- make each attention-side RPC carry more useful work
- revisit the current layer-by-layer synchronous choreography

### 2. Prefill/decode disaggregation has overhead, even before remote attention

Observed issue:

- `split_gpu_full_decode` is slower than `monolithic_gpu`.

Why this matters:

- PD split is not free, even though it is not the dominant current bottleneck.

Optimization directions:

- overlap prefill completion, KV shipping, and decode startup where possible
- reduce duplicate control overhead around request handoff
- later explore pipelined multi-request execution

### 3. Ray RPC transport and tensor marshalling remain a measurable tax

Observed issue:

- The current prototype uses Ray RPC for correctness and convenience.
- The stage timings suggest that compute time alone does not explain latency.

Why this matters:

- Fine-grained actor interactions make transport overhead much more visible.

Optimization directions:

- continue quantifying transport cost through stage timing breakdowns
- introduce a cleaner transport abstraction before deeper optimization
- evaluate lighter transports or RDMA only after call granularity is improved

### 4. Full model copies on both GPU nodes constrain scaling

Observed issue:

- The current design requires a full model copy on both the prefill and decode
  dense nodes.
- This is why `Qwen-7B` is not currently a practical first baseline on the
  available `V100-16GB` nodes.

Why this matters:

- Model memory limits affect which experiments are even possible.

Optimization directions:

- stay on `Qwen-1_8B` for the main baseline phase
- later investigate lower-memory loading or more explicit model partitioning
- revisit larger models only after the system baseline is more mature

### 5. Batch=1 correctness-first decode limits throughput

Observed issue:

- The current pipeline is intentionally simple and decode-oriented.

Why this matters:

- Many disaggregated designs become more favorable only when fixed overhead is
  amortized over larger batches or longer decode traces.

Optimization directions:

- add small-batch experiments after the current correctness path is stable
- compare whether remote attention overhead is amortized better under batching
- avoid overgeneralizing from very short, tiny-sample runs

### 6. Current benchmark scale is still early-stage

Observed issue:

- Existing Qwen experiments are still small and primarily diagnostic.

Why this matters:

- The current matrix is enough to identify bottlenecks, but not enough for a
  final empirical claim.

Optimization directions:

- increase sample count and repeat count for core baselines
- sweep context length and output length under the unified benchmark entry
- standardize outputs, configuration logging, and commit tracking for paper use

## Recommended Optimization Order

### Stage 1: Fix the naive PIM invocation pattern

- batch heads
- reduce per-launch overhead
- remove subprocess/file-heavy execution patterns

### Stage 2: Fix remote attention interaction granularity

- reduce `.4 <-> .7` synchronization frequency
- make each attention-side call do more work

### Stage 3: Re-evaluate scaling regimes

- rerun head sweep
- rerun context sweep at longer lengths
- introduce small batch sizes

### Stage 4: Revisit deeper compute-path changes

- reconsider whether additional attention stages should move
- only then revisit `AV`, more advanced kernels, or more aggressive cache
  policies

### Stage 5: Revisit transport optimization

- evaluate lower-level transports after upper-layer granularity is improved
- otherwise transport tuning risks optimizing the wrong bottleneck

## Paper-Oriented Summary

The current roadmap implies the following narrative:

- The present framework already isolates the relevant bottlenecks.
- PIM-specific work should focus first on amortizing host-side orchestration.
- Non-PIM system work should focus first on making remote attention less
  fine-grained.
- Only after those steps is it reasonable to expect the PIM-based design to
  approach or surpass the CPU remote-attention baseline in the tested setup.
