# Resident-KV PIM Baseline Design

Date: 2026-04-24

## Purpose

This document defines the next implementation target after the helper-based
`pim_naive` baseline:

- keep DPU allocation persistent
- keep request KV cache resident on the PIM node
- reduce per-step KV movement from host to DPU
- preserve a correctness-first implementation boundary

This is intended to be the first baseline that actually exercises a
PIM-resident KV workflow, rather than only a persistent host-side DPU runner.

## Why This Is The Next Step

Current state:

- the helper-based naive PIM path is now close to `disagg_cpu`
- host orchestration was a first-order bottleneck and has been substantially
  reduced
- the current baseline still does not keep KV cache resident on DPU memory

Therefore the next meaningful question is no longer:

- "can we remove obvious subprocess/file overhead?"

It is now:

- "does a resident-KV PIM workflow expose a new performance regime?"

## Scope Of This Baseline

This baseline is intentionally limited.

Included:

- one-time DPU allocation and binary load per attention actor lifecycle
- KV residency on the attention node with explicit placement metadata
- incremental KV append per decode step
- query-only hot-path submission plus incremental `k_new/v_new`
- correctness comparison against `CpuAttentionBackend`

Not included:

- aggressive multi-request packing
- complex fragmentation control
- request migration across DPUs
- distributed softmax fully inside DPUs
- full `AV` offload as a hard requirement for the first version
- overlap between transport and PIM execution

## Position In The Baseline Ladder

The paper/system narrative should now distinguish these two baselines:

### Baseline A: Helper-Based Naive PIM

Properties:

- persistent DPU runner
- batched QK submission
- no temp files / no per-call process spawn
- KV still prepared and shipped from host-side hot path

Research question:

- how much of the original slowdown came from host orchestration?

### Baseline B: Resident-KV PIM

Properties:

- persistent DPU runner
- KV resident on DPU-managed storage
- incremental append for decode
- reduced hot-path KV movement

Research question:

- once host orchestration is controlled, does PIM-side KV residency create a
  measurable advantage region?

## First-Version Design Choice

The recommended first version uses:

- head-wise or head-group-wise KV sharding
- `QK` on the PIM side
- softmax and `AV` initially on the host side

Reason:

- it isolates the effect of KV residency and incremental append
- it avoids taking on distributed softmax and distributed `AV` at the same
  time
- it keeps the result easier to validate against the CPU oracle

This is a deliberate engineering choice, not the final architecture.

## High-Level Workflow

### 1. Attention Actor Startup

On `AttentionNode` initialization:

1. allocate DPUs once
2. load DPU binary once
3. initialize a long-lived runner / helper process if needed
4. create host-side metadata tables for:
   - request registry
   - layer placement
   - head-group placement
   - DPU capacity / free space

Expected invariant:

- DPU setup is not repeated per request or per decode step

### 2. Request Init After Prefill

When prefill finishes on `.3`:

1. the initial per-layer KV is sent to `.7`
2. the attention backend partitions each layer's KV by head-group
3. each partition is written into the assigned DPU region
4. host metadata records:
   - request id
   - current context length
   - layer count
   - per-layer, per-head-group DPU location
   - current append offset

Expected invariant:

- once initialization finishes, the request's prefill KV is no longer treated
  as transient host-side input for each decode step

### 3. Decode Step

For each decode layer call from `.4`:

1. `.4` sends:
   - `query`
   - `k_new`
   - `v_new`
   - request id
   - layer index
2. `.7` looks up the request metadata
3. `.7` appends `k_new/v_new` into the correct DPU partition
4. `.7` routes the corresponding query heads to the relevant DPU partitions
5. DPU computes local `QK`
6. host gathers local score blocks or compact reduction outputs
7. host performs:
   - softmax
   - `AV`
   - final context assembly
8. `.7` returns attention context to `.4`

Expected invariant:

- the hot path does not resend the whole historical KV window from host memory

### 4. Request Free

When the request finishes:

1. the host marks the request's DPU regions as reclaimable
2. append offsets and placement metadata are removed
3. the DPU runtime remains alive for future requests

Expected invariant:

- request teardown reclaims request-local KV state without destroying the DPU
  runtime

## Data Layout Recommendation

## Sharding Choice

Use head-wise or head-group-wise sharding first.

Example:

- if a layer has `H` heads and `G` groups
- each group owns a contiguous subset of heads
- each group is assigned to one DPU or one small DPU set

Why this is recommended:

- query routing is simple
- KV append is simple
- output context assembly is simple
- correctness comparison against CPU attention is straightforward

Avoid for the first version:

- arbitrary request-wise striping across all DPUs
- token-major layouts that complicate append and head slicing

## Per-Partition Storage

For each request/layer/head-group partition, maintain:

- `K_cache`
- `V_cache`
- current logical sequence length
- maximum allocated length
- DPU address or slot metadata

The simplest first baseline should favor:

- contiguous append
- fixed-size capacity reservation per request where practical
- whole-request reclaim on free

## Host Metadata Structures

The first implementation should have explicit metadata, even if not yet
optimal.

Suggested structures:

```python
RequestState:
  request_id
  context_len
  num_layers
  layer_states: list[LayerState]

LayerState:
  head_groups: list[HeadGroupState]

HeadGroupState:
  dpu_id
  head_start
  head_end
  seq_len
  capacity
  k_slot
  v_slot
```

This may later move partly into C/C++ or a lower-level runtime, but the first
version should keep the state explicit and inspectable.

## Compute Boundary For Baseline V1

Baseline V1 should keep this split:

- DPU:
  - local `QK`
  - optional local max/sum-exp helper stats if convenient
- Host CPU on `.7`:
  - global softmax
  - `AV`
  - final context concatenation

Why this is acceptable:

- it isolates the effect of resident KV and incremental append
- it avoids overloading the first implementation with the hardest distributed
  reduction logic

## Correctness Strategy

`CpuAttentionBackend` remains the oracle.

Required checks:

1. for selected requests/layers, compare resident-KV PIM `QK` outputs against
   CPU reference
2. compare final attention context against CPU backend on small cases
3. record numerical drift metrics in benchmark output
4. fail loudly on metadata inconsistencies:
   - wrong context length
   - append offset mismatch
   - invalid DPU mapping

## Simplifying Constraints For The First Version

These constraints are recommended on purpose:

- batch size `= 1`
- one request occupies a simple contiguous region per partition
- no eviction or paging
- no in-place compaction during request lifetime
- no concurrent writes to the same request partition from multiple threads
- no attempt to optimize for maximum DPU occupancy yet

This keeps the baseline interpretable.

## Main Challenges To Expect

### 1. KV Capacity Management

Problem:

- DPU memory is limited
- initial reservation policy affects usable context length

Mitigation:

- start with conservative maximum context
- make capacity visible in logs and benchmark records

### 2. Append Correctness

Problem:

- an off-by-one in append offset or per-layer sequence length will silently
  corrupt attention

Mitigation:

- keep explicit host metadata
- add assertions on every append in the debug path

### 3. Score Gathering Cost

Problem:

- if the DPU returns large score blocks every step, host-device traffic may
  remain high

Mitigation:

- accept this in V1 if needed
- record transfer volume explicitly
- treat distributed softmax/`AV` as the next optimization stage

### 4. Multi-Request Fragmentation

Problem:

- general allocators are complex and can obscure the first result

Mitigation:

- use simple whole-request reservation and reclaim for V1
- postpone advanced packing until after correctness and basic scaling are clear

## Implementation Plan

### Step 1: Design/Interface Landing

- add this document
- define the resident-KV state model in the backend code
- separate helper-based naive path from resident-KV path conceptually

### Step 2: Host-Side Resident Metadata Prototype

- add host-side request/layer/head-group state objects
- make `init_request` build persistent placement metadata
- make `free_request` reclaim metadata and slots

### Step 3: DPU Storage Prototype

- extend the UPMEM host/runtime path so partitions can be initialized once and
  appended later
- keep the initial capacity and layout simple

### Step 4: Decode-Path Conversion

- change decode submission from "ship a temporary key window" to
  "append incremental KV + query resident storage"

### Step 5: Validation

- small `opt-125m` end-to-end correctness
- selected Qwen smoke
- compare against CPU backend

### Step 6: Benchmark Refresh

- rerun:
  - unified baseline matrix
  - mixed-head sweep if still applicable
  - context sweep

## Success Criteria

The resident-KV baseline is successful if:

1. the request lifecycle is correct end to end
2. KV is no longer rebuilt or fully resent on each decode step
3. the backend remains numerically close to the CPU oracle
4. the resulting benchmark can be compared directly with:
   - `disagg_cpu`
   - helper-based `disagg_pim_naive`

## What Comes After This Baseline

Only after resident-KV baseline stability should we push into:

- distributed softmax
- `AV` offload
- compressed or reduced score return paths
- more advanced multi-request DPU packing
- transport-level overlap or reduced `.4 <-> .7` synchronization

This ordering keeps the paper narrative clean:

1. naive orchestration-heavy PIM baseline
2. helper-based naive PIM baseline
3. resident-KV PIM baseline
4. deeper PIM-side attention reductions
