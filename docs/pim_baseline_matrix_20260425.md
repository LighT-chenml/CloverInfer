# PIM Baseline Matrix

Date: 2026-04-25

## Hardware Confirmation

GPU nodes:

- `192.168.123.3`
  - `Tesla V100-PCIE-16GB`
  - compute capability `7.0`
- `192.168.123.4`
  - `Tesla V100-PCIE-16GB`
  - compute capability `7.0`

Validation:

- `torch 2.9.1+cu128`
- CUDA visible on both GPU nodes
- `float16` matmul runs successfully on V100

Conclusion:

- V100 absolutely supports `fp16`
- so using `fp16` for resident-KV compression is hardware-consistent with the
  current cluster

## Current Baselines

All results below use:

- model: `OPT-125M`
- dataset: `dataset/humaneval.jsonl`
- `max_new_tokens = 2`
- `concurrency = 4`

### Baseline A: 4 DPU + fp32 resident KV

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu4_legacy_identity.jsonl`

Observed:

- `max_usage_ratio = 0.99169921875`
- `max_fallback_allocations = 28`
- `max_dpu_capacity_fallbacks = 28`

Meaning:

- this is the raw capacity-bottleneck baseline
- the full-DPU path cannot hold the concurrent workload

### Baseline B: 4 DPU + fp16 resident KV

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu4_fp16.jsonl`

Observed:

- `max_usage_ratio = 0.5855712890625`
- `max_fallback_allocations = 0`
- `max_dpu_capacity_fallbacks = 0`

Meaning:

- resident-KV compression alone is enough to recover the full-DPU path
- this is currently the strongest capacity optimization in the constrained
  `4 DPU` setting

### Baseline C: 8 DPU + fp32 + legacy/identity

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu8_legacy_identity.jsonl`

Observed:

- `max_usage_ratio = 0.78076171875`
- `max_fallback_allocations = 0`

Meaning:

- more PIM resources already remove overflow
- but pressure is still unevenly distributed

### Baseline D: 8 DPU + fp32 + balanced/rotated

Artifact:

- `artifacts/pim_allocator_trace_opt125m_humaneval4_conc4_dpu8_balanced_rotated.jsonl`

Observed:

- `max_usage_ratio = 0.6114501953125`
- `max_fallback_allocations = 0`

Meaning:

- extra DPU resources plus explicit grouping/placement reduce peak allocator
  pressure further
- this is the current best `fp32` placement-aware baseline

## Takeaways

There are now two clearly different ways to recover the full-DPU path:

1. increase effective PIM capacity
   - more DPU resources
   - better grouping / placement

2. reduce resident KV footprint
   - `fp16 resident KV`

For the current cluster and workload:

- `4 DPU` is not salvageable by placement alone
- `4 DPU + fp16` is salvageable
- `8 DPU + fp32` is also salvageable

## Recommended Comparison Set

For the next round of experiments and future paper figures, the most useful
comparison set is:

1. `4 DPU + fp32`
   - capacity bottleneck baseline

2. `4 DPU + fp16`
   - resident compression baseline

3. `8 DPU + fp32 + balanced + rotated`
   - scale-out / placement baseline

These three points already isolate three different stories:

- overflow under constrained PIM capacity
- recovery via KV compression
- recovery via more PIM resources plus better placement
