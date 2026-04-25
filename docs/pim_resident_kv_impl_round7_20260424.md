# Resident-KV PIM Implementation Round 7

Date: 2026-04-24

## Goal

Turn the DPU resident-KV helper from:

- logical-slot reuse only

into:

- logical-slot reuse plus reusable MRAM allocation ranges

This round targeted the first real steady-state issue in the helper:

- `FREE` previously released only slot metadata
- per-DPU `elem_offset` allocation still grew monotonically
- longer-running experiments would eventually hit avoidable DPU-capacity
  failures

## Code Changes

Updated files:

- `src/pim/upmem_kvslot/host_kvslot.c`
- `src/core/resident_kv_store.py`

### DPU-side host helper allocator

The helper now tracks reusable free ranges per DPU.

Added state:

- `elem_count` in each logical slot
- sorted free-range lists per DPU
- free-range count per DPU

Added allocator behavior:

- allocate from an existing free range first
- otherwise allocate from the monotonic tail
- on `FREE`, return the released interval to the free-range list
- coalesce adjacent free ranges
- reclaim tail-aligned free ranges back into `next_free_elem`

This means the helper can now recycle both:

- logical slot ids
- the MRAM pool space used by those slots

### Python store debug

`UpmemKVSlotStore` now reports:

- `dpu_allocate_failures`

This helps distinguish:

- successful DPU placement with real allocator reuse
- silent fallback caused by DPU-side exhaustion or helper errors

## Validation

### Direct backend smoke on `.7`

Observed result:

```python
{
  "shadow_diff": 0.0,
  "dpu_allocations": 8,
  "dpu_allocate_failures": 0,
  "fallback_allocations": 0,
  "dpu_free_ops": 8,
  "live_slots_end": 0,
}
```

Interpretation:

- correctness remains unchanged
- no new allocation failures were introduced
- request teardown still fully frees logical slots

### Long sequential reuse smoke on `.7`

Stress test:

- one persistent backend/helper instance
- 80 sequential requests
- each request allocates, decodes, and frees 12 resident groups

Observed checkpoints:

```python
{'req': 1, 'dpu_allocations': 48, 'dpu_allocate_failures': 0, 'fallback_allocations': 0}
{'req': 32, 'dpu_allocations': 1536, 'dpu_allocate_failures': 0, 'fallback_allocations': 0}
{'req': 64, 'dpu_allocations': 3072, 'dpu_allocate_failures': 0, 'fallback_allocations': 0}
{'req': 80, 'dpu_allocations': 3840, 'dpu_allocate_failures': 0, 'fallback_allocations': 0}
```

Interpretation:

- the helper no longer collapses under repeated request churn
- allocator reuse is now good enough to avoid artificial host fallback on this
  workload
- this is stronger evidence than the round-6 slot-id-only reuse check

### Three-machine end-to-end smoke

Validated command:

```bash
/home/cml/anaconda3/envs/clover_infer/bin/python tests/verify_cluster_placement.py \
  --address 192.168.123.4:26379 \
  --model /home/cml/CloverInfer/model/opt-125m \
  --attention-backend pim_naive \
  --pim-resident-store-backend upmem_kvslot \
  --no-pim-qk-mixed-enabled \
  --pim-num-dpus 4 \
  --pim-length 8 \
  --max-new-tokens 2
```

Observed resident-store debug:

- `dpu_allocations = 48`
- `dpu_free_ops = 48`
- `dpu_allocate_failures = 0`
- `fallback_allocations = 0`
- `resident_shadow_max_abs_diff = 0.0`

Interpretation:

- allocator changes did not break distributed execution
- the functional OPT-125M path still lands all resident groups on DPU
- the resident path remains numerically aligned with the CPU oracle

## What Changed Conceptually

Before this round, the system had:

- DPU-backed resident placement
- slot-id reuse
- no real DPU-space reuse

After this round, the system has:

- DPU-backed resident placement
- slot-id reuse
- first reusable DPU MRAM allocator

That makes the current baseline much more credible for longer-running functional
experiments.

## Remaining Limitation

The allocator is still intentionally simple:

- first-fit free-range reuse
- no compaction
- no fragmentation statistics yet

So the next likely issue is not basic exhaustion anymore, but fragmentation and
layout quality under more varied sequence lengths or mixed workloads.

## Suggested Next Step

The next useful implementation step is observability plus workload pressure:

1. expose per-DPU allocator stats in debug output
2. run longer dataset-driven experiments with varied prompt lengths
3. measure when fragmentation starts to matter
4. only then decide whether a stronger allocator or KV layout is necessary
