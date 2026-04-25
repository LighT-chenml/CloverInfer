# PIM Allocator Trace Round 2

Date: 2026-04-24

## Goal

Move from sequential allocator traces to overlapping requests so the system can
begin exposing:

- concurrent resident occupancy
- request-lifecycle skew
- real free-range creation

## Code Changes

Updated file:

- `tests/trace_pim_allocator.py`

Added:

- `--concurrency`
- fixed-size in-flight submission using `ray.wait`
- per-request metadata:
  - `completion_index`
  - `inflight_at_submit`
  - submit timestamps

Updated run summary to include:

- `concurrency`
- `max_live_slot_count`
- `out_of_order_completions`
- `requests_with_overlap_snapshot`
- `requests_with_free_ranges`

## Validation Run

Command:

```bash
/home/cml/anaconda3/envs/clover_infer/bin/python tests/trace_pim_allocator.py \
  --address 192.168.123.4:26379 \
  --data dataset/humaneval.jsonl \
  --model /home/cml/CloverInfer/model/opt-125m \
  --limit 6 \
  --concurrency 2 \
  --max-new-tokens 2 \
  --pim-resident-store-backend upmem_kvslot \
  --no-pim-qk-mixed-enabled \
  --pim-num-dpus 4 \
  --pim-length 8 \
  --output artifacts/pim_allocator_trace_opt125m_humaneval6_conc2.jsonl
```

Observed summary:

- `num_requests = 6`
- `concurrency = 2`
- `max_latency = 2.0108 s`
- `avg_latency = 1.4710 s`
- `max_usage_ratio = 0.62841796875`
- `max_free_range_count = 1.0`
- `max_dpu_allocate_failures = 0`
- `max_fallback_allocations = 0`
- `out_of_order_completions = 0`

Per-request highlights:

- requests 1, 3, 5:
  - `max_live_slot_count = 24`
  - `max_usage_ratio` between `0.54` and `0.63`
  - no free ranges in the captured snapshot
- requests 2, 4, 6:
  - `max_live_slot_count = 12`
  - `max_free_range_count = 1`
  - `min_largest_free_range` between about `246k` and `332k` elems

## Interpretation

This is the first trace run that clearly shows overlapping-request allocator
behavior.

What we learned:

- overlapping requests do raise resident occupancy materially
  - sequential run peak was `12` live slots
  - concurrency-2 run peak reached `24` live slots
- allocator fragmentation is now observable in the real distributed path
  - some request snapshots already show `free_range_count = 1`
- there is still no functional failure
  - no DPU allocation failure
  - no host fallback

Interesting detail:

- even though completions still happened in-order on this small run, overlap was
  already enough to create free ranges
- that means we do not need severe completion reordering before fragmentation
  begins to appear

## What This Means

The allocator challenge has now moved from:

- "can we observe anything beyond empty/full?"

to:

- "how do free ranges evolve as overlap and prompt diversity increase?"

This is exactly the kind of trace needed before discussing allocator or
placement optimizations.

## Suggested Next Step

Push pressure in one of these directions:

1. raise overlap
   - `--concurrency 4`
   - keep `humaneval` first for controlled debugging

2. raise prompt diversity/length
   - use `dataset/longbench/*.jsonl`
   - keep concurrency small at first

Best next move:

- try `humaneval` with `--concurrency 4`
- then switch to a longer-prompt LongBench subset once the overlap path is
  stable
