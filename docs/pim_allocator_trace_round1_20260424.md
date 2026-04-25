# PIM Allocator Trace Round 1

Date: 2026-04-24

## Goal

Start running dataset-driven allocator traces using the real three-machine
 cluster and capture allocator state before request teardown.

This round also fixed the first real bug that appeared immediately once we
moved from toy smoke tests to real prompts.

## Real Bug Found

The first trace attempt on `dataset/humaneval.jsonl` failed with:

- `RuntimeError: kvslot helper returned incomplete output: Slot 0 capacity exceeded`

Root cause:

- resident KV slot capacity was initialized as `max(pim_length, prefill_seq_len)`
- this ignored the future decode budget
- real prompts plus even one decode append could exceed the allocated slot size

This bug did not show up in earlier tiny smoke tests because those tests used
very short prompts and tiny generation budgets.

## Fix

Updated files:

- `src/core/attention_backend.py`
- `src/core/nodes.py`
- `src/core/scheduler.py`

Change:

- added `decode_reserve_tokens` to the resident init path
- scheduler now passes `max_new_tokens` into `AttentionNode.init_request(...)`
- backend allocates resident capacity as:
  - `max(pim_length, prefill_seq_len + decode_reserve_tokens)`

This preserves the old behavior for callers that do not pass a reserve, while
making real request traces safe.

## Trace Script

Added:

- `tests/trace_pim_allocator.py`

This script:

- runs sequential real requests against the existing three-machine Ray cluster
- collects `attention_backend_before_free`
- extracts `resident_store_debug.allocator_stats`
- writes one JSONL row per request
- prints lightweight allocator summaries during the run

Output fields include:

- request-level latency stats
- `dpu_allocations`
- `fallback_allocations`
- `dpu_allocate_failures`
- full `allocator_stats`
- condensed allocator summaries such as:
  - `max_usage_ratio`
  - `max_free_range_count`
  - `min_largest_free_range`
  - `max_live_slot_count`

## Validation Run

Command:

```bash
/home/cml/anaconda3/envs/clover_infer/bin/python tests/trace_pim_allocator.py \
  --address 192.168.123.4:26379 \
  --data dataset/humaneval.jsonl \
  --model /home/cml/CloverInfer/model/opt-125m \
  --limit 4 \
  --max-new-tokens 2 \
  --pim-resident-store-backend upmem_kvslot \
  --no-pim-qk-mixed-enabled \
  --pim-num-dpus 4 \
  --pim-length 8 \
  --output artifacts/pim_allocator_trace_opt125m_humaneval4.jsonl
```

Observed summary:

- `num_requests = 4`
- `max_latency = 1.7035 s`
- `avg_latency = 1.2355 s`
- `max_usage_ratio = 0.31640625`
- `max_free_range_count = 0.0`
- `max_dpu_allocate_failures = 0`
- `max_fallback_allocations = 0`

Per-request allocator summaries:

- request 1:
  - `max_usage_ratio = 0.31640625`
  - `max_live_slot_count = 12`
- request 2:
  - `max_usage_ratio = 0.31201171875`
  - `max_live_slot_count = 12`
- request 3:
  - `max_usage_ratio = 0.235107421875`
  - `max_live_slot_count = 12`
- request 4:
  - `max_usage_ratio = 0.3076171875`
  - `max_live_slot_count = 12`

Interpretation:

- the trace pipeline now works on real dataset samples
- the capacity-planning bug is fixed
- no fallback or DPU allocation failures occurred in this small real workload
- free-range fragmentation did not appear yet in sequential single-request runs

## What This Means

We now have the first usable end-to-end dataset-driven allocator trace path.

That is important because it shifts future challenge analysis from:

- synthetic reasoning

to:

- request-by-request evidence from real workloads

## Suggested Next Step

Increase pressure in one of two directions:

1. longer sequential runs
   - raise `--limit`
   - use longer-prompt datasets from `dataset/longbench/`

2. overlapping requests
   - modify the trace driver to issue concurrent requests
   - this is more likely to expose actual free-range fragmentation and placement
     imbalance
