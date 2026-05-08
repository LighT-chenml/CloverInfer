# Clover Scheduler Components README

## Files

- `src/core/clover_planner.py`
- `src/core/clover_scheduler_components.py`
- `src/core/clover_pipeline_sim.py`

## What These Modules Are For

- `clover_planner.py`
  - build a reusable sharding plan from request metadata
  - report per-DPU token load and balance quality
- `clover_scheduler_components.py`
  - build micro-batches with injected prediction and capacity policies
- `clover_pipeline_sim.py`
  - simulate transfer, DPU, reduction, and FC overlap without running Ray

## How To Plug In A Real Latency Model

Replace the stub predictors passed into
`CapacityAwareMicroBatchScheduler(...)`:

```python
def predict_pim_time(batch_requests, sharding_plan):
    # use real measured features here
    # examples:
    # - max tokens on any DPU
    # - number of active ranksets
    # - number of grouped head slices
    return model.predict(...)

def predict_host_time(total_tokens):
    return host_model.predict(total_tokens)
```

Recommended feature inputs:

- max / avg DPU token load
- number of requests in the batch
- context length bucket
- stripe width
- rankset count
- head-group count

## How To Plug In A Real Capacity Checker

Replace `default_capacity_checker(...)` with a function that queries live DPU
allocator state:

```python
def capacity_checker(batch_requests, sharding_plan, max_capacity_per_dpu):
    # query resident store / allocator stats here
    # examples:
    # - free slots per DPU
    # - largest free contiguous range
    # - temporary query / reduction workspace
    return {
        "ok": ...,
        "peak_load": ...,
        "usage_ratio": ...,
    }
```

Recommended real checks:

- resident KV growth on target DPUs
- temporary Q / score / context buffers
- reduction workspace on host
- fragmentation-sensitive largest-free-range checks

## How To Use The Simulator

1. Build a `CapacityAwareMicroBatchScheduler`.
2. Build a `SimplePipelineSimulator`.
3. Run once with `optimized=True`.
4. Run again with `optimized=False`.
5. Compare:
   - `tokens_per_s`
   - per-stage utilization
   - per-batch timeline logs

The simulator returns stage windows that can be rendered into a Gantt chart by a
later notebook or plotting script.
