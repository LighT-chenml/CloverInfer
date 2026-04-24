# PIM Batched-QK Optimization Round 3

Date: 2026-04-24

## Goal

Validate the first optimization step from the roadmap:

- reduce naive PIM fixed overhead by batching mixed-head QK work
- avoid the previous pattern of one host-side QK invocation per mixed head

This round does not attempt to change scheduler logic or transport. It only
changes the attention-side naive PIM host path.

## Implementation

Code changes:

- `src/core/attention_backend.py`
  - add `_run_qk_scores_batch`
  - change mixed-head decode from per-head calls to one batched call
  - add `qk_batch_calls` debug counter
  - rebuild `host_qk` automatically when source files are newer than the binary
- `src/pim/upmem_qk/host_qk.c`
  - add a reusable DPU runner
  - extend file mode to support multi-query batches
  - reuse a single DPU session across batched queries in the same process

Commit:

- `4e221d9 pim: batch qk mixed heads in host path`

## Environment Note

The current machine `192.168.123.4` does not have a complete local UPMEM build
toolchain:

- error observed locally: `dpu-upmem-dpurte-clang: No such file or directory`

Therefore, functional validation for the UPMEM-backed path in this round was
performed through the real attention node on `192.168.123.7`, where the UPMEM
toolchain is already available and previous smoke tests were passing.

## Validation

### 1. Real three-machine smoke still passes

Configuration:

- model: `Qwen-1_8B`
- topology:
  - `192.168.123.3`: prefill GPU
  - `192.168.123.4`: decode dense GPU
  - `192.168.123.7`: attention `pim_naive`
- dataset: `dataset/humaneval.jsonl`
- `limit = 1`
- `max_new_tokens = 3`

Observed result:

- request succeeded end to end
- `qk_check_failures = 0`
- `qk_batch_calls = 48`
- `qk_mixed_count = 96`

Interpretation:

- The new path was actually exercised.
- For this request, the naive PIM backend performed one batched QK call per
  decode layer, rather than one call per mixed head.

### 2. Two-sample CPU vs PIM comparison after batching

Output:

- `artifacts/baseline_comparison_qwen_pim_batched.jsonl`

Measured summary:

| Baseline | Avg Latency (s) | Avg Throughput (tok/s) |
| --- | ---: | ---: |
| `disagg_cpu` | `2.225` | `1.386` |
| `disagg_pim_naive` | `4.117` | `0.740` |

## Before / After Comparison

Reference before-change result:

- `artifacts/baseline_comparison_qwen_v2.jsonl`

Comparison for `disagg_pim_naive`:

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Avg latency (s) | `5.764` | `4.117` | `-28.6%` |
| Avg throughput (tok/s) | `0.522` | `0.740` | `+41.7%` |
| Avg attention decode compute (s) | `3.495` | `1.947` | `-44.3%` |

## What This Means

Measured conclusion:

- Batching mixed-head QK work reduces both end-to-end latency and
  attention-side compute time substantially.

Interpretation:

- This strongly supports the earlier challenge analysis that the naive PIM path
  was dominated by invocation granularity and host-side orchestration overhead.
- The optimization does not yet make naive PIM competitive with the CPU remote
  attention baseline, but it clearly moves the implementation in the right
  direction.

## Remaining Gap

Even after this optimization:

- `disagg_pim_naive` is still slower than `disagg_cpu`
- remote attention is still much slower than the `split_gpu_full_decode`
  baseline

So the main story after round 3 is:

1. the challenge diagnosis was correct
2. reducing invocation granularity helps materially
3. there is still substantial work left before the PIM path becomes competitive

## Recommended Next Steps

1. Keep this round as the first optimization reference point.
2. Next, reduce remaining host-side overhead beyond batched heads:
   - persistent buffers
   - less temporary-file traffic
   - less repeated input packing
3. After that, rerun:
   - the unified Qwen baseline matrix
   - the mixed-head sweep
   - the context sweep
