# Baseline Compare HumanEval limit=8 concurrency=4 max_new_tokens=4

| baseline | out tok/s | req/s | avg latency(s) | avg TTFT(s) | avg TPOT(s) |
|---|---:|---:|---:|---:|---:|
| monolithic_gpu | 17.107 | 4.277 | 0.227 | 0.115 | 0.037 |
| pd | 10.734 | 2.684 | 0.372 | 0.129 | 0.081 |
| cpu_attention | 5.313 | 1.328 | 2.922 | 0.328 | 0.865 |
| naive_pim | 1.999 | 0.500 | 7.724 | 0.321 | 2.468 |
| cloverinfer | 2.407 | 0.602 | 6.417 | 0.342 | 2.025 |

## CloverInfer Debug
```json
{
  "resident_append_fallbacks": 1,
  "resident_context_fallbacks": 2,
  "resident_runtime_fallbacks": 3,
  "planner_segment_materialized_count": 348,
  "qk_full_batch_calls": 10,
  "softmax_av_fused_batch_calls": 10,
  "active_dpus": 8,
  "active_ratio": 0.25,
  "max_live_slot_count": 36.0,
  "max_usage_ratio": 0.135498046875,
  "slot_capacity_reroutes": 3,
  "dpu_allocate_failures": 0,
  "failure_reasons": {
    "No DPU KV slot capacity remains in allowed placement set: preferred_dpu=0 allowed_dpus=[0, 1, 2, 3, 4, 5, 6, 7] slot_spill_alloc_enabled=False emergency_slot_spill_enabled=False": 1
  },
  "qk_fallback_rounds": 36,
  "av_fallback_rounds": 49,
  "qk_batched_round_total_ms": 354.254382,
  "qk_fallback_round_total_ms": 608.587534,
  "av_batched_round_total_ms": 166.752933,
  "av_fallback_round_total_ms": 1209.203539
}
```
