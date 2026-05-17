# CloverInfer Motivation Experiments

Output directory: `artifacts/motivation_imbalance_model_static_smoke_20260517`
Completed result rows: 2
Failed cases: 0

## Metric Definitions

- Experiment 1 uses `dpu_imbalance_ratio = max_live_elems / avg_live_elems_active` as the main straggler metric; `tail_latency_proxy_tokens = max_live_elems` is the proxy for the slowest DPU. `dpu_imbalance_ratio_all` is only an auxiliary under-utilization metric because it counts idle DPUs in the denominator.
- Experiment 2 uses `throughput_gap_ratio = abs(dense_throughput_tps - pim_throughput_tps) / max(dense_throughput_tps, pim_throughput_tps)`; larger means the dense side and PIM side are less aligned.

## Experiment 1: DPU Load Imbalance

| experiment | workload | length | baseline | mean imbalance | p95 imbalance | tail proxy | all-DPU imbalance | mean spread | live CV | all-DPU CV | block fill | tok/s | peak DPU live elems |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imbalance_model | mixed_synthetic | native | Clover Planner | 1.000 | 1.000 | 768.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768] |
| imbalance_model | mixed_synthetic | native | Naive Static PIM | 2.000 | 2.000 | 1536.000 | 2.000 | 1.917 | 0.685 | 0.685 | 0.000 | 0.000 | [832, 640, 1536, 64, 832, 640, 1536, 64, 832, 640, 1536, 64, 832, 640, 1536, 64] |

