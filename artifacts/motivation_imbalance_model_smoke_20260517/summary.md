# CloverInfer Motivation Experiments

Output directory: `artifacts/motivation_imbalance_model_smoke_20260517`
Completed result rows: 2
Failed cases: 0

## Metric Definitions

- Experiment 1 uses `dpu_imbalance_ratio = max_live_elems / avg_live_elems_active`; larger means the slowest DPU carries more resident KV load than the average active DPU.
- Experiment 2 uses `throughput_gap_ratio = abs(dense_throughput_tps - pim_throughput_tps) / max(dense_throughput_tps, pim_throughput_tps)`; larger means the dense side and PIM side are less aligned.

## Experiment 1: DPU Load Imbalance

| experiment | workload | length | baseline | mean imbalance | p95 imbalance | all-DPU imbalance | mean spread | live CV | all-DPU CV | block fill | tok/s | peak DPU live elems |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imbalance_model | mixed_synthetic | native | Clover Planner | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768] |
| imbalance_model | mixed_synthetic | native | Naive Head-Only PIM | 1.000 | 1.000 | 4.000 | 0.000 | 0.000 | 1.732 | 0.000 | 0.000 | [3072, 3072, 3072, 3072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |

