# CloverInfer Motivation Experiments

Output directory: `artifacts/motivation_revised_20260517`
Completed result rows: 22
Failed cases: 0

## Metric Definitions

- Experiment 1 reports `dpu_imbalance_ratio = max_live_elems / avg_live_elems_active` for active-DPU skew and `dpu_imbalance_ratio_all = max_live_elems / avg_live_elems_all` for end-to-end DPU under-utilization; the all-DPU metric is the main motivation metric when D >> H.
- Experiment 2 uses `throughput_gap_ratio = abs(dense_throughput_tps - pim_throughput_tps) / max(dense_throughput_tps, pim_throughput_tps)`; larger means the dense side and PIM side are less aligned.

## Experiment 1: DPU Load Imbalance

| experiment | workload | length | baseline | mean imbalance | p95 imbalance | all-DPU imbalance | mean spread | live CV | all-DPU CV | block fill | tok/s | peak DPU live elems |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imbalance_model | mixed_synthetic | 256 | Clover Planner | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512] |
| imbalance_model | mixed_synthetic | 256 | Naive Head-Only PIM | 1.000 | 1.000 | 4.000 | 0.000 | 0.000 | 1.732 | 0.000 | 0.000 | [2048, 2048, 2048, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| imbalance_model | mixed_synthetic | 512 | Clover Planner | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |
| imbalance_model | mixed_synthetic | 512 | Naive Head-Only PIM | 1.000 | 1.000 | 4.000 | 0.000 | 0.000 | 1.732 | 0.000 | 0.000 | [4096, 4096, 4096, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| imbalance_model | mixed_synthetic | native | Clover Planner | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088] |
| imbalance_model | mixed_synthetic | native | Naive Head-Only PIM | 1.000 | 1.000 | 4.000 | 0.000 | 0.000 | 1.732 | 0.000 | 0.000 | [4352, 4352, 4352, 4352, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| imbalance_model | qasper | 256 | Clover Planner | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512] |
| imbalance_model | qasper | 256 | Naive Head-Only PIM | 1.000 | 1.000 | 4.000 | 0.000 | 0.000 | 1.732 | 0.000 | 0.000 | [2048, 2048, 2048, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| imbalance_model | qasper | 512 | Clover Planner | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |
| imbalance_model | qasper | 512 | Naive Head-Only PIM | 1.000 | 1.000 | 4.000 | 0.000 | 0.000 | 1.732 | 0.000 | 0.000 | [4096, 4096, 4096, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| imbalance_model | qasper | native | Clover Planner | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [9273, 9273, 9272, 9272, 9273, 9273, 9272, 9272, 9273, 9273, 9272, 9272, 9273, 9273, 9272, 9272] |
| imbalance_model | qasper | native | Naive Head-Only PIM | 1.000 | 1.000 | 4.000 | 0.000 | 0.000 | 1.732 | 0.000 | 0.000 | [37090, 37090, 37090, 37090, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| imbalance_model | sharegpt | 256 | Clover Planner | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512] |
| imbalance_model | sharegpt | 256 | Naive Head-Only PIM | 1.000 | 1.000 | 4.000 | 0.000 | 0.000 | 1.732 | 0.000 | 0.000 | [2048, 2048, 2048, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| imbalance_model | sharegpt | 512 | Clover Planner | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |
| imbalance_model | sharegpt | 512 | Naive Head-Only PIM | 1.000 | 1.000 | 4.000 | 0.000 | 0.000 | 1.732 | 0.000 | 0.000 | [4096, 4096, 4096, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| imbalance_model | sharegpt | native | Clover Planner | 1.002 | 1.002 | 1.002 | 0.003 | 0.001 | 0.001 | 0.000 | 0.000 | [388, 387, 387, 387, 388, 387, 387, 387, 388, 387, 387, 387, 388, 387, 387, 387] |
| imbalance_model | sharegpt | native | Naive Head-Only PIM | 1.000 | 1.000 | 4.000 | 0.000 | 0.000 | 1.732 | 0.000 | 0.000 | [1549, 1549, 1549, 1549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |

## Experiment 2: Dense/PIM Throughput Mismatch

| configured max batch | baseline | observed max batch | dense tok/s | PIM tok/s | gap ratio | end-to-end tok/s | batch histogram |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CloverInfer | 1 | 29.123 | 2.188 | 0.925 | 2.629 | {"1": 8} |
| 2 | CloverInfer | 2 | 29.695 | 2.098 | 0.929 | 2.903 | {"1": 2, "2": 3} |
| 4 | CloverInfer | 4 | 31.284 | 2.232 | 0.929 | 3.169 | {"1": 1, "3": 1, "4": 1} |
| 8 | CloverInfer | 5 | 32.927 | 1.724 | 0.948 | 3.291 | {"3": 1, "5": 1} |

