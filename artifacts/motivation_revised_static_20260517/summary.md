# CloverInfer Motivation Experiments

Output directory: `artifacts/motivation_revised_static_20260517`
Completed result rows: 22
Failed cases: 0

## Metric Definitions

- Experiment 1 uses `dpu_imbalance_ratio = max_live_elems / avg_live_elems_active` as the main straggler metric; `tail_latency_proxy_tokens = max_live_elems` is the proxy for the slowest DPU. `dpu_imbalance_ratio_all` is only an auxiliary under-utilization metric because it counts idle DPUs in the denominator.
- Experiment 2 uses `throughput_gap_ratio = abs(dense_throughput_tps - pim_throughput_tps) / max(dense_throughput_tps, pim_throughput_tps)`; larger means the dense side and PIM side are less aligned.

## Experiment 1: DPU Load Imbalance

| experiment | workload | length | baseline | mean imbalance | p95 imbalance | tail proxy | all-DPU imbalance | mean spread | live CV | all-DPU CV | block fill | tok/s | peak DPU live elems |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imbalance_model | mixed_synthetic | 256 | Clover Planner | 1.000 | 1.000 | 512.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512] |
| imbalance_model | mixed_synthetic | 256 | Naive Static PIM | 1.000 | 1.000 | 512.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512] |
| imbalance_model | mixed_synthetic | 512 | Clover Planner | 1.000 | 1.000 | 1024.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |
| imbalance_model | mixed_synthetic | 512 | Naive Static PIM | 1.000 | 1.000 | 1024.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |
| imbalance_model | mixed_synthetic | native | Clover Planner | 1.000 | 1.000 | 1088.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088, 1088] |
| imbalance_model | mixed_synthetic | native | Naive Static PIM | 2.353 | 2.353 | 2560.000 | 2.353 | 2.059 | 0.799 | 0.799 | 0.000 | 0.000 | [832, 640, 2560, 320, 832, 640, 2560, 320, 832, 640, 2560, 320, 832, 640, 2560, 320] |
| imbalance_model | qasper | 256 | Clover Planner | 1.000 | 1.000 | 512.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512] |
| imbalance_model | qasper | 256 | Naive Static PIM | 1.000 | 1.000 | 512.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512] |
| imbalance_model | qasper | 512 | Clover Planner | 1.000 | 1.000 | 1024.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |
| imbalance_model | qasper | 512 | Naive Static PIM | 1.000 | 1.000 | 1024.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |
| imbalance_model | qasper | native | Clover Planner | 1.000 | 1.000 | 9273.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [9273, 9273, 9272, 9272, 9273, 9273, 9272, 9272, 9273, 9273, 9272, 9272, 9273, 9273, 9272, 9272] |
| imbalance_model | qasper | native | Naive Static PIM | 1.081 | 1.081 | 10025.000 | 1.081 | 0.179 | 0.071 | 0.071 | 0.000 | 0.000 | [8367, 8945, 10025, 9753, 8367, 8945, 10025, 9753, 8367, 8945, 10025, 9753, 8367, 8945, 10025, 9753] |
| imbalance_model | sharegpt | 256 | Clover Planner | 1.000 | 1.000 | 512.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512] |
| imbalance_model | sharegpt | 256 | Naive Static PIM | 1.000 | 1.000 | 512.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512] |
| imbalance_model | sharegpt | 512 | Clover Planner | 1.000 | 1.000 | 1024.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |
| imbalance_model | sharegpt | 512 | Naive Static PIM | 1.000 | 1.000 | 1024.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024] |
| imbalance_model | sharegpt | native | Clover Planner | 1.002 | 1.002 | 388.000 | 1.002 | 0.003 | 0.001 | 0.001 | 0.000 | 0.000 | [388, 387, 387, 387, 388, 387, 387, 387, 388, 387, 387, 387, 388, 387, 387, 387] |
| imbalance_model | sharegpt | native | Naive Static PIM | 2.990 | 2.990 | 1158.000 | 2.990 | 2.869 | 1.160 | 1.160 | 0.000 | 0.000 | [124, 220, 47, 1158, 124, 220, 47, 1158, 124, 220, 47, 1158, 124, 220, 47, 1158] |

## Experiment 2: Dense/PIM Throughput Mismatch

| configured max batch | baseline | observed max batch | dense tok/s | PIM tok/s | gap ratio | end-to-end tok/s | batch histogram |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CloverInfer | 1 | 28.592 | 1.974 | 0.931 | 2.587 | {"1": 8} |
| 2 | CloverInfer | 2 | 31.515 | 1.719 | 0.945 | 2.919 | {"2": 4} |
| 4 | CloverInfer | 4 | 30.867 | 1.911 | 0.938 | 2.970 | {"1": 1, "3": 1, "4": 1} |
| 8 | CloverInfer | 7 | 30.727 | 2.152 | 0.930 | 3.543 | {"1": 1, "7": 1} |

