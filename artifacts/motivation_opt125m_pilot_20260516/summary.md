# CloverInfer Motivation Experiments

Output directory: `artifacts/motivation_opt125m_pilot_20260516`
Completed result rows: 22
Failed cases: 0

## Metric Definitions

- Experiment 1 uses `dpu_imbalance_ratio = max_live_elems / avg_live_elems_active`; larger means the slowest DPU carries more resident KV load than the average active DPU.
- Experiment 2 uses `throughput_gap_ratio = abs(dense_throughput_tps - pim_throughput_tps) / max(dense_throughput_tps, pim_throughput_tps)`; larger means the dense side and PIM side are less aligned.

## Experiment 1: DPU Load Imbalance

| workload | length | baseline | mean imbalance | p95 imbalance | mean spread | block fill | tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| humaneval | 128 | CloverInfer | 1.000 | 1.000 | 0.000 | 0.508 | 1.039 |
| humaneval | 128 | Naive PIM | 1.000 | 1.000 | 0.000 | 0.508 | 0.977 |
| humaneval | 256 | CloverInfer | 1.000 | 1.000 | 0.000 | 0.504 | 0.849 |
| humaneval | 256 | Naive PIM | 1.000 | 1.000 | 0.000 | 0.504 | 0.789 |
| humaneval | 512 | CloverInfer | 1.000 | 1.000 | 0.000 | 0.669 | 0.714 |
| humaneval | 512 | Naive PIM | 1.000 | 1.000 | 0.000 | 0.669 | 0.598 |
| qasper | 128 | CloverInfer | 1.000 | 1.000 | 0.000 | 0.508 | 1.073 |
| qasper | 128 | Naive PIM | 1.000 | 1.000 | 0.000 | 0.508 | 1.016 |
| qasper | 256 | CloverInfer | 1.000 | 1.000 | 0.000 | 0.504 | 0.870 |
| qasper | 256 | Naive PIM | 1.000 | 1.000 | 0.000 | 0.504 | 0.791 |
| qasper | 512 | CloverInfer | 1.000 | 1.000 | 0.000 | 0.669 | 0.727 |
| qasper | 512 | Naive PIM | 1.000 | 1.000 | 0.000 | 0.669 | 0.633 |
| sharegpt | 128 | CloverInfer | 1.000 | 1.000 | 0.000 | 0.508 | 1.082 |
| sharegpt | 128 | Naive PIM | 1.000 | 1.000 | 0.000 | 0.508 | 1.021 |
| sharegpt | 256 | CloverInfer | 1.000 | 1.000 | 0.000 | 0.504 | 0.861 |
| sharegpt | 256 | Naive PIM | 1.000 | 1.000 | 0.000 | 0.504 | 0.795 |
| sharegpt | 512 | CloverInfer | 1.000 | 1.000 | 0.000 | 0.669 | 0.736 |
| sharegpt | 512 | Naive PIM | 1.000 | 1.000 | 0.000 | 0.669 | 0.646 |

## Experiment 2: Dense/PIM Throughput Mismatch

| configured max batch | baseline | observed max batch | dense tok/s | PIM tok/s | gap ratio | end-to-end tok/s | batch histogram |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CloverInfer | 1 | 28.926 | 36.138 | 0.200 | 2.556 | {"1": 8} |
| 2 | CloverInfer | 2 | 30.928 | 15.506 | 0.499 | 2.857 | {"2": 4} |
| 4 | CloverInfer | 4 | 28.448 | 35.986 | 0.209 | 3.036 | {"1": 1, "3": 1, "4": 1} |
| 8 | CloverInfer | 7 | 32.471 | 36.182 | 0.103 | 3.209 | {"1": 1, "7": 1} |

