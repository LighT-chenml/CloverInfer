# Motivation 1 DPU Load Sweep

This sweep fixes the request distribution and model head count, then varies only the number of DPUs.
The plotted quantities are the naive static PIM `max DPU load` and `average DPU load`.

- Figure: `artifacts/motivation_imbalance_dpu_sweep_20260517/imbalance_dpu_load.svg`
- Plot CSV: `artifacts/motivation_imbalance_dpu_sweep_20260517/imbalance_dpu_sweep.csv`

| D | D/H | avg load | max load | max/avg |
| --- | --- | --- | --- | --- |
| 16 | 4 | 12160 | 36352 | 2.989 |
| 32 | 8 | 6080 | 34304 | 5.642 |
| 64 | 16 | 3040 | 33280 | 10.947 |
| 128 | 32 | 1520 | 32768 | 21.558 |
