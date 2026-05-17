# Motivation 2 Dynamic Micro-Batch Sweep

This sweep shows that the best fixed micro-batch size depends on context length distribution and capacity pressure.
The plotted gap ratio is `|T_pim - T_host| / max(T_pim, T_host)`, so lower is better.
The gap optimum minimizes this ratio among capacity-feasible fixed batch sizes.

- Figure: `artifacts/motivation_microbatch_dynamic_20260517/microbatch_gap_ratio.svg`
- Plot CSV: `artifacts/motivation_microbatch_dynamic_20260517/microbatch_gap_curve.csv`

| scenario | min len | max len | seq CV | gap-opt B | min gap | throughput-opt B | throughput | p95 capacity at gap-opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short | 128 | 128 | 0.000 | 32 | 0.193 | 32 | 481.182 | 0.250 |
| medium | 512 | 512 | 0.000 | 4 | 0.150 | 32 | 198.560 | 0.125 |
| long | 2048 | 2048 | 0.000 | 1 | 0.057 | 8 | 52.784 | 0.125 |
| very_long | 4096 | 4096 | 0.000 | 1 | 0.467 | 4 | 26.674 | 0.250 |
| mixed | 128 | 2048 | 0.885 | 2 | 0.159 | 16 | 120.882 | 0.133 |
| bursty | 128 | 4096 | 1.225 | 2 | 0.387 | 8 | 76.540 | 0.266 |
| capacity_pressure | 1024 | 4096 | 0.423 | 1 | 0.307 | 4 | 36.860 | 0.250 |
