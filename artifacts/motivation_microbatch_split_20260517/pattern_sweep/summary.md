# Motivation 2 Pattern Sweep

This figure fixes the base context length and varies request-shape patterns such as mixed, bursty, and capacity-pressure. It isolates workload shape from length scale.

- Figure: `artifacts/motivation_microbatch_split_20260517/pattern_sweep/microbatch_gap_ratio.svg`
- Plot CSV: `artifacts/motivation_microbatch_split_20260517/pattern_sweep/microbatch_gap_curve.csv`

| scenario | min len | max len | seq CV | gap-opt B | min gap | throughput-opt B | throughput | p95 capacity at gap-opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| homogeneous | 2048 | 2048 | 0.000 | 1 | 0.057 | 8 | 52.784 | 0.125 |
| mixed | 128 | 2048 | 0.885 | 2 | 0.159 | 16 | 120.882 | 0.133 |
| bursty | 128 | 4096 | 1.225 | 2 | 0.387 | 8 | 76.540 | 0.266 |
| capacity_pressure | 1024 | 4096 | 0.423 | 1 | 0.307 | 4 | 36.860 | 0.250 |
