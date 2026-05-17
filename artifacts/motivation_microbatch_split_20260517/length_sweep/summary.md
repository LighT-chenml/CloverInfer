# Motivation 2 Length Sweep

This figure fixes the request pattern to homogeneous batches and only changes the context-length scale. It shows that the best fixed micro-batch size moves as sequences get longer.

- Figure: `artifacts/motivation_microbatch_split_20260517/length_sweep/microbatch_gap_ratio.svg`
- Plot CSV: `artifacts/motivation_microbatch_split_20260517/length_sweep/microbatch_gap_curve.csv`

| scenario | min len | max len | seq CV | gap-opt B | min gap | throughput-opt B | throughput | p95 capacity at gap-opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short | 128 | 128 | 0.000 | 32 | 0.193 | 32 | 481.182 | 0.250 |
| medium | 512 | 512 | 0.000 | 4 | 0.150 | 32 | 198.560 | 0.125 |
| long | 2048 | 2048 | 0.000 | 1 | 0.057 | 8 | 52.784 | 0.125 |
| very_long | 4096 | 4096 | 0.000 | 1 | 0.467 | 4 | 26.674 | 0.250 |
