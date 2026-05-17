# CloverInfer Motivation Experiments

Output directory: `artifacts/_motivation_dryrun2`
Completed result rows: 0
Failed cases: 0

## Metric Definitions

- Experiment 1 uses `dpu_imbalance_ratio = max_live_elems / avg_live_elems_active`; larger means the slowest DPU carries more resident KV load than the average active DPU.
- Experiment 2 uses `throughput_gap_ratio = abs(dense_throughput_tps - pim_throughput_tps) / max(dense_throughput_tps, pim_throughput_tps)`; larger means the dense side and PIM side are less aligned.

