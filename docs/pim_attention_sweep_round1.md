# PIM Attention Sweep Round 1

Date: 2026-04-24

## Setup

- Topology:
  - `192.168.123.3`: prefill GPU node
  - `192.168.123.4`: decode dense GPU node
  - `192.168.123.7`: attention CPU/PIM node
- Model: `opt-125m`
- Transport/runtime: Ray
- Benchmark script:
  - `scripts/pim/run_attention_sweep.sh`
  - `tests/benchmark_attention_sweep.py`
- Output file:
  - `artifacts/attention_sweep_heads_timing.jsonl`

Command used:

```bash
scripts/pim/run_attention_sweep.sh \
  --attention-backends cpu,pim_naive \
  --pim-qk-mixed-heads-list 0,1,2,4,8,12 \
  --repeats 3 \
  --max-new-tokens 8 \
  --output /home/cml/CloverInfer/artifacts/attention_sweep_heads_timing.jsonl
```

## Results

| Backend | Mixed Heads | Avg Latency (s) | Slowdown vs CPU | Avg Throughput (tok/s) | Attention Decode Compute (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 0 | 2.447 | 1.00x | 3.311 | 0.062 |
| `pim_naive` | 0 | 2.448 | 1.00x | 3.309 | 0.065 |
| `pim_naive` | 1 | 5.020 | 2.05x | 1.603 | 2.564 |
| `pim_naive` | 2 | 7.590 | 3.10x | 1.057 | 5.130 |
| `pim_naive` | 4 | 12.070 | 4.93x | 0.663 | 9.619 |
| `pim_naive` | 8 | 21.508 | 8.79x | 0.372 | 18.991 |
| `pim_naive` | 12 | 31.455 | 12.85x | 0.254 | 29.006 |

Correctness observations:

- All runs completed successfully.
- `final_qk_check_failures = 0` for all `pim_naive` cases.
- `mixed_heads = 0` is effectively identical to the CPU baseline, which is the
  expected control result.

## Key Findings

1. The current naive PIM baseline is correctness-stable but performance-poor.
2. Once PIM QK is enabled, end-to-end latency grows roughly with the number of
   mixed heads.
3. The dominant cost is the attention-node decode path, not prefill or dense
   compute.
4. `prepare_attention`, `finish_layer`, and `sample_next_token` stay relatively
   flat across the sweep, so they are not the main bottleneck in this round.
5. The primary optimization target is therefore the `.7` attention-side QK path:
   host-side orchestration, subprocess/file exchange, and UPMEM launch cadence.

## Interpretation

This is a good naive baseline result.

It shows that:

- the framework is correctly isolating the effect of the PIM-backed attention
  path
- the current implementation is not yet competitive with CPU attention
- the challenge is now measurable rather than speculative

The most important signal is that `attention_decode_compute_s` scales almost
linearly with `mixed_heads`. That strongly suggests the current implementation
is dominated by repeated per-head QK invocation overhead rather than a fixed
transport tax alone.

## Recommended Next Steps

1. Keep this round as the official naive baseline reference.
2. Add a second round that sweeps context length while keeping `mixed_heads`
   fixed, to see when KV size starts to dominate.
3. Replace the current per-head/per-layer file-based QK invocation with a more
   batched host path before touching `AV`.
4. After QK batching is improved, rerun the same sweep to measure whether the
   slope with respect to `mixed_heads` improves.
