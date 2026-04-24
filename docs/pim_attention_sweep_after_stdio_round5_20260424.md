# PIM Attention Sweep After Stdio Helper

Date: 2026-04-24

## Setup

- Topology:
  - `192.168.123.3`: prefill GPU node
  - `192.168.123.4`: decode dense GPU node
  - `192.168.123.7`: attention CPU/PIM node
- Model: `opt-125m`
- Transport/runtime: Ray
- Benchmark script:
  - `tests/benchmark_attention_sweep.py`
- Output file:
  - `artifacts/attention_sweep_heads_stdio_round4.jsonl`
- Fixed configuration:
  - `max_new_tokens = 8`
  - `repeats = 3`
  - `pim_qk_mixed_enabled = true`
  - `pim_num_dpus = 4`
  - `pim_length = 128`

Command used:

```bash
/home/cml/anaconda3/envs/clover_infer/bin/python tests/benchmark_attention_sweep.py \
  --address 192.168.123.4:26379 \
  --model /home/cml/CloverInfer/model/opt-125m \
  --attention-backends cpu,pim_naive \
  --pim-qk-mixed-heads-list 0,1,2,4,8,12 \
  --repeats 3 \
  --max-new-tokens 8 \
  --output /home/cml/CloverInfer/artifacts/attention_sweep_heads_stdio_round4.jsonl
```

## Results

| Backend | Mixed Heads | Avg Latency (s) | Slowdown vs CPU | Avg Throughput (tok/s) | Attention Decode Compute (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 0 | `2.671` | `1.00x` | `3.054` | `0.069` |
| `pim_naive` | 0 | `2.629` | `0.98x` | `3.082` | `0.070` |
| `pim_naive` | 1 | `2.852` | `1.07x` | `2.853` | `0.270` |
| `pim_naive` | 2 | `2.938` | `1.10x` | `2.767` | `0.353` |
| `pim_naive` | 4 | `3.174` | `1.19x` | `2.542` | `0.499` |
| `pim_naive` | 8 | `3.482` | `1.30x` | `2.318` | `0.796` |
| `pim_naive` | 12 | `3.731` | `1.40x` | `2.164` | `1.107` |

Correctness observations:

- All runs completed successfully.
- `final_qk_check_failures = 0` for all `pim_naive` cases.
- Example helper-path debug record for `mixed_heads = 2`:
  - `qk_batch_calls = 84`
  - `qk_helper_started = true`
  - `qk_helper_restarts = 1`
  - `qk_mixed_count = 168`

## Comparison Against Round 1

Round 1 reference:

- `docs/pim_attention_sweep_round1.md`

Key latency improvements:

| Mixed Heads | Round 1 Latency (s) | Post-Stdio Latency (s) | Speedup |
| ---: | ---: | ---: | ---: |
| `1` | `5.020` | `2.852` | `1.76x` |
| `2` | `7.590` | `2.938` | `2.58x` |
| `4` | `12.070` | `3.174` | `3.80x` |
| `8` | `21.508` | `3.482` | `6.18x` |
| `12` | `31.455` | `3.731` | `8.43x` |

Key attention-decode-compute improvements:

| Mixed Heads | Round 1 Attention Decode Compute (s) | Post-Stdio Attention Decode Compute (s) | Speedup |
| ---: | ---: | ---: | ---: |
| `1` | `2.564` | `0.270` | `9.49x` |
| `2` | `5.130` | `0.353` | `14.51x` |
| `4` | `9.619` | `0.499` | `19.27x` |
| `8` | `18.991` | `0.796` | `23.87x` |
| `12` | `29.006` | `1.107` | `26.20x` |

## Interpretation

This rerun changes the story materially:

1. The naive PIM path no longer explodes with mixed-head count the way it did in
   round 1.
2. The slope with respect to `mixed_heads` is still present, but it is much
   flatter.
3. The dominant problem in round 1 was therefore not raw DPU arithmetic alone;
   it was largely host-side orchestration around each QK call.
4. After batching and persistent-helper reuse, naive PIM becomes much closer to
   the remote CPU baseline even when more heads use the QK path.

Paper-usable phrasing:

- "The original mixed-head sensitivity of the naive PIM backend was largely a
  host-orchestration artifact. After replacing per-call process/file overhead
  with batched QK submission and a persistent stdio helper, the latency slope
  versus mixed-head count flattens substantially."

## Recommended Next Steps

1. Re-run the context sweep using the current stdio-helper path.
2. Re-run the unified Qwen baseline matrix so the official baseline table uses
   the improved naive PIM backend.
3. Only after those refreshed baselines, revisit longer-context or more
   aggressive kernel changes such as `AV` offload.
