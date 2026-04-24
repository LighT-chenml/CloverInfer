# PIM Context Sweep After Stdio Helper

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
  - `artifacts/context_sweep_after_stdio_round5.jsonl`
- Fixed configuration:
  - `pim_qk_mixed_heads = 2`
  - `max_new_tokens = 8`
  - `repeats = 3`

Command used:

```bash
/home/cml/anaconda3/envs/clover_infer/bin/python tests/benchmark_attention_sweep.py \
  --address 192.168.123.4:26379 \
  --model /home/cml/CloverInfer/model/opt-125m \
  --attention-backends cpu,pim_naive \
  --pim-qk-mixed-heads-list 2 \
  --prompt-token-lengths 16,64,256,512 \
  --repeats 3 \
  --max-new-tokens 8 \
  --output /home/cml/CloverInfer/artifacts/context_sweep_after_stdio_round5.jsonl
```

## Results

| Prompt Tokens | Backend | Avg Latency (s) | Slowdown vs CPU | Avg Throughput (tok/s) | Attention Decode Compute (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| `16` | CPU | `2.712` | `1.00x` | `2.992` | `0.072` |
| `16` | `pim_naive` | `3.133` | `1.16x` | `2.611` | `0.371` |
| `64` | CPU | `2.780` | `1.00x` | `2.910` | `0.089` |
| `64` | `pim_naive` | `3.215` | `1.16x` | `2.517` | `0.494` |
| `256` | CPU | `2.983` | `1.00x` | `2.711` | `0.175` |
| `256` | `pim_naive` | `3.466` | `1.16x` | `2.330` | `0.690` |
| `512` | CPU | `3.283` | `1.00x` | `2.454` | `0.306` |
| `512` | `pim_naive` | `3.825` | `1.17x` | `2.104` | `0.842` |

Correctness observations:

- All runs completed successfully.
- `final_qk_check_failures = 0` for all `pim_naive` cases.
- `max_qk_mixed_last_abs_diff` remained small in all runs.

## Comparison Against Round 2

Round 2 reference:

- `docs/pim_context_sweep_round2.md`

Latency comparison:

| Prompt Tokens | Round 2 Naive PIM Latency (s) | Post-Stdio Naive PIM Latency (s) | Speedup |
| ---: | ---: | ---: | ---: |
| `16` | `7.547` | `3.133` | `2.41x` |
| `64` | `7.554` | `3.215` | `2.35x` |
| `256` | `8.053` | `3.466` | `2.32x` |
| `512` | `8.425` | `3.825` | `2.20x` |

Attention decode compute comparison:

| Prompt Tokens | Round 2 Naive PIM Attention Decode Compute (s) | Post-Stdio Attention Decode Compute (s) | Speedup |
| ---: | ---: | ---: | ---: |
| `16` | `5.118` | `0.371` | `13.80x` |
| `64` | `5.108` | `0.494` | `10.34x` |
| `256` | `5.437` | `0.690` | `7.88x` |
| `512` | `5.638` | `0.842` | `6.70x` |

## Interpretation

This rerun updates the previous conclusion in an important way:

1. The old round 2 result was dominated by fixed orchestration cost.
2. After batching plus persistent-helper reuse, naive PIM still has a fixed
   premium over CPU attention, but the cost now grows more clearly with context.
3. This suggests the system is moving toward a more meaningful regime where
   actual KV/QK work matters again, rather than being completely buried by
   process/file overhead.
4. Even so, up to `512` prompt tokens, naive PIM still does not beat the remote
   CPU baseline.

Paper-usable phrasing:

- "Once host orchestration overhead is reduced, the naive PIM backend begins to
  exhibit more context-sensitive behavior, but it remains slower than remote CPU
  attention in the tested `16-512` token range."

## Recommended Next Steps

1. Refresh the unified Qwen baseline matrix with the current helper-based naive
   PIM backend.
2. Extend the context sweep beyond `512` tokens to test whether the remaining
   gap keeps shrinking at longer contexts.
3. If longer contexts still do not expose an advantage, shift focus to deeper
   PIM-specific optimizations rather than only host orchestration.
