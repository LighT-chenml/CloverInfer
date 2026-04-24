# PIM Context Sweep Round 2

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
  - `artifacts/context_sweep_round2.jsonl`
- Fixed configuration:
  - `pim_qk_mixed_heads = 2`
  - `max_new_tokens = 8`
  - `repeats = 3`

Command used:

```bash
scripts/pim/run_attention_sweep.sh \
  --attention-backends cpu,pim_naive \
  --pim-qk-mixed-heads-list 2 \
  --prompt-token-lengths 16,64,256,512 \
  --repeats 3 \
  --max-new-tokens 8 \
  --output /home/cml/CloverInfer/artifacts/context_sweep_round2.jsonl
```

## Results

| Prompt Tokens | Backend | Avg Latency (s) | Slowdown vs CPU | Avg Throughput (tok/s) | Attention Decode Compute (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | CPU | 2.395 | 1.00x | 3.375 | 0.065 |
| 16 | `pim_naive` | 7.547 | 3.15x | 1.062 | 5.118 |
| 64 | CPU | 2.532 | 1.00x | 3.185 | 0.077 |
| 64 | `pim_naive` | 7.554 | 2.98x | 1.059 | 5.108 |
| 256 | CPU | 2.768 | 1.00x | 2.912 | 0.161 |
| 256 | `pim_naive` | 8.053 | 2.91x | 0.995 | 5.437 |
| 512 | CPU | 2.971 | 1.00x | 2.715 | 0.294 |
| 512 | `pim_naive` | 8.425 | 2.84x | 0.951 | 5.638 |

Correctness observations:

- All runs completed successfully.
- `final_qk_check_failures = 0` for all `pim_naive` cases.
- `max_qk_mixed_last_abs_diff` remained small in all runs.

## Key Findings

1. Longer prompts do increase cost for both CPU attention and `pim_naive`.
2. CPU attention decode compute rises clearly with context length:
   - `0.065s` at 16 tokens
   - `0.294s` at 512 tokens
3. `pim_naive` attention decode compute also rises, but much more slowly in this
   range:
   - `5.118s` at 16 tokens
   - `5.638s` at 512 tokens
4. End-to-end slowdown vs CPU drops only modestly as context grows:
   - `3.15x` at 16 tokens
   - `2.84x` at 512 tokens
5. The naive PIM implementation is still dominated by fixed per-layer/per-head
   QK invocation overhead, not by context-length-sensitive KV work alone.

## Interpretation

This round complements the head sweep:

- the head sweep showed almost linear scaling with `mixed_heads`
- the context sweep shows only mild improvement in relative competitiveness as
  context increases from `16` to `512`

Together, these results suggest that the current naive baseline is bottlenecked
primarily by the invocation pattern of the PIM QK path rather than by raw KV
volume alone.

## Recommended Next Steps

1. Keep this round as the reference context sweep for the naive baseline.
2. Prioritize batching or fusing the current per-head QK host path.
3. After batching is implemented, rerun both:
   - the head sweep from round 1
   - this context sweep from round 2
4. If needed, extend the next context sweep to `1024` and `1536` tokens to see
   whether the crossover trend strengthens at longer contexts.
