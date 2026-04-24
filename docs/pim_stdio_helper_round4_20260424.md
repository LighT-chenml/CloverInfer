# PIM Stdio-Helper Optimization Round 4

Date: 2026-04-24

## Goal

Apply the second fixed-overhead reduction after batched mixed-head QK:

- remove temporary-file exchange from the hot QK path
- avoid spawning a fresh `host_qk` process for every batched call
- keep a persistent helper process on the attention node

## Implementation

Code changes:

- `src/pim/upmem_qk/host_qk.c`
  - add `--stdio` mode
  - accept binary requests from `stdin`
  - return binary scores on `stdout`
  - reuse a single DPU runner inside the helper process
- `src/core/attention_backend.py`
  - add `_ensure_qk_helper`
  - route batched QK through a persistent helper process
  - add helper debug fields:
    - `qk_helper_started`
    - `qk_helper_restarts`

This round keeps the same scheduler and the same high-level backend interface.

## Functional Validation

Single-request Qwen smoke:

- output file:
  - `artifacts/humaneval_qwen_pim_naive_stdio_smoke.jsonl`
- key observations:
  - request succeeded end to end
  - `qk_batch_calls = 48`
  - `qk_helper_started = true`
  - `qk_helper_restarts = 1`
  - `qk_check_failures = 0`

This confirms the helper-based path is active and correctness remains stable.

## Two-Sample Comparison

Output:

- `artifacts/baseline_comparison_qwen_pim_stdio.jsonl`

Measured summary:

| Baseline | Avg Latency (s) | Avg Throughput (tok/s) |
| --- | ---: | ---: |
| `disagg_cpu` | `2.245` | `1.376` |
| `disagg_pim_naive` | `2.668` | `1.146` |

Compared with the previous batched-file round:

- previous `disagg_pim_naive` latency: `4.117 s`
- new `disagg_pim_naive` latency: `2.668 s`
- latency reduction vs previous round: about `35.2%`

## Five-Sample Comparison

Output:

- `artifacts/baseline_comparison_qwen_pim_stdio_limit5.jsonl`

Measured summary:

| Baseline | Avg Latency (s) | Avg Throughput (tok/s) |
| --- | ---: | ---: |
| `disagg_cpu` | `2.159` | `1.430` |
| `disagg_pim_naive` | `2.518` | `1.219` |

Derived comparison:

- `disagg_pim_naive / disagg_cpu = 1.17x` in latency

This `limit=5` run is a better current reference than the single-request or
two-request smoke numbers.

## Multi-Round Progress Summary

Using the same Qwen real-model setup, the naive PIM path improved as follows:

| Version | Avg Latency (s) | Avg Attention Decode Compute (s) | Notes |
| --- | ---: | ---: | --- |
| Original naive PIM | `5.764` | `3.495` | per-head calls, subprocess/file path |
| Round 3 batched mixed-head QK | `4.117` | `1.947` | one batched call per layer |
| Round 4 stdio helper | `2.518` | workload-dependent | persistent helper, no temp files |

Measured from the available comparisons:

- original -> round 3 latency drop: about `28.6%`
- round 3 -> round 4 latency drop: about `38.8%` when comparing the more stable
  `limit=5` round 4 result against the earlier two-sample round 3 result
- original -> round 4 latency drop: about `56.3%`

## Interpretation

This round strengthens two conclusions:

1. The original diagnosis was correct:
   - naive PIM was dominated by fixed host-side overhead, not only arithmetic
2. A substantial part of that overhead came from process/file orchestration:
   - removing temporary files and moving to a persistent helper materially
     narrows the gap to the CPU remote-attention baseline

Current state:

- the PIM path is still slower than `disagg_cpu`
- but the gap is now much smaller than before
- in the `limit=5` run, the slowdown is only about `1.17x`

This is a much stronger paper result than the original naive baseline, because
it shows a concrete optimization path from:

- "PIM path is much worse than CPU"

to:

- "once orchestration overhead is reduced, naive PIM approaches the CPU remote
  attention baseline"

## Caveat

The exact gain from helper reuse across requests is still somewhat noisy in very
small sample runs. The strongest and most stable conclusion from this round is
the benefit of removing process/file overhead from the hot path, not the exact
marginal gain of cross-request helper reuse by itself.

## Recommended Next Steps

1. Keep the stdio-helper design as the new baseline implementation.
2. Re-run:
   - mixed-head sweep
   - context sweep
   - unified baseline matrix
3. If the improved naive PIM path remains close to the CPU baseline under
   longer contexts, then the next paper question becomes whether PIM begins to
   outperform CPU once context-sensitive KV cost dominates.
