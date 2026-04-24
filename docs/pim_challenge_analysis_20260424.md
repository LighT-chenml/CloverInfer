# PIM Challenge Analysis Notes

Date: 2026-04-24

## Purpose

This note records the current bottleneck analysis for the PD + AF split
prototype and the naive PIM attention baseline. The goal is to preserve
paper-usable observations, not just working notes.

The key rule in this document is:

- first state what was measured
- then state what is inferred from those measurements

## Measurement Sources

- Unified Qwen baseline comparison:
  - `artifacts/baseline_comparison_qwen_v2.jsonl`
  - `tests/benchmark_baselines.py`
- Naive PIM mixed-head sweep:
  - `docs/pim_attention_sweep_round1.md`
  - `artifacts/attention_sweep_heads_timing.jsonl`
- Post-stdio mixed-head sweep:
  - `docs/pim_attention_sweep_after_stdio_round5_20260424.md`
  - `artifacts/attention_sweep_heads_stdio_round4.jsonl`
- Naive PIM context sweep:
  - `docs/pim_context_sweep_round2.md`
  - `artifacts/context_sweep_round2.jsonl`
- Post-stdio context sweep:
  - `docs/pim_context_sweep_after_stdio_round5_20260424.md`
  - `artifacts/context_sweep_after_stdio_round5.jsonl`

## Main Baseline Matrix

Setup:

- model: `Qwen-1_8B`
- dataset: `dataset/humaneval.jsonl`
- samples: first `2`
- `max_new_tokens = 3`

Measured summary:

| Baseline | Avg Latency (s) | Avg TTFT (s) | Avg Throughput (tok/s) |
| --- | ---: | ---: | ---: |
| `monolithic_gpu` | `0.242` | `0.185` | `22.59` |
| `split_gpu_full_decode` | `0.841` | `0.540` | `4.21` |
| `disagg_cpu` | `2.224` | `0.530` | `1.39` |
| `disagg_pim_naive` | `5.764` | `0.534` | `0.52` |

Derived latency ratios:

- `split_gpu_full_decode / monolithic_gpu = 3.48x`
- `disagg_cpu / split_gpu_full_decode = 2.64x`
- `disagg_pim_naive / disagg_cpu = 2.59x`
- `disagg_pim_naive / monolithic_gpu = 23.85x`

## Evidence 1: Prefill/Decode Split Is Not the Dominant Problem

Measured facts:

- Moving from `monolithic_gpu` to `split_gpu_full_decode` increases latency by
  about `3.48x`.
- Moving from `split_gpu_full_decode` to `disagg_cpu` increases latency by an
  additional `2.64x`.

Inference:

- The current overhead of prefill/decode disaggregation is real, but it is not
  the largest bottleneck in the present system.
- A larger extra cost appears only after attention is moved off the dense GPU
  node.

Paper-usable phrasing:

- "In the current prototype, disaggregating prefill from decode is not the
  dominant source of slowdown; the larger penalty appears when decode-time
  attention becomes remote."

## Evidence 2: Remote Attention Is the Main Current Bottleneck

Measured facts from `disagg_cpu` records:

- `prepare_attention_rpc_s` is about `0.49-0.73 s`
- `attention_decode_rpc_s` is about `0.46-0.49 s`
- `finish_layer_rpc_s` is about `0.39-0.42 s`
- `attention_decode_compute_s` on the CPU attention node is only about
  `0.088-0.123 s`
- decode layers per request: `48`

Inference:

- Even in the CPU baseline, remote attention cost is not only raw attention
  compute.
- RPC cadence and per-layer cross-node synchronization are already a meaningful
  component of end-to-end decode latency.
- The remote-attention overhead is therefore a combination of:
  - compute on `.7`
  - per-layer RPC overhead
  - fine-grained decode choreography between `.4` and `.7`

This means that improving only the arithmetic kernel may not be enough; the
interaction pattern itself is also part of the challenge.

## Evidence 3: Naive PIM Is Dominated by Attention-Side Decode Cost

Measured facts from `disagg_pim_naive` records:

- `attention_decode_rpc_s` is about `3.88-3.93 s`
- `attention_decode_compute_s` is about `3.47-3.52 s`
- `prepare_attention_rpc_s` is only about `0.55-0.77 s`
- `finish_layer_rpc_s` is only about `0.45-0.47 s`
- `qk_check_failures = 0`
- `qk_mixed_last_max_abs_diff` stayed below about `1e-3`

Inference:

- The naive PIM slowdown is concentrated in the attention-node decode path.
- Correctness is stable enough that the current issue is performance, not
  numerical failure.
- Since the non-attention RPC stages stay in the sub-second range while
  `attention_decode_rpc_s` rises into the multi-second range, the first
  optimization target should remain the `.7` backend path rather than the dense
  node pipeline.

## Evidence 4: The Naive PIM Cost Scales With Mixed-Head Invocation Count

Measured facts from round 1 head sweep:

| Mixed Heads | Avg Latency (s) | Attention Decode Compute (s) |
| ---: | ---: | ---: |
| `0` | `2.448` | `0.065` |
| `1` | `5.020` | `2.564` |
| `2` | `7.590` | `5.130` |
| `4` | `12.070` | `9.619` |
| `8` | `21.508` | `18.991` |
| `12` | `31.455` | `29.006` |

Inference:

- The naive PIM path scales roughly with the number of mixed heads.
- This strongly suggests the main cost is repeated host-side invocation and
  orchestration per head, rather than a mostly fixed transport cost.
- The current PIM backend should therefore be understood as a per-head QK
  invocation prototype, not as a throughput-oriented attention engine.

## Evidence 5: The Naive PIM Cost Is Weakly Sensitive to Context Length in the Tested Range

Measured facts from round 2 context sweep:

| Prompt Tokens | CPU Attention Compute (s) | Naive PIM Attention Compute (s) |
| ---: | ---: | ---: |
| `16` | `0.065` | `5.118` |
| `64` | `0.077` | `5.108` |
| `256` | `0.161` | `5.437` |
| `512` | `0.294` | `5.638` |

Inference:

- CPU attention cost grows clearly with context length, which is expected.
- Naive PIM cost grows only mildly in this range.
- Therefore the present naive implementation is dominated by fixed invocation
  overhead, not by the context-sensitive KV-read cost alone.

This is an important paper point:

- the current prototype has not yet reached the regime where PIM's potential
  bandwidth advantage can dominate its orchestration overhead

## Current Challenge Statement

The present measurements support the following challenge statement:

1. PD split alone is not the primary blocker.
2. The main systems challenge is remote decode-time attention.
3. Within remote attention, the current naive PIM bottleneck is dominated by
   fine-grained host orchestration and repeated QK invocation overhead.
4. The current implementation does not yet expose a regime where PIM-side data
   movement advantages outweigh host-side launch and marshaling costs.

## Optimization Priorities Suggested by the Data

Priority 1:

- Reduce per-head and per-layer invocation overhead on `.7`
- Replace repeated subprocess / file-based QK execution with a more persistent
  or batched path

Status update:

- a first batched mixed-head host-path optimization was implemented and
  validated in `docs/pim_batched_qk_round3_20260424.md`
- this reduced `disagg_pim_naive` average latency from about `5.764s` to about
  `4.117s` on the current two-sample Qwen smoke comparison
- the result strengthens the conclusion that invocation granularity was a real
  bottleneck rather than a speculative one
- a second optimization replaced temporary-file/process-heavy QK execution with
  a persistent stdio helper, recorded in
  `docs/pim_stdio_helper_round4_20260424.md`
- in a `limit=5` Qwen comparison, this brought `disagg_pim_naive` down to about
  `2.518s`, versus about `2.159s` for `disagg_cpu`
- this further strengthens the conclusion that host-side orchestration was a
  first-order bottleneck
- a post-stdio mixed-head sweep, recorded in
  `docs/pim_attention_sweep_after_stdio_round5_20260424.md`, showed that the
  previous latency explosion versus `mixed_heads` was dramatically reduced
- for example, at `mixed_heads = 12`, average latency dropped from about
  `31.455s` in round 1 to about `3.731s` after batching + persistent helper
- this indicates the original mixed-head scaling was dominated far more by
  invocation granularity than by unavoidable DPU-side arithmetic cost

Priority 2:

- Batch more work into each attention-side call
- Increase the useful work per launch before revisiting more advanced kernels

Priority 3:

- Re-evaluate longer contexts after host-side batching is improved
- The current `16-512` token sweep is still dominated by fixed overhead
- a post-stdio context sweep, recorded in
  `docs/pim_context_sweep_after_stdio_round5_20260424.md`, shows that naive PIM
  is now more context-sensitive than before, but still remains slower than the
  remote CPU baseline in the tested `16-512` token range
- this suggests the system is finally approaching a regime where actual KV/QK
  work is visible again, although no PIM advantage is yet exposed at this scale

Priority 4:

- Only after the above, consider more aggressive compute-path changes such as
  moving additional attention stages beyond QK or revisiting `AV`

## Writing Guidance

The current evidence supports a challenge-driven paper narrative:

- demonstrate a real three-node PD + AF split framework
- quantify where the overhead enters using a layered baseline matrix
- show that remote attention is the main bottleneck
- show that naive PIM currently exposes the critical orchestration challenge
- motivate optimized PIM execution as the next step

What the current evidence does not yet support:

- a strong claim that the present PIM-based design already outperforms the best
  conventional baseline
- a strong claim that PIM benefits have already emerged at the tested scale

This distinction should be preserved in later writing.
