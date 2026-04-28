# CloverInfer Optimization Status (2026-04-28)

This note captures the current CloverInfer-specific optimization status while
keeping `disagg_pim_naive` / `PimNaiveAttentionBackend` intact as the
comparison baseline.

## Scope

- CloverInfer logic stays isolated in
  `src/core/clover_attention_backend.py`.
- Baseline `pim_naive` behavior is intentionally preserved.

## Confirmed Wins

### 1. Sampled shadow checks as default

Current default:

- `clover_shadow_check_token_interval = 4`
- `clover_shadow_check_layer_interval = 4`

Why kept:

- reduces resident shadow validation frequency
- lowers `resident_shadow_check_s`
- produces measurable end-to-end improvement versus full `(1, 1)` checks

Key artifact:

- `artifacts/clover_shadow_sweep_tok256_gen8_stage1.jsonl`

### 2. Host fast path avoids per-layer resident materialization

CloverInfer host path now:

- computes score/context from CPU shadow KV
- materializes resident KV only for sampled validation

Why kept:

- removes full-KV reconstruction from the hot path
- clearly reduces `prepare_decode_record_s`
- reduces `resident_materialize_s`

Key artifacts:

- `artifacts/pim_trace_cloverinfer_defaultsampled_materializeopt.jsonl`
- `artifacts/clover_shadow_sweep_tok256_gen8_stage2_materializeopt.jsonl`

### 3. Host path skips qk_mixed by default

Current default:

- `clover_host_qk_mixed_enabled = false`

Why kept:

- in host resident-store mode, `qk_mixed` was mostly redundant validation-like
  work rather than acceleration
- removing it gives a real end-to-end win

Key artifacts:

- `artifacts/pim_trace_cloverinfer_host_nomixed_default.jsonl`
- `artifacts/clover_shadow_sweep_tok256_gen8_stage3_host_nomixed.jsonl`

Observed stage result:

- stage2 `(4,4)` latency: about `3.0427s`
- stage3 `(4,4)` latency: about `2.8941s`

### 4. CPU shadow KV uses preallocated buffers

CloverInfer now keeps independent CPU shadow buffers with reserved capacity and
in-place append.

Why kept:

- removes per-token `torch.cat` growth cost from the main path
- sharply reduces `cpu_shadow_append_s`
- lowers `prepare_decode_record_s`

Key artifacts:

- `artifacts/pim_trace_cloverinfer_host_shadowbuf.jsonl`
- `artifacts/clover_shadow_sweep_tok256_gen8_stage4_shadowbuf.jsonl`

## Experiments That Were Not Worth Keeping

### 1. Batched host context compute

Tried:

- grouping host fallback records
- batched `softmax + matmul` for context compute

Why not kept:

- reduced `host_context_compute_s` locally
- increased finalize-side orchestration overhead
- no convincing end-to-end benefit

Artifact:

- `artifacts/clover_shadow_sweep_tok256_gen8_stage5_batchctx.jsonl`

### 2. Dense residual handle / scheduler-side RPC micro-optimization

Tried:

- keeping dense residual locally
- passing only a small handle back for `finish_layer`

Why not kept:

- technically valid
- savings were too small because the dominant cross-actor payloads still remain
- added state-management complexity without meaningful payoff

Artifacts:

- `artifacts/pim_trace_cloverinfer_schedulerdensefused.jsonl`
- `artifacts/pim_trace_cloverinfer_densehandle.jsonl`

## Current Recommended CloverInfer Defaults

- attention backend: `cloverinfer`
- resident store backend: `host`
- `clover_shadow_check_token_interval = 4`
- `clover_shadow_check_layer_interval = 4`
- `clover_host_qk_mixed_enabled = false`
- CPU shadow enabled
- shadow checks enabled
- op profiling enabled during optimization work

## Recommended Benchmark Entry

For the current best-maintained host-path benchmark, use:

- `tests/benchmark_clover_host_best.py`

This script is intended to reduce run-to-run noise by:

- keeping the best-known CloverInfer host defaults fixed
- supporting warmup runs
- supporting larger repeat counts
- emitting scheduler/actor/clover timing summaries in one place

## Best Current Reference Result

For the repeated `prompt_token_length=256`, `max_new_tokens=8`, `repeats=3`
host-path comparison, the most practically useful current result is stage4:

- artifact:
  `artifacts/clover_shadow_sweep_tok256_gen8_stage4_shadowbuf.jsonl`
- average latency: about `2.9105s`

Even though stage3 was slightly lower in one short run, stage4 keeps the CPU
shadow append optimization and remains the best maintained implementation state.

## Recommended Next Work

The next optimization round should focus on one of:

1. More stable benchmarking
2. Larger-grain communication reduction instead of small RPC/local-cache tricks

Recommended first:

- expand benchmark repeats
- test longer decode lengths
- separate scheduler overhead from actor compute more aggressively

This avoids overfitting to noisy short runs while preserving the baseline
comparison story.
