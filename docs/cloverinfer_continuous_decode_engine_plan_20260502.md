# CloverInfer Continuous Decode Engine (Step-Batched) Plan (2026-05-02)

This document records the implementation plan for a "continuous batching" decode
engine that batches decoding by **decode step** (scheme A).

Goal:
- keep the existing `disagg_pim_naive` path intact as the baseline
- implement CloverInfer's high-throughput decode orchestration as an independent
  code path
- make attention-side batching stable and large even if KV cache lives on the
  attention/PIM node

Non-goals (Phase 1):
- full GPU-side continuous batching with KV cache on GPU
- multi-step speculative decoding
- mixing different models in one engine

## 1. Current Situation (Why We Need This)

Today each request runs its own decode loop inside `GlobalScheduler.submit_request`
and the dense/GPU side is effectively `batch=1`. Attention batching happens only
opportunistically (wavefront queues) and often collapses to batch size 1 because
requests drift across `(step, layer)` timing.

Result:
- PIM helper sees many small QK rounds and high launch overhead
- we cannot reliably amortize per-layer attention overhead across requests

## 2. Target Execution Model (Scheme A: Step-Batched)

At each decode step `t`, we build a dynamic batch of active requests
(`B <= decode_continuous_max_batch_size`), and run the following iteration:

1. Dense: `start_token_batch` (for step 1 only) or carry hidden from previous step
2. For each layer `L`:
   1. Dense: `prepare_attention_batch(hidden_B, L, request_ids_B, context_lens_B)`
      outputs per-request:
      - `residual_B` (for `finish_layer`)
      - `query_B`, `key_new_B`, `value_new_B` (for attention + KV append)
      - `score_scale` (model-dependent, can be scalar shared by the batch)
   2. Attention: `decode_layer_batch(payloads_B)` where each payload is one request
      for the same `(step, layer)` and includes the Q/K/V for KV append + attention.
   3. Dense: `finish_layer_batch(residual_B, context_B, L)` -> next hidden_B
3. Dense: `sample_next_token_batch(hidden_B)` -> next token ids
4. Update each request state (generated tokens, done flags, context_len increment)
5. Repeat for the next step with the updated active set.

Key property:
- within a step, all requests in the batch are perfectly aligned at the same
  layer, so attention-side batching is stable.

## 3. Interfaces to Add

### 3.1 DecodeDenseNode batch RPCs (GPU)

Add batch variants without removing existing single-item RPCs:
- `start_token_batch(token_ids: List[int], positions: List[int]) -> {hidden: Tensor[B,1,H]}`
- `prepare_attention_batch(hidden: Tensor[B,1,H], layer_idx: int, request_ids: List[str], context_lens: List[int])`
  returns:
  - `residual: Tensor[B,1,H]` (CPU or GPU, but consistent)
  - `query: Tensor[B,heads,dim]` (CPU for transport to attention)
  - `key: Tensor[B,heads,dim]` (CPU)
  - `value: Tensor[B,heads,dim]` (CPU)
  - `score_scale: float`
- `finish_layer_batch(residual: Tensor[B,1,H], context: Tensor[B,1,H], layer_idx: int) -> {hidden: Tensor[B,1,H]}`
- `sample_next_token_batch(hidden: Tensor[B,1,H]) -> {token_ids: List[int]}`

Implementation constraints:
- initial implementation targets OPT first (opt-125m), then Qwen
- `seq_len` stays 1 for decode; `B` can be > 1

### 3.2 Scheduler-side continuous decode engine

Add a new code path (do not change the baseline API):
- `GlobalScheduler.submit_request_continuous(...)` or a config flag that selects
  the engine when `cluster_config.decode_continuous_batching_enabled=True`.

Engine state (in scheduler actor):
- active request table: prompt_len, generated_ids, current_token_id, hidden, done
- per-step barrier: collect up to `max_batch_size` items or wait `batch_window_s`
- per-layer pipeline: call dense batch RPC -> attention batch RPC -> dense batch RPC

Correctness checks (optional, Clover-only):
- reuse Clover shadow check infrastructure in `CloverInferAttentionBackend`

## 4. Phase Plan

### Phase 1: Minimal working engine for OPT (batch > 1)
- implement the Dense batch RPCs on `DecodeDenseNode`
- implement step-batched scheduler loop for `max_new_tokens=2` and `limit small`
- ensure attention batching sees `batch_size=B` consistently
- add tracing output fields to confirm effective batch size and helper rounds

Acceptance criteria:
- `scheduler_attention_batching.max_observed_size` becomes >= 2 at concurrency=2
- `helper_dpu_profile.qk_rounds_total` decreases vs. baseline for same workload

### Phase 2: Make it robust for longer decode and Qwen
- handle EOS per-request while keeping batch packed (drop completed requests)
- support Qwen rotary position handling via `context_lens_B` inputs
- add backpressure / admission control for SLO experiments

### Phase 3: Performance improvements
- reduce CPU copies between Dense -> Scheduler -> Attention (use shared memory or
  object store references if needed)
- combine attention RPCs to fused helper commands where possible

## 5. Risks / Tradeoffs

- Increased scheduler complexity (centralized engine) vs. per-request loop.
- Batch formation can increase tail latency under strict SLO; need a small
  `batch_window_s` and an optional "max_wait" policy.
- FP16 resident KV can reduce capacity fallback but may increase per-round compute;
  batching must be large enough to amortize helper launch overhead.

## 6. Related Files

- Scheduler: `src/core/scheduler.py`
- Dense node: `src/core/nodes.py`, `src/core/model_adapter.py`
- Attention: `src/core/attention_backend.py`, `src/core/clover_attention_backend.py`
- Experiments: `tests/trace_pim_allocator.py`, `tests/benchmark_*`

