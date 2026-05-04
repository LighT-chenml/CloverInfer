## Dense-side Continuous Batching Plan

### Goal

Change decode execution from the current per-request loop into a scheduler-driven
continuous batching loop where each decode iteration advances a wavefront batch
through both sides of the split:

1. Dense `start_token`
2. For each layer:
   - Dense `prepare_attention`
   - Attention `decode_layer_batch`
   - Dense `finish_layer`
3. Dense `sample_next_token`

The scheduler, not each request coroutine, becomes the owner of decode
progression. Requests join a shared decode queue and are advanced in batches.

### Why The Current Path Is Not Enough

- `submit_request()` currently runs a private decode loop per request.
- Dense compute is dispatched through only single-item RPCs.
- Scheduler-side batching exists only around the attention decode hop.
- This means request alignment helps only the attention node, while dense work
  still pays per-request RPC and kernel launch overhead.

### Target Execution Model

### 1. Request Lifecycle

- `submit_request()` performs prompt prefill and attention-side KV init.
- The request is converted into a scheduler-owned decode state.
- The request is placed into a shared active queue.
- A background decode driver wakes up and schedules batches until each request
  finishes or emits EOS.

### 2. Decode Wavefront

For a selected batch of active requests at decode step `s`:

- run dense `start_token_batch`
- iterate layers `0..L-1`
- at each layer:
  - run dense `prepare_attention_batch`
  - run attention `decode_layer_batch`
  - run dense `finish_layer_batch`
- run dense `sample_next_token_batch`
- update per-request decode state
- retire finished requests and requeue unfinished requests

This is the concrete “Dense and Attention bounce back and forth per batch”
execution style we want.

### 3. Batching Policy

- Use a scheduler-owned active decode queue.
- Each scheduling round pops up to
  `cluster_config.decode_continuous_batch_max_size` requests.
- Requests that are not finished are appended back to the queue after the round.
- This is a simple round-robin wavefront policy intended to get the structure in
  place first. Smarter policies can be layered on later.

### Metrics

Per request:

- `ttft`: prefill start to first token availability
- `latency`: request start to final output availability
- `tpot`: `(latency - ttft) / max(total_tokens - 1, 1)`
- `throughput`: `total_tokens / latency`
- `total_tokens`

Scheduler summary additions:

- dense continuous batching config and observed batch stats
- per-stage RPC and actor timing, still split into scheduler vs actor views

### Implementation Stages

1. Add Dense batch primitives to `CausalModelAdapter` and `DecodeDenseNode`.
2. Introduce scheduler-owned decode request state and a background decode loop.
3. Convert `submit_request()` into:
   - prefill
   - attention init
   - enqueue
   - await completion future
4. Reuse attention batched decode for per-layer batch execution.
5. Add scheduler metrics for dense continuous batching.
6. Run decoding and integration smoke tests.

### Compatibility Notes

- Keep existing single-request interfaces in place for compatibility.
- Preserve the existing metrics keys consumed by benchmark scripts.
- Preserve attention backend instrumentation already used by current traces.
