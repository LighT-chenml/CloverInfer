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

### Follow-up Scheduler Optimizations

After the first correctness pass, two throughput-oriented scheduler refinements
were added:

1. Decode batch aggregation window
   - Add `decode_continuous_batch_window_s`.
   - The decode driver may wait briefly before flushing an underfilled batch when
     more active decode requests are expected to arrive.
   - Metrics now expose flush reasons and total wait time so the policy can be
     tuned with data instead of guesswork.

2. Async request completion
   - Finished requests no longer force the decode wave to synchronously perform
     `free_request`, final token decoding, and metrics collection inline.
   - Those completion steps now run in background tasks so the hot decode path
     can keep advancing surviving requests.

### Early Measurement Notes

On local `opt-125m` CPU smoke runs:

- A small aggregation window alone gave only modest gains when requests were
  already submitted at the same time, because later decode waves naturally
  formed full batches.
- Moving request completion out of the decode hot path gave the clearer win:
  average throughput improved from about `1.25 tok/s` to about `1.30 tok/s`
  on a `limit=4`, `max_new_tokens=8`, `decode_continuous_batch_max_size=4`
  benchmark.

This suggests the next valuable scheduler work should focus less on initial
batch formation and more on reducing non-decode stalls inside the active wave.

### DPU Allocator Follow-up

Once the scheduler-side continuous batch path became stable, the next bottleneck
shifted to DPU KV placement and space management:

- static `rotated` / `rank_spread` placement can keep requests spread out, but
  it does not react to current live KV pressure
- blocked-group allocation previously appended blocks by fixed DPU progression
  from a base DPU, which makes long/short request mixtures drift toward skew
- runtime debug exposed raw per-DPU allocator stats, but not a standard summary
  that benchmark scripts could compare directly

The current follow-up implementation adds:

- `load_aware` placement as a new DPU placement policy
- runtime-load-aware target selection for both whole-group and blocked-group
  allocations
- allocator-visible actual `physical_dpu` returned to the attention side so
  head-group metadata reflects real placement, not only the requested hint
- standardized resident-store summaries:
  - `allocator_summary`
  - `dpu_balance_summary`
  - `rank_balance_summary`
  - `block_summary`

This keeps the scheduler and benchmark layer focused on TTFT / TPOT /
Throughput, while making DPU utilization and placement imbalance directly
observable in the same result payloads.

### Current Local Bring-up Finding

On the current local machine, `upmem_kvslot` did not actually execute on DPUs
during benchmark validation. The immediate root cause was environmental rather
than scheduling-related:

- the configured helper binary path
  `src/pim/upmem_kvslot/build/host_kvslot` was missing
- the local environment also lacks `dpu-upmem-dpurte-clang` /
  `dpu-pkg-config`, so the helper cannot currently be rebuilt in-place

Without that helper binary, resident KV allocation silently fell back to host
storage and made DPU utilization appear as zero. The runtime now raises a clear
startup error when the helper binary is missing so this failure mode is explicit.

### 2026-05-05 Decode Batching Follow-up

While validating the real three-machine topology
(`192.168.123.4` prefill, `192.168.123.3` decode-dense, `192.168.123.7`
attention/DPU), decode continuous batching exposed a correctness gap in the
dense batched path:

- `DecodeDenseNode.prepare_attention_batch()` could receive a hidden state that
  had been wrapped as a singleton Python `list` during batched decode
  progression
- `CausalModelAdapter.prepare_attention()` then called `.to(...)` on that value
  and failed before any DPU work could start

The current fix makes the dense adapter normalize singleton containers at the
entry points used by batched decode:

- `prepare_attention()`
- `prepare_attention_batch()`
- `sample_next_token()`
- `sample_next_token_batch()`

This keeps the continuous batching hot path robust even if a scheduler or RPC
boundary returns `[tensor]` instead of `tensor` for a single request item.

At the same time, `start_token_batch()` now supports `llama` so the shared
dense batching path covers the model families we want to validate next:

- Qwen
- OPT
- Llama

### 2026-05-05 DPU Space Management Follow-up

After the first real three-machine UPMEM run succeeded, the resident-store
summary showed that DPU utilization was no longer zero, but host fallback was
still too high:

- all 4 DPUs were active
- DPU load balance was already even
- but a noticeable number of allocations still fell back to host storage

The first concrete allocator issue was internal fragmentation in the blocked KV
path:

- blocked allocation always rounded each initial block up to `block_tokens`
- this meant a logical `capacity=128` tail block still consumed a full
  `256-token` DPU block at allocation time

### 2026-05-05 Attention Locality Follow-up

The next throughput issue is not only capacity but locality:

- `load_aware` placement helped balance live KV usage, but it still treated all
  candidate DPUs with similar free space as effectively interchangeable
- for blocked KV groups, later blocks in the same logical sequence could drift
  away from the previous block and even cross rank boundaries
- on the helper side, launch rounds were previously built in arrival order, so
  shape-compatible items from the same rank were often mixed together with
  unrelated items and pushed onto the fallback path

The current follow-up tightens both sides around attention locality:

1. Python resident-store placement
   - blocked append allocation now uses the most recent block placement as its
     locality anchor instead of only the group's initial base DPU
   - `load_aware` scoring now explicitly prefers:
     - same-rank candidates
     - nearby DPUs in circular topology distance
     - then lower live load / slot pressure

2. UPMEM helper round construction
   - batched QK / AV rounds now default to shape-compatible grouping
   - round builders now prefer items that share the seed rank before filling the
     rest of the round with other compatible items

The intent is to preserve DPU bandwidth advantages for the active attention
window instead of paying to re-spread one logical sequence across many distant
DPUs and ranks.

### 2026-05-05 QK Path Bring-up Follow-up

After the locality pass, the next question was whether the main QK path should
move fully onto the current UPMEM helper path.

Measured on the same three-machine `opt-125m`, `4 DPU`, `limit=4`,
`max_new_tokens=4` setup:

- locality-improved resident AV path with `qk_mixed` still enabled:
  about `0.68 tok/s`
- full `qk_full + softmax_av_fused` path with shadow checks disabled:
  about `0.56 tok/s`
- locality-improved resident AV path with `qk_mixed` disabled:
  about `0.80 tok/s`

This gives two concrete conclusions:

1. The current full-QK UPMEM path is functionally working
   - QK rounds became fully batched on 4 DPUs
   - but launch-dominated kernel cost is still too high for this workload

2. For throughput-oriented benchmarking right now, the better operating point is
   - keep the improved locality-aware resident AV path
   - disable `qk_mixed` shadow traffic by default in the Humaneval benchmark

So the near-term optimization priority is not “force full QK on DPU at all
costs”, but:

- keep benchmark defaults aligned with real throughput measurement
- continue reducing helper/kernel launch overhead before revisiting full-QK as
  the default path
- later layers with smaller logical capacity therefore exhausted DPU capacity
  earlier than necessary and increased host fallback

The allocator now builds blocked initial allocations from the actual block
layout produced by `(capacity, seq_len)`:

- full blocks still use `block_tokens`
- the final short block now uses its real logical capacity instead of being
  over-expanded to `block_tokens`

This is a correctness-preserving space-efficiency fix intended to reduce:

- `fallback_allocations`
- `dpu_capacity_fallbacks`
- unnecessary DPU block waste in later/lighter head groups

### 2026-05-05 More-DPU Follow-up

After enabling `16 DPU` and `128 DPU` cluster runs, the measured behavior did
not match the naive expectation that “more DPUs should always be faster”:

- `4 DPU` with locality-aware resident AV and `qk_mixed` disabled remained the
  best throughput point so far at about `0.80 tok/s`
- `16 DPU` removed allocator fallback entirely, but throughput dropped to about
  `0.63 tok/s`
- `128 DPU` on the same resident AV path stayed in the same range at about
  `0.63 tok/s`
- full `qk_full + softmax_av_fused` also functionally scaled to `16` and
  `128 DPU`, but throughput regressed further because the current helper path is
  still launch-dominated

The key topology/placement lesson is:

- more allocated DPUs increased total KV capacity
- but it also increased the physical DPU/rank set touched by each decode wave
- the current helper launch path does not yet amortize that wider fan-out well

In other words, the current bottleneck is no longer only “fit more KV”, but
also “keep each decode step local enough that host-side QK/AV launch overhead
does not dominate”.

### 2026-05-05 Request-Local Stripe Placement

To address that fan-out problem, the resident KV placement policy now adds a
request-local stripe constraint on top of the existing `load_aware` allocator:

1. Request-level stripe selection
   - when a request is initialized, the attention backend computes a preferred
     DPU stripe for that request
   - stripe width is chosen from:
     - the maximum resident group count needed by any layer
     - an approximate capacity requirement derived from the request’s total KV
       live elements

2. Group allocation inside the stripe
   - initial per-layer head groups prefer DPUs inside that stripe
   - the resident store now accepts an `allowed_dpus` hint so
     `choose_physical_dpu()` can stay inside the stripe unless there is no
     viable candidate

3. Block append locality inside the stripe
   - blocked KV append continues to use the latest block as the locality anchor
   - but now also keeps later blocks inside the same request stripe whenever
     possible

The intended effect is:

- preserve the KV-capacity benefit of using more total DPUs
- while reducing the number of distinct physical DPUs each request touches
- so QK/AV rounds stay denser and more local on the DPU side
- and the host helper pays less per-step launch/set construction overhead

This is the current bridge between:

- the user requirement that most QK/AV work should live on DPU
- and the observed reality that naive “spread wider across more DPUs” currently
  hurts throughput

### 2026-05-05 128-DPU Locality Follow-up

The first two `128 DPU` locality experiments established the two failure modes
we need to avoid:

- too-wide request placement
  - a request stripe that effectively spans all `128` DPUs keeps per-DPU KV
    pressure low
  - but each decode wave touches nearly the whole machine, so helper
    launch/set-construction overhead dominates
- too-narrow request placement
  - aggressively compacting each request down to about `6` active DPUs did cut
    fan-out sharply and usually stayed inside `1` active rank per AV round
  - but too many resident groups then stacked onto each active DPU, so
    `av_rounds_total` jumped from about `135` to about `244`

That means the next placement target is the middle ground:

- prefer a single physical rank when possible
- use a medium-width stripe inside that rank
- keep enough width that same-layer groups can still spread across multiple
  DPUs instead of serializing behind one tiny hot set

The implementation follow-up therefore changes request stripe selection from:

- primarily capacity-driven shrinking

to:

- capacity floor plus a minimum parallel width derived from resident
  head-group count
- rank-aware stripe construction using helper topology when available
- stripe-local rotation so different layers do not all pin their first groups
  to the same few DPUs

The success criterion for this next round is not “minimum touched DPUs at any
cost”, but:

- fewer touched DPUs than the old full-width `128 DPU` path
- fewer helper rounds than the over-compact path
- recovered or improved throughput with stable TTFT / TPOT

### 2026-05-05 Dynamic KV Placement Follow-up

Static request placement helped find a better locality point, but it still has
an obvious limitation:

- stripe width is chosen only once at request init
- later decode growth can make that stripe too narrow for the request's real
  runtime footprint
- widening every request up front hurts locality for shorter or colder requests

The next implementation step therefore adds a first dynamic-KV mechanism:

- append-time stripe expansion
- no migration of existing KV blocks yet
- only future block allocations can use the widened stripe

This stage is intentionally conservative:

1. keep old blocks in place
   - avoids background migration complexity and correctness risk
2. widen only when request growth justifies it
   - use request context growth and resident group structure as the trigger
3. apply widening through existing `allowed_dpus` control
   - each resident group updates its allowed stripe
   - later blocked appends can place new blocks on the wider stripe

This gives us a usable intermediate point between:

- purely static locality-aware placement
- and full dynamic KV migration

The expected benefit is:

- short requests stay local
- longer-running requests gain more DPU parallel headroom over time
- continuous batching can react better to changing active-request pressure
