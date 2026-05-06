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

### 2026-05-05 Dynamic KV Placement Follow-up

To make dynamic KV placement materially affect real decode traffic, blocked KV
allocation now distinguishes between:

- `base` blocks used for prefill-era bulk allocation
- `growth` blocks used for decode-era append allocation

This is needed because stripe expansion alone does not move already-allocated
`256-token` base blocks. Without a smaller decode-tail allocation unit, a
request can widen its preferred stripe while still appending into the original
physical block for a long time.

The first correctness pass with `growth_block_tokens=64` confirmed that dynamic
placement was finally real:

- request footprints started reporting non-zero `stripe_expand_count`
- later blocks were allocated on the widened stripe instead of only the base
  DPU set
- blocked-slot debug showed mixed `base` and `growth` physical blocks

However, throughput regressed because the current helper protocol still treats
each physical block as a distinct QK/AV batch item:

- one logical blocked request with multiple `growth` blocks becomes multiple
  helper items
- the helper round builder allows only one item per DPU per round
- this sharply increased `av_rounds_total`, `av_fallback_rounds`, and aggregate
  active-rank traffic

The current mitigation is intentionally conservative:

- keep dynamic stripe expansion
- keep base-block rollover so widened stripes can take effect during decode
- increase `growth_block_tokens` to `128`

The working hypothesis is that `128` is still small enough to let decode-time
KV placement react to stripe growth, but large enough to avoid the severe
round-fragmentation penalty seen with `64`.

If this still leaves throughput below the best static-locality point, the next
step should be helper-side coalescing of multiple blocks from the same logical
request into fewer AV/QK round items instead of further shrinking decode block
granularity.

### 2026-05-05 Helper Submit-Order Follow-up

The first valid `growth_block_tokens=128` cluster run confirmed that physical
block fragmentation did improve:

- blocked slots dropped to `2` blocks per logical slot in many cases instead of
  `3`
- but helper-side `av_rounds_total` and fallback traffic stayed high
- throughput still regressed to roughly `0.72 tok/s`

That shifts the immediate bottleneck from pure DPU space fragmentation to
helper-side round construction quality.

As an intermediate step before changing the helper protocol itself, the
resident store now reorders outgoing DPU batch items before submitting them to
the helper:

- group by helper rank first
- then by round-shape compatibility (`heads`, `window`, `head_dim`)
- preserve logical correctness by restoring original segment order when merging
  outputs back into one logical request

This does not yet reduce the number of physical blocks, but it gives the
existing greedy helper round builder a better input order so same-rank,
shape-compatible items are more likely to land in the same launch round.

If this still does not recover throughput, the next step remains true protocol
coalescing:

- multiple contiguous blocks from one logical request should become fewer AV/QK
  helper items
- otherwise dynamic KV placement will keep paying one helper item per physical
  growth block

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

### 2026-05-05 Helper AV Batching Follow-up

The first grouped-AV helper change proved correctness, but the full-cluster
throughput improvement was still too small:

- `groupedav_full`
  - `avg_throughput = 0.7227 tok/s`
  - `avg_tpot = 1.3999 s`
  - `av_fallback_rounds = 891`

That result showed the real remaining issue was not just rank locality, but
that a large fraction of AV work was still falling back to per-item launches.

The next helper-side change therefore widened the batched-AV eligibility:

- allow one AV round to batch items with different `seq_len`
  - as long as `group_heads`, `head_dim`, dtype, and mode still match
- transfer weights with round-local zero padding up to the max byte size in
  that round
- run grouped AV through the same round builder / batched launcher instead of
  one launch per grouped item

This keeps correctness simple while removing the previous “identical
`padded_weight_bytes` only” restriction that split otherwise compatible work
back into fallback launches.

Real cluster validation on `opt-125m`, `128 DPU`, `limit=2`,
`max_new_tokens=64`:

- previous best helper-locality path: `groupedav_full`
  - `avg_latency = 88.5567 s`
  - `avg_ttft = 0.3656 s`
  - `avg_tpot = 1.3999 s`
  - `avg_throughput = 0.7227 tok/s`
  - `attention_decode_compute_s = 29.8158 s`
  - `av_rounds_total = 3915`
  - `av_batched_rounds = 3024`
  - `av_fallback_rounds = 891`
- new batched-pad helper path: `groupedav_batchedpad_full`
  - `avg_latency = 63.7910 s`
  - `avg_ttft = 0.3834 s`
  - `avg_tpot = 1.0065 s`
  - `avg_throughput = 1.0033 tok/s`
  - `attention_decode_compute_s = 17.5744 s`
  - `av_rounds_total = 1536`
  - `av_batched_rounds = 1536`
  - `av_fallback_rounds = 0`

What this establishes:

- helper AV fallback elimination is a first-order throughput lever
- dynamic blocked KV placement can perform well once helper launch fragmentation
  is removed
- the next optimization target should move to:
  - QK partial / fused helper batching
  - scheduler policy on top of this improved AV baseline

### 2026-05-05 QK Grouped / Mixed-Window Follow-up

After the AV-only path reached about `1.00 tok/s`, the next question was
whether the full decode QK path could be made competitive on the same real
three-machine topology.

First, the helper-side grouped QK path was brought up and validated:

- grouped same-DPU blocked segments can now be submitted as one logical QK item
- `qk_full` correctness stayed within shadow-check tolerance

Smoke results on `opt-125m`, `128 DPU`, `limit=1`, `max_new_tokens=8`:

- original `qk_full_smoke`
  - `throughput = 0.8308 tok/s`
  - `qk_rounds_total = 224`
  - `qk_fallback_rounds = 56`
- grouped-QK `qkfull_grouped_smoke`
  - `throughput = 0.8631 tok/s`
  - `qk_rounds_total = 217`
  - `qk_fallback_rounds = 49`

So grouped QK was real and modestly helpful in the single-request case.

The next helper follow-up then tried to improve the actual continuous-batch
decode case by:

- allowing batched QK rounds to mix different `window` sizes with round-local
  zero padding
- moving grouped-QK items onto the same round builder / batched launcher path
  instead of launching each grouped item individually

However, full cluster validation on the real continuous-batch workload
(`opt-125m`, `128 DPU`, `limit=2`, `max_new_tokens=64`,
`decode_continuous_batch_max_size=2`) showed that this was not enough:

- previous grouped-QK full path: `qkfull_grouped_full`
  - `avg_latency = 114.0265 s`
  - `avg_ttft = 0.4031 s`
  - `avg_tpot = 1.8035 s`
  - `avg_throughput = 0.5613 tok/s`
  - `attention_decode_compute_s = 42.6225 s`
  - `qk_rounds_total = 3626`
  - `qk_fallback_rounds = 593`
  - `qk_active_ranks_total = 3664`
- mixed-window grouped-QK full path: `qkfull_grouped_full_v2`
  - `avg_latency = 116.6117 s`
  - `avg_ttft = 0.4112 s`
  - `avg_tpot = 1.8445 s`
  - `avg_throughput = 0.5488 tok/s`
  - `attention_decode_compute_s = 44.0752 s`
  - `qk_rounds_total = 3767`
  - `qk_fallback_rounds = 723`
  - `qk_active_ranks_total = 3816`

This gives a fairly strong practical conclusion:

- helper-side grouped QK alone is not enough to make full-QK decode competitive
- widening QK round eligibility without stronger placement/packing control can
  actually increase fallback pressure and active-rank traffic
- the current throughput-optimal operating point remains:
  - AV-only resident path
  - `qk_full` disabled for throughput runs

So the next useful QK work should probably not be more small helper-only round
relaxations. The more promising directions are:

- request-/step-level packing that keeps same-wave QK items closer in window and
  placement before they reach the helper
- or a larger protocol change such as more fused QK-to-AV execution that avoids
  materializing so many standalone raw-score launches

### 2026-05-05 Scheduler Packing Follow-up

Based on the failed mixed-window QK helper pass, the next low-risk step moves
up one layer in the stack:

- keep helper protocol unchanged
- improve which requests are paired into the same decode wave

The current scheduler follow-up changes decode wave selection from pure FIFO to
an `oldest-first + best-fit` policy:

- preserve fairness by always seeding the wave with the oldest pending request
- fill the remaining wave slots using attention-side packing hints captured at
  `init_request`
- the best-fit score currently prefers:
  - same helper-rank requests
  - overlapping preferred DPU stripes
  - smaller context-length gaps
  - similar stripe widths

The attention node now returns a small packing hint to the scheduler:

- `rank_index`
- `preferred_dpu_stripe`
- `stripe_width`

Scheduler metrics now also expose whether this policy is actually doing useful
work:

- `reordered_flushes`
- `same_rank_flushes`
- `mixed_rank_flushes`
- `avg_context_span`

This is intended to answer a very specific question in the next validation
round:

- can we reduce QK active-rank pressure and fallback rounds by pairing more
  compatible requests before they hit the helper?

Formal validation on the comparable AV-only workload
(`opt-125m`, `128 DPU`, `limit=2`, `max_new_tokens=64`,
`decode_continuous_batch_max_size=2`) showed a useful boundary:

- new artifact: `groupedav_schedpack_full`
  - `avg_latency = 64.9371 s`
  - `avg_ttft = 0.9781 s`
  - `avg_tpot = 1.0152 s`
  - `avg_throughput = 0.9900 tok/s`
  - scheduler:
    - `reordered_flushes = 0`
    - `same_rank_flushes = 0`
    - `mixed_rank_flushes = 62`
    - `avg_context_span = 2.90625`
  - helper:
    - `av_rounds_total = 1512`
    - `av_batched_rounds = 1512`
    - `av_fallback_rounds = 0`
    - `av_active_ranks_total = 3000`
    - `av_max_active_ranks = 2`

Compared with the previous best AV-only baseline
`groupedav_batchedpad_full`:

- old best AV-only:
  - `avg_ttft = 0.3834 s`
  - `avg_tpot = 1.0065 s`
  - `avg_throughput = 1.0033 tok/s`
- scheduler-packed AV-only:
  - `avg_ttft = 0.9781 s`
  - `avg_tpot = 1.0152 s`
  - `avg_throughput = 0.9900 tok/s`

So dequeue-time batch packing by itself was not enough in the real `limit=2`
case:

- once the two requests were initially placed onto different ranks, the decode
  queue had almost no freedom left to create same-rank waves
- the data confirms this directly because all observed multi-request flushes
  stayed mixed-rank

This points the next optimization one step earlier in the lifecycle:

- coordinate request placement / initial stripe selection so concurrently active
  requests are more likely to start on the same rank
- preserve that locality during later stripe growth instead of recomputing a
  fresh rank from `request_id` hash

### 2026-05-05 Placement-Coordinated Follow-up

The next follow-up therefore moves from decode dequeue policy to attention
request initialization:

- initial `preferred_dpu_stripe` selection now tries to reuse an already-active
  helper rank when one exists
- ties are still broken conservatively using rank live load and stripe-width
  fit, so this does not completely ignore allocator pressure
- when no active rank exists yet, the code falls back to the previous
  rank-local hash-based stripe choice
- later stripe expansion now stays inside the request's current rank instead of
  re-deriving a potentially different rank from `request_id` hash

Backend observability was also extended so the next run can confirm whether
placement steering actually activated:

- `init_rank_locality_reuse_count`
- `init_rank_hash_fallback_count`
- `init_rank_last_reason`

Formal validation on the same comparable workload
(`opt-125m`, `128 DPU`, `limit=2`, `max_new_tokens=64`,
`decode_continuous_batch_max_size=2`) confirmed that the placement steering was
really active after syncing the updated attention backend onto the attention
worker:

- artifact: `groupedav_placepack_full_v2`
  - `avg_latency = 66.1157 s`
  - `avg_ttft = 0.9162 s`
  - `avg_tpot = 1.0349 s`
  - `avg_throughput = 0.97 tok/s`
  - scheduler:
    - `same_rank_flushes = 62`
    - `mixed_rank_flushes = 0`
  - backend:
    - `init_rank_locality_reuse_count = 1`
    - `init_rank_hash_fallback_count = 1`
    - `init_rank_last_reason = active_rank_reuse`
  - helper:
    - `av_active_ranks_total = 1512`
    - `av_max_active_ranks = 1`

This is an important directional result:

- the new placement policy did exactly what it was supposed to do for helper
  locality
- compared with the earlier mixed-rank runs, active-rank traffic was cut in
  half and decode waves became consistently same-rank

But the throughput result also shows the next problem very clearly:

- locality improved
- yet end-to-end throughput still regressed versus the previous AV-only best
  baseline (`~1.00 tok/s`)

So the next optimization should not be “more same-rank at any cost”. The better
next target is likely:

- keep same-rank placement as a constraint
- but choose a smoother intra-rank stripe anchor so the two active requests do
  not over-concentrate on awkward DPU subsets or wraparound stripes
- then re-measure whether we can keep `av_max_active_ranks = 1` without paying
  the extra TTFT / TPOT penalty

That follow-up was then implemented by tightening the intra-rank stripe choice:

- avoid wraparound windows inside a rank
- prefer a contiguous rank-local window
- avoid overlap with already-active same-rank request stripes when possible
- require stripe expansion windows to contain the old stripe instead of shifting
  it elsewhere

Formal validation on the same workload:

- artifact: `groupedav_placepack_full_v3`
  - `avg_latency = 63.8342 s`
  - `avg_ttft = 0.8226 s`
  - `avg_tpot = 1.0002 s`
  - `avg_throughput = 1.0026 tok/s`
  - scheduler:
    - `same_rank_flushes = 62`
    - `mixed_rank_flushes = 0`
  - helper:
    - `av_active_ranks_total = 1512`
    - `av_max_active_ranks = 1`
  - backend:
    - `init_rank_locality_reuse_count = 1`
    - `init_rank_hash_fallback_count = 1`

This is the first placement-aware result that preserves the locality win without
giving back throughput:

- compared with `placepack_full_v2`, the same-rank property stayed intact
- `av_active_ranks_total` stayed at the reduced single-rank level
- throughput recovered from `0.97 tok/s` back to essentially the old best AV
  baseline

The remaining gap versus the old best baseline is now mostly TTFT:

- old best AV-only: `avg_ttft = 0.3834 s`
- placement-aware v3: `avg_ttft = 0.8226 s`

So the next step should likely focus on admission / prefill-side overlap or
attention init cost rather than helper-rank locality itself, because the decode
steady-state throughput is now roughly back where we want it.

### 2026-05-05 TTFT Fix And Placement Revalidation

The high `avg_ttft` reported by `groupedav_placepack_full_v3` turned out not to
be a real placement penalty, but a scheduler accounting bug:

- `first_token_time` had been captured after `attention.init_request`
- this incorrectly charged attention-side KV init time into TTFT

The scheduler was then fixed so `first_token_time` is recorded immediately
after `prefill.process_prompt` returns the first token.

With only the TTFT accounting fix applied, the next comparable run was:

- artifact: `groupedav_placepack_full_v4`
  - `avg_latency = 64.1200 s`
  - `avg_ttft = 0.3930 s`
  - `avg_tpot = 1.0253 s`
  - `avg_throughput = 0.9848 tok/s`
  - scheduler:
    - `same_rank_flushes = 62`
    - `mixed_rank_flushes = 0`
  - backend:
    - `init_rank_locality_reuse_count = 1`
    - `init_rank_hash_fallback_count = 1`
    - `init_rank_last_reason = active_rank_reuse`

This clarified the TTFT story:

- the placement-aware path was no longer materially worse than the old best AV
  baseline on TTFT
- the earlier `v3` TTFT gap was mostly measurement error

One later experiment then tried to make active-rank reuse more conservative by
requiring a larger active cohort before reusing a rank. That run was:

- artifact: `groupedav_placepack_full_v5`
  - `avg_latency = 66.2960 s`
  - `avg_ttft = 0.3988 s`
  - `avg_tpot = 1.0566 s`
  - `avg_throughput = 0.9558 tok/s`
  - scheduler:
    - `same_rank_flushes = 0`
    - `mixed_rank_flushes = 62`
  - backend:
    - `init_rank_locality_reuse_count = 0`
    - `init_rank_hash_fallback_count = 2`
    - `init_rank_last_reason = insufficient_active_cohort`

That result established a second important point:

- the conservative active-cohort gate was harmful
- once locality reuse was blocked, the system fell back to the old mixed-rank
  helper pattern and lost throughput again

There was also an operational wrinkle during this validation:

- the harmful `insufficient_active_cohort` gate had already been reverted
  locally
- but `192.168.123.7` still had the stale attention backend
- so the attention worker had to be explicitly re-synced before trusting any
  further locality results

After re-syncing `src/core/attention_backend.py` onto `192.168.123.7`, the same
comparable workload was rerun twice:

- model: `opt-125m`
- topology:
  - `192.168.123.4` prefill GPU / benchmark launcher
  - `192.168.123.3` decode dense GPU
  - `192.168.123.7` attention / UPMEM
- config:
  - `128 DPU`
  - `limit=2`
  - `max_new_tokens=64`
  - `decode_continuous_batch_max_size=2`
  - `resident_store_backend=upmem_kvslot`
  - `dpu_placement_policy=load_aware`
  - `qk_full=false`

Repeated validation results:

- artifact: `groupedav_placepack_full_v6`
  - `avg_latency = 60.6732 s`
  - `avg_ttft = 0.4115 s`
  - `avg_tpot = 0.9565 s`
  - `avg_throughput = 1.0549 tok/s`
  - scheduler:
    - `same_rank_flushes = 62`
    - `mixed_rank_flushes = 0`
  - helper:
    - `av_rounds_total = 1536`
    - `av_batched_rounds = 1536`
    - `av_fallback_rounds = 0`
    - `av_active_ranks_total = 1536`
    - `av_max_active_ranks = 1`
  - backend:
    - `init_rank_locality_reuse_count = 1`
    - `init_rank_hash_fallback_count = 1`
    - `init_rank_last_reason = active_rank_reuse`
  - resident store:
    - `active_dpus = 8`
    - `active_rank_count = 1`

- artifact: `groupedav_placepack_full_v7`
  - `avg_latency = 61.3000 s`
  - `avg_ttft = 0.4040 s`
  - `avg_tpot = 0.9666 s`
  - `avg_throughput = 1.0441 tok/s`
  - scheduler:
    - `same_rank_flushes = 62`
    - `mixed_rank_flushes = 0`
  - helper:
    - `av_rounds_total = 1536`
    - `av_batched_rounds = 1536`
    - `av_fallback_rounds = 0`
    - `av_active_ranks_total = 1536`
    - `av_max_active_ranks = 1`
  - backend:
    - `init_rank_locality_reuse_count = 1`
    - `init_rank_hash_fallback_count = 1`
    - `init_rank_last_reason = active_rank_reuse`
  - resident store:
    - `active_dpus = 8`
    - `active_rank_count = 1`

These repeated runs give the current best practical conclusion:

- once the stale remote gate is removed, placement-aware same-rank reuse is not
  just a locality win, but also a throughput win
- throughput improved beyond the old AV-only best baseline:
  - old best AV-only `groupedav_batchedpad_full`: `1.0033 tok/s`
  - post-fix placement-aware reruns: `1.0549 tok/s` and `1.0441 tok/s`
- TTFT stayed close to the old baseline range (`0.3834 s` old best vs
  `0.4115 s` / `0.4040 s` repeated reruns)
- helper behavior is now cleaner and more stable:
  - single-rank AV rounds
  - zero AV fallback rounds
  - all active decode waves remained same-rank

So the current optimization frontier is no longer “recover locality without
losing throughput”. That has now been achieved on the real cluster. The next
useful step should instead focus on increasing useful DPU parallelism and
dynamic KV placement quality while preserving this single-rank helper locality.

### 2026-05-06 Same-Rank DPU Breadth Follow-up

After the locality-aware path was stable again, the next question was whether
we could keep the single-rank helper behavior but activate more DPUs inside that
rank.

The first concrete issue was that the previous placement-aware path still left
some DPU bandwidth unused:

- the request stripe had already expanded to width `12`
- but because the request started at width `8` and only widened later, the
  base resident groups were still biased toward the initial narrower subset
- on the real run, only `8` DPUs ended up active even though the final allowed
  stripe had width `12`

The first fix moved that stripe growth earlier:

- for small-group requests (`max_layer_groups <= 4`) whose prompt already
  exceeds the base resident length, initialize directly at width `12`
  instead of starting at `8` and widening later

Validation on the same comparable workload:

- artifact: `groupedav_placepack_full_v9`
  - `avg_latency = 61.0777 s`
  - `avg_ttft = 0.4669 s`
  - `avg_tpot = 0.9621 s`
  - `avg_throughput = 1.0479 tok/s`
  - scheduler:
    - `same_rank_flushes = 62`
    - `mixed_rank_flushes = 0`
  - helper:
    - `av_active_ranks_total = 1536`
    - `av_max_active_ranks = 1`
    - `av_batched_launch_ns = 22561867814`
  - resident store:
    - `active_dpus = 12`
    - `max_usage_ratio = 0.28125`
  - backend:
    - `stripe_width = 12`
    - `stripe_expand_count = 0`
    - `last_stripe_update_reason = init`

This established that earlier stripe widening was real and useful:

- active DPU count improved from `8` to `12`
- helper launch time improved
- throughput stayed above the previous `v6` / `v7` locality reruns

The next experiment then widened the same-rank init stripe further for these
small-group requests:

- if the prompt already overshoots the base resident length by a small margin,
  start directly at width `16`

Validation result:

- artifact: `groupedav_placepack_full_v10`
  - `avg_latency = 59.4433 s`
  - `avg_ttft = 0.4963 s`
  - `avg_tpot = 0.9357 s`
  - `avg_throughput = 1.0767 tok/s`
  - scheduler:
    - `same_rank_flushes = 62`
    - `mixed_rank_flushes = 0`
  - helper:
    - `av_active_ranks_total = 1536`
    - `av_max_active_ranks = 1`
    - `av_batched_launch_ns = 20317241862`
  - resident store:
    - `active_dpus = 15`
    - `active_slot_counts = [2, 4, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 4, 2]`
    - `max_usage_ratio = 0.28125`
  - backend:
    - `stripe_width = 16`
    - `stripe_expand_count = 0`
    - `last_stripe_update_reason = init`

This is the current best throughput result on the comparable AV-only resident
path:

- old best AV-only baseline `groupedav_batchedpad_full`:
  - `1.0033 tok/s`
- placement-aware same-rank reruns after remote re-sync:
  - `v6 = 1.0549 tok/s`
  - `v7 = 1.0441 tok/s`
- early width-12 init:
  - `v9 = 1.0479 tok/s`
- early width-16 init:
  - `v10 = 1.0767 tok/s`

Current practical takeaway:

- preserving single-rank AV locality is still the right constraint
- but we do not need to stay artificially narrow inside that rank
- for the current `opt-125m` comparable workload, widening the same-rank stripe
  to `16` improves useful DPU breadth and produces the best throughput so far
- the remaining tradeoff to watch is TTFT, which rose modestly on `v10` even
  while TPOT and end-to-end throughput improved
