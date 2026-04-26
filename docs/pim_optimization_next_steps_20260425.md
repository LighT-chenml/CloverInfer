# PIM Optimization Next Steps

Date: 2026-04-25

## Purpose

This document records the near-term optimization direction after the current
resident-KV baselines.

The goal here is not paper writing. It is to keep the implementation work
focused on the most promising technical paths.

## Current State

We now have three useful baselines:

1. `4 DPU + fp32 resident KV`
   - capacity bottleneck baseline
   - overflows under `concurrency = 4`

2. `4 DPU + fp16 resident KV`
   - compression baseline
   - recovers the full-DPU path

3. `8 DPU + fp32 + balanced + rotated`
   - scale-out / placement baseline
   - also recovers the full-DPU path

Key observation:

- in the constrained `4 DPU` setting, reducing resident-KV footprint is more
  effective than placement tuning

## Mainline Priorities

### 1. Stress the current best baselines harder

Before adding more mechanisms, test whether the present conclusions still hold
under heavier workloads:

- longer prompts
- larger `max_new_tokens`
- higher concurrency
- real models, starting with `Qwen-1_8B`

This should answer:

- does `4 DPU + fp16` remain stable under longer decode?
- when is `8 DPU + fp32` still insufficient?
- when do we need both scale-out and resident-KV compression?

Current answer from the first heavier real-model trace:

- `Qwen-1_8B`, `max_new_tokens = 8`, `concurrency = 2`
- `4 DPU + fp16` still overflows
- `8 DPU + fp32` still overflows
- `8 DPU + fp16` recovers the full-resident path

### 2. Push real PIM-side attention computation

The most important architectural gap is still:

- KV is resident on the PIM side
- but attention computation is not yet fully staying there

Recommended order:

1. move `QK` computation further onto the PIM side
2. then implement `AV`
3. reduce host-side materialization as much as possible

The real target is not only resident KV, but also reducing return traffic from
the attention node.

### 3. Keep resident-KV compression as a first-class optimization axis

The successful `fp16` result means resident representation should remain a core
optimization dimension.

Likely next sub-directions:

- validate `fp16` under heavier workloads
- measure numerical stability on larger models
- later consider more aggressive compression:
  - `int8 KV`
  - mixed-precision KV
  - asymmetric K/V formats

## Secondary Priorities

### 4. Revisit allocator / reserve policy later

At the current lightweight workload, reserve policy is not the main bottleneck.
It matters more once:

- generation length grows
- per-request lifetime becomes longer
- append-heavy workloads dominate

This should stay on the list, but behind resident-KV footprint reduction and
actual PIM-side compute.

### 5. Improve batching / pipeline overlap

After the core backend path is more mature, worthwhile system optimizations
include:

- continuous batching
- overlapping dense and attention stages better
- reducing RPC overhead

These are important, but they are not the next bottleneck to solve first.

## Sparse Attention

Sparse attention is worth considering for a PIM-based design, but not all forms
of sparsity are equally suitable.

### More promising for PIM

- sliding-window attention
- block-sparse attention
- local + global structured sparsity

Why:

- memory access is more regular
- block mapping is easier
- metadata overhead is lower
- placement and capacity accounting stay manageable

### Less suitable as the first sparse target

- top-k token sparse patterns
- query-dependent irregular sparse attention
- runtime-dynamic sparse connectivity

Why:

- index management becomes heavy
- memory access becomes irregular
- DPU balance becomes harder
- control overhead can erase theoretical savings

### Recommendation

If sparse attention is explored, start with:

- block-sparse
- or sliding-window

Do not start with highly irregular sparse schemes.

## Grouped-Query Attention

GQA should absolutely be considered, but it changes the PIM story.

### Why it matters

GQA reduces:

- KV-head count
- KV-cache footprint
- bandwidth pressure

### What that means for PIM

Positive:

- PIM attention can still help with disaggregated KV ownership
- long context and concurrency can still create attention-node pressure

Negative:

- one of PIM's clearest advantages is large KV capacity / bandwidth pressure
- GQA directly weakens that advantage
- relative benefit versus strong GPU baselines may shrink

### Practical implication

GQA is best treated as:

- a more realistic and more difficult evaluation setting

not as the first architecture-driving assumption.

### Recommendation

- include GQA models in evaluation soon
- but do not redesign the whole system around GQA first
- first establish the dense-attention and resident-KV optimization story

## Suggested Execution Order

Recommended next sequence:

1. run the current best baselines on heavier workloads
   - especially `8 DPU + fp16 + balanced + rotated`
   - and compare against `4 DPU + fp16`
   - and `8 DPU + fp32 + balanced + rotated`

2. switch to a more realistic model
   - use `Qwen-1_8B` first on the current `V100-16GB` cluster
   - keep `Qwen-7B` as a later target after reducing dense-side memory pressure

3. continue moving `QK` onto the PIM side

4. move `AV` onto the PIM side

5. only after the dense path is mature:
   - add structured sparse-attention exploration

6. keep GQA as a realistic evaluation branch throughout

## Bottom Line

The current mainline should remain:

- resident-KV footprint reduction
- real PIM-side attention compute
- heavier and more realistic workloads

Sparse attention and GQA are both important, but they should be treated as
controlled next-stage branches rather than replacing the current mainline.

## Scheme 2 Follow-Up

The shared-owner refactor has now reached a useful checkpoint:

- resident KV and QK-mixed traffic share the same `upmem_kvslot` helper
- startup no longer runs a separate `upmem_dot` allocation smoke in this mode
- the current real-model trace is stable at both `512 DPU` and `1020 DPU`
- QK has now moved from CPU-inside-helper to a real DPU launch path
- AV has now moved to a first real DPU-side resident-`V` path with standalone
  correctness checks passing at `fp32` and `fp16`

The next actions should separate system issues from algorithm issues.

### Immediate system follow-up

- always check for stale helper processes before concluding that `.7` cannot
  allocate a requested DPU count
- especially look for leftover `host_qk --stdio` or `host_kvslot --stdio`
  processes from previous experiments
- note that the resident-KV path now also auto-closes its own `host_kvslot`
  helper when the last live DPU slot is freed

### Immediate backend follow-up

- keep Scheme 2 as the mainline
- do not reintroduce dual-owner `host_qk` allocation for large-DPU runs
- add clearer diagnostics when helper startup fails because a previous helper is
  still occupying DPUs
- treat the current DPU-QK path as the new functional baseline, even though it
  is not yet performance-optimized

### Immediate optimization takeaway

- the attempted `QK active16` launch cap did not help at either `512 DPU` or
  `1020 DPU`
- treat it as a negative result and do not keep iterating on that specific
  idea first

### Next compute step

- validate the new resident-AV path in end-to-end cluster runs
- then optimize the combined DPU-side `QK + AV` path
- priority candidates:
  - reduce per-call host/DPU copy overhead
  - batch more work into one launch
  - reduce or eliminate host-side key materialization on the mixed-QK path

### Early end-to-end result after AV integration

The first small end-to-end comparison now says:

- the resident-AV path is functionally active
  - host-side `resident_materialize_ops` dropped to `0`
  - resident `AV` calls replaced them in decode
- but the naive `upmem_kvslot + resident AV` path is still much slower than
  the host-store path on small `OPT-125M` traces

Current interpretation:

- the next bottleneck is no longer “can we route AV through DPU?”
- the next bottleneck is “how do we make DPU-side AV amortize copy and launch
  costs enough to win?”

So the optimization order should now bias toward:

1. reducing AV input/output traffic per decode step
2. fusing or batching more work per launch
3. reducing the residual host-side work around mixed-QK

### AV batching checkpoint

We have now tried the first `AV` batching step:

- one Python/helper `AV_BATCH` call per layer
- but helper-internal execution is still one `AV` launch per item

Result:

- end-to-end latency on the small `OPT-125M` trace was essentially unchanged
- this is an expected but now confirmed negative result

Immediate implication:

- do not spend more time on protocol-only batching
- move the optimization focus into `host_kvslot` itself

Updated near-term order:

1. batch `AV` work inside the helper by DPU and launch asynchronously when
   possible
2. after that, revisit the mixed `QK` path and reduce resident-key
   materialization / host-side assembly
3. then decide whether a deeper DPU-kernel refactor is needed for multi-slot
   `AV` or resident-key-native `QK`

### AV helper-side async checkpoint

We also completed the next `AV` batching step:

- inside `host_kvslot`, one batch is now partitioned into rounds
- each round launches at most one `AV` item per DPU asynchronously
- responses are written back in original order after all rounds finish

Result:

- correctness remained stable
- small-trace latency still did not improve
- this is another useful negative result

Immediate implication:

- stop spending time on finer-grained `AV` host/helper scheduling for now
- the remaining bottleneck is likely more structural than orchestration-only

Updated near-term order:

1. reduce mixed-`QK` host-side resident-key assembly / materialization
2. then re-measure whether `AV` becomes relatively more visible
3. only if `AV` is still dominant, consider a deeper multi-slot kernel design
   rather than more helper-side scheduling tweaks

### Resident-slot mixed-QK checkpoint

We have now completed the first resident-slot mixed-`QK` baseline:

- mixed-`QK` now reads resident `K` directly from `upmem_kvslot`
- it no longer repacks a CPU-materialized key window for the helper
- minimal correctness check passed with `qk_mixed_last_max_abs_diff = 0.0`

Small-trace result:

- end-to-end latency became worse, not better

Immediate implication:

- the direction is still correct architecturally
- but the current implementation is too naive to win yet

Updated near-term order:

1. batch or fuse resident-slot mixed-`QK` across heads / slots to reduce
   per-head kernel overhead
2. after that, re-measure whether mixed-`QK` or resident-`AV` is the larger
   remaining bottleneck
3. only then decide whether the next deeper refactor should target:
   - multi-head resident-slot `QK`
   - or multi-slot `AV`

### Grouped resident-slot mixed-QK checkpoint

We have now completed the next step on that path:

- mixed heads are grouped by resident slot before sending work to
  `upmem_kvslot`
- one slot item can now carry multiple local heads / queries
- this reduces some per-head protocol and helper setup overhead

Small-trace result:

- grouped resident-slot mixed-`QK` is slightly better than the first
  ungrouped resident-slot baseline
- but it is still slower than the earlier resident-AV baseline without this
  resident-slot mixed-`QK` path

Immediate implication:

- grouping by slot helps, so this line is worth continuing
- but the remaining bottleneck is now clearly inside helper/kernel execution
  rather than only in Python-side packing

Updated near-term order:

1. implement true multi-head resident-slot `QK` inside one helper/kernel item
   instead of still launching one kernel per head
2. after that, consider overlapping multiple slot groups / DPUs in the helper
3. only then re-evaluate whether the next deeper target should return to
   resident-`AV`

### True multi-head resident-slot QK checkpoint

We have now completed that next step:

- one grouped slot item executes one real multi-head resident-slot `QK` kernel
- the grouped path no longer falls back to one kernel launch per head
- minimal three-machine correctness still passed with exact mixed-`QK`
  agreement

Small-trace result:

- this version is materially faster than both:
  - the earlier grouped resident-slot mixed-`QK` baseline
  - the original resident-AV-only baseline on the same workload

Immediate implication:

- the mainline should stay on the mixed-`QK` optimization path for now
- we now have direct evidence that fixing execution granularity inside
  `host_kvslot` / DPU kernels can beat the earlier resident-AV baseline
- protocol-only batching should remain de-prioritized relative to
  helper/kernel-structure changes

Updated near-term order:

1. overlap multiple grouped slot items across DPUs inside `host_kvslot`
   without undoing the new multi-head kernel granularity
2. re-measure whether mixed-`QK` is still the dominant remaining bottleneck or
   whether resident-`AV` becomes the next limiter
3. only after that, choose between:
   - deeper resident-`AV` kernel fusion
   - or further mixed-`QK` tiling / WRAM-staging improvements

### Helper-side cross-DPU overlap checkpoint

We have now tried the next obvious helper-side step on top of the true
multi-head resident-slot `QK` kernel:

- preload all grouped slot items for one `QK_SLOT_BATCH`
- launch at most one grouped item per physical DPU asynchronously in each
  round
- read results back and return them in original order

Small-trace result:

- end-to-end latency stayed essentially unchanged relative to the immediately
  previous multi-head-kernel baseline

Immediate implication:

- after fixing the per-head kernel granularity problem, the next bottleneck is
  likely no longer plain host/helper slot scheduling
- do not spend more time on finer-grained cross-slot helper overlap for now

Updated near-term order:

1. optimize the multi-head slot kernel itself:
   - reduce repeated MRAM reads for query rows
   - stage query/head metadata into WRAM once per grouped item
   - revisit tasklet work partitioning across `(head, token)` tiles
2. then re-measure whether mixed-`QK` still dominates relative to resident-`AV`
3. only after that, decide whether to:
   - continue kernel-level mixed-`QK` optimization
   - or switch the mainline back to deeper resident-`AV` fusion

### Per-row query WRAM staging checkpoint

We have now completed the first kernel-internal staging step:

- one grouped slot-`QK` head row stages its query row into WRAM once
- the token loop then reuses that WRAM row while reading only resident `K`
- correctness remained exact in the current mixed-`QK` check path

Small-trace result:

- this improved both latency and TPOT materially over the previous true
  multi-head slot-kernel baseline

Immediate implication:

- the current mainline should continue inside the slot-`QK` kernel rather than
  going back to helper scheduling
- repeated MRAM traffic on the query side was a real bottleneck, not just a
  micro-optimization target

Updated near-term order:

1. continue kernel-level mixed-`QK` optimization:
   - revisit tasklet work partitioning across `(head, token)` tiles
   - reduce repeated `K` unpack / scalar-read overhead
   - explore small tiled staging for `K` where WRAM permits
2. then re-measure whether resident-`AV` becomes the next dominant limiter
3. only after that, choose whether the next engineering focus should be:
   - deeper mixed-`QK` kernel refactoring
   - or resident-`AV` kernel fusion / bandwidth reduction

### Correctness reset after grouped multi-head readback bug

We later found that the earlier fastest grouped multi-head numbers were not a
safe basis for further optimization.

Root cause:

- grouped multi-head DPU `QK` wrote score rows with one layout
- helper-side readback interpreted the same MRAM region with a different
  compact layout
- real decode could therefore produce corrupted later-head score rows,
  including `NaN`

Practical consequence:

- all optimization decisions after this point should use only post-fix
  measurements
- the first corrected row-stage reruns were around `1.91 s` to `1.94 s`, not
  `~1.20 s`

### Token-parallel compact-score checkpoint

We then replaced the slower correct row-stage kernel with a token-parallel
compact-score variant.

What changed:

- each tasklet owns different token positions within the decode window
- each tasklet computes the full dot product for those tokens
- grouped scores remain in the compact layout that matches helper readback
- per-token barrier/reduction overhead is removed

Checkpoint result:

- exact mixed-`QK` verification remained intact
- small-trace performance improved to about:
  - `avg_latency = 1.8203 s`
  - `avg_tpot = 1.6384 s`

Immediate implication:

- tasklet work partitioning is now a stronger lever than additional helper
  overlap
- the next kernel work should focus on `K`-side read efficiency rather than
  reworking the response protocol again

Updated near-term order:

1. reduce `read_k_value()` overhead inside the grouped mixed-`QK` kernel
2. explore small tiled / packed `K` reads that preserve the compact score
   layout
3. re-measure whether resident `AV` or mixed-`QK` is now the dominant limiter

### First K-read optimization checkpoint

We have now completed the first small `K`-side optimization pass inside the
token-parallel grouped mixed-`QK` kernel.

Result:

- a conservative fp32 pair-read fast path gave a small positive result
- a larger four-float read attempt regressed

Immediate implication:

- the right next step is not simply increasing contiguous read size
- the better direction is likely a more structured `K` tiling / staging design
  that matches tasklet ownership and MRAM access order more carefully

Updated near-term order:

1. keep the fp32 pair-read fast path as the current kernel baseline
2. design a small tiled `K`-read / staging experiment
3. measure whether that reduces MRAM access overhead more effectively than
   larger direct reads
4. only after that, re-evaluate whether the next main bottleneck has shifted
   toward resident `AV`

### Small K-tile staging checkpoint

We have now completed that next tiled-read experiment.

Result:

- reading a small aligned `fp32` `K` tile directly into a WRAM `float` buffer
  improved the small-trace baseline more clearly than the earlier pair-read
  micro-optimization

Immediate implication:

- structured small-tile staging is a better direction than simply increasing
  direct read block size
- the grouped mixed-`QK` mainline should now keep this `K`-tile variant as the
  active kernel baseline

Updated near-term order:

1. keep the small `fp32` `K`-tile staging kernel as the current baseline
2. refine tile shape / loop structure carefully:
   - tile size
   - tail handling
   - alignment-sensitive fast paths
3. then re-measure whether mixed-`QK` still dominates over resident `AV`

### Base-hoisting cleanup checkpoint

We also tried a more "engineering cleanup" style optimization on top of the
small `K`-tile kernel:

- hoist more token-base arithmetic
- hoist more fast-path checks

Result:

- correctness remained fine
- performance regressed relative to the simpler `ktile8float` kernel

Immediate implication:

- do not spend more time on pure index-hoisting cleanup right now
- keep focusing on memory-access structure and tile design instead

### K-tile size sweep checkpoint

We also checked whether growing the direct `fp32` staging tile beyond `8`
would keep helping.

Result:

- `16-float` staging regressed relative to the `8-float` tile baseline

Immediate implication:

- the next step should not be "try even larger tiles"
- the current local optimum appears to be a small structured tile rather than a
  maximal direct read block

Updated near-term order:

1. keep `ktile8float` as the current kernel baseline
2. stop spending time on larger direct tile sizes for now
3. move to a different structural direction:
   - different tasklet/tile ownership
   - or re-evaluating whether resident `AV` has become the more promising
     optimization target

### Attribution checkpoint: mixed-QK vs resident AV

We then explicitly disabled mixed-`QK` while keeping resident `AV` enabled.

Result:

- latency dropped materially when mixed-`QK` was removed
- but the remaining no-mixed-`QK` latency was still large enough that
  resident `AV` was clearly worth direct optimization

Immediate implication:

- grouped mixed-`QK` still matters
- but it is no longer the only sensible mainline
- it is now justified to optimize resident `AV` directly

### First resident-AV optimization checkpoint

We then implemented a same-head adjacent-dim fast path in the `AV` kernel.

Result:

- the optimization improved end-to-end latency with mixed-`QK` enabled
- it also improved latency when mixed-`QK` was disabled

Immediate implication:

- the `AV` gain is real and independent
- resident `AV` should remain an active optimization branch rather than only a
  secondary concern

Updated near-term order:

1. keep:
   - `ktile8float` for grouped mixed-`QK`
   - same-head adjacent-dim fast path for resident `AV`
2. continue exploring resident `AV` structure next
3. only then decide whether the next stronger lever is:
   - more `AV` kernel restructuring
   - or a return to grouped mixed-`QK`

### Resident-AV weight-tile checkpoint

We next pushed one step deeper into the resident-`AV` kernel.

What changed:

- for same-head adjacent-dim output pairs, stage `8` consecutive attention
  weights together in WRAM
- keep the earlier paired-`V` reuse path on top of that

Important reset:

- the first implementation of this idea produced faster numbers
- but those numbers were invalid because the staged weight path broke aligned
  packed-weight decoding
- the tell was `resident_av_shadow_max_abs_diff` jumping to about `2.03`

So the workflow lesson here is:

1. do not trust a faster `AV` result unless the resident-`AV` shadow diff stays
   near the earlier tolerance
2. keep the invalid first `avweighttile8` run only as a debugging note, not as
   an optimization result

We then fixed the bug by:

- reading aligned packed `uint64` weight pairs from MRAM
- unpacking the needed `fp32` weights in WRAM
- preserving the original logical index semantics

Corrected result:

- mixed-`QK` + aligned `avweighttile8`:
  - `avg_latency = 1.7764 s`
  - `avg_tpot = 1.5986 s`
- no mixed-`QK` + aligned `avweighttile8`:
  - `avg_latency = 1.6168 s`
  - `avg_tpot = 1.4386 s`
- resident-`AV` shadow diff returned to the earlier small-tolerance regime

Immediate implication:

- the underlying direction is still valid after the bug fix
- staged weight reuse is a real resident-`AV` lever, not just a measurement
  artifact
- but every deeper resident-`AV` kernel change now needs explicit shadow-diff
  scrutiny because silent layout bugs are easy to introduce

Updated near-term order:

1. keep:
   - grouped mixed-`QK` `ktile8float`
   - resident-`AV` pair-dim fast path
   - aligned resident-`AV` `weight_tile[8]`
2. continue with the next resident-`AV` structural step:
   - reduce repeated `V`-side MRAM traffic further
   - or change tasklet/output ownership to improve reuse
3. only after that, decide whether the next stronger lever is:
   - more resident-`AV` kernel restructuring
   - or switching back to grouped mixed-`QK`

### Resident-AV pair-block checkpoint

We also tried a more aggressive resident-`AV` restructuring step:

- let one tasklet compute a small same-head block of output pairs together
- reuse one weight row across more output dims than the earlier pair-dim path

What we hoped:

- move from “reuse weight for 2 dims” toward “reuse weight for a small dim
  block”
- create a more natural next step toward deeper `V`-side reuse

Result:

- correctness still passed
- but small-trace performance regressed materially:
  - `avg_latency = 1.8621 s`

Immediate implication:

- this direction increases per-tasklet work too much in the current kernel
  structure
- the extra same-head reuse did not compensate for the lost granularity /
  parallel balance

Decision:

- do not keep this pair-block kernel in the active code path
- record it as a useful negative result and revert to the aligned
  `weight_tile[8]` baseline

Updated near-term order:

1. keep the current trustworthy resident-`AV` baseline:
   - pair-dim fast path
   - aligned `weight_tile[8]`
2. avoid “one tasklet owns many output pairs” style changes for now
3. instead, try lighter-weight resident-`AV` improvements:
   - token-side / `V`-side staging
   - or reduced per-token `V` unpack overhead without enlarging output blocks

### Resident-AV pair-block-2 checkpoint

We then tested a smaller version of that same idea.

What changed:

- instead of one tasklet owning `4` output pairs, let it own only `2`
- this allows one aligned `16B` `V` read per token for `4` consecutive dims
- the hope was to keep the extra reuse while avoiding the larger granularity
  penalty seen in the `pair-block-4` attempt

Result:

- correctness still passed
- mixed-`QK` small trace was essentially tied with the current best:
  - `avg_latency = 1.7754 s`
- but no-mixed-`QK` resident-`AV` attribution regressed:
  - `avg_latency = 1.6294 s`
  - compared with the aligned `weight_tile[8]` baseline:
    - `avg_latency = 1.6168 s`

Immediate implication:

- the tiny mixed-`QK` difference is too small to trust as a real win
- once mixed-`QK` is removed, the resident-`AV` path itself is slower
- so this is not a robust resident-`AV` improvement

Decision:

- do not keep `pair-block-2` in the active code path
- treat both `pair-block-4` and `pair-block-2` as evidence that enlarging
  output ownership is the wrong local direction for this kernel

Updated near-term order:

1. keep the aligned `weight_tile[8]` resident-`AV` baseline
2. stop exploring larger output-block ownership for now
3. focus next on lighter resident-`AV` ideas:
   - smaller `V` unpack / decode overhead
   - token-side staging that does not change output granularity

### Resident-AV float-pair read checkpoint

We also tried a very small micro-optimization inside the current same-head
pair path.

What changed:

- keep the aligned `weight_tile[8]` structure
- but replace the manual `uint64 -> two fp32` unpack with a direct
  `float[2]` DMA read for the aligned pair

What we wanted to test:

- whether the current bottleneck still contains a meaningful amount of
  per-token unpack overhead
- or whether almost all remaining cost is already in MRAM access / launch
  structure

Result:

- correctness still passed
- mixed-`QK` small trace improved slightly:
  - `avg_latency = 1.7663 s`
- but no-mixed-`QK` attribution regressed:
  - `avg_latency = 1.6303 s`
  - compared with the aligned `weight_tile[8]` baseline:
    - `avg_latency = 1.6168 s`

Immediate implication:

- this is not a robust resident-`AV` improvement
- direct float-pair DMA may help one mixed path slightly, but it does not
  improve the resident-`AV` kernel itself consistently

Decision:

- do not keep this micro-optimization in the active code path
- treat it as another sign that the remaining problem is more structural than
  unpack-only

Updated near-term order:

1. keep the aligned `weight_tile[8]` baseline
2. stop spending time on tiny `V` unpack micro-optimizations for now
3. move the next effort up one level:
   - host/helper amortization around resident `AV`
   - or a more structural kernel/data-layout change rather than local unpack

### Helper-side AV batch push-xfer checkpoint

We then tried the first real helper-side structural optimization for
resident-`AV`.

What changed:

- inside one `AV_BATCH` launch round, when:
  - there is at most one item per physical DPU
  - all items have matching padded weight/context sizes
  - and the total DPU count is small (`<= 16`)
- the helper now uses batched UPMEM xfers:
  - one set-wide push for `runtime_slot_args`
  - one set-wide push for `av_weights_bits`
  - one set-wide launch
  - one set-wide pull for `av_context_bits`
- instead of doing `copy_to + launch + copy_from` separately for each item

Result:

- mixed-`QK` small trace dropped to:
  - `avg_latency = 1.1833 s`
- no-mixed-`QK` small trace dropped to:
  - `avg_latency = 1.0241 s`

Immediate implication:

- this is the first resident-`AV` optimization in a while that is clearly
  larger than local measurement noise
- the main small-trace bottleneck was still helper-side transfer / launch
  granularity, not only kernel arithmetic
- moving the optimization one level up was the right decision

Important scope note:

- the fast path is deliberately gated to small DPU counts
- we should not blindly reuse this exact whole-set launch pattern for
  `512/1020 DPU` experiments without rethinking the active-set issue

Updated near-term order:

1. keep this helper-side AV batch push-xfer path as the active small-DPU
   resident-`AV` baseline
2. re-measure whether mixed-`QK` is now the dominant remaining limiter again
3. for larger-DPU regimes, design a variant that preserves the same batching
   idea without forcing a whole-machine launch for tiny rounds

### Mixed-QK helper-batching checkpoint

We tested the most obvious follow-up after the AV helper win:
apply the same whole-set batched helper pattern to `QK_SLOT_BATCH`.

Result:

- correctness stayed clean
- but the optimization did not improve the small `4 DPU` trace
- serial attribution gave:
  - mixed-`QK`: `1.2104 s`
  - no mixed-`QK`: `1.0835 s`
- compared with the previous trustworthy AV-batched baseline:
  - mixed-`QK`: `1.1653 s`
  - no mixed-`QK`: `1.0260 s`

Immediate implication:

- the helper-side mixed-`QK` path is not the same kind of bottleneck that
  resident `AV` was
- even though the implementation is structurally analogous, the cost center
  does not move in the same way

Updated direction:

1. keep the current helper-side mixed-`QK` batching code as an experiment
   checkpoint, not as the promoted baseline
2. stop prioritizing more "whole-set helper batching" variants for grouped
   mixed-`QK` in the small-DPU regime
3. shift the next effort back to mixed-`QK` structure itself:
   - reduce host-side packing/copy work
   - revisit grouped-query layout / row staging
   - or design an active-set launch variant that does not broadcast the whole
     runner set
