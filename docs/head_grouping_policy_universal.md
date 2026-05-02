# Universal Head Grouping Policy

This repository supports multiple attention backends:

- `pim_naive` (baseline)
- `cloverinfer` (CloverInfer system path)

Both can use the same `pim_head_grouping_policy` knob to control how a layer's
heads are partitioned into resident KV groups.

## Why "universal"

Different model families (OPT, Qwen, Llama) vary in:

- `num_heads`
- `head_dim`
- typical `seq_len` during decode

A grouping strategy that is tuned to a single model shape can become brittle
when any of those change. The goal of the `universal` policy is:

- behave reasonably across a wide range of shapes without per-model tuning
- keep groups wide at short context (avoid tiny groups and overhead)
- split more as per-head KV work grows (increase parallelism when it matters)

## How It Works (High Level)

For a layer with KV cache shaped `[seq_len, num_heads, head_dim]`, the policy:

1. Computes `per_head_live_elems = seq_len * head_dim`.
2. Picks a `target_heads_per_group` tier based on `per_head_live_elems`.
3. Converts that into a group count:
   `ceil(num_heads / target_heads_per_group)`.
4. Clamps the result by:
   - hardware limit (`<= num_dpus`)
   - per-group shape limit (`heads_per_group <= 32`)

This keeps the policy stable across OPT/Qwen/Llama-ish shapes while still
reacting to larger contexts.

## Usage

Set the cluster config (or CLI flag in test scripts) to:

- `pim_head_grouping_policy = universal`

Example:

```bash
python tests/trace_pim_allocator.py \
  --pim-head-grouping-policy universal \
  --pim-dpu-placement-policy rotated
```

