# Qwen Baseline Refresh After Stdio Helper

Date: 2026-04-24

## Setup

- Dataset: `dataset/humaneval.jsonl`
- Limit: first `5` samples
- Model: `/home/cml/CloverInfer/model/Qwen-1_8B`
- `max_new_tokens = 3`
- Output:
  - `artifacts/baseline_comparison_qwen_stdio_limit5_refresh.jsonl`

Command used:

```bash
/home/cml/anaconda3/envs/clover_infer/bin/python tests/benchmark_baselines.py \
  --address 192.168.123.4:26379 \
  --data dataset/humaneval.jsonl \
  --limit 5 \
  --model /home/cml/CloverInfer/model/Qwen-1_8B \
  --model-name qwen-1_8b \
  --max-new-tokens 3 \
  --baselines monolithic_gpu,split_gpu_full_decode,disagg_cpu,disagg_pim_naive \
  --output /home/cml/CloverInfer/artifacts/baseline_comparison_qwen_stdio_limit5_refresh.jsonl
```

## Results

| Baseline | Avg Latency (s) | Avg TTFT (s) | Avg Throughput (tok/s) |
| --- | ---: | ---: | ---: |
| `monolithic_gpu` | `0.292` | `0.177` | `15.664` |
| `split_gpu_full_decode` | `0.739` | `0.401` | `4.862` |
| `disagg_cpu` | `2.143` | `0.400` | `1.434` |
| `disagg_pim_naive` | `2.425` | `0.406` | `1.268` |

Derived ratios:

- `split_gpu_full_decode / monolithic_gpu = 2.54x`
- `disagg_cpu / split_gpu_full_decode = 2.90x`
- `disagg_pim_naive / disagg_cpu = 1.13x`
- `disagg_pim_naive / monolithic_gpu = 8.32x`

## Comparison Against Earlier Baseline

Earlier reference:

- `artifacts/baseline_comparison_qwen_v2.jsonl`

Key change:

| Baseline | Earlier Avg Latency (s) | Refreshed Avg Latency (s) |
| --- | ---: | ---: |
| `monolithic_gpu` | `0.242` | `0.292` |
| `split_gpu_full_decode` | `0.841` | `0.739` |
| `disagg_cpu` | `2.224` | `2.143` |
| `disagg_pim_naive` | `5.764` | `2.425` |

Main improvement:

- `disagg_pim_naive` improved from `5.764s` to `2.425s`
- this is about `2.38x` lower latency than the original naive PIM baseline

## Interpretation

This refreshed baseline matrix changes the system story in an important way:

1. The remote-attention baseline gap remains real.
2. But the naive PIM path is no longer dramatically worse than remote CPU
   attention.
3. The current helper-based naive PIM backend is now close enough to `disagg_cpu`
   that the next bottlenecks are worth analyzing as system-design questions
   rather than obvious implementation bugs.

Paper-usable phrasing:

- "After removing major host orchestration overheads, the naive PIM attention
  baseline narrows to within about `1.13x` of the remote CPU attention baseline
  on the refreshed five-sample Qwen-1.8B comparison."

## What This Means For Next Steps

1. The helper-based naive PIM path is now a credible baseline for future PIM
   architecture work.
2. The next major design step should focus on resident KV placement on DPUs,
   rather than only more host-side invocation cleanup.
3. Future comparisons should clearly separate:
   - helper-based naive PIM
   - resident-KV PIM
   - any deeper PIM-side softmax or `AV` optimization
