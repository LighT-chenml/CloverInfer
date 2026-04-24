# Qwen Real-Model Smoke Notes

Date: 2026-04-24

## Goal

Validate whether the current PD + AF split prototype can run with a real Qwen model and a real dataset sample, instead of only OPT toy models and synthetic prompts.

## What Was Verified

### 1. Qwen-1_8B adapter smoke on GPU

- Machine: `192.168.123.4`
- Device: `cuda` (`Tesla V100-PCIE-16GB`)
- Model: `model/Qwen-1_8B`
- Prompt:

```python
from typing import List


def add(a, b):
    return
```

- Result:
  - model load time: `87.837 s`
  - prefill time: `0.358 s`
  - prompt tokens: `13`
  - returned KV layers: `24`
  - peak CUDA memory: `3560 MB`

This confirms the tracked runtime code can now load Qwen, prefill, collect KV cache, and decode token ids successfully.

### 2. End-to-end real-dataset smoke

- Dataset sample: `dataset/humaneval.jsonl`, first record (`HumanEval/0`)
- Runtime topology for this smoke:
  - prefill: CPU
  - attention: CPU
  - decode dense: GPU
- Model: `model/Qwen-1_8B`
- `max_new_tokens=3`

- Result:
  - request succeeded end-to-end
  - generated text: `'他们的ը册'`
  - TTFT: `15.824 s`
  - latency: `17.579 s`
  - TPOT: `0.878 s`
  - decode steps: `2`
  - decode layers executed: `48`

This validates the full scheduler path:

- `prefill -> init attention cache -> start_token -> prepare_attention -> attention decode -> finish_layer -> sample_next_token -> decode_tokens`

### 3. Three-machine Qwen placement verification with remote CPU attention

- Command:

```bash
python tests/verify_cluster_placement.py \
  --address 192.168.123.4:26379 \
  --model /home/cml/CloverInfer/model/Qwen-1_8B \
  --prompt 'from typing import List\n\n\ndef add(a, b):\n    return' \
  --max-new-tokens 3 \
  --attention-backend cpu
```

- Placement:
  - prefill: `192.168.123.3`, `cuda`
  - attention: `192.168.123.7`, `cpu`
  - decode dense: `192.168.123.4`, `cuda`
- Result:
  - request succeeded end to end on the intended three-machine topology
  - TTFT: about `0.47 s`
  - latency: about `2.24 s`
  - total tokens: `3`

This confirms the real Qwen path is not limited to single-machine smoke. The
three-node PD + AF split can already place actors on the intended machines and
complete generation over Ray.

### 4. Three-machine Qwen smoke with `pim_naive` attention backend

- Command:

```bash
python tests/verify_cluster_placement.py \
  --address 192.168.123.4:26379 \
  --model /home/cml/CloverInfer/model/Qwen-1_8B \
  --prompt 'from typing import List\n\n\ndef add(a, b):\n    return' \
  --max-new-tokens 3 \
  --attention-backend pim_naive
```

- Placement:
  - prefill: `192.168.123.3`, `cuda`
  - attention: `192.168.123.7`, `cpu`, backend=`pim_naive`
  - decode dense: `192.168.123.4`, `cuda`
- Backend debug:
  - UPMEM dot smoke: `PASS`
  - `qk_check_failures = 0`
  - `qk_mixed_count = 96`
  - last mixed-head diff: about `6.3e-4`
- Result:
  - request succeeded end to end
  - TTFT: about `0.45 s`
  - latency: about `5.13 s`
  - attention decode compute: about `2.99 s`
  - total tokens: `3`

This is the first successful real-model three-machine run through the naive PIM
backend path. It is still much slower than the CPU attention baseline, but the
goal of this phase is correctness and integration, not competitiveness.

### 5. Three-machine Qwen real-dataset smoke with `pim_naive`

- Dataset: first three samples from `dataset/humaneval.jsonl`
- Model: `/home/cml/CloverInfer/model/Qwen-1_8B`
- Topology:
  - prefill GPU on `192.168.123.3`
  - decode dense GPU on `192.168.123.4`
  - attention `pim_naive` on `192.168.123.7`
- `max_new_tokens = 3`

- Results:
  - `HumanEval/0`
    - latency: `5.90 s`
    - TTFT: `0.71 s`
    - attention decode compute: `3.42 s`
    - `qk_check_failures = 0`
  - `HumanEval/1`
    - latency: `5.14 s`
    - TTFT: `0.35 s`
    - attention decode compute: `3.33 s`
    - `qk_check_failures = 0`
  - `HumanEval/2`
    - latency: `4.99 s`
    - TTFT: `0.28 s`
    - attention decode compute: `3.28 s`
    - `qk_check_failures = 0`

This confirms the `pim_naive` path is stable across multiple real requests, not
just a single placement-check example.

## Qwen-7B Feasibility Note

Current hardware per GPU node is `V100-16GB`.

- Rough fp16 weight memory for `Qwen-7B`: about `13.04 GB`
- Physical GPU memory on current nodes: about `15.77 GB`

Conclusion:

- Under the current design, prefill node and decode-dense node each need a full model copy.
- `Qwen-7B` is therefore not a practical first real-model baseline on the current 16GB topology.
- Even if weights barely load, there is too little headroom left for runtime buffers, activations, and KV-related overhead.

## Recommended Next Step

Use `Qwen-1_8B` as the first real-model baseline for:

- three-machine functional validation
- real dataset smoke runs
- early PIM attention experiments

Keep `Qwen-7B` as a later target after introducing one of the following:

- lower-memory weight loading
- model sharding / tensor parallel style changes
- smaller dense-side model copy
- larger-memory GPUs

## Current Status

As of `2026-04-24`, the following is already working:

- real-model Qwen prefill/decode on the tracked runtime code
- three-machine PD + AF split placement
- remote CPU attention baseline
- remote `pim_naive` attention baseline
- multi-request real-dataset smoke on the three-machine cluster

The next step is no longer basic functional bring-up. The focus can move to
repeatable benchmark runs and optimization of the naive PIM path.

## Small Reproducible Benchmark Entry

`tests/benchmark_humaneval.py` now supports the three-machine Qwen layout and
both attention backends directly.

Example CPU baseline command:

```bash
python tests/benchmark_humaneval.py \
  --address 192.168.123.4:26379 \
  --model /home/cml/CloverInfer/model/Qwen-1_8B \
  --model-name qwen-1_8b \
  --attention-backend cpu \
  --prefill-resource prefill_gpu \
  --decode-dense-resource decode_dense_gpu \
  --attention-resource attention_pim \
  --use-gpu-for-prefill \
  --use-gpu-for-decode-dense \
  --max-new-tokens 3 \
  --limit 2 \
  --sequential \
  --output /home/cml/CloverInfer/artifacts/humaneval_qwen_cpu_smoke.jsonl
```

Observed summary:

- average latency: about `2.18 s`
- average TTFT: about `0.53 s`
- average TPOT: about `0.83 s`
- average throughput: about `1.42 tok/s`

Example `pim_naive` baseline command:

```bash
python tests/benchmark_humaneval.py \
  --address 192.168.123.4:26379 \
  --model /home/cml/CloverInfer/model/Qwen-1_8B \
  --model-name qwen-1_8b \
  --attention-backend pim_naive \
  --prefill-resource prefill_gpu \
  --decode-dense-resource decode_dense_gpu \
  --attention-resource attention_pim \
  --use-gpu-for-prefill \
  --use-gpu-for-decode-dense \
  --max-new-tokens 3 \
  --limit 2 \
  --sequential \
  --output /home/cml/CloverInfer/artifacts/humaneval_qwen_pim_naive_smoke.jsonl
```

Observed summary:

- average latency: about `5.48 s`
- average TTFT: about `0.53 s`
- average TPOT: about `2.47 s`
- average throughput: about `0.55 tok/s`

These numbers match the earlier manual smoke conclusions: the three-machine
Qwen path is stable, while `pim_naive` is currently correctness-oriented and
substantially slower than the remote CPU attention baseline.

## Unified Baseline Matrix

`tests/benchmark_baselines.py` now provides a single entry point for the main
Qwen baseline comparisons:

- `monolithic_gpu`
- `split_gpu_full_decode`
- `disagg_cpu`
- `disagg_pim_naive`

Example command:

```bash
python tests/benchmark_baselines.py \
  --data dataset/humaneval.jsonl \
  --limit 2 \
  --model /home/cml/CloverInfer/model/Qwen-1_8B \
  --model-name qwen-1_8b \
  --max-new-tokens 3 \
  --baselines monolithic_gpu,split_gpu_full_decode,disagg_cpu,disagg_pim_naive \
  --address 192.168.123.4:26379 \
  --prefill-resource prefill_gpu \
  --decode-dense-resource decode_dense_gpu \
  --attention-resource attention_pim \
  --output /home/cml/CloverInfer/artifacts/baseline_comparison_qwen_v2.jsonl
```

Observed summary on `2026-04-24`:

| Baseline | Avg Latency (s) | Avg TTFT (s) | Avg Throughput (tok/s) | Relative to Monolithic |
| --- | ---: | ---: | ---: | ---: |
| `monolithic_gpu` | `0.242` | `0.185` | `22.59` | `1.00x` |
| `split_gpu_full_decode` | `0.841` | `0.540` | `4.21` | `3.48x` |
| `disagg_cpu` | `2.224` | `0.530` | `1.39` | `9.20x` |
| `disagg_pim_naive` | `5.764` | `0.534` | `0.52` | `23.84x` |

Useful relative comparisons:

- `split_gpu_full_decode` vs `monolithic_gpu`:
  about `3.5x` slower in latency
- `disagg_cpu` vs `split_gpu_full_decode`:
  about `2.6x` slower in latency
- `disagg_pim_naive` vs `disagg_cpu`:
  about `2.6x` slower in latency

Interpretation:

- Prefill/decode disaggregation by itself is not the dominant problem.
- The largest additional cost comes from making attention remote.
- The current naive PIM path is still slower than the remote CPU attention
  reference, so it should be treated as a correctness and challenge baseline,
  not as a performance claim.

## Paper Framing Note

The current results do not support a strong claim that the disaggregated design
is already faster than conventional execution. A better framing is:

1. The framework demonstrates that PD split and AF split are feasible on a real
   model and a real three-machine deployment.
2. The baseline matrix separates three sources of overhead:
   - monolithic execution cost
   - prefill/decode disaggregation cost
   - remote-attention cost
3. The current bottleneck is remote attention execution, especially the naive
   PIM path, which is exactly the systems challenge the later optimized PIM
   design is meant to address.

In other words, the present prototype is already useful for a paper, but the
paper should currently read more like a challenge-driven systems study than a
"we already beat the baseline" paper.
