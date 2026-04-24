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
