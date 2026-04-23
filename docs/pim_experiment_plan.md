# PIM-Based Disaggregated LLM Inference Experiment Plan

This document records the implementation and evaluation plan for a simplified
experiment framework for prefill-decoding disaggregation and attention-FFN
separation, with a PIM-capable attention node.

## Goals

The immediate goal is to rebuild CloverInfer into a reliable experimental
framework rather than a performance-optimized serving system. The first working
version should prioritize correctness, observability, and clear component
boundaries.

Target properties:

- Prefill and decoding run as separate nodes.
- Decoding is split into dense/FFN execution and attention execution.
- The attention node can run without GPU.
- The attention implementation is backend-pluggable: CPU first, PIM later.
- Cross-machine placement is explicit and reproducible.
- Metrics expose time spent in prefill, dense decode, attention, transport, and
  end-to-end generation.
- RDMA is treated as an optimization layer, not a requirement for the initial
  correctness path.

## Cluster Layout

The initial cluster has three machines:

| IP | Role | Hardware | Planned Actor |
| --- | --- | --- | --- |
| `192.168.123.3` | Prefill node | GPU | `PrefillNode` |
| `192.168.123.4` | Decode dense node | GPU | `DecodeDenseNode`, scheduler/head candidate |
| `192.168.123.7` | Attention node | CPU + PIM, no GPU | `AttentionNode` |

All machines will use conda-managed environments. Code will be synchronized via
git. If missing software, dependency conflicts, CUDA/driver issues, SSH issues,
or permission issues appear during setup, stop and report the exact error before
working around it.

## Cross-Machine Development Workflow

The preferred workflow is:

1. Develop and test locally in this repository.
2. Commit changes to git.
3. Pull or clone the same commit on all three machines.
4. Create or update the same conda environment on all machines.
5. Start a Ray cluster with explicit node resources.
6. Launch the experiment from the scheduler/head machine.

Avoid manually editing different versions of the code on different machines.
If direct SSH access is available, it can be used for inspection, environment
setup, and running scripts, but code should still be synchronized through git.

Example Ray resource layout:

```bash
# 192.168.123.4: Ray head and decode dense GPU node
ray start --head --node-ip-address=192.168.123.4 --port=26379 \
  --num-gpus=1 \
  --resources='{"decode_dense_gpu": 1}'

# 192.168.123.3: prefill GPU node
ray start --address=192.168.123.4:26379 --node-ip-address=192.168.123.3 \
  --num-gpus=1 \
  --resources='{"prefill_gpu": 1}'

# 192.168.123.7: CPU/PIM attention node
ray start --address=192.168.123.4:26379 --node-ip-address=192.168.123.7 \
  --num-gpus=0 \
  --resources='{"attention_pim": 1}'
```

Actors should be placed using these resources rather than relying on Ray's
default scheduling.

## Proposed Architecture

The current `AttnNode` and `FFNNode` split should be replaced by a clearer
decode split:

```text
PrefillNode, GPU:
  prompt -> prompt KV cache
  prompt -> first generated token
  ship initial KV to AttentionNode

DecodeDenseNode, GPU:
  token embedding
  positional embedding
  layer norm
  Q/K/V projection
  send q, k_new, v_new to AttentionNode
  receive attention context
  output projection
  FFN
  final norm + lm_head

AttentionNode, CPU/PIM:
  own KV cache
  ingest prefill KV
  append decode K/V
  compute attention(q, KV cache)
  return attention context
```

This makes the PIM research question clearer: the attention node owns the
bandwidth-bound KV-cache attention path, while the GPU dense node owns GEMM-heavy
model components.

## Backend Abstraction

The attention node should use a pluggable backend interface:

```python
class AttentionBackend:
    def init_request(self, request_id, layer_kv): ...
    def decode_layer(self, layer_idx, request_id, q, k_new, v_new, position): ...
    def free_request(self, request_id): ...
```

Initial backends:

- `CpuAttentionBackend`: PyTorch CPU reference implementation. This is the first
  correctness target and the temporary stand-in for PIM.
- `TorchGpuAttentionBackend`: optional local GPU reference/baseline.
- `PimAttentionBackend`: future integration point for the PIM runtime, C++
  extension, or vendor API.

The scheduler and decode pipeline should not depend on the concrete backend.

## Implementation Phases

### Phase 1: Correctness-First Refactor

- Replace current ad hoc node interfaces with typed request and layer messages.
- Implement `PrefillNode`, `DecodeDenseNode`, and `AttentionNode`.
- Implement CPU attention backend on the no-GPU attention node.
- Fix KV cache lifecycle: initialize, append exactly once per token, free.
- Correctly handle positional embeddings and generated-token accounting.
- Use Ray transport only.
- Add a small end-to-end test using a local model such as `opt-125m`.

Exit criteria:

- A prompt can be processed end to end with prefill on one actor, dense decode on
  another actor, and attention on a third actor.
- The CPU attention path produces stable output and does not leak request state.
- The test can run locally in a simulated placement mode.

### Phase 2: Cross-Machine Ray Bring-Up

- Add scripts for conda environment setup and Ray cluster start/stop.
- Add placement checks to verify each actor runs on the expected IP.
- Run the same small end-to-end test across the three machines.
- Record per-stage latency metrics.

Exit criteria:

- `PrefillNode` runs on `192.168.123.3`.
- `DecodeDenseNode` runs on `192.168.123.4`.
- `AttentionNode` runs on `192.168.123.7`.
- Generation completes over Ray RPC.

### Phase 3: Baselines and Instrumentation

- Add monolithic GPU decode baseline.
- Add prefill/decode disaggregated GPU baseline.
- Add GPU dense + remote CPU attention baseline.
- Add structured benchmark output, preferably JSONL.
- Measure TTFT, TPOT, throughput, attention time, dense time, transport time,
  and KV transfer volume.

Exit criteria:

- The framework can run repeatable experiments over sequence length, batch size,
  and generated-token count.
- Results include enough timing breakdown to identify bottlenecks.

### Phase 4: PIM Integration

- Implement `PimAttentionBackend` behind the same backend interface.
- Keep CPU backend as correctness reference.
- Add backend-level correctness comparison for selected requests/layers.
- Compare CPU attention and PIM attention under the same transport and scheduler.

Exit criteria:

- PIM attention can replace CPU attention without changing scheduler logic.
- PIM correctness can be checked against CPU reference for small cases.

### Phase 5: Transport Optimization

- Revisit RDMA only after the Ray RPC path is correct and measured.
- Add a transport abstraction if needed:
  - `RayTransport`
  - `SocketTransport`
  - `RDMATransport`
- Benchmark transport overhead independently from attention backend time.

Exit criteria:

- RDMA or another low-level transport shows measurable benefit over Ray transport
  for the target tensor sizes.

## Baselines

Use the following comparisons to evaluate the PIM-based design:

| Baseline | Layout | Purpose |
| --- | --- | --- |
| A. Monolithic GPU | Prefill + decode entirely on one GPU node | Main correctness/performance reference |
| B. Prefill/decode split GPU | `192.168.123.3` prefill, `192.168.123.4` full decode | Isolate prefill-decoding disaggregation overhead |
| C. GPU dense + remote CPU attention | `192.168.123.4` dense, `192.168.123.7` CPU attention | CPU stand-in for PIM and transport reference |
| D. GPU dense + remote PIM attention | `192.168.123.4` dense, `192.168.123.7` PIM attention | Target design |
| E. Transport ablation | Same compute layout, different transports | Separate transport cost from backend cost |

Expected result: short contexts and small batches may not favor remote
attention/PIM because communication overhead can dominate. The design is more
likely to show benefits for longer contexts, larger KV cache reads, and batched
decode.

## Experiment Matrix

Start small and increase scale only after correctness is stable.

Recommended dimensions:

- Model: start with `opt-125m`; later test Llama/Qwen variants if supported.
- Prompt/context length: `512`, `2048`, `4096`, `8192`, `16384+`.
- Batch size: `1`, `4`, `8`, `16`.
- Generated tokens: fixed `128` or `256`.
- Backend: monolithic GPU, remote CPU attention, remote PIM attention.
- Transport: Ray first; RDMA later.

Key metrics:

- TTFT
- TPOT
- tokens/s
- end-to-end latency
- prefill time
- dense decode time
- attention backend time
- transport time
- KV cache memory footprint
- bytes transferred per token and per request

## Naive PIM Baseline

The first PIM version should be a correctness-first baseline, not an optimized
implementation. It should stay simple enough to serve as a long-term comparison
point for later PIM optimizations.

Definition:

- Keep the current three-node topology unchanged.
- Keep the scheduler, actor placement, and message flow unchanged.
- Replace only the attention backend on `192.168.123.7`.
- Preserve `CpuAttentionBackend` as the correctness oracle.
- Add `PimNaiveAttentionBackend` with the same public interface.

Naive baseline requirements:

- No overlap between transport and PIM compute.
- No decode batching beyond the current correctness path.
- No KV compression, paging, or speculative cache policy.
- No fusion across layers or tokens.
- One synchronous backend call per decode layer per token.
- Prefer the simplest tensor layout that is easy to validate and measure.

Recommended decomposition for the first UPMEM-backed baseline:

- `AttentionNode` still owns KV cache lifecycle.
- Host CPU on `192.168.123.7` is responsible for orchestration, data marshaling,
  and result collection.
- DPU code handles the smallest useful kernel first, even if the host still does
  part of the attention computation.
- If full attention is too much for the first step, use staged milestones:
  - Stage A: DPU computes dot products or partial `QK^T`.
  - Stage B: host CPU performs softmax.
  - Stage C: DPU computes weighted value accumulation.
  - Stage D: revisit whether softmax should also move to PIM.

This staged path is acceptable for the naive baseline as long as the framework
can switch between `cpu` and `pim_naive` under the same scheduler logic.

### UPMEM-Oriented Implementation Sketch

The likely implementation split is:

- Host side:
  - `dpu_alloc`
  - `dpu_load`
  - transfer inputs with host-to-DPU copy APIs
  - `dpu_launch`
  - transfer outputs back
  - maintain DPU group/rank ownership and error handling
- DPU side:
  - C kernel compiled with the UPMEM toolchain
  - explicit WRAM/MRAM layout
  - tasklet-level work partitioning
  - simple reductions and profiling hooks

The framework should isolate this into a backend-specific module so the rest of
the codebase never directly depends on raw UPMEM host/runtime APIs.

### Planned Comparisons Involving Naive PIM

- CPU attention baseline vs naive PIM baseline:
  same transport, same scheduler, same actor layout
- naive PIM baseline vs optimized PIM backend:
  isolates gains from layout, overlap, batching, and cache policy
- monolithic GPU vs naive PIM baseline:
  shows the total cost/benefit of the full design

### Expected Challenges To Measure Explicitly

- Host-to-DPU and DPU-to-host transfer overhead
- KV cache placement and update policy on the attention node
- Whether softmax on host dominates overall layer latency
- Per-layer launch overhead on the DPU runtime
- Head-level or token-level work partition granularity
- Precision and numerical drift relative to the CPU backend
- Capacity limits when context length grows
- Whether long-context decode is needed before PIM benefits appear

These should be reported as measured bottlenecks rather than guessed ones.

## UPMEM Readiness Notes

Before writing the production `PimNaiveAttentionBackend`, confirm:

- exact SDK version and driver version on `192.168.123.7`
- available number of ranks/DPUs
- working host compile path for `libdpu`
- working DPU compile path for the device kernel
- whether the backend will use a C/C++ extension, subprocess wrapper, or a thin
  standalone helper binary

The first milestone does not need to integrate directly with PyTorch custom ops.
A standalone backend wrapper that exchanges plain buffers is acceptable if it is
easier to validate and profile.

## Immediate Refactor Checklist

- Rename or replace `FFNNode` with `DecodeDenseNode`.
- Move QKV projection out of the attention node and into the dense GPU node.
- Make the attention node own KV cache and attention computation only.
- Remove duplicate KV append behavior.
- Add positional embedding handling for decode steps.
- Add request cleanup for all nodes.
- Make model dimensions come from the loaded model config where possible.
- Replace hardcoded topology with explicit cluster config.
- Make RDMA optional and disabled by default.
- Add actor placement checks and structured metrics.

## Operational Notes

- Use conda on all three machines.
- Use git for code synchronization.
- Report missing packages, CUDA mismatch, Ray startup failure, SSH failure, or
  permission issues immediately.
- Do not rely on the current RDMA path until the correctness-first Ray path works.
- Keep PIM integration behind a backend interface so that the rest of the
  framework remains stable while the PIM design evolves.
