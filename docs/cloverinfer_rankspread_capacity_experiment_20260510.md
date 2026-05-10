# CloverInfer Rank-Spread Capacity Experiment 2026-05-10

Environment:

- Ray head and prefill GPU: `192.168.123.4`
- Decode dense GPU: `192.168.123.3`
- CloverInfer attention with UPMEM PIM: `192.168.123.7`
- Model: `/home/cml/CloverInfer/model/opt-125m`
- Dataset: `dataset/humaneval.jsonl`
- Decode: `max_new_tokens=4`, `concurrency=4`, `pim_num_dpus=32`, `pim_resident_store_backend=upmem_kvslot`

Recommended command shape:

```bash
CLOVER_BENCHMARK_PY_MODULES=src CLOVER_BENCHMARK_PYTHONPATH= \
RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor \
/home/cml/anaconda3/envs/clover_infer/bin/python tests/benchmark_baselines.py \
  --address 192.168.123.4:26379 \
  --baselines cloverinfer \
  --data dataset/humaneval.jsonl \
  --limit 16 \
  --concurrency 4 \
  --max-new-tokens 4 \
  --model /home/cml/CloverInfer/model/opt-125m \
  --model-name opt-125m \
  --dtype float16 \
  --prefill-resource prefill_gpu \
  --decode-dense-resource decode_dense_gpu \
  --attention-resource attention_pim \
  --pim-num-dpus 32 \
  --pim-resident-store-backend upmem_kvslot \
  --pim-length 128 \
  --pim-block-tokens 256 \
  --pim-dpu-placement-policy rank_spread \
  --decode-continuous-batch-window-ms 20 \
  --decode-continuous-batch-max-size 8 \
  --clover-capacity-aware-batching-enabled \
  --clover-capacity-aware-max-tokens-per-dpu 4096 \
  --clover-capacity-aware-lookahead-window 3 \
  --clover-capacity-aware-time-gap-threshold 0.05
```

Observed results:

| run | req | conc | latency | tpot | req/s | tok/s | wall | ooo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rotated/off | 8 | 4 | 5.979 | 1.900 | 0.644 | 2.578 | 12.414 | 0 |
| rotated/on | 8 | 4 | 5.908 | 1.875 | 0.647 | 2.590 | 12.356 | 3 |
| rank_spread/off | 8 | 4 | 5.640 | 1.792 | 0.684 | 2.735 | 11.701 | 0 |
| rank_spread/on | 8 | 4 | 5.413 | 1.719 | 0.700 | 2.799 | 11.431 | 0 |
| rotated/off | 16 | 4 | 5.004 | 1.597 | 0.781 | 3.124 | 20.489 | 5 |
| rotated/on | 16 | 4 | 5.071 | 1.620 | 0.765 | 3.061 | 20.911 | 0 |
| rank_spread/off | 16 | 4 | 4.816 | 1.534 | 0.808 | 3.234 | 19.790 | 0 |
| rank_spread/on | 16 | 4 | 4.701 | 1.500 | 0.825 | 3.298 | 19.405 | 0 |

Notes:

- `rank_spread` placement is currently the best default for this workload.
- Capacity-aware batching is positive under `rank_spread`: `+2.3% tok/s` at 8 requests and `+2.0% tok/s` at 16 requests.
- The UPMEM helper still has a hard `64 slots/DPU` table limit; at 16 requests, max live slots per DPU reaches 64.
- Global slot spill can eliminate host fallback but was much slower in this workload (`0.974 tok/s` at 16 requests), so it is exposed as an experimental flag but should remain off by default.
- Helper-side `CLOVER_KVSLOT_RANK_SPREAD_ALLOC=1` entered a pathological slow path in the 16-request run and is not recommended yet.
