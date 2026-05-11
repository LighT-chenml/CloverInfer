# HumanEval 8x4 Baseline Summary, 2026-05-11

Command artifact:
`artifacts/baseline_afd_cpu_clover_hybrid_window0_humaneval_8x4_20260511.jsonl`

Workload:
- Dataset: `dataset/humaneval.jsonl`
- Limit: 8 prompts
- Concurrency: 4
- Max new tokens: 4
- Model: `/home/cml/CloverInfer/model/opt-125m`
- Placement: prefill GPU on `prefill_gpu`, dense decode GPU on `decode_dense_gpu`, attention/PIM node on `attention_pim`

Configuration:
- AFD baseline: `AFD -> disagg_afd`
- CPU-Attention baseline: `CPU-Attention -> disagg_cpu`
- CloverInfer: `disagg_cloverinfer`
- CloverInfer short-context hybrid threshold: `--clover-cpu-fast-path-max-context-tokens 256`
- CloverInfer shadow checks/profiling disabled for performance run
- Global decode continuous batch window: `0 ms`

Results:

| Baseline | Output throughput | Avg latency | Avg TPOT | Avg TTFT |
| --- | ---: | ---: | ---: | ---: |
| AFD | 4.901705 tok/s | 3.150179 s | 0.943620 s | 0.319319 s |
| CPU-Attention | 5.230365 tok/s | 2.941481 s | 0.873194 s | 0.321900 s |
| CloverInfer | 5.287313 tok/s | 2.892275 s | 0.860433 s | 0.310974 s |

Conclusion:
- AFD baseline is wired and runnable through `tests/benchmark_baselines.py`.
- CloverInfer now exceeds CPU-Attention on this short-context HumanEval 8x4 run by about 1.09%.
- The winning short-context setting uses the CloverInfer CPU fast path to avoid expensive UPMEM launch overhead for prompts at or below 256 context tokens.
- Larger-margin gains likely require optimizing the true PIM path for longer-context workloads, where PIM compute should amortize launch and host-reduction overhead better.

Validation:
- `python -m compileall -q src/core tests/benchmark_baselines.py tests/test_attention_backend_planner_integration.py`
- Direct execution of all `tests/test_attention_backend_planner_integration.py::test_*` functions
