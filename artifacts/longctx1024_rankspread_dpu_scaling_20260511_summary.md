# Long Context DPU/Rank-Spread Scaling, 2026-05-11

Workload: OPT-125M, prompt tokens 1024, max_new_tokens 4, concurrency 1 unless noted, true CloverInfer PIM, fp32 resident KV, CPU fast path disabled.

| Run | DPUs | Rank alloc | Init reason | TPS | Latency | TPOT | Attention s | Active DPUs | Active ranks | Blocks | QK rounds | AV rounds | QK batched/fallback | AV batched/fallback | QK max ranks | AV max ranks |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| longctx1024_clover_pim_fp32_limit2_c1_tok4_20260511.jsonl | 32 | 0 | no_active_rank | 0.399787 | 9.795473 | 3.033848 | 6.274431 | 16 | 1/1 | 960 | 531 | 261 | 420/111 | 201/60 | 1 | 1 |
| longctx1024_clover_rankspread_d32_limit1_c1_tok4_20260511.jsonl | 32 | 1 | no_active_rank | 0.552132 | 7.188291 | 2.107185 | 4.146690 | 2 | 1/16 | 121 | 237 | 54 | 126/111 | 54/0 | 1 | 1 |
| longctx1024_clover_rankspread_d64_limit1_c1_tok4_20260511.jsonl | 64 | 1 | no_active_rank | 0.426971 | 9.301782 | 2.808935 | 6.199732 | 4 | 1/16 | 252 | 267 | 198 | 216/51 | 135/63 | 1 | 1 |
| longctx1024_clover_rankspread_d128_limit1_c1_tok4_20260511.jsonl | 128 | 1 | no_active_rank | 0.484878 | 7.941477 | 2.364662 | 4.478971 | 8 | 1/16 | 512 | 411 | 213 | 330/81 | 192/21 | 1 | 1 |
| longctx1024_clover_rankspread_crossrank_d32_limit1_c1_tok4_20260511.jsonl | 32 | 1 | rank_spread_alloc_cross_rank | 0.270450 | 14.702469 | 4.633509 | 10.724271 | 16 | 16/16 | 960 | 837 | 261 | 0/837 | 0/261 | 8 | 16 |

Conclusion:
- Cross-rank allocation is now functional, but the current helper cannot batch multi-rank rounds efficiently; the cross-rank run falls back for all QK/AV rounds and is slower.
- The best run in this quick sweep is the rank-spread allocation with a very narrow single-rank stripe, because it dramatically reduces block/round count. That points the next optimization at coarser grouping/segmentation, not simply more DPU ranks.
